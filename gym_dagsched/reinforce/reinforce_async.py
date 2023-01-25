import sys
from dataclasses import dataclass

sys.path.append('./gym_dagsched/data_generation/tpch/')
import os

import numpy as np
import torch
from torch.distributions import Categorical
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch
from torch_scatter import segment_add_csr

from ..envs.dagsched_env_async_wrapper import DagSchedEnvAsyncWrapper
from ..utils.metrics import avg_job_duration
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.baselines import compute_baselines
from ..utils.pyg import add_adj
from ..utils.diff_returns import DifferentialReturnsCalculator
from ..utils.hidden_prints import HiddenPrints




def train(model,
          optim_class,
          optim_lr,
          n_sequences,
          num_envs,
          discount,
          entropy_weight_init,
          entropy_weight_decay,
          entropy_weight_min,
          num_job_arrivals, 
          num_init_jobs, 
          job_arrival_rate,
          num_workers,
          max_time_mean_init,
          max_time_mean_growth,
          max_time_mean_ceil,
          writer):
    '''trains the model on different job arrival sequences. 
    Multiple episodes are run on each sequence in parallel.
    '''

    # use torch.distributed for IPC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    datagen_state = np.random.RandomState(0)

    diff_return_calc = \
        DifferentialReturnsCalculator(discount)

    procs, conns = \
        setup_workers(num_envs, 
                      model,  
                      optim_class, 
                      optim_lr,
                      datagen_state)

    max_time_mean = max_time_mean_init
    entropy_weight = entropy_weight_init

    for i in range(n_sequences):
        # sample the max wall duration of the current episode
        # max_time = np.random.exponential(max_time_mean)
        max_time = np.inf

        print('training on sequence '
              f'{i+1} with max wall time = ' 
              f'{max_time*1e-3:.1f}s',
              flush=True)

        # send episode data to each of the workers.
        for conn in conns:
            conn.send(
                (num_job_arrivals, 
                 num_init_jobs, 
                 job_arrival_rate, 
                 num_workers, 
                 max_time, 
                 entropy_weight))

        # wait for rewards and wall times
        rewards_list, wall_times_list = \
            list(zip(*[conn.recv() for conn in conns]))

        diff_returns_list = \
            diff_return_calc.calculate(wall_times_list,
                                       rewards_list)

        baselines_list = \
            compute_baselines(wall_times_list,
                              diff_returns_list)

        value_losses = []
        # send advantages
        for (returns, 
             baselines, 
             conn) in zip(diff_returns_list,
                          baselines_list,
                          conns):
            advantages = returns - baselines
            value_losses += [np.sum(advantages**2)]
            conn.send(advantages)

        # wait for model update
        (action_losses,
         entropies,
         avg_job_durations, 
         completed_job_counts) = \
            list(zip(*[conn.recv() for conn in conns]))

        if writer:
            # log episode stats to tensorboard
            episode_stats = {
                'avg job duration': np.mean(avg_job_durations),
                'max wall time': max_time,
                'completed jobs count': np.mean(completed_job_counts),
                'avg reward per sec': \
                    1e5 * diff_return_calc.avg_per_step_reward,
                'avg return': \
                    np.mean([returns[0] for returns in diff_returns_list]),
                'action loss': np.mean(action_losses),
                'entropy': np.mean(entropies),
                'value loss': np.mean(value_losses),
                'episode length': \
                    np.mean([len(rewards) for rewards in rewards_list])
            }

            for name, stat in episode_stats.items():
                writer.add_scalar(name, stat, i)

        # increase the mean episode duration
        max_time_mean = \
            min(max_time_mean + max_time_mean_growth,
                max_time_mean_ceil)

        # decrease the entropy weight
        entropy_weight = \
            max(entropy_weight - entropy_weight_decay, 
                entropy_weight_min)

    cleanup_workers(conns, procs)




def setup_workers(num_envs, 
                  model, 
                  optim_class,
                  optim_lr,
                  datagen_state):
    procs = []
    conns = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        proc = Process(target=episode_runner, 
                       args=(rank,
                             num_envs,
                             conn_sub,
                             model,
                             optim_class,
                             optim_lr,
                             datagen_state))
        procs += [proc]
        proc.start()

    return procs, conns




def cleanup_workers(conns, procs):
    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]




def setup_worker(rank, world_size):
    sys.stdout = open(f'log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    dist.init_process_group(dist.Backend.GLOO, 
                            rank=rank, 
                            world_size=world_size)

    # IMPORTANT! Each worker needs to produce 
    # unique rollouts, which are determined
    # by the rng seeds.
    torch.manual_seed(rank)
    np.random.seed(rank)




def cleanup_worker():
    dist.destroy_process_group()




def episode_runner(rank,
                   num_envs,
                   conn,
                   model, 
                   optim_class,
                   optim_lr,
                   datagen_state):
    '''worker target which runs episodes 
    and trains the model by communicating 
    with the main process and other workers
    '''
    setup_worker(rank, num_envs)

    env = DagSchedEnvAsyncWrapper(rank, datagen_state)

    model = DDP(model.to(device), 
                device_ids=[device])

    optim = optim_class(model.parameters(),
                        lr=optim_lr)
    
    i = 0
    while data := conn.recv():
        i += 1
        # receive episode data from parent process
        run_iteration(i, 
                      rank, 
                      num_envs, 
                      env, 
                      model, 
                      optim, 
                      conn,
                      data)

    cleanup_worker()




def run_iteration(i, 
                  rank, 
                  num_envs, 
                  env, 
                  model, 
                  optim, 
                  conn, 
                  data):
    (num_job_arrivals, 
     num_init_jobs, 
     job_arrival_rate, 
     n_workers,
     max_wall_time, 
     entropy_weight) = data

    prof = Profiler()
    # prof = None

    if prof:
        prof.enable()

    with HiddenPrints():
        experience = \
            run_episode(env,
                        num_job_arrivals, 
                        num_init_jobs, 
                        job_arrival_rate, 
                        n_workers,
                        max_wall_time,
                        model)

    wall_times, rewards = \
        list(zip(*[(exp.wall_time, exp.reward) 
                    for exp in experience]))

    # notify main proc that returns and wall times are ready
    conn.send((np.array(rewards), 
                np.array(wall_times)))

    # wait for main proc to compute advantages
    advantages = conn.recv()
    advantages = torch.from_numpy(advantages).float()

    action_loss, entropy_loss = \
        learn_from_experience(model,
                              optim,
                              experience,
                              advantages,
                              entropy_weight,
                              num_envs)

    if rank == 0 and i % 10 == 0:
        torch.save(model.module.state_dict(), 'model.pt')

    if prof:
        prof.disable()

    # send episode stats
    conn.send((
        action_loss, 
        entropy_loss / len(experience),
        avg_job_duration(env) * 1e-3,
        env.n_completed_jobs
    ))




@dataclass
class Experience:
    dag_batch: object
    valid_ops_mask: object
    valid_prlsm_lim_mask: object
    num_jobs: int
    num_source_workers: int
    wall_time: float
    action: object
    reward: float




def run_episode(env,
                num_job_arrivals,
                num_init_jobs,
                job_arrival_rate,
                num_workers,
                max_wall_time,
                model):  

    obs = env.reset(num_job_arrivals, 
                    num_init_jobs, 
                    job_arrival_rate, 
                    num_workers,
                    max_wall_time)

    done = False

    # maintain a cached dag batch,
    # since graph structure doesn't
    # always change (i.e. number of
    # nodes may stay the same). Node
    # features always get updated.
    dag_batch_device = None
    job_ptr = None

    # save experience from each step
    # of the episode, to later be used
    # in leaning
    experience = []
    
    while not done:
        # unpack the current observation
        (did_update_dag_batch,
         dag_batch,
         valid_ops_mask,
         valid_prlsm_lim_mask,
         active_job_ids, 
         num_source_workers,
         _) = obs

        if did_update_dag_batch:
            # the whole dag batch object was
            # updated because the number of
            # nodes changed, so send the new
            # object to the GPU
            job_ptr = dag_batch.ptr.numpy()
            dag_batch_clone = dag_batch.clone()
            dag_batch_clone.edge_index = None
            dag_batch_device = dag_batch_clone \
                .to(device, non_blocking=True)
        else:
            # only send new node features to
            # the GPU
            dag_batch_device.x = dag_batch.x \
                .to(device, non_blocking=True)
        
        
        node_scores, dag_scores = \
            invoke_agent(model,
                        dag_batch_device,
                        valid_ops_mask,
                        valid_prlsm_lim_mask)

        raw_action, env_action = \
            sample_action(node_scores, 
                          dag_scores,
                          job_ptr,
                          active_job_ids)

        obs, reward, done = env.step(env_action)

        *_, wall_time = obs

        experience += \
            [Experience(dag_batch.clone(),
                        valid_ops_mask,
                        valid_prlsm_lim_mask,
                        len(active_job_ids),
                        num_source_workers,
                        wall_time,
                        raw_action,
                        reward)]

    return experience




def invoke_agent(model,
                 dag_batch_device,
                 valid_ops_mask,
                 valid_prlsm_lim_mask):
           
    with torch.no_grad():
        # no computational graphs needed during 
        # the episode, only model outputs.
        node_scores, dag_scores = \
            model(dag_batch_device)

    node_scores = node_scores.cpu()
    node_scores[~valid_ops_mask] = float('-inf')

    dag_scores = dag_scores.cpu()
    valid_prlsm_lim_mask = \
        torch.from_numpy(valid_prlsm_lim_mask)
    dag_scores.masked_fill_(~valid_prlsm_lim_mask, 
                            float('-inf'))

    return node_scores, dag_scores




def sample_action(node_scores, 
                  dag_scores, 
                  job_ptr,
                  active_job_ids):
    # select the next operation to schedule
    op_sample = \
        Categorical(logits=node_scores) \
            .sample()

    op_env, job_idx = \
        translate_op(op_sample.item(), 
                     job_ptr,
                     active_job_ids)

    # select the number of workers to schedule
    num_workers_sample = \
        Categorical(logits=dag_scores[job_idx]) \
            .sample()
    num_workers_env = 1 + num_workers_sample.item()

    # action that's recorded in experience, 
    # later used during training
    raw_action = (op_sample, 
                  job_idx, 
                  num_workers_sample)

    # action that gets sent to the env
    env_action = (op_env, num_workers_env)

    return raw_action, env_action




def translate_op(op, job_ptr, active_jobs_ids):
    '''Returns:
    - `op`: translation of the policy sample so that 
    the environment can find the corresponding operation
    - `job_idx`: index of the job that the selected op 
    belongs to
    '''
    job_idx = (op >= job_ptr).sum() - 1

    job_id = active_jobs_ids[job_idx]
    active_op_idx = op - job_ptr[job_idx]
    
    op = (job_id, active_op_idx)

    return op, job_idx




def learn_from_experience(model,
                          optim,
                          experience,
                          advantages,
                          entropy_weight,
                          num_envs):
    (total_loss,
     action_loss,
     entropy_loss) = \
        compute_loss(model,
                     experience,
                     advantages,
                     entropy_weight)
    
    # compute gradients
    optim.zero_grad()
    total_loss.backward()

    # we want the sum of grads over all the
    # workers, but DDP gives average, so
    # scale the grads back
    for param in model.parameters():
        param.grad.mul_(num_envs)

    # update model parameters
    optim.step()

    return action_loss.item(), \
           entropy_loss.item()



def compute_loss(model,
                 experience,
                 advantages,
                 entropy_weight):
    (dag_batch_list, 
     num_dags_per_obs,
     valid_ops_mask_list, 
     valid_prlsm_lim_mask_list,
     actions) = \
        list(zip(*[(exp.dag_batch, 
                    exp.num_jobs,
                    exp.valid_ops_mask,
                    exp.valid_prlsm_lim_mask,
                    exp.action)
                   for exp in experience]))
                   
    nested_dag_batch, num_nodes_per_dag = \
        construct_nested_dag_batch(dag_batch_list, 
                                   num_dags_per_obs)

    # re-feed all the inputs from the entire
    # episode back through the model, this time
    # recording a computational graph
    model_outputs = \
        model(nested_dag_batch,
              torch.tensor(num_dags_per_obs))

    # move model outputs to CPU
    (node_scores_batch, 
     dag_scores_batch, 
     num_nodes_per_obs, 
     obs_indptr) = \
        [t.cpu() for t in model_outputs]

    # calculate attributes of the actions
    # which will be used to construct loss
    action_lgprobs, action_entropies = \
        action_attributes(node_scores_batch, 
                          dag_scores_batch,
                          num_nodes_per_dag,
                          num_nodes_per_obs,
                          obs_indptr,
                          valid_ops_mask_list,
                          valid_prlsm_lim_mask_list,
                          actions)

    # compute loss
    action_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    total_loss = action_loss + \
                entropy_weight * entropy_loss

    return total_loss, \
           action_loss, \
           entropy_loss



def construct_nested_dag_batch(dag_batch_list, 
                               num_dags_per_obs):
    '''extracts the inputs to the model from each
    step of the episode, then stacks them into a
    large batch.
    '''
    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    num_nodes_per_dag = nested_dag_batch.batch.bincount()
    ptr = num_nodes_per_dag.cumsum(dim=0)
    nested_dag_batch.ptr = \
        torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = sum(num_dags_per_obs)
    add_adj(nested_dag_batch)
    nested_dag_batch.to(device)

    return nested_dag_batch, \
           num_nodes_per_dag




def action_attributes(node_scores_batch, 
                      dag_scores_batch,
                      num_nodes_per_dag,
                      num_nodes_per_obs,
                      obs_indptr,
                      valid_ops_mask_list,
                      valid_prlsm_lim_mask_list,
                      actions):

    node_scores_batch, dag_scores_batch = \
        mask_outputs(node_scores_batch, 
                     dag_scores_batch,
                     valid_ops_mask_list,
                     valid_prlsm_lim_mask_list)

    (node_selection_batch, 
     dag_idx_batch, 
     dag_selection_batch) = \
        [torch.tensor(lst) for lst in zip(*actions)]

    (all_node_probs, 
     node_lgprob_batch, 
     node_entropy_batch) = \
        node_action_attributes_batch(node_scores_batch,
                                     node_selection_batch,
                                     num_nodes_per_obs)

    dag_probs = compute_dag_probs(all_node_probs, 
                                  num_nodes_per_dag)
    
    dag_idx_batch += obs_indptr[:-1]
    dag_lgprob_batch, dag_entropy_batch = \
        dag_action_attributes(dag_scores_batch, 
                              dag_idx_batch, 
                              dag_selection_batch,
                              dag_probs,
                              obs_indptr)

    # normalize entropy
    num_workers = dag_scores_batch.shape[1]
    entropy_scale = get_entropy_scale(num_workers, 
                                      num_nodes_per_obs)

    action_lgprob_batch = \
        node_lgprob_batch + dag_lgprob_batch

    action_entropy_batch = \
        entropy_scale * (node_entropy_batch + \
                         dag_entropy_batch)

    return action_lgprob_batch, \
           action_entropy_batch




def mask_outputs(node_scores_batch, 
                 dag_scores_batch,
                 valid_ops_mask_list,
                 valid_prlsm_lim_mask_list):
    # mask node scores
    valid_ops_mask_batch = np.concatenate(valid_ops_mask_list)
    node_scores_batch[~valid_ops_mask_batch] = float('-inf')

    # mask dag scores
    valid_prlsm_lim_mask_batch = \
        torch.from_numpy(np.vstack(valid_prlsm_lim_mask_list))
    dag_scores_batch.masked_fill_(~valid_prlsm_lim_mask_batch,
                                  float('-inf'))

    return node_scores_batch, \
           dag_scores_batch




def compute_dag_probs(all_node_probs, num_nodes_per_dag):
    '''for each dag, compute the probability of it
    being selected by summing over the probabilities
    of each of its nodes being selected
    '''
    dag_indptr = num_nodes_per_dag.cumsum(0)
    dag_indptr = torch.cat([torch.tensor([0]), dag_indptr], 0)
    dag_probs = segment_add_csr(all_node_probs, dag_indptr)
    return dag_probs




def get_entropy_scale(num_workers, num_nodes_per_obs):
    entropy_norm = torch.log(num_workers * num_nodes_per_obs)
    entropy_scale = 1 / torch.max(torch.tensor(1), entropy_norm)
    return entropy_scale




def node_action_attributes_batch(node_scores_batch,
                                 node_selection_batch,
                                 num_nodes_per_obs):
    '''splits the node score/selection batches into subbatches 
    (see subroutine below), then for each subbatch, comptues 
    attributes (action log-probability and entropy) using 
    vectorized computations. Finally, merges the attributes 
    from the subbatches together. This is faster than 
    either 
    - separately computing attributes for each sample in the
      batch, because vectorized computations are not utilized
      at all, or
    - stacking the whole batch together with padding and doing 
      one large vectorized computation, because the backward 
      pass becomes very expensive
    '''

    (node_scores_subbatches, 
     node_selection_subbatches, 
     subbatch_node_counts) = \
        split_node_experience_into_subbatches(node_scores_batch, 
                                              node_selection_batch, 
                                              num_nodes_per_obs)

    # for each subbatch, compute the node
    # action attributes, vectorized
    node_probs_list = []
    node_lgprob_list = []
    node_entropy_list = []

    for (node_scores_subbatch, 
         node_selection_subbatch,
         node_count) in zip(node_scores_subbatches,
                          node_selection_subbatches,
                          subbatch_node_counts):
        node_scores_subbatch = \
            node_scores_subbatch.view(-1, node_count)

        node_probs, node_lgprob_subbatch, node_entropy_subbatch = \
            node_action_attributes(node_scores_subbatch,
                                   node_selection_subbatch)

        node_probs_list += [torch.flatten(node_probs)]
        node_lgprob_list += [node_lgprob_subbatch]
        node_entropy_list += [node_entropy_subbatch]

    ## concatenate the subbatch attributes together

    # for each node ever seen, records the probability
    # that that node is selected out of all the
    # nodes within its observation
    all_node_probs = torch.cat(node_probs_list)

    # for each observation, records the log probability
    # of its node selection
    node_lgprob_batch = torch.cat(node_lgprob_list)

    # for each observation, records its node entropy
    node_entropy_batch = torch.cat(node_entropy_list)

    return all_node_probs, \
           node_lgprob_batch, \
           node_entropy_batch




def split_node_experience_into_subbatches(node_scores_batch, 
                                          node_selection_batch, 
                                          num_nodes_per_obs):
    '''splits the node score/selection batches into
    subbatches, where each each sample within a subbatch
    has the same node count.
    '''
    batch_size = len(num_nodes_per_obs)

    # find indices where op count changes
    op_count_change_mask = \
        num_nodes_per_obs[:-1] != num_nodes_per_obs[1:]
    ptr = 1 + op_count_change_mask.nonzero().squeeze()
    if ptr.shape == torch.Size():
        # ptr is zero-dimentional; not allowed in torch.cat
        ptr = ptr.unsqueeze(0)
    ptr = torch.cat([torch.tensor([0]), 
                     ptr, 
                     torch.tensor([batch_size])])

    # unique op count within each subbatch
    subbatch_node_counts = num_nodes_per_obs[ptr[:-1]]

    # number of samples in each subbatch
    subbatch_sizes = ptr[1:] - ptr[:-1]

    # split node scores into subbatches
    node_scores_split = \
        torch.split(node_scores_batch, 
                    list(subbatch_node_counts * subbatch_sizes))

    # split node selections into subbatches
    node_selection_split = \
        torch.split(node_selection_batch, 
                    list(subbatch_sizes))

    return node_scores_split, \
           node_selection_split, \
           subbatch_node_counts




def node_action_attributes(node_scores, node_selection):
    c = Categorical(logits=node_scores)
    node_lgprob = c.log_prob(node_selection)
    node_entropy = c.entropy()
    return c.probs, node_lgprob, node_entropy




def dag_action_attributes(dag_scores_batch, 
                          dag_idx_batch, 
                          dag_selection_batch,
                          dag_probs,
                          obs_indptr):
    dag_lgprob_batch = \
        Categorical(logits=dag_scores_batch[dag_idx_batch]) \
            .log_prob(dag_selection_batch)

    # can't have rows where all the entries are
    # -inf when computing entropy, so for all such 
    # rows, set the first entry to be 0. then the 
    # entropy for these rows becomes 0.
    inf_counts = torch.isinf(dag_scores_batch).sum(1)
    allinf_rows = (inf_counts == dag_scores_batch.shape[1])
    dag_scores_batch[allinf_rows, 0] = 0

    # compute expected entropy over dags for each obs.
    # each dag is weighted by the probability of it 
    # being selected. sum is segmented over observations.
    entropy_per_dag = \
        Categorical(logits=dag_scores_batch).entropy()
    dag_entropy_batch = \
        segment_add_csr(dag_probs * entropy_per_dag, 
                        obs_indptr)
    
    return dag_lgprob_batch, dag_entropy_batch