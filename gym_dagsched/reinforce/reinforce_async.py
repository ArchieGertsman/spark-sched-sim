import sys
from time import sleep
from attr import dataclass
from torch_scatter import segment_add_csr


sys.path.append('./gym_dagsched/data_generation/tpch/')
import os

import torch
from torch.distributions import Categorical
import numpy as np
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

from ..envs.dagsched_env_async_wrapper import DagSchedEnvAsyncWrapper
from ..utils.metrics import avg_job_duration
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.baselines import compute_baselines
from ..utils.pyg import add_adj
from ..utils.diff_returns import DifferentialReturnsCalculator




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

    datagen_state = np.random.RandomState(69)#42)

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
        max_time = 2e6

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

        # send advantages
        for (returns, 
             baselines, 
             conn) in zip(diff_returns_list,
                          baselines_list,
                          conns):
            advantages = returns - baselines
            conn.send(advantages)

        # wait for model update
        (actor_losses, 
         advantage_losses, 
         entropy_losses,
         avg_job_durations, 
         completed_job_counts) = \
            list(zip(*[conn.recv() for conn in conns]))

        if writer:
            # log episode stats to tensorboard
            episode_stats = {
                'actor loss': np.mean(actor_losses),
                'advantage loss': np.mean(advantage_losses),
                'entropy_loss': np.mean(entropy_losses),
                'avg job duration': np.mean(avg_job_durations),
                'completed jobs count': np.mean(completed_job_counts),
                'avg reward per sec': \
                    1e5 * diff_return_calc.avg_per_step_reward,
                'avg return': np.mean([returns[0] 
                                       for returns in diff_returns_list])
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

    # torch.autograd.set_detect_anomaly(True)




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


    experience = \
        run_episode(env,
                    num_job_arrivals, 
                    num_init_jobs, 
                    job_arrival_rate, 
                    n_workers,
                    max_wall_time,
                    model)

    # dist.barrier()
    # print('SLEEPING', flush=True)
    # sleep(10)

    wall_times, rewards = \
        list(zip(*[(exp.wall_time, exp.reward) 
                    for exp in experience]))

    # notify main proc that returns and wall times are ready
    conn.send((np.array(rewards), 
                np.array(wall_times)))

    # wait for main proc to compute advantages
    advantages = conn.recv()
    advantages = torch.from_numpy(advantages).float()

    actor_loss, advantage_loss, entropy_loss = \
        learn_from_experience(model,
                                optim,
                                experience,
                                advantages,
                                entropy_weight,
                                num_envs)

    if rank == 0 and i % 10 == 0:
        torch.save(model.state_dict(), 'model.pt')

    if prof:
        prof.disable()

    # send episode stats
    conn.send((
        actor_loss, 
        advantage_loss, 
        entropy_loss,
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
            dag_batch_device = dag_batch.clone() \
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
    invalid_row, invalid_col = \
        (~valid_prlsm_lim_mask).nonzero()
    dag_scores[invalid_row, invalid_col] = float('-inf')

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
    (actor_loss,
     advantage_loss,
     entropy_loss) = \
        compute_loss(model,
                     experience,
                     advantages,
                     entropy_weight)

    dist.barrier()
    
    # compute gradients
    optim.zero_grad()
    actor_loss.backward()

    # we want the sum of grads over all the
    # workers, but DDP gives average, so
    # scale the grads back
    torch.cuda.synchronize()
    for param in model.parameters():
        param.grad.mul_(num_envs)

    # update model parameters
    optim.step()

    optim.zero_grad(set_to_none=True)

    return actor_loss.item(), \
           advantage_loss.item(), \
           entropy_loss.item()



def compute_loss(model,
                 experience,
                 advantages,
                 entropy_weight):
    batched_model_inputs = \
        extract_batched_model_inputs(experience)

    # re-feed all the inputs from the entire
    # episode back through the model, this time
    # recording a computational graph
    model_outputs = model(*batched_model_inputs)
    torch.cuda.synchronize()

    # move model outputs to CPU
    (node_scores_batch, 
     dag_scores_batch, 
     op_counts, 
     obs_indptr) = \
        [t.cpu() for t in model_outputs]

    # calculate attributes of the actions
    # which will be used to construct loss
    action_lgprobs, action_entropies = \
        action_attributes(node_scores_batch, 
                          dag_scores_batch,
                          op_counts,
                          obs_indptr,
                          experience)

    # compute loss
    advantage_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    actor_loss = \
        advantage_loss + entropy_weight * entropy_loss

    return actor_loss, \
           advantage_loss, \
           entropy_loss



def extract_batched_model_inputs(experience):
    '''extracts the inputs to the model from each
    step of the episode, then stacks them into a
    large batch.
    '''
    dag_batch_list, job_counts = \
        list(zip(*[(exp.dag_batch, exp.num_jobs)
                   for exp in experience]))

    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    ptr = nested_dag_batch.batch.bincount().cumsum(dim=0)
    nested_dag_batch.ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = sum(job_counts)
    add_adj(nested_dag_batch)
    nested_dag_batch.to(device)

    job_counts = torch.tensor(job_counts)

    return nested_dag_batch, job_counts




def action_attributes(node_scores_batch, 
                      dag_scores_batch,
                      op_counts,
                      obs_indptr,
                      experience):

    valid_ops_mask_list, valid_prlsm_lim_mask_list = \
        list(zip(*[(exp.valid_ops_mask,
                    exp.valid_prlsm_lim_mask)
                   for exp in experience]))

    # mask node scores
    valid_ops_mask_batch = np.concatenate(valid_ops_mask_list)
    node_scores_batch[(~valid_ops_mask_batch).nonzero()] = float('-inf')

    # mask dag scores
    valid_prlsm_lim_mask_batch = \
        np.vstack(valid_prlsm_lim_mask_list)
    invalid_row, invalid_col = \
        (~valid_prlsm_lim_mask_batch).nonzero()
    dag_scores_batch[invalid_row, invalid_col] = float('-inf')

    # batch the actions
    actions = list(zip(*[exp.action for exp in experience]))

    (node_selection_batch, 
     dag_idx_batch, 
     dag_selection_batch) = \
        [torch.tensor(lst) for lst in actions]


    node_lgprob_batch, node_entropy_batch = \
        node_action_attributes_batch(node_scores_batch,
                                     node_selection_batch,
                                     op_counts)
    
    dag_idx_batch += obs_indptr[:-1]
    dag_lgprob_batch, dag_entropy_batch = \
        dag_action_attributes(dag_scores_batch, 
                              dag_idx_batch, 
                              dag_selection_batch,
                              obs_indptr)

    # normalize entropy
    num_workers = dag_scores_batch.shape[1]
    entropy_norm = torch.log(num_workers * op_counts)
    entropy_scale = 1 / torch.max(torch.tensor(1), entropy_norm)

    action_lgprob_batch = node_lgprob_batch + dag_lgprob_batch
    action_entropy_batch = \
        entropy_scale * (node_entropy_batch + dag_entropy_batch)

    return action_lgprob_batch, \
           action_entropy_batch




def node_action_attributes_batch(node_scores_batch,
                                 node_selection_batch,
                                 node_counts):
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
                                              node_counts)

    # for each subbatch, compute the node
    # action attributes, vectorized
    node_lgprob_list = []
    node_entropy_list = []

    for (node_scores_subbatch, 
         node_selection_subbatch,
         op_count) in zip(node_scores_subbatches,
                          node_selection_subbatches,
                          subbatch_node_counts):
        node_scores_subbatch = \
            node_scores_subbatch.view(-1, op_count)

        node_lgprob_subbatch, node_entropy_subbatch = \
            node_action_attributes(node_scores_subbatch,
                                   node_selection_subbatch)

        node_lgprob_list += [node_lgprob_subbatch]
        node_entropy_list += [node_entropy_subbatch]

    # concatenate the subbatch attributes together
    return torch.cat(node_lgprob_list), \
           torch.cat(node_entropy_list)




def split_node_experience_into_subbatches(node_scores_batch, 
                                          node_selection_batch, 
                                          node_counts):
    '''splits the node score/selection batches into
    subbatches, where each each sample within a subbatch
    has the same node count.
    '''
    # find indices where op count changes
    op_count_change_mask = node_counts[:-1] != node_counts[1:]
    ptr = 1 + op_count_change_mask.nonzero().squeeze()
    ptr = torch.cat([torch.tensor([0]), 
                     ptr, 
                     torch.tensor([len(node_counts)])])

    # unique op count within each subbatch
    subbatch_node_counts = node_counts[ptr[:-1]]

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
    c_node = Categorical(logits=node_scores)
    node_lgprob = c_node.log_prob(node_selection)
    node_entropy = c_node.entropy()
    return node_lgprob, node_entropy




def dag_action_attributes(dag_scores, 
                          dag_idx, 
                          dag_selection,
                          obs_indptr):
    dag_lgprob = \
        Categorical(logits=dag_scores[dag_idx]) \
            .log_prob(dag_selection)

    # can't have rows where all the entries are
    # -inf when computing entropy, so for all such 
    # rows, set the first entry to be 0. then the 
    # entropy for these rows becomes 0.
    inf_counts = torch.isinf(dag_scores).sum(1)
    allinf_rows = (inf_counts == dag_scores.shape[1])
    dag_scores[allinf_rows, 0] = 0

    dag_entropy = Categorical(logits=dag_scores) \
                    .entropy()

    dag_entropy = segment_add_csr(dag_entropy, 
                                  obs_indptr)
    
    return dag_lgprob, dag_entropy