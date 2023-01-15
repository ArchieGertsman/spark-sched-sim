import sys
from attr import dataclass


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
          n_workers,
          initial_mean_ep_len,
          ep_len_growth,
          min_ep_len,
          writer):
    '''trains the model on different job arrival sequences. 
    Multiple episodes are run on each sequence in parallel.
    '''

    # use torch.distributed for IPC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    datagen_state = np.random.RandomState(69)

    diff_return_calc = \
        DifferentialReturnsCalculator(discount)

    procs, conns = \
        setup_workers(num_envs, 
                      model,  
                      optim_class, 
                      optim_lr,
                      datagen_state)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    for epoch in range(n_sequences):
        # sample the length of the current episode
        # ep_len = np.random.geometric(1/mean_ep_len)
        # ep_len = max(ep_len, min_ep_len)
        # ep_len = min(ep_len, 4500)
        max_ep_len = mean_ep_len

        print('beginning training on sequence',
              epoch+1, 'with ep_len =', max_ep_len, 
              flush=True)

        # send episode data to each of the workers.
        for conn in conns:
            conn.send(
                (num_job_arrivals, 
                 num_init_jobs, 
                 job_arrival_rate, 
                 n_workers, 
                 max_ep_len, 
                 entropy_weight))

        # wait for rewards and wall times
        rewards_list, wall_times_list = \
            list(zip(*[conn.recv() for conn in conns]))

        diff_returns_list, avg_per_step_reward = \
            diff_return_calc.calculate(wall_times_list, 
                                       rewards_list)

        baselines_list = \
            compute_baselines(diff_returns_list, 
                              wall_times_list)

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
            episode_stats = {
                'actor loss': np.mean(actor_losses),
                'advantage loss': np.mean(advantage_losses),
                'entropy_loss': np.mean(entropy_losses),
                'avg job duration': np.mean(avg_job_durations),
                'completed jobs count': np.mean(completed_job_counts),
                'avg reward per sec': avg_per_step_reward * 1e-5,
                'avg return': np.mean([returns[0] 
                                       for returns in diff_returns_list])
            }

            for name, stat in episode_stats.items():
                writer.add_scalar(name, stat, epoch)

        # increase the mean episode length
        mean_ep_len += ep_len_growth

        # decrease the entropy weight
        entropy_weight = max(
            entropy_weight - entropy_weight_decay, 
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

    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    torch.cuda.set_per_process_memory_fraction(1/world_size, device=device)

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
    
    while data := conn.recv():
        # receive episode data from parent process
        (num_job_arrivals, 
         num_init_jobs, 
         job_arrival_rate, 
         n_workers,
         max_ep_len, 
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
                        max_ep_len,
                        model)

        wall_times, rewards = \
            list(zip(*[
                (exp.wall_time, exp.reward) 
                for exp in experience]))

        # notify main proc that returns and wall times are ready
        conn.send((np.array(rewards), 
                   np.array((0,)+wall_times)))

        # wait for main proc to compute advantages
        advantages = conn.recv()
        advantages = torch.from_numpy(advantages)

        actor_loss, advantage_loss, entropy_loss = \
            learn_from_experience(model,
                                  optim,
                                  experience,
                                  advantages,
                                  entropy_weight,
                                  num_envs)

        # if rank == 0 and i % 10 == 0:
        #     torch.save(model.state_dict(), 'model.pt')
        
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

    cleanup_worker()




def learn_from_experience(model,
                          optim,
                          experience,
                          advantages,
                          entropy_weight,
                          num_envs):
    batched_model_inputs = \
        extract_batched_model_inputs(experience)

    # re-feed all the inputs from the entire
    # episode back through the model, this time
    # recording a computational graph
    node_scores_batch, dag_scores_batch = \
        model(*batched_model_inputs)

    # calculate attributes of the actions
    # which will be used to construct loss
    action_lgprobs, action_entropies = \
        action_attributes(node_scores_batch, 
                          dag_scores_batch, 
                          experience)

    # compute loss
    advantage_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    actor_loss = \
        advantage_loss + entropy_weight * entropy_loss
    
    # compute gradients
    optim.zero_grad()
    actor_loss.backward()
    torch.cuda.synchronize()

    # we want the sum of grads over all the
    # workers, but DDP gives average, so
    # scale the grads back
    for param in model.parameters():
        param.grad.mul_(num_envs)

    # update model parameters
    optim.step()

    return actor_loss.item(), \
           advantage_loss.item(), \
           entropy_loss.item()




def extract_batched_model_inputs(experience):
    '''extracts the inputs to the model at each
    step of the episode from the experience list,
    and then stacks them into a large batch.
    '''
    dag_batch_list, job_counts, worker_counts = \
        list(zip(*[
            (exp.dag_batch,
             exp.num_jobs,
             exp.num_source_workers)
            for exp in experience
        ]))

    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    ptr = nested_dag_batch.batch.bincount().cumsum(dim=0)
    nested_dag_batch.ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
    add_adj(nested_dag_batch)
    nested_dag_batch.to(device)

    worker_counts = torch.tensor(worker_counts)

    job_counts = torch.tensor(job_counts)

    return nested_dag_batch, worker_counts, job_counts




def action_attributes(node_scores_batch, 
                      dag_scores_batch, 
                      experience):
    action_lgprobs = []
    action_entropies = []

    for node_scores, dag_scores, exp in zip(node_scores_batch,
                                            dag_scores_batch,
                                            experience):
        dag_scores = \
            dag_scores.view(exp.num_jobs, 
                            exp.num_source_workers)

        node_selection, dag_idx, dag_selection = exp.action

        node_scores = node_scores.cpu()
        node_scores[~exp.valid_ops_mask] = float('-inf')
        node_lgprob, node_entropy = \
            node_action_attributes(node_scores, 
                                   node_selection)

        dag_scores = dag_scores.cpu()
        dag_lgprob, dag_entropy = \
            dag_action_attributes(dag_scores, 
                                  dag_idx, 
                                  dag_selection)

        num_nodes = node_scores.numel()
        entropy_norm = np.log(exp.num_source_workers * num_nodes)
        entropy_scale = 1 / max(1, entropy_norm)

        action_lgprob = node_lgprob + dag_lgprob
        action_entropy = entropy_scale * (node_entropy + dag_entropy)

        action_lgprobs += [action_lgprob]
        action_entropies += [action_entropy]

    return torch.stack(action_lgprobs), \
           torch.stack(action_entropies)




def node_action_attributes(node_scores, node_selection):
    c_node = Categorical(logits=node_scores)
    node_lgprob = c_node.log_prob(node_selection)
    node_entropy = c_node.entropy()
    return node_lgprob, node_entropy




def dag_action_attributes(dag_scores, 
                          dag_idx, 
                          dag_selection):
    dag_lgprob = \
        Categorical(logits=dag_scores[dag_idx]) \
            .log_prob(dag_selection)

    dag_entropy = \
        Categorical(logits=dag_scores) \
            .entropy().sum()
    
    return dag_lgprob, dag_entropy




@dataclass
class Experience:
    dag_batch: object
    valid_ops_mask: object
    num_jobs: int
    num_source_workers: int
    wall_time: float
    action: object
    reward: float




def run_episode(
    env,
    num_job_arrivals,
    num_init_jobs,
    job_arrival_rate,
    num_workers,
    ep_len,
    model
):  
    obs = env.reset(num_job_arrivals, 
                    num_init_jobs, 
                    job_arrival_rate, 
                    num_workers)
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
    
    for _ in range(ep_len):
        if done:
            break

        # unpack the current observation
        (did_update_dag_batch,
         dag_batch,
         valid_ops_mask, 
         active_job_ids, 
         num_source_workers,
         wall_time) = obs

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
                         num_source_workers,
                         valid_ops_mask)

        raw_action, parsed_action = \
            sample_action(node_scores, 
                          dag_scores,
                          job_ptr,
                          active_job_ids)

        obs, reward, done = env.step(parsed_action)

        experience += [Experience(
            dag_batch.clone(),
            valid_ops_mask,
            len(active_job_ids),
            num_source_workers,
            wall_time,
            raw_action,
            reward
        )]

    return experience




def invoke_agent(
    model,
    dag_batch_device,
    worker_count,
    valid_ops_mask
):
    with torch.no_grad():
        # no computational graphs needed during 
        # the episode, only model outputs.
        node_scores, dag_scores = \
            model(dag_batch_device, worker_count)

    node_scores = node_scores.cpu()
    node_scores[~valid_ops_mask] = float('-inf')

    dag_scores = dag_scores.cpu()
    job_count = dag_batch_device.num_graphs
    dag_scores = dag_scores.view(job_count, 
                                 worker_count)

    return node_scores, dag_scores




def sample_action(
    node_scores, 
    dag_scores, 
    job_ptr,
    active_job_ids,
):
    # select the next operation to schedule
    op_sample = \
        Categorical(logits=node_scores) \
            .sample()

    op_env, job_idx = \
        translate_op(
            op_sample.item(), 
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
    - `op`: translation of the policy sample so
    that the environment can find the correct
    operation
    - `job_idx`: index of the job that the
    selected op belongs to
    '''
    job_idx = (op >= job_ptr).sum() - 1

    job_id = active_jobs_ids[job_idx]
    active_op_idx = op - job_ptr[job_idx]
    
    op = (job_id, active_op_idx)

    return op, job_idx