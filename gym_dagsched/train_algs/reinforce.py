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
import gymnasium as gym

from .common import collect_rollout, construct_nested_dag_batch
from ..envs.dagsched_env import DagSchedEnv
from ..wrappers.decima_wrapper import DecimaWrapper
from ..utils.metrics import avg_job_duration
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.baselines import compute_baselines
from ..utils.graph import add_adj
from ..utils.diff_returns import DifferentialReturnsCalculator
from ..utils.hidden_prints import HiddenPrints




def train(agent,
          num_epochs=500,
          world_size=4,
          writer=None,
          discount=.99,
          entropy_weight_init=1.,
          entropy_weight_decay=1e-3,
          entropy_weight_min=1e-4,
          num_job_arrivals=0, 
          num_init_jobs=20,
          job_arrival_rate=0,
          num_workers=10,
          max_time_mean_init=np.inf,
          max_time_mean_growth=0,
          max_time_mean_ceil=np.inf,
          moving_delay=2000):
    '''trains the model on different job arrival sequences. 
    Multiple episodes are run on each sequence in parallel.
    '''

    if num_job_arrivals > 0 and job_arrival_rate == 0:
        raise Exception('job arrival rate must be positive '
                        'when jobs are streaming')

    # use torch.distributed for IPC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    diff_return_calc = \
        DifferentialReturnsCalculator(discount)

    env_kwargs = {
        'num_workers': num_workers,
        'num_init_jobs': num_init_jobs,
        'num_job_arrivals': num_job_arrivals,
        'job_arrival_rate': job_arrival_rate,
        'moving_delay': moving_delay
    }

    procs, conns = \
        setup_workers(world_size, agent, env_kwargs)

    max_time_mean = max_time_mean_init
    entropy_weight = entropy_weight_init

    for i in range(num_epochs):
        # sample the max wall duration of the current episode
        max_time = np.random.exponential(max_time_mean)

        print('training on sequence '
              f'{i+1} with max wall time = ' 
              f'{max_time*1e-3:.1f}s',
              flush=True)

        # send episode data to each of the workers.
        for conn in conns:
            conn.send((max_time, entropy_weight))

        # wait for rewards and wall times
        rewards_list, wall_times_list = \
            zip(*[conn.recv() for conn in conns])

        diff_returns_list = \
            diff_return_calc.calculate(wall_times_list,
                                       rewards_list)

        baselines_list, stds_list = \
            compute_baselines(wall_times_list,
                              diff_returns_list)

        value_losses = []
        # send advantages
        for (returns, 
             baselines,
             stds,
             conn) in zip(diff_returns_list,
                          baselines_list,
                          stds_list,
                          conns):
            advantages = (returns - baselines) / (stds + 1e-8)
            value_losses += [np.sum(advantages**2)]
            conn.send(advantages)

        # wait for model update
        (action_losses,
         entropies,
         avg_job_durations, 
         completed_job_counts) = \
            zip(*[conn.recv() for conn in conns])

        if writer:
            # log episode stats to tensorboard
            episode_stats = {
                'avg job duration': np.mean(avg_job_durations),
                'max wall time': max_time,
                'completed jobs count': np.mean(completed_job_counts),
                'avg reward per sec': diff_return_calc.avg_per_step_reward,
                'avg return': np.mean([returns[0] for returns in diff_returns_list]),
                'action loss': np.mean(action_losses),
                'entropy': np.mean(entropies),
                'value loss': np.mean(value_losses),
                'episode length': np.mean([len(rewards) for rewards in rewards_list])
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




def setup_workers(num_envs, agent, env_kwargs):
    procs = []
    conns = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        proc = Process(target=trainer, 
                       args=(rank,
                             num_envs,
                             conn_sub,
                             agent,
                             env_kwargs))
        procs += [proc]
        proc.start()

    return procs, conns




def cleanup_workers(conns, procs):
    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]




def setup_worker(rank, world_size):
    sys.stdout = open(f'ignore/log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    dist.init_process_group(dist.Backend.GLOO, 
                            rank=rank, 
                            world_size=world_size)

    # IMPORTANT! Each worker needs to produce 
    # unique rollouts, which are determined
    # by the rng seeds.
    torch.manual_seed(rank)




def cleanup_worker():
    dist.destroy_process_group()




def trainer(rank, num_envs, conn, agent, env_kwargs):
    '''worker target which runs episodes 
    and trains the model by communicating 
    with the main process and other workers
    '''
    setup_worker(rank, num_envs)

    env = gym.make('gym_dagsched:gym_dagsched/DagSchedEnv-v0', **env_kwargs)
    # env = DecimaWrapper(base_env)
    agent.build(ddp=True, device=device)
    
    epoch = 0
    while data := conn.recv():
        prof = Profiler()
        prof.enable()

        stats = run_epoch(env, agent, data, epoch, conn)

        prof.disable()

        if rank == 0 and (epoch+1) % 10 == 0:
            state_dict = \
                agent.actor_network.module.state_dict()
            torch.save(state_dict, 'model.pt')

        conn.send(stats)
        epoch += 1

    cleanup_worker()




def run_epoch(env, agent, data, seed, conn):
    max_wall_time, entropy_weight = data

    with HiddenPrints():
        rollout = collect_rollout(env, 
                                agent, 
                                seed=seed, 
                                options={'max_wall_time': max_wall_time})

    wall_times, rewards = \
        zip(*[(obs.wall_time, obs.reward) for obs in rollout])

    # notify main proc that returns and wall times are ready
    conn.send((np.array(rewards), 
               np.array(wall_times)))

    # wait for main proc to compute advantages
    advantages = conn.recv()
    advantages = torch.from_numpy(advantages).float()

    (total_loss,
     action_loss,
     entropy_loss) = \
        compute_loss(agent,
                     rollout,
                     advantages,
                     entropy_weight)
    
    agent.update_parameters(total_loss, dist.get_world_size())

    return action_loss, \
           entropy_loss / len(rollout), \
           avg_job_duration(env) * 1e-3, \
           env.num_completed_jobs



def compute_loss(agent,
                 rollout,
                 advantages,
                 entropy_weight):
    (dag_batches, 
     num_dags_per_obs,
     valid_ops_masks, 
     valid_prlsm_lim_masks,
     actions) = \
        zip(*[(obs.dag_batch, 
               obs.num_jobs,
               obs.valid_ops_mask,
               obs.valid_prlsm_lim_mask,
               obs.action)
              for obs in rollout])
                   
    nested_dag_batch, num_nodes_per_dag = \
        construct_nested_dag_batch(dag_batches, 
                                   num_dags_per_obs)

    valid_ops_masks = np.concatenate(valid_ops_masks)
    valid_prlsm_lim_masks = \
        torch.from_numpy(np.vstack(valid_prlsm_lim_masks))

    obsns = (nested_dag_batch.to(device),
             torch.tensor(num_dags_per_obs),
             num_nodes_per_dag,
             valid_ops_masks,
             valid_prlsm_lim_masks)

    # calculate attributes of the actions
    # which will be used to construct loss
    action_lgprobs, action_entropies = \
        agent.evaluate_actions(obsns, actions)

    # compute loss
    action_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    total_loss = action_loss + entropy_weight * entropy_loss

    return total_loss, action_loss.item(), entropy_loss.item()