import sys
from typing import NamedTuple

sys.path.append('./gym_dagsched/data_generation/tpch/')
import os

import numpy as np
import torch
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
import gymnasium as gym

from ..wrappers.decima_wrappers import (
    DecimaObsWrapper,
    DecimaActWrapper
)
from ..utils import metrics
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.baselines import compute_baselines
from ..utils.returns_calculator import ReturnsCalculator
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

    # computes differential returns by default
    return_calc = ReturnsCalculator(discount)

    env_kwargs = {
        'num_workers': num_workers,
        'num_init_jobs': num_init_jobs,
        'num_job_arrivals': num_job_arrivals,
        'job_arrival_rate': job_arrival_rate,
        'moving_delay': moving_delay
    }

    procs, conns = \
        start_rollout_workers(world_size, agent, env_kwargs)

    max_time_mean = max_time_mean_init
    entropy_weight = entropy_weight_init

    for epoch in range(num_epochs):
        # sample the max wall duration of the current episode
        max_time = np.random.exponential(max_time_mean)

        print('training on sequence '
              f'{epoch+1} with max wall time = ' 
              f'{max_time*1e-3:.1f}s',
              flush=True)

        # send episode data
        [conn.send((max_time, entropy_weight)) for conn in conns]

        # rewards and wall times
        rewards_list, wall_times_list = \
            zip(*[conn.recv() for conn in conns])

        returns_list = \
            return_calc.calculate(rewards_list, wall_times_list)

        baselines_list, stds_list = \
            compute_baselines(wall_times_list, returns_list)

        # send advantages
        gen = zip(returns_list,
                  baselines_list,
                  stds_list,
                  conns)
        for returns, baselines, stds, conn in gen:
            advantages = (returns - baselines) / (stds + 1e-8)
            conn.send(advantages)

        # receive rollout stats
        rollout_stats_list = [conn.recv() for conn in conns]

        if writer:
            ep_lens = [len(rewards) for rewards in rewards_list]
            write_stats(
                writer,
                epoch,
                rollout_stats_list,
                returns_list,
                ep_lens,
                max_time,
                return_calc.avg_per_step_reward()
            )

        # increase the mean episode duration
        max_time_mean = \
            min(max_time_mean + max_time_mean_growth,
                max_time_mean_ceil)

        # decrease the entropy weight
        entropy_weight = \
            max(entropy_weight - entropy_weight_decay, 
                entropy_weight_min)

    end_rollout_workers(conns, procs)



def start_rollout_workers(num_envs, agent, env_kwargs):
    procs = []
    conns = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        proc = Process(target=rollout_worker, 
                       args=(rank,
                             num_envs,
                             conn_sub,
                             agent,
                             env_kwargs))
        procs += [proc]
        proc.start()

    return procs, conns



def end_rollout_workers(conns, procs):
    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]



def write_stats(writer,
                epoch,
                rollout_stats_list, 
                returns_list,
                ep_lens,
                max_time,
                avg_per_step_reward):
    (action_losses,
     entropies,
     avg_job_durations, 
     completed_job_counts) = zip(*rollout_stats_list)

    episode_stats = {
        'avg job duration': np.mean(avg_job_durations),
        'max wall time': max_time,
        'completed jobs count': np.mean(completed_job_counts),
        'avg reward per sec': avg_per_step_reward,
        'avg return': np.mean([returns[0] for returns in returns_list]),
        'action loss': np.mean(action_losses),
        'entropy': np.mean(entropies),
        'episode length': np.mean(ep_lens)
    }

    for name, stat in episode_stats.items():
        writer.add_scalar(name, stat, epoch)




## rollout workers

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



def rollout_worker(rank, num_envs, conn, agent, env_kwargs):
    '''worker target which runs episodes 
    and trains the model by communicating 
    with the main process and other workers
    '''
    setup_worker(rank, num_envs)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))
    agent.build(device=device, ddp=True)
    
    epoch = 0
    while data := conn.recv():
        prof = Profiler()
        prof.enable()

        stats = run_epoch(env, agent, data, conn, seed=epoch)

        prof.disable()

        if rank == 0 and (epoch+1) % 10 == 0:
            state_dict = \
                agent.actor_network.module.state_dict()
            torch.save(state_dict, 'model.pt')

        conn.send(stats)
        epoch += 1

    cleanup_worker()



def run_epoch(env, agent, data, conn, seed=42):
    max_wall_time, entropy_weight = data

    env_options = {'max_wall_time': max_wall_time}

    with HiddenPrints():
        rollout_buffer = \
            collect_rollout(env, 
                            agent, 
                            seed=seed, 
                            options=env_options)

    (wall_times, 
     rewards,
     obsns,
     actions) = \
        zip(*((exp.wall_time, 
               exp.reward,
               exp.obs,
               exp.action) 
              for exp in rollout_buffer))

    conn.send((np.array(rewards), np.array(wall_times)))

    advantages = conn.recv()
    advantages = torch.from_numpy(advantages).float()

    (total_loss,
     action_loss,
     entropy_loss) = \
        compute_loss(agent,
                     obsns,
                     actions,
                     advantages,
                     entropy_weight)
    
    agent.update_parameters(total_loss, dist.get_world_size())

    return action_loss, \
           entropy_loss / len(rollout_buffer), \
           metrics.avg_job_duration(env) * 1e-3, \
           env.num_completed_jobs
    



class Experience(NamedTuple):
    obs: object
    wall_time: float
    action: object
    reward: float



def collect_rollout(env, agent, seed, options):  
    obs, info = env.reset(seed=seed, options=options)
    done = False

    rollout_buffer = []

    while not done:
        action = agent(obs)

        new_obs, reward, terminated, truncated, info = \
            env.step(action)

        done = (terminated or truncated)

        exp = Experience(obs, info['wall_time'], action, reward)
        rollout_buffer += [exp]

        obs = new_obs

    return rollout_buffer



def compute_loss(agent, obsns, actions, advantages, entropy_weight):
    action_lgprobs, action_entropies = \
        agent.evaluate_actions(obsns, actions)

    action_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    total_loss = action_loss + entropy_weight * entropy_loss

    return total_loss, action_loss.item(), entropy_loss.item()