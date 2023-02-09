import sys
from typing import List, Tuple, NamedTuple, Optional

sys.path.append('./gym_dagsched/data_generation/tpch/')
import os

import numpy as np
import torch
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.core import ObsType, ActType

from ..agents.decima_agent import DecimaAgent
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




def train(
    agent: DecimaAgent,
    num_iterations: int = 500,
    world_size: int = 4,
    writer: Optional[SummaryWriter] = None,
    discount: float = .99,
    entropy_weight_init: float = 1.,
    entropy_weight_decay: float = 1e-3,
    entropy_weight_min: float = 1e-4,
    num_job_arrivals: int = 0, 
    num_init_jobs: int = 20,
    job_arrival_rate: int = 0,
    num_workers: int = 10,
    max_time_mean_init: float = np.inf,
    max_time_mean_growth: float = 0.,
    max_time_mean_ceil: float = np.inf,
    moving_delay: float = 2000
) -> None:
    '''trains the model on different job arrival sequences. 
    For each job sequence, 
    - multiple rollouts are collected in parallel, asynchronously
    - the rollouts are gathered at the center to compute advantages, and
    - the advantages are scattered and models are updated using DDP
    '''

    if not torch.cuda.is_available():
        raise Exception('at least one GPU is needed for DDP')

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
        start_rollout_workers(world_size, env_kwargs)

    agent.build(device)

    max_time_mean = max_time_mean_init
    entropy_weight = entropy_weight_init

    for iteration in range(num_iterations):
        # sample the max wall duration of the current episode
        max_time = np.random.exponential(max_time_mean)

        print('training on sequence '
              f'{iteration+1} with max wall time = ' 
              f'{max_time*1e-3:.1f}s',
              flush=True)

        # send episode data
        state_dict = agent.actor_network.state_dict()
        [conn.send((state_dict, max_time)) for conn in conns]

        # rewards and wall times
        (rollout_buffers,
         avg_job_durations,
         completed_job_counts) = \
            zip(*[conn.recv() for conn in conns])


        prof = Profiler()
        prof.enable()


        (obsns_list,
         wall_times_list,
         actions_list,
         rewards_list) = zip(*((buff.obsns, 
                                buff.wall_times, 
                                buff.actions, 
                                buff.rewards)
                               for buff in rollout_buffers))

        returns_list = \
            return_calc.calculate(rewards_list, wall_times_list)

        baselines_list, stds_list = \
            compute_baselines(wall_times_list, returns_list)

        advantages_list = []
        gen = zip(returns_list, baselines_list, stds_list)
        advantages_list = [(returns - baselines) / (stds + 1e-8)
                           for returns, baselines, stds in gen]

        # flatten rollout data from all the workers
        all_obsns = [obs for obsns in obsns_list for obs in obsns]
        all_actions = [act for actions in actions_list for act in actions]
        all_advantages = np.hstack(advantages_list)
        
        action_loss, entropy =\
            train_iteration(
                agent,
                all_obsns,
                all_actions,
                all_advantages,
                entropy_weight,
            )

        torch.cuda.synchronize()
        prof.disable()

        if writer:
            ep_lens = [len(rewards) for rewards in rewards_list]
            write_stats(
                writer,
                iteration,
                action_loss,
                entropy,
                avg_job_durations,
                completed_job_counts,
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



def train_iteration(
    agent: DecimaAgent,
    all_obsns: list[ObsType],
    all_actions: list[ActType],
    all_advantages: np.ndarray,
    entropy_weight: float,
    num_epochs: int = 10, 
    batch_size: int = 64
) -> None:

    all_sample_idxs = np.random.permutation(np.arange(len(all_obsns)))
    split_ind = np.arange(batch_size, len(all_sample_idxs), batch_size)
    mini_batches = np.split(all_sample_idxs, split_ind)

    action_losses = []
    entropies = []

    for _ in range(num_epochs):
        for idx in mini_batches:
            obsns = [all_obsns[i] for i in idx]
            actions = [all_actions[i] for i in idx]
            advantages = torch.from_numpy(all_advantages[idx]).float()

            total_loss, action_loss, entropy_loss = \
                compute_loss(
                    agent, 
                    obsns, 
                    actions, 
                    advantages, 
                    entropy_weight
                )

            action_losses += [action_loss]
            entropies += [entropy_loss / len(idx)]

            agent.update_parameters(total_loss)

    return np.mean(action_losses), np.mean(entropies)



def start_rollout_workers(
    num_envs: int, 
    env_kwargs: dict
) -> Tuple[List[Process], List[Connection]]:

    procs = []
    conns = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        proc = Process(target=rollout_worker, 
                       args=(rank,
                             num_envs,
                             conn_sub,
                             env_kwargs))
        procs += [proc]
        proc.start()

    return procs, conns



def end_rollout_workers(
    conns: List[Connection], 
    procs: List[Process]
) -> None:

    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]



def write_stats(
    writer: SummaryWriter,
    epoch: int,
    action_loss: float,
    entropy: float,
    avg_job_durations: list[float],
    completed_job_counts: list[int],
    returns_list: List[np.ndarray],
    ep_lens: List[int],
    max_time: float,
    avg_per_step_reward: float
) -> None:

    episode_stats = {
        'avg job duration': np.mean(avg_job_durations),
        'max wall time': max_time,
        'completed jobs count': np.mean(completed_job_counts),
        'avg reward per sec': avg_per_step_reward,
        'avg return': np.mean([returns[0] for returns in returns_list]),
        'action loss': action_loss,
        'entropy': entropy,
        'episode length': np.mean(ep_lens)
    }

    for name, stat in episode_stats.items():
        writer.add_scalar(name, stat, epoch)




## rollout workers

def setup_worker(rank: int, world_size: int) -> None:
    sys.stdout = open(f'ignore/log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    dist.init_process_group(dist.Backend.GLOO, 
                            rank=rank, 
                            world_size=world_size)

    # IMPORTANT! Each worker needs to produce 
    # unique rollouts, which are determined
    # by the rng seed
    torch.manual_seed(rank)



def cleanup_worker() -> None:
    dist.destroy_process_group()



def rollout_worker(
    rank: int, 
    num_envs: int, 
    conn: Connection, 
    env_kwargs: dict
) -> None:
    '''collects rollouts and trains the model by communicating 
    with the main process and other workers
    '''
    setup_worker(rank, num_envs)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    agent = DecimaAgent(env_kwargs['num_workers'])
    agent.build(device)
    
    epoch = 0
    while data := conn.recv():
        state_dict, max_wall_time = data
        agent.actor_network.load_state_dict(state_dict)
        
        prof = Profiler()
        prof.enable()

        env_options = {'max_wall_time': max_wall_time}

        with HiddenPrints():
            rollout_buffer = \
                collect_rollout(
                    env, 
                    agent, 
                    seed=epoch, 
                    options=env_options
                )

        avg_job_duration = metrics.avg_job_duration(env) * 1e-3
        conn.send((
            rollout_buffer, 
            avg_job_duration, 
            env.num_completed_jobs
        ))

        prof.disable()

        epoch += 1

    cleanup_worker()
    




class RolloutBuffer:
    obsns: list[ObsType] = []
    wall_times: list[float] = []
    actions: list[ActType] = []
    rewards: list[float] = []

    def add(self, obs, wall_time, action, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]



def collect_rollout(
    env, 
    agent: DecimaAgent, 
    seed: int, 
    options: dict
) -> RolloutBuffer:

    obs, info = env.reset(seed=seed, options=options)
    done = False

    rollout_buffer = RolloutBuffer()

    while not done:
        action = agent(obs)

        new_obs, reward, terminated, truncated, info = \
            env.step(action)

        done = (terminated or truncated)

        rollout_buffer.add(obs, info['wall_time'], action, reward)

        obs = new_obs

    return rollout_buffer



def compute_loss(
    agent: DecimaAgent, 
    obsns: List[dict], 
    actions: List[dict], 
    advantages: torch.Tensor, 
    entropy_weight: float
) -> Tuple[torch.Tensor, float, float]:

    action_lgprobs, action_entropies = \
        agent.evaluate_actions(obsns, actions)

    action_loss = -advantages @ action_lgprobs
    entropy_loss = action_entropies.sum()
    total_loss = action_loss + entropy_weight * entropy_loss

    return total_loss, action_loss.item(), entropy_loss.item()