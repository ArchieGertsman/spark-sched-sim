from abc import ABC, abstractmethod
from typing import Optional
import shutil
import os
import sys
from multiprocessing.connection import Connection

import numpy as np
import gymnasium as gym
import torch
from torch.multiprocessing import Pipe, Process
from torch.utils.tensorboard import SummaryWriter

from ..agents.decima_agent import DecimaAgent
from ..wrappers.decima_wrappers import (
    DecimaObsWrapper,
    DecimaActWrapper
)
from ..utils import metrics
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.returns_calculator import ReturnsCalculator
from ..utils.hidden_prints import HiddenPrints
from ..utils.rollout_buffer import RolloutBuffer
from ..utils.baselines import compute_baselines




class BaseAlg(ABC):

    def __init__(
        self,
        env_kwargs: dict,
        num_iterations: int,
        num_epochs: int,
        batch_size: int,
        num_envs: int,
        seed: int,
        log_dir: str,
        summary_writer_dir: Optional[str],
        model_save_dir: str,
        model_save_freq: int,
        optim_class: torch.optim.Optimizer,
        optim_lr: float,
        gamma: float,
        max_time_mean_init: float,
        max_time_mean_growth: float,
        max_time_mean_clip_range: float,
        entropy_weight_init: float,
        entropy_weight_decay: float,
        entropy_weight_min: float
    ):  
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_envs = num_envs

        self.log_dir = log_dir
        self.summary_writer_path = summary_writer_dir
        self.model_save_path = model_save_dir
        self.model_save_freq = model_save_freq

        self.max_time_mean = max_time_mean_init
        self.max_time_mean_growth = max_time_mean_growth
        self.max_time_mean_clip_range = max_time_mean_clip_range

        self.entropy_weight = entropy_weight_init
        self.entropy_weight_decay = entropy_weight_decay
        self.entropy_weight_min = entropy_weight_min

        self.agent = \
            DecimaAgent(
                env_kwargs['num_workers'],
                optim_class=optim_class,
                optim_lr=optim_lr)

        # computes differential returns by default, which is
        # helpful for maximizing average returns
        self.return_calc = ReturnsCalculator(gamma)

        self.env_kwargs = env_kwargs

        torch.manual_seed(seed)
        self.np_random_max_time = np.random.RandomState(seed)

        self.procs = []
        self.conns = []



    def train(self) -> None:
        '''trains the model on different job arrival sequences. 
        For each job sequence, 
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters
            are updated, and
        - new model parameters are scattered to the rollout workers
        '''

        self._setup()

        for iteration in range(self.num_iterations):
            # sample the max wall duration of the current episode
            max_time = self.np_random_max_time.exponential(self.max_time_mean)
            max_time = np.clip(
                max_time, 
                self.max_time_mean - self.max_time_mean_clip_range,
                self.max_time_mean + self.max_time_mean_clip_range
            )

            self._log_iteration_start(iteration, max_time)

            state_dict = self.agent.actor_network.state_dict()
            if (iteration+1) % self.model_save_freq == 0:
                torch.save(state_dict, f'{self.model_save_path}/model.pt')
            
            # scatter updated model params and max wall time
            env_options = {'max_wall_time': max_time}
            [conn.send((state_dict, env_options)) for conn in self.conns]

            # gather rollouts and env stats
            (rollout_buffers,
             avg_job_durations,
             completed_job_counts) = \
                zip(*[conn.recv() for conn in self.conns])

            prof = Profiler().enable()
            
            action_loss, entropy = self._learn_from_rollouts(rollout_buffers)

            torch.cuda.synchronize()
            prof.disable()

            if self.summary_writer:
                ep_lens = [len(buff) for buff in rollout_buffers]
                self._write_stats(
                    iteration,
                    action_loss,
                    entropy,
                    avg_job_durations,
                    completed_job_counts,
                    ep_lens,
                    max_time
                )

            self._update_vars()

        self._cleanup()



    ## internal methods

    @abstractmethod
    def _learn_from_rollouts(
        self,
        rollout_buffers: list[RolloutBuffer]
    ) -> tuple[float, float]:
        '''unique to each training algorithm'''
        pass



    def _compute_advantages(
        self,
        rewards_list: list[np.ndarray],
        wall_times_list: list[np.ndarray]
    ) -> list[np.ndarray]:

        returns_list = \
            self.return_calc.calculate(rewards_list, wall_times_list)

        baselines_list, stds_list = \
            compute_baselines(wall_times_list, returns_list)

        advantages_list = [
                  (returns   -   baselines)  /  (stds + 1e-8)
            for    returns,      baselines,      stds \
            in zip(returns_list, baselines_list, stds_list)
        ]  

        return advantages_list



    def _setup(self) -> None:
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.mkdir(self.log_dir)
        sys.stdout = open(f'{self.log_dir}/main.out', 'a')
        
        print('cuda available:', torch.cuda.is_available())

        torch.multiprocessing.set_start_method('forkserver')
        
        self.summary_writer = None
        if self.summary_writer_path:
            self.summary_writer = SummaryWriter(self.summary_writer_path)

        self.agent.build(device)

        self._start_rollout_workers()



    def _cleanup(self) -> None:
        self._terminate_rollout_workers()

        if self.summary_writer:
            self.summary_writer.close()



    @classmethod
    def _log_iteration_start(cls, i, max_time):
        print_str = f'training on sequence {i+1}'
        if max_time < np.inf:
            print_str += f' (max wall time = {max_time*1e-3:.1f}s)'
        print(print_str, flush=True)



    def _start_rollout_workers(self) -> None:
        self.procs = []
        self.conns = []

        for rank in range(self.num_envs):
            conn_main, conn_sub = Pipe()
            self.conns += [conn_main]

            proc = Process(target=rollout_worker, 
                        args=(rank,
                              conn_sub,
                              self.env_kwargs))
            self.procs += [proc]
            proc.start()



    def _terminate_rollout_workers(self) -> None:
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def _write_stats(
        self,
        epoch: int,
        action_loss: float,
        entropy: float,
        avg_job_durations: list[float],
        completed_job_counts: list[int],
        ep_lens: list[int],
        max_time: float
    ) -> None:

        episode_stats = {
            'avg job duration': np.mean(avg_job_durations),
            'max wall time': max_time * 1e-3,
            'completed jobs count': np.mean(completed_job_counts),
            'avg reward per sec': self.return_calc.avg_per_step_reward(),
            'action loss': action_loss,
            'entropy': entropy,
            'episode length': np.mean(ep_lens)
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)



    def _update_vars(self) -> None:
        # increase the mean episode duration
        self.max_time_mean += self.max_time_mean_growth

        # decrease the entropy weight
        self.entropy_weight = np.clip(
            self.entropy_weight - self.entropy_weight_decay,
            self.entropy_weight_min,
            None
        )




## rollout workers

def setup_worker(rank: int, env_kwargs: dict) -> None:
    # log each of the processes to separate files
    sys.stdout = open(f'ignore/log/proc/{rank}.out', 'a')

    # torch multiprocessing is very slow without this
    torch.set_num_threads(1)

    # IMPORTANT! Each worker needs to produce unique 
    # rollouts, which are determined by the rng seed
    torch.manual_seed(rank)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    agent = DecimaAgent(env_kwargs['num_workers'])
    agent.build(device=device)

    return env, agent



def rollout_worker(
    rank: int, 
    conn: Connection, 
    env_kwargs: dict
) -> None:
    '''collects rollouts and trains the model by communicating 
    with the main process and other workers
    '''
    env, agent = setup_worker(rank, env_kwargs)
    
    iteration = 0
    while data := conn.recv():
        state_dict, env_options = data

        # load updated model parameters
        agent.actor_network.load_state_dict(state_dict)
        
        prof = Profiler().enable()

        with HiddenPrints():
            rollout_buffer = \
                collect_rollout(
                    env, 
                    agent, 
                    seed=iteration, 
                    options=env_options
                )

        # send rollout buffer and stats to center
        avg_job_duration = metrics.avg_job_duration(env) * 1e-3
        conn.send((
            rollout_buffer, 
            avg_job_duration, 
            env.num_completed_jobs
        ))

        prof.disable()

        iteration += 1



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