from abc import ABC, abstractmethod
import shutil
from typing import List, Tuple, Optional
from itertools import chain
import os
import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')

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
from ..utils.rollout_buffer import RolloutBuffer




class BaseAlg(ABC):

    def __init__(
        self,
        env_kwargs: dict,
        num_iterations: int = 500,
        num_epochs: int = 4,
        batch_size: int = 512,
        num_envs: int = 4,
        seed: int = 42,
        log_dir: str = 'log',
        summary_writer_dir: Optional[str] = None,
        model_save_dir: str = 'models',
        model_save_freq: int = 20,
        optim_class: torch.optim.Optimizer = torch.optim.Adam,
        optim_lr: float = 3e-4,
        gamma: float = .99,
        max_time_mean_init: float = np.inf,
        max_time_mean_growth: float = 0.,
        max_time_mean_ceil: float = np.inf,
        entropy_weight_init: float = 1.,
        entropy_weight_decay: float = 1e-3,
        entropy_weight_min: float = 1e-4
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
        self.max_time_mean_ceil = max_time_mean_ceil

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
        self.np_random = np.random.RandomState(seed)

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
            max_time = self.np_random.exponential(self.max_time_mean)

            self._log_iteration_start(iteration, max_time)

            state_dict = self.agent.actor_network.state_dict()
            if (iteration+1) % self.model_save_freq == 0:
                torch.save(state_dict, f'{self.model_save_path}/model.pt')
            
            # scatter updated model params and max wall time
            [conn.send((state_dict, max_time)) for conn in self.conns]

            # gather rollouts and env stats
            (rollout_buffers,
             avg_job_durations,
             completed_job_counts) = \
                zip(*[conn.recv() for conn in self.conns])

            prof = Profiler()
            prof.enable()
            
            action_loss, entropy = self._run_train_iteration(rollout_buffers)

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
    def _run_train_iteration(
        self,
        rollout_buffers: list[RolloutBuffer]
    ) -> tuple[float, float]:
        pass



    def _setup(self) -> None:
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.mkdir(self.log_dir)
        sys.stdout = open(f'{self.log_dir}/main.out', 'a')
        
        print('cuda available:', torch.cuda.is_available())

        torch.multiprocessing.set_start_method('forkserver')
        
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
            print_str += f' (max wall time = {max_time*1e-3})'
        print(print_str, flush=True)



    def _compute_loss(
        self,
        obsns: List[dict], 
        actions: List[dict], 
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:

        action_lgprobs, action_entropies = \
            self.agent.evaluate_actions(obsns, actions)

        action_loss = -advantages @ action_lgprobs
        entropy_loss = action_entropies.sum()
        total_loss = action_loss + self.entropy_weight * entropy_loss

        return total_loss, action_loss.item(), entropy_loss.item()



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
        ep_lens: List[int],
        max_time: float
    ) -> None:

        episode_stats = {
            'avg job duration': np.mean(avg_job_durations),
            'max wall time': max_time,
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
        self.max_time_mean = \
            min(self.max_time_mean + self.max_time_mean_growth,
                self.max_time_mean_ceil)

        # decrease the entropy weight
        self.entropy_weight = \
            max(self.entropy_weight - self.entropy_weight_decay, 
                self.entropy_weight_min)




## rollout workers

def setup_worker(rank: int) -> None:
    sys.stdout = open(f'ignore/log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    # IMPORTANT! Each worker needs to produce 
    # unique rollouts, which are determined
    # by the rng seed
    torch.manual_seed(rank)



def rollout_worker(
    rank: int, 
    conn: Connection, 
    env_kwargs: dict
) -> None:
    '''collects rollouts and trains the model by communicating 
    with the main process and other workers
    '''
    setup_worker(rank)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    agent = DecimaAgent(env_kwargs['num_workers'])
    agent.build(device)
    
    iteration = 0
    while data := conn.recv():
        state_dict, max_wall_time = data

        # load updated model parameters
        agent.actor_network.load_state_dict(state_dict)

        env_options = {'max_wall_time': max_wall_time}
        
        prof = Profiler()
        prof.enable()

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