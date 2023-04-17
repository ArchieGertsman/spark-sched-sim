from abc import ABC, abstractmethod
from typing import Optional, Iterable
import shutil
import os
import sys

import numpy as np
import torch
from torch.multiprocessing import Pipe, Process
from torch.utils.tensorboard import SummaryWriter

from spark_sched_sim.schedulers import DecimaScheduler
from .rollouts import RolloutBuffer, rollout_worker
from .utils.profiler import Profiler
from .utils.returns_calculator import ReturnsCalculator
from .utils.device import device




class BaseAlg(ABC):
    '''Base class for training algorithms, which must
    implement the abstract `_compute_loss` method
    '''

    def __init__(
        self,
        env_kwargs: dict,
        num_iterations: int,
        num_epochs: int,
        batch_size: Optional[int],
        num_envs: int,
        seed: int,
        log_dir: str,
        summary_writer_dir: Optional[str],
        model_save_dir: str,
        model_save_freq: int,
        optim_class: torch.optim.Optimizer,
        optim_lr: float,
        max_grad_norm: float,
        gamma: float,
        max_time_mean_init: float,
        max_time_mean_growth: float,
        max_time_mean_ceil: float,
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
        self.max_time_mean_ceil = max_time_mean_ceil

        self.entropy_weight = entropy_weight_init
        self.entropy_weight_decay = entropy_weight_decay
        self.entropy_weight_min = entropy_weight_min

        # computes differential returns by default, which is
        # helpful for maximizing average returns
        self.return_calc = ReturnsCalculator()
        self.gamma = gamma

        self.env_kwargs = env_kwargs

        torch.manual_seed(seed)
        self.np_random_max_time = np.random.RandomState(seed)
        self.dataloader_gen = torch.Generator()
        self.dataloader_gen.manual_seed(seed)

        self.agent = \
            DecimaScheduler(
                env_kwargs['num_executors'],
                optim_class=optim_class,
                optim_lr=optim_lr,
                max_grad_norm=max_grad_norm
            )

        self.procs = []
        self.conns = []

        self.num_job_arrivals = 20



    def train(self) -> None:
        '''trains the model on different job arrival sequences. 
        For each job sequence, 
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters
            are updated, and
        - new model parameters are scattered to the rollout executors
        '''

        self._setup()

        for i in range(self.num_iterations):
            max_time = 6e6
            # max_time = self.np_random_max_time.geometric(1/self.max_time_mean)
            # max_time = self.np_random_max_time.triangular(0, 0, 2.5*self.max_time_mean)
            # max_time = 1e6

            # max_time = np.inf
            # max_time = 1e6
            # num_job_arrivals = self.np_random_max_time.randint(
            #     self.num_job_arrivals // 2,
            #     min(self.num_job_arrivals * 2, 201)
            # )
            # num_job_arrivals = 30

            self._log_iteration_start(i, max_time)

            actor_sd = self.agent.actor.state_dict()
            critic_sd = None # self.agent.critic.state_dict()
            if (i+1) % self.model_save_freq == 0:
                torch.save(actor_sd, f'{self.model_save_path}/model.pt')
            
            # scatter
            env_seed = i
            env_options = {
                'max_wall_time': max_time #,
                # 'num_job_arrivals': 20
            }
            N = self.num_envs # // 2
            for j, conn in enumerate(self.conns):
                # env_seed = i * N + (j % N)
                env_seed = i
                # env_seed = 4*i + int(4 * (j / self.num_envs))
                conn.send((
                    actor_sd, 
                    critic_sd,
                    env_seed, 
                    env_options
                ))

            # gather
            (rollout_buffers,
             avg_job_durations,
             avg_num_jobs,
             completed_job_counts,
             job_arrival_counts) = \
                zip(*[conn.recv() for conn in self.conns])

            with Profiler():
                policy_loss, entropy_loss, value_loss, approx_kl_div = \
                    self._learn_from_rollouts(rollout_buffers)
                self.agent.lr_scheduler.step()
                torch.cuda.synchronize()

            if self.summary_writer:
                ep_lens = [len(buff) for buff in rollout_buffers if buff]
                self._write_stats(
                    i,
                    np.abs(policy_loss),
                    np.abs(entropy_loss),
                    avg_job_durations[:N],
                    avg_num_jobs[:N],
                    completed_job_counts[:N],
                    job_arrival_counts[:N],
                    ep_lens[:N],
                    max_time,
                    approx_kl_div,
                    value_loss,
                    [buff.norm_return for buff in rollout_buffers if buff]
                )

            self._update_vars(i)

        self._cleanup()



    ## internal methods

    @abstractmethod
    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        pass



    def _setup(self) -> None:
        # torch.autograd.set_detect_anomaly(True)

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

            proc = Process(
                target=rollout_worker, 
                args=(
                    rank, 
                    self.num_envs, 
                    conn_sub, 
                    self.env_kwargs, 
                    self.log_dir
                )
            )

            self.procs += [proc]
            proc.start()



    def _terminate_rollout_workers(self) -> None:
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def _write_stats(
        self,
        epoch: int,
        policy_loss: float,
        entropy_loss: float,
        avg_job_durations: list[float],
        avg_num_jobs,
        completed_job_counts: list[int],
        job_arrival_counts: list[int],
        ep_lens: list[int],
        max_time: float,
        approx_kl_div: float,
        value_loss: float,
        returns
    ) -> None:

        episode_stats = {
            'avg job duration': np.mean([x for x in avg_job_durations if x is not None]),
            'max wall time': max_time * 1e-3,
            'completed jobs count': np.mean([x for x in completed_job_counts if x is not None]),
            'job arrival count': np.mean([x for x in job_arrival_counts if x is not None]),
            # 'avg num jobs': np.mean([x for x in avg_num_jobs if x is not None]),
            # 'norm. -return': -np.mean(returns),
            'rolling avg num jobs': -self.return_calc.avg_num_jobs,
            'policy loss': policy_loss,
            'entropy': entropy_loss,
            'episode length': np.mean(ep_lens),
            'max time mean': self.max_time_mean * 1e-3,
            'entropy weight': self.entropy_weight,
            # 'KL div': approx_kl_div,
            # 'value loss': value_loss
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)



    def _update_vars(self, iteration) -> None:
        # geometrically increase the mean episode duration
        self.max_time_mean = min(
            self.max_time_mean * self.max_time_mean_growth, 
            self.max_time_mean_ceil
        )

        if (iteration+1) % 10 == 0:
            self.num_job_arrivals += 1

        # geometrically decrease the entropy weight
        self.entropy_weight = max(
            self.entropy_weight - self.entropy_weight_decay,
            self.entropy_weight_min
        )