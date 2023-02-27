from abc import ABC, abstractmethod
from typing import Optional, Iterable
import shutil
import os
import sys
from itertools import chain

import numpy as np
from gymnasium.core import ObsType, ActType
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.multiprocessing import Pipe, Process
from torch.utils.tensorboard import SummaryWriter

from ..agents.decima_agent import DecimaAgent
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.returns_calculator import ReturnsCalculator
from .rollouts import RolloutBuffer, RolloutDataset, rollout_worker
from ..utils.baselines import compute_baselines
from ..utils.graph import ObsBatch, collate_obsns




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
        max_time_mean_clip_range: float,
        entropy_weight_init: float,
        entropy_weight_decay: float,
        entropy_weight_min: float,
        target_kl: Optional[float]
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

        self.target_kl = target_kl

        # computes differential returns by default, which is
        # helpful for maximizing average returns
        self.return_calc = ReturnsCalculator(gamma) #, size=20000)

        self.env_kwargs = env_kwargs

        torch.manual_seed(seed)
        self.np_random_max_time = np.random.RandomState(seed)
        self.dataloader_gen = torch.Generator()
        self.dataloader_gen.manual_seed(seed)

        self.agent = \
            DecimaAgent(
                env_kwargs['num_workers'],
                optim_class=optim_class,
                optim_lr=optim_lr,
                max_grad_norm=max_grad_norm)

        self.procs = []
        self.conns = []

        self.avg_reward = -1000
        self.avg_value = -1000

        self.alpha = .1
        self.lam = .95
        self.nu = 1.



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
            max_time = self._sample_max_time()

            self._log_iteration_start(iteration, max_time)

            state_dict = self.agent.state_dict()
            if (iteration+1) % self.model_save_freq == 0:
                torch.save(state_dict, f'{self.model_save_path}/model.pt')
            
            # scatter
            env_options = {'max_wall_time': max_time}
            [conn.send((state_dict, env_options)) for conn in self.conns]

            # gather
            (rollout_buffers,
             avg_job_durations,
             completed_job_counts,
             job_arrival_counts) = \
                zip(*[conn.recv() for conn in self.conns])

            with Profiler():
                policy_loss, entropy_loss, value_loss, approx_kl_div = \
                    self._learn_from_rollouts(rollout_buffers)
                
                torch.cuda.synchronize()

            if self.summary_writer:
                ep_lens = [len(buff) for buff in rollout_buffers]
                self._write_stats(
                    iteration,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    avg_job_durations,
                    completed_job_counts,
                    job_arrival_counts,
                    ep_lens,
                    max_time,
                    approx_kl_div
                )

            self._update_vars()

        self._cleanup()



    ## internal methods

    @abstractmethod
    def _compute_loss(
        self,
        obsns: ObsBatch,
        actions: Tensor,
        advantages: Tensor,
        old_lgprobs: Tensor
    ) -> tuple[Tensor, float, float]:
        '''Loss calculation unique to each algorithm

        Returns: 
            tuple (total_loss, policy_loss, entropy_loss),
            where total_loss is differentiable and the other losses
            are just scalars for logging.
        '''
        pass



    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:

        dataloader = self._make_dataloader(rollout_buffers)

        policy_losses = []
        entropy_losses = []
        value_losses = []
        approx_kl_divs = []

        continue_training = True

        # run multiple learning epochs with minibatching
        for _ in range(self.num_epochs):
            if not continue_training:
                break

            for obsns, actions, returns, advantages, old_lgprobs in dataloader:
                (total_loss, 
                 action_loss, 
                 entropy_loss, 
                 value_loss, 
                 approx_kl_div) = \
                    self._compute_loss(
                        obsns, 
                        actions,
                        returns,
                        advantages,
                        old_lgprobs
                    )

                policy_losses += [action_loss]
                entropy_losses += [entropy_loss]
                value_losses += [value_loss]
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.agent.update_parameters(total_loss)

        return np.mean(policy_losses), \
               np.mean(entropy_losses), \
               np.mean(value_losses), \
               np.mean(approx_kl_divs)



    def _make_dataloader(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> DataLoader:
        '''creates a dataset out of the new rollouts, and returns a 
        dataloader that loads minibatches from that dataset
        '''

        # separate the rollout data into lists
        (obsns_list, 
         actions_list, 
         wall_times_list, 
         rewards_list, 
         values_list, 
         lgprobs_list) = \
            zip(*((buff.obsns, 
                   buff.actions, 
                   buff.wall_times, 
                   buff.rewards,
                   buff.values,
                   buff.lgprobs)
                  for buff in rollout_buffers)) 

        # flatten observations and actions into a dict for fast access time
        obsns = {i: obs for i, obs in enumerate(chain(*obsns_list))}
        actions = {i: act for i, act in enumerate(chain(*actions_list))}

        
        rew = np.hstack(rewards_list)
        rew = rew[rew != 0]
        self.avg_reward += self.alpha * (rew.mean() - self.avg_reward)

        val = np.hstack(values_list)
        self.avg_value += self.alpha * (val.mean() - self.avg_value)


        all_advantages = []

        for rewards, values in zip(rewards_list, values_list):
            rewards = np.hstack(rewards)
            values = np.hstack(values)

            dv = values[1:] - values[:-1]
            dv = np.concatenate([dv, np.array([0])])
            deltas = rewards - self.avg_reward + dv

            advantages = []
            for t in reversed(range(len(deltas))):
                adv = deltas[t]
                if t < len(deltas)-1:
                    adv += self.lam * deltas[t+1]
                advantages += [adv]

            all_advantages += advantages

        
        advantages = torch.from_numpy(np.hstack(all_advantages)).float()
        values = torch.from_numpy(val).float()

        value_targets = advantages + values - self.nu * self.avg_value

        

        old_lgprobs = torch.from_numpy(np.hstack(lgprobs_list))
        
        rollout_dataset = \
            RolloutDataset(
                obsns, 
                actions, 
                value_targets,
                advantages, 
                old_lgprobs
            )

        if self.batch_size is not None:
            dataloader = \
                DataLoader(
                    dataset=rollout_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=RolloutDataset.collate,
                    generator=self.dataloader_gen
                )
        else:
            dataloader = \
                DataLoader(
                    dataset=rollout_dataset,
                    batch_size=len(rollout_dataset),
                    collate_fn=RolloutDataset.collate
                )

        return dataloader



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

            proc = Process(
                target=rollout_worker, 
                args=(rank, self.num_envs, conn_sub, self.env_kwargs)
            )

            self.procs += [proc]
            proc.start()



    def _terminate_rollout_workers(self) -> None:
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def _sample_max_time(self):
        max_time = self.np_random_max_time.uniform(
            low=self.max_time_mean - self.max_time_mean_clip_range,
            high=self.max_time_mean + self.max_time_mean_clip_range
        )
        # max_time = self.np_random_max_time.exponential(self.max_time_mean)
        # max_time = np.clip(
        #     max_time, 
        #     self.max_time_mean - self.max_time_mean_clip_range,
        #     self.max_time_mean + self.max_time_mean_clip_range
        # )
        return max_time



    def _write_stats(
        self,
        epoch: int,
        policy_loss: float,
        value_loss: float,
        entropy_loss: float,
        avg_job_durations: list[float],
        completed_job_counts: list[int],
        job_arrival_counts: list[int],
        ep_lens: list[int],
        max_time: float,
        approx_kl_div: float
    ) -> None:

        episode_stats = {
            'avg job duration': np.mean(avg_job_durations),
            'max wall time': max_time * 1e-3,
            'completed jobs count': np.mean(completed_job_counts),
            'job arrival count': np.mean(job_arrival_counts),
            'avg reward': self.avg_reward,
            'avg value': self.avg_value,
            'policy loss': policy_loss,
            'value loss': value_loss,
            'entropy': -entropy_loss,
            'episode length': np.mean(ep_lens),
            'KL div': approx_kl_div
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)



    def _update_vars(self) -> None:
        # increase the mean episode duration
        self.max_time_mean += self.max_time_mean_growth

        self.max_time_mean_clip_range += 2e3

        # decrease the entropy weight
        self.entropy_weight = np.clip(
            self.entropy_weight - self.entropy_weight_decay,
            self.entropy_weight_min,
            None
        )
