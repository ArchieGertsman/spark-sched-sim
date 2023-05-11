from abc import ABC, abstractmethod
from typing import Optional, Iterable
import shutil
import os
import sys
from copy import deepcopy
import json

import numpy as np
import torch
from torch.multiprocessing import Pipe, Process
from torch.utils.tensorboard import SummaryWriter

from spark_sched_sim.schedulers import DecimaScheduler
from .rollout_worker import RolloutBuffer, rollout_worker
from .utils import Profiler, ReturnsCalculator, device




class Trainer(ABC):
    '''Base training algorithm class. Each algorithm must implement the abstract method 
    `_learn_from_rollouts` 
    '''

    def __init__(
        self,
        num_iterations,
        num_envs,
        log_options,
        model_save_options,
        time_limit_options,
        entropy_options,
        env_kwargs,
        model_kwargs,
        seed
    ):  
        self.num_iterations = num_iterations
        self.num_envs = num_envs
        self.log_options = log_options
        self.model_save_options = model_save_options
        self.env_kwargs = env_kwargs
        self.model_kwargs = model_kwargs

        self.time_limit_options = time_limit_options
        self.time_limit_mean = time_limit_options['init']

        self.entropy_options = entropy_options
        self.entropy_weight = entropy_options['init']

        self.return_calc = ReturnsCalculator()

        torch.manual_seed(seed)
        self.np_random_time_limit = np.random.RandomState(seed)

        self.agent = DecimaScheduler(env_kwargs['num_executors'], **model_kwargs)



    def train(self) -> None:
        '''trains the model on different job arrival sequences. 
        For each job sequence:
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters are updated, and
        - new model parameters are scattered to the rollout workers
        '''

        self._setup()

        # every n'th iteration, save the best model from the past n iterations,
        # where `n = self.model_save_freq`
        best_state = None

        for i in range(self.num_iterations):
            time_limit = self._sample_time_limit()

            self._log_iteration_start(i, time_limit)

            actor_sd = deepcopy(self.agent.actor.state_dict())

            # move params to GPU for learning
            self.agent.actor.to(device, non_blocking=True)
            
            # scatter
            env_seed = i
            env_options = {'time_limit': time_limit}
            for conn in self.conns:
                conn.send((
                    actor_sd, 
                    env_seed, 
                    env_options
                ))

            # gather
            (rollout_buffers,
             avg_job_durations,
             completed_job_counts,
             job_arrival_counts) = \
                zip(*[conn.recv() for conn in self.conns])

            # update parameters
            with Profiler():
                policy_loss, entropy_loss = \
                    self._learn_from_rollouts(rollout_buffers)
                
                # return params to CPU before scattering state dict to rollout workers
                self.agent.actor.cpu()

            # check if model is the current best
            if not best_state or self.return_calc.avg_num_jobs < best_state['avg_num_jobs']:
                best_state = {
                    'iteration': i,
                    'avg_num_jobs': np.round(self.return_calc.avg_num_jobs, 3),
                    'state_dict': actor_sd,
                    'time_limit': int(time_limit * 1e-3),
                    'time_limit_mean': int(self.time_limit_mean * 1e-3),
                    'completed_job_count': int(np.mean(completed_job_counts))
                }

            if (i+1) % self.model_save_options['freq'] == 0:
                self._save_best_model(i, best_state)
                best_state = None

            if self.summary_writer:
                ep_lens = [len(buff) for buff in rollout_buffers if buff]
                self._write_stats(
                    i,
                    policy_loss,
                    entropy_loss,
                    avg_job_durations,
                    completed_job_counts,
                    job_arrival_counts,
                    ep_lens,
                    time_limit
                )

            self._update_hyperparams()

        self._cleanup()



    ## internal methods

    @abstractmethod
    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        pass



    def _setup(self) -> None:
        # logging
        proc_log_dir = self.log_options['proc_dir']
        shutil.rmtree(proc_log_dir, ignore_errors=True)
        os.mkdir(proc_log_dir)
        sys.stdout = open(f'{proc_log_dir}/main.out', 'a')
        
        try:
            self.summary_writer = SummaryWriter(self.log_options['tensorboard_dir'])
        except:
            self.summary_writer = None

        # model checkpoints
        shutil.rmtree(self.model_save_options['dir'], ignore_errors=True)
        os.mkdir(self.model_save_options['dir'])

        # torch
        print('cuda available:', torch.cuda.is_available())
        torch.multiprocessing.set_start_method('forkserver')
        # torch.autograd.set_detect_anomaly(True)

        self.agent.train()

        self._start_rollout_workers()



    def _cleanup(self) -> None:
        self._terminate_rollout_workers()

        if self.summary_writer:
            self.summary_writer.close()

        print('training complete.', flush=True)



    @classmethod
    def _log_iteration_start(cls, i, time_limit):
        print_str = f'training on sequence {i+1}'
        if time_limit < np.inf:
            print_str += f' (max wall time = {time_limit*1e-3:.1f}s)'
        print(print_str, flush=True)



    def _save_best_model(self, i, best_state):
        save_path = self.model_save_options['dir']
        dir = f'{save_path}/{i+1}'
        os.mkdir(dir)
        best_sd = best_state.pop('state_dict')
        torch.save(best_sd, f'{dir}/model.pt')
        with open(f'{dir}/state.json', 'w') as fp:
            json.dump(best_state, fp)



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
                    conn_sub, 
                    self.env_kwargs, 
                    self.model_kwargs,
                    self.log_options['proc_dir']
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
        completed_job_counts: list[int],
        job_arrival_counts: list[int],
        ep_lens: list[int],
        time_limit: float
    ) -> None:

        episode_stats = {
            'policy loss': np.abs(policy_loss),
            'entropy': np.abs(entropy_loss),
            'avg job duration': np.mean([x for x in avg_job_durations if x is not None]),
            'completed jobs count': np.mean([x for x in completed_job_counts if x is not None]),
            'job arrival count': np.mean([x for x in job_arrival_counts if x is not None]),
            'episode length': np.mean(ep_lens),
            'max time': time_limit * 1e-3,
            'rolling avg num jobs': self.return_calc.avg_num_jobs,
            'entropy weight': self.entropy_weight
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)



    def _sample_time_limit(self):
        time_limit = self.np_random_time_limit.exponential(self.time_limit_mean)

        # not too short
        time_limit = max(time_limit, 1e5)

        return time_limit



    def _update_hyperparams(self) -> None:
        # geometrically increase the mean episode duration
        self.time_limit_mean = min(
            self.time_limit_mean * self.time_limit_options['factor'], 
            self.time_limit_options['ceil']
        )

        # arithmetically decrease the entropy weight
        self.entropy_weight = max(
            self.entropy_weight - self.entropy_options['delta'],
            self.entropy_options['floor']
        )
