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

from spark_sched_sim.schedulers import NeuralScheduler
from .rollout_worker import *
from .utils import Profiler, ReturnsCalculator



class Trainer(ABC):
    '''Base training algorithm class. Each algorithm must implement the abstract method 
    `learn_from_rollouts` 
    '''

    def __init__(
        self,
        scheduler_cls,
        device,
        log_options,
        checkpoint_options,
        env_kwargs,
        model_kwargs,
        seed,
        async_rollouts
    ):  
        assert issubclass(scheduler_cls, NeuralScheduler)
        self.scheduler_cls = scheduler_cls

        self.device = device
        self.log_options = log_options
        self.checkpoint_options = checkpoint_options
        self.env_kwargs = env_kwargs
        self.model_kwargs = model_kwargs
        self.async_rollouts = async_rollouts

        self.return_calc = ReturnsCalculator()

        torch.manual_seed(seed) # call before initializing model

        self.agent = scheduler_cls(
            env_kwargs['num_executors'], **model_kwargs)



    def train(self, num_iterations, num_envs) -> None:
        '''trains the model on different job arrival sequences. 
        For each job sequence:
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters are updated, and
        - new model parameters are scattered to the rollout workers
        '''

        self._setup(num_envs)

        # every n'th iteration, save the best model from the past n iterations,
        # where `n = self.model_save_freq`
        best_state = None

        print('Beginning training.\n', flush=True)

        for i in range(num_iterations):
            actor_sd = deepcopy(self.agent.actor.state_dict())

            # move params to GPU for learning
            self.agent.actor.to(self.device, non_blocking=True)
            
            # scatter
            for conn in self.conns:
                conn.send({'actor_sd': actor_sd})

            # gather
            results = [conn.recv() for conn in self.conns]

            rollout_buffers, stats_list = zip(*[
                (res['rollout_buffer'], res['stats']) 
                for res in results if res])

            # update parameters
            # with Profiler():
            learning_stats = self.learn_from_rollouts(rollout_buffers)
            
            # return params to CPU before scattering updated state dict 
            # to the rollout workers
            self.agent.actor.to('cpu', non_blocking=True)

            # check if model is the current best
            if not best_state \
                or self.return_calc.avg_num_jobs < best_state['avg_num_jobs']:
                best_state = self._capture_state(
                    i, actor_sd, stats_list)

            if (i+1) % self.checkpoint_options['freq'] == 0:
                self._checkpoint(i, best_state)
                best_state = None

            if self.summary_writer:
                ep_lens = [len(buff) for buff in rollout_buffers if buff]
                self._write_stats(
                    i,
                    learning_stats,
                    stats_list,
                    ep_lens
                )

            if self.agent.lr_scheduler:
                self.agent.lr_scheduler.step()

            print(f'Iteration {i+1} complete. Avg. # jobs: ' 
                  f'{self.return_calc.avg_num_jobs:.3f}', 
                  flush=True)

        self._cleanup()


    
    @abstractmethod
    def learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        pass


    ## internal methods

    def _setup(self, num_envs) -> None:
        # logging
        proc_log_dir = self.log_options['proc_dir']
        shutil.rmtree(proc_log_dir, ignore_errors=True)
        os.mkdir(proc_log_dir)
        sys.stdout = open(f'{proc_log_dir}/main.out', 'a')
        
        try:
            tensorboard_dir = self.log_options['tensorboard_dir']
            self.summary_writer = SummaryWriter(tensorboard_dir)
        except:
            self.summary_writer = None

        # model checkpoints
        shutil.rmtree(self.checkpoint_options['dir'], ignore_errors=True)
        os.mkdir(self.checkpoint_options['dir'])

        # torch
        torch.multiprocessing.set_start_method('forkserver')
        # print('cuda available:', torch.cuda.is_available())
        # torch.autograd.set_detect_anomaly(True)

        self.agent.train()

        self._start_rollout_workers(num_envs)



    def _cleanup(self) -> None:
        self._terminate_rollout_workers()

        if self.summary_writer:
            self.summary_writer.close()

        print('\nTraining complete.', flush=True)



    def _capture_state(self, i, actor_sd, stats_list):
        return {
            'iteration': i,
            'avg_num_jobs': np.round(self.return_calc.avg_num_jobs, 3),
            'state_dict': actor_sd,
            'completed_job_count': int(np.mean([
                stats['num_completed_jobs'] for stats in stats_list]))
        }



    def _checkpoint(self, i, best_state):
        save_path = self.checkpoint_options['dir']
        dir = f'{save_path}/{i+1}'
        os.mkdir(dir)
        best_sd = best_state.pop('state_dict')
        torch.save(best_sd, f'{dir}/model.pt')
        with open(f'{dir}/state.json', 'w') as fp:
            json.dump(best_state, fp)



    def _start_rollout_workers(self, num_envs) -> None:
        self.procs = []
        self.conns = []

        for rank in range(num_envs):
            conn_main, conn_sub = Pipe()
            self.conns += [conn_main]

            proc = Process(
                target = RolloutWorkerAsync() if self.async_rollouts \
                    else RolloutWorkerSync(), 
                args = (
                    rank, 
                    conn_sub,
                    self.scheduler_cls,
                    self.env_kwargs, 
                    self.model_kwargs,
                    self.log_options['proc_dir']
                ))

            self.procs += [proc]
            proc.start()



    def _terminate_rollout_workers(self) -> None:
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def _write_stats(
        self,
        epoch: int,
        learning_stats,
        stats_list,
        ep_lens: list[int]
    ) -> None:

        episode_stats = learning_stats | {
            'avg job duration': \
                np.mean([stats['avg_job_duration'] for stats in stats_list]),
            'completed jobs count': \
                np.mean([stats['num_completed_jobs'] for stats in stats_list]),
            'job arrival count': \
                np.mean([stats['num_job_arrivals'] for stats in stats_list]),
            'episode length': np.mean(ep_lens),
            'rolling avg num jobs': self.return_calc.avg_num_jobs
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)