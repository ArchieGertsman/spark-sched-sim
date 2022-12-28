from time import time
import shutil
import os

import torch
from torch.multiprocessing import Process, Pipe
import numpy as np
from torch_geometric.data import Batch

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
from .env_run import env_run
from .shared_obs import SharedObs
from ..utils.pyg import construct_subbatch, chunk_feature_tensor




class VecDagSchedEnv:

    def __init__(self, num_envs, datagen_state):
        self.n = num_envs
        self.datagen_state = datagen_state
        self.datagen = TPCHDataGen(datagen_state)




    ## API methods

    @property
    def num_jobs_per_env(self):
        return self.active_job_msk_batch.sum(-1)



    @property
    def num_ops_per_job(self):
        return self.op_counts.masked_select(self.active_job_msk_batch)



    def run(self):
        '''starts `self.n` subprocesses and creates
        pipes for communicting with them
        '''
        self.procs = []
        self.conns = []
        for rank in range(self.n):
            conn_main, conn_sub = Pipe()
            self.conns += [conn_main]
            proc = Process(target=env_run, args=(rank, self.datagen_state, conn_sub))
            self.procs += [proc]
            proc.start()



    def terminate(self):
        '''closes pipes and joins subprocesses'''
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def reset(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals, n_init_jobs, mjit)
        workers = self.datagen.workers(n_workers)

        self.n_total_jobs = n_init_jobs + n_job_arrivals
        self.n_workers = len(workers)
        self.op_counts = torch.tensor([
            len(e.job.ops) for _,_,e in initial_timeline.pq])
        self.max_ops = self.op_counts.max()

        self._reset_dag_batch(initial_timeline)

        self._reset_shared_obs_data()

        self.prev_episode_stats_batch = self._reset_envs(
            n_job_arrivals, n_init_jobs, mjit, n_workers)

        print('done resetting')

        return self._observe()



    def get_prev_episode_stats(self):
        if self.prev_episode_stats_batch[0] is None:
            return None

        avg_job_duration_batch, n_completed_jobs_batch = \
            list(zip(*self.prev_episode_stats_batch))

        avg_job_duration_mean = np.mean(avg_job_duration_batch)
        n_completed_jobs_mean = np.mean(n_completed_jobs_batch)
        return avg_job_duration_mean, n_completed_jobs_mean

        

    def step(self, action_batch):
        op_id_batch, prlvl_batch = action_batch
        self._step_envs(op_id_batch, prlvl_batch)
        return self._observe(), self.reward_batch, self.done_batch



    def translate_op_selections(self, op_idx_batch):
        job_idx_batch = torch.zeros_like(op_idx_batch)
        op_id_batch = [None] * len(op_idx_batch)

        gen = zip(self.done_batch, self.active_job_msk_batch)

        i = 0
        n_jobs_traversed = 0
        for done, active_job_msk in gen:
            if done.item():
                continue

            op_idx = op_idx_batch[i].item()

            active_job_ids = active_job_msk.nonzero().flatten()

            op_counts = self.op_counts[active_job_ids]
            cum = torch.cumsum(op_counts, 0)
            j = (op_idx >= cum).sum()

            job_idx_batch[i] = n_jobs_traversed + j

            job_id = active_job_ids[j].item()
            op_id = op_idx - (cum[j-1].item() if j>0 else 0)
            op_id_batch[i] = (job_id, op_id)

            n_jobs_traversed += len(active_job_ids)
            i += 1

        return job_idx_batch, op_id_batch




    ## reset helpers

    def _reset_dag_batch(self, initial_timeline):
        data_list = [
            e.job.init_pyg_data() 
            for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list * self.n)



    def _reset_shared_obs_data(self):
        self.feature_tensor_chunks = \
            chunk_feature_tensor(self.dag_batch)

        self.active_job_msk_batch = \
            torch.zeros((self.n, self.n_total_jobs), 
            dtype=torch.bool)

        self.op_msk_batch = \
            torch.zeros((self.n, self.n_total_jobs, self.max_ops), 
            dtype=torch.bool)

        self.prlvl_msk_batch = \
            torch.zeros((self.n, self.n_total_jobs, self.n_workers), 
            dtype=torch.bool)

        self.reward_batch = torch.zeros(self.n)

        self.done_batch = torch.zeros(self.n, dtype=torch.bool)



    def _reset_envs(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        it = zip(
            self.conns,
            self.active_job_msk_batch,
            self.op_msk_batch,
            self.prlvl_msk_batch,
            self.reward_batch,
            self.done_batch
        )

        for env_idx, (conn, active_job_msk, op_msk, prlvl_msk, reward, done) in enumerate(it):
            shared_obs = SharedObs(
                self._get_env_feature_tensor_chunks(env_idx),
                active_job_msk,
                op_msk,
                prlvl_msk,
                reward,
                done)

            conn.send(('reset', (
                n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers, 
                shared_obs
            )))

        prev_episode_stats_batch = [conn.recv() for conn in self.conns]
        return prev_episode_stats_batch



    def _get_env_feature_tensor_chunks(self, env_idx):
        '''get feature tensor chunks that correspond with 
        this environment only
        '''
        start = env_idx * self.n_total_jobs
        end = start + self.n_total_jobs
        return self.feature_tensor_chunks[start : end]




    ## step helpers

    def _step_envs(self, op_id_batch, prlvl_batch):
        i = 0
        for done, conn in zip(self.done_batch, self.conns):
            if not done.item():
                op_id = op_id_batch[i]
                # shift selection from {0,...,max-1} to {1,...,max}
                # by adding 1
                prlvl = 1 + prlvl_batch[i].item()
                action = (op_id, prlvl)
                i += 1
            else:
                action = None

            conn.send(('step', action))

        [conn.recv() for conn in self.conns]



    def _observe(self):
        if self.done_batch.all():
            return None

        dag_batch = self._construct_dag_batch()
        op_msk_batch = self.op_msk_batch[self.active_job_msk_batch]
        prlvl_msk_batch = self.prlvl_msk_batch[self.active_job_msk_batch]
        
        obs = (dag_batch, op_msk_batch, prlvl_msk_batch)
        return obs



    def _construct_dag_batch(self):
        done = torch.repeat_interleave(self.done_batch, self.n_total_jobs)

        # include job dags whose env is not done and that are currently active
        mask = ~done & self.active_job_msk_batch.flatten()

        subbatch = construct_subbatch(self.dag_batch, mask)
        return subbatch

