from time import time

import torch
from torch.multiprocessing import Process, Pipe
import numpy as np
from torch_geometric.data import Batch

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen

from .env_run import env_run
from .shared_obs import SharedObs
from ..utils.misc import construct_subbatch




class VecDagSchedEnv:

    def __init__(self, n):
        self.n = n
        self.datagen = TPCHDataGen(np.random.RandomState())




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
            proc = Process(target=env_run, args=(rank, self.datagen, conn_sub))
            self.procs += [proc]
            proc.start()



    def terminate(self):
        '''closes pipes and joins subprocesses'''
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def reset(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        t_reset = time()

        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals, n_init_jobs, mjit)
        workers = self.datagen.workers(n_workers)

        self.n_job_arrivals = n_job_arrivals
        self.n_workers = len(workers)
        self.op_counts = \
            torch.tensor([len(e.job.ops) for _,_,e in initial_timeline.pq])
        self.max_ops = self.op_counts.max()

        self._reset_dag_batch(initial_timeline)

        self._reset_shared_obs_data()

        self.prev_episode_stats = \
            self._reset_envs(n_init_jobs, mjit, n_workers)

        print('done resetting')

        self.t_reset = time() - t_reset
        self.t_step = 0
        self.t_parse = 0
        self.t_subbatch = 0

        return self._observe()



    def get_prev_episode_stats(self):
        if self.prev_episode_stats is None:
            return None
        
        avg_job_duration_batch, n_completed_jobs_batch = self.prev_episode_stats

        avg_job_duration_mean = np.mean(avg_job_duration_batch)
        n_completed_jobs_mean = np.mean(n_completed_jobs_batch)
        return avg_job_duration_mean, n_completed_jobs_mean

        

    def step(self, action_batch):
        op_id_batch, prlvl_batch = action_batch
        self._step_envs(op_id_batch, prlvl_batch)
        return self._observe(), self.reward_batch, self.done_batch



    def find_job_indices(self, op_idx_batch):
        job_idx_batch = torch.zeros_like(op_idx_batch)
        op_id_batch = [None] * len(op_idx_batch)

        it = enumerate(zip(
            op_idx_batch, 
            self.active_job_msk_batch
        ))

        n_jobs_traversed = 0
        for i, (op_idx, active_job_msk) in it:
            op_idx = op_idx.item()
            active_job_ids = active_job_msk.nonzero().flatten()

            op_counts = self.op_counts[active_job_ids]
            cum = torch.cumsum(op_counts, 0)
            j = (op_idx >= cum).sum()

            job_idx_batch[i] = n_jobs_traversed + j

            job_id = active_job_ids[j]
            op_id = op_idx - (cum[j-1] if j>0 else 0)
            op_id_batch[i] = (job_id, op_id)

            n_jobs_traversed += len(active_job_ids)

        return job_idx_batch, op_id_batch




    ## reset helpers

    def _reset_dag_batch(self, initial_timeline):
        data_list = [e.job.init_pyg_data() for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list * self.n)



    def _reset_shared_obs_data(self):
        self.feature_tensor_chunks = self._chunk_feature_tensor()

        self.active_job_msk_batch = \
            torch.zeros((self.n, self.n_job_arrivals), dtype=torch.bool)

        self.op_msk_batch = \
            torch.zeros((self.n, self.n_job_arrivals, self.max_ops), dtype=torch.bool)

        self.prlvl_msk_batch = \
            torch.zeros((self.n, self.n_job_arrivals, self.n_workers), dtype=torch.bool)

        self.reward_batch = torch.zeros(self.n)

        self.done_batch = torch.zeros(self.n, dtype=torch.bool)



    def _chunk_feature_tensor(self):
        '''returns a list of chunks of the feature tensor,
        where there is one chunk per job, per environment,
        i.e. the list has length `self.n * self.n_job_arrivals`.
        each chunk has as many rows as there are operations
        in the corresponding job.
        '''
        ptr = self.dag_batch._slice_dict['x']
        num_nodes_per_graph = (ptr[1:] - ptr[:-1]).tolist()
        feature_tensor_chunks = torch.split(self.dag_batch.x, num_nodes_per_graph)
        return feature_tensor_chunks



    def _reset_envs(self, n_init_jobs, mjit, n_workers):
        it = zip(
            self.conns,
            self.active_job_msk_batch,
            self.op_msk_batch,
            self.prlvl_msk_batch,
            self.reward_batch,
            self.done_batch
        )

        for env_idx, conn, active_job_msk, op_msk, prlvl_msk, reward, done in enumerate(it):
            shared_obs = SharedObs(
                self._get_env_feature_tensor_chunks(env_idx),
                active_job_msk,
                op_msk,
                prlvl_msk,
                reward,
                done)

            conn.send(('reset', (
                self.n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers, 
                shared_obs
            )))
            
        avg_job_duration_batch, n_completed_jobs_batch = \
            list(zip(*[conn.recv() for conn in self.conns]))

        return avg_job_duration_batch, n_completed_jobs_batch



    def _get_env_feature_tensor_chunks(self, env_idx):
        '''get feature tensor chunks that correspond with 
        this environment only
        '''
        start = env_idx * self.n_job_arrivals
        end = start + self.n_job_arrivals
        return self.feature_tensor_chunks[start : end]




    ## step helpers

    def _step_envs(self, op_id_batch, prlvl_batch):
        i = 0
        for done, conn in zip(self.done_batch, self.conns):
            if not done.item():
                action = (op_id_batch[i], prlvl_batch[i].item())
                i += 1
            else:
                action = None

            conn.send(('step', action))

        t = time()
        [conn.recv() for conn in self.conns]
        self.t_step += time() - t



    def _observe(self):
        if self.done_batch.all():
            return None

        dag_batch = self._construct_dag_batch()
        op_msk_batch = self.op_msk_batch[self.active_job_msk_batch].flatten(end_dim=1)
        prlvl_msk_batch = self.prlvl_msk_batch[self.active_job_msk_batch].flatten(end_dim=1)
        
        obs = (dag_batch, op_msk_batch, prlvl_msk_batch)
        return obs



    def _construct_dag_batch(self):
        done = torch.repeat_interleave(self.done_batch, self.n_job_arrivals)

        # include job dags that are currently active and whose env is not done and 
        mask = ~done & self.active_job_msk_batch.flatten()

        subbatch = construct_subbatch(self.dag_batch, mask)
        return subbatch

