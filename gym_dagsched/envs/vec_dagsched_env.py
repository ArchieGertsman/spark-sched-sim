from dataclasses import dataclass
from typing import List
from time import time
from torch.multiprocessing import Process, Pipe


import torch
import numpy as np
from torch_geometric.data import Batch

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen

from .env_run import env_run
from ..utils.misc import construct_subbatch



@dataclass
class SharedObs:
    '''contains all of the shared memory tensors
    that each environment needs to communicate
    its observations with the main process,
    without having to send data through pipes. 
    Each environment gets its own instance of 
    this dataclass.
    '''

    # list of feature tensors for each job
    # in the environment. Rows correspond
    # to operations within the job, and columns
    # are the features.
    # shape[i]: torch.Size([num_ops_per_job[i], num_features])
    feature_tensor_chunks: List[torch.Tensor]

    # mask that indicated which jobs are
    # currently active
    # shape: torch.Size([num_job_arrivals])
    active_job_msk: torch.BoolTensor

    # mask that indicates which operations
    # are valid selections in each job.
    # shape: torch.Size([num_job_arrivals, max_ops_in_a_job])
    op_msk: torch.BoolTensor

    # mask that indicates which parallelism
    # limits are valid for each job
    # shape: torch.Size([num_job_arrivals, num_workers])
    prlvl_msk: torch.BoolTensor

    # reward signal
    # shape: torch.Size([])
    reward: torch.Tensor

    # whether or not the episode is done
    # shape: torch.Size([])
    done: torch.BoolTensor



class VecDagSchedEnv:
    def __init__(self, n):
        self.n = n
        self.datagen = TPCHDataGen(np.random.RandomState())



    def run(self):
        self.procs = []
        self.conns = []
        for rank in range(self.n):
            conn_main, conn_sub = Pipe()
            self.conns += [conn_main]
            proc = Process(target=env_run, args=(rank, self.datagen, conn_sub))
            self.procs += [proc]
            proc.start()



    def terminate(self):
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def reset(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        t_reset = time()

        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals, n_init_jobs, mjit)
        workers = self.datagen.workers(n_workers)

        self.n_job_arrivals = n_job_arrivals
        self._init_dag_batch(initial_timeline)

        self.n_workers = len(workers)
        self.op_counts = \
            torch.tensor([len(e.job.ops) for _,_,e in initial_timeline.pq])
        self.max_ops = self.op_counts.max()

        self._reset_shared_obs_data()

        it = zip(
            self.conns,
            self.active_job_msk_chunks,
            self.op_msk_chunks,
            self.prlvl_msk_chunks,
            self.reward_chunks,
            self.done_chunks
        )

        for i, conn, active_job_msk, op_msk, prlvl_msk, reward, done in enumerate(it):
            # get feature tensor chunks that correspond with 
            # this environment only
            start = i * self.n_job_arrivals
            end = start + self.n_job_arrivals
            feature_tensor_chunks = self.feature_tensor_chunks[start : end]

            shared_obs = SharedObs(
                feature_tensor_chunks,
                active_job_msk,
                op_msk,
                prlvl_msk,
                reward,
                done)

            conn.send(('reset', (n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs)))
            
        avg_job_duration_batch, n_completed_jobs_batch = \
            list(zip(*[conn.recv() for conn in self.conns]))

        print('done resetting')

        self.t_reset = time() - t_reset
        self.t_step = 0
        self.t_parse = 0
        self.t_subbatch = 0

        return self._parse_prev_episode_stats(avg_job_duration_batch, n_completed_jobs_batch)


    
    def _reset_shared_obs_data(self):
        self.feature_tensor_chunks = self._get_feature_tensor_chunks()

        self.active_job_msk_batch, self.active_job_msk_chunks = \
            self._zeros_batch((self.n_job_arrivals,))

        self.op_msk_batch, self.op_msk_chunks = \
            self._zeros_batch((self.n_job_arrivals, self.max_ops))

        self.prlvl_msk_batch, self.prlvl_msk_chunks = \
            self._zeros_batch((self.n_job_arrivals, self.n_workers))

        self.reward_batch, self.reward_chunks = self._zeros_batch((1,))

        self.done_batch, self.done_chunks = self._zeros_batch((1,))



    def _zeros_batch(self, shape):
        shape = list(shape)
        shape[0] *= self.n
        batch = torch.zeros(shape)
        chunks = torch.chunk(batch, self.n)
        return batch, chunks



    def _get_feature_tensor_chunks(self):
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



    @classmethod
    def _parse_prev_episode_stats(cls, avg_job_duration_batch, n_completed_jobs_batch):
        if avg_job_duration_batch[0] is not None:
            avg_job_duration_mean = np.mean(avg_job_duration_batch)
            n_completed_jobs_mean = np.mean(n_completed_jobs_batch)
        else:
            avg_job_duration_mean, n_completed_jobs_mean = None, None
        return avg_job_duration_mean, n_completed_jobs_mean

        

    def step(self, op_id_batch=None, prlvl_batch=None):
        i = 0
        for done, conn in zip(self.done_batch, self.conns):
            if not done and prlvl_batch is not None:
                op_id, prlvl = op_id_batch[i], prlvl_batch[i].item()
                i += 1
            else:
                op_id, prlvl = (None, None), None

            conn.send(('step', (op_id, prlvl)))

        t = time()
        [conn.recv() for conn in self.conns]
        self.t_step += time() - t

        return self._observe(), self.reward_batch, self.done_batch



    def _observe(self):
        if self.done_batch.all():
            return None

        dag_batch = self._construct_dag_batch()
        op_msk_batch = self.op_msk_batch[self.active_job_msk_batch]
        prlvl_msk_batch = self.prlvl_msk_batch[self.active_job_msk_batch]
        
        obs = (dag_batch, op_msk_batch, prlvl_msk_batch)
        return obs

    

    def _init_dag_batch(self, initial_timeline):
        data_list = [e.job.init_pyg_data() for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list * self.n)



    def _construct_dag_batch(self):
        done = torch.repeat_interleave(self.done_batch, self.n_job_arrivals)

        # include job dags that are currently active and whose env is not done and 
        mask = ~done & self.active_job_msk_batch

        subbatch = construct_subbatch(self.dag_batch, mask)
        return subbatch



    def find_job_indices(self, op_idx_batch):
        job_idx_batch = torch.zeros_like(op_idx_batch)
        op_id_batch = [None] * len(op_idx_batch)

        it = enumerate(zip(
            op_idx_batch, 
            self.active_job_msk_chunks
        ))

        n_jobs_traversed = 0
        for i, (op_idx, active_job_msk) in it:
            active_job_ids = active_job_msk.nonzero().flatten()

            op_counts = self.op_counts[active_job_ids]
            cum = torch.cumsum(op_counts, 0)
            j = (op_idx >= cum).sum()

            job_idx_batch[i] = n_jobs_traversed + j

            job_id = active_job_ids[j]
            op_id = op_idx - (cum[j-1] if j>0 else 0)
            op_id_batch[i] = (job_id.item(), op_id.item())

            n_jobs_traversed += len(active_job_ids)

        return job_idx_batch, op_id_batch

