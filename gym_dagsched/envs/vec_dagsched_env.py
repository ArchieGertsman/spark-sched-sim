from collections import defaultdict
from time import time
from torch.multiprocessing import Process, Pipe


import torch
import numpy as np
from torch_geometric.data import Batch

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen

from .env_run import env_run
from ..entities.operation import FeatureIdx
from ..utils.misc import construct_subbatch




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
        self.n_ops_per_job = torch.tensor([len(e.job.ops) for _,_,e in initial_timeline.pq])
        self.max_ops = self.n_ops_per_job.max()

        self.shared_obs_tensor_sizes = [ \
            1,                                      # participating
            self.n_job_arrivals * self.max_ops,       # op mask
            self.n_job_arrivals * (self.n_workers+1),     # prlvl pask
            self.n_job_arrivals,                      # active job mask
            1,  # num available workers
            1,  # reward
            1,  # done
        ]

        shared_obs_tensor_shape = (sum(self.shared_obs_tensor_sizes),)

        ptr = self.dag_batch._slice_dict['x']
        num_nodes_per_graph = (ptr[1:] - ptr[:-1]).tolist()
        x_ptrs_all = torch.split(self.dag_batch.x, num_nodes_per_graph)

        self.split_shared_obs_tensors = []

        for i, conn in enumerate(self.conns):
            start = i * self.n_job_arrivals
            end = start + self.n_job_arrivals
            x_ptrs = x_ptrs_all[start : end]

            shared_obs_tensor = torch.empty(shared_obs_tensor_shape)
            split_shared_obs_tensor = torch.split(shared_obs_tensor, self.shared_obs_tensor_sizes)
            self.split_shared_obs_tensors += [split_shared_obs_tensor]

            conn.send(('reset', (n_job_arrivals, n_init_jobs, mjit, n_workers, x_ptrs, shared_obs_tensor)))
            
        avg_job_duration_batch, n_completed_jobs_batch = \
            list(zip(*[conn.recv() for conn in self.conns]))
        print('done resetting')

        if avg_job_duration_batch[0] is not None:
            avg_job_duration_mean = np.mean(avg_job_duration_batch)
            n_completed_jobs_mean = np.mean(n_completed_jobs_batch)
        else:
            avg_job_duration_mean, n_completed_jobs_mean = None, None

        self.participating_msk = torch.ones(self.n, dtype=torch.bool)

        self.t_reset = time() - t_reset
        self.t_step = 0
        self.t_parse = 0
        self.t_subbatch = 0

        return avg_job_duration_mean, n_completed_jobs_mean

        

    def step(self, op_id_batch=None, prlvl_batch=None):
        i = 0
        for participating, conn in zip(self.participating_msk, self.conns):
            if participating and prlvl_batch is not None:
                op_id, prlvl = op_id_batch[i], prlvl_batch[i].item()
                i += 1
            else:
                op_id, prlvl = (None, None), None

            conn.send(('step', (op_id, prlvl)))

        t = time()
        [conn.recv() for conn in self.conns]
        self.t_step += time() - t

        t = time()
        op_msk_batch, prlvl_msk_batch, rewards, dones = self._parse_observations()
        self.t_parse += time() - t
        
        if op_msk_batch is not None:
            t = time()
            subbatch = self._construct_subbatch()
            self.t_subbatch += time() - t

            obs = (subbatch, op_msk_batch, prlvl_msk_batch)
            return obs, rewards, False
        else:
            if all(dones):
                return None, None, True
            else:
                # recursively keep trying until at least one env has available actions
                return self.step(None, None)



    def _parse_observations(self):
        participating_msk = []
        op_msk_batch = []
        prlvl_msk_batch = []
        active_job_ids_batch = []
        n_avail_workers_batch = []
        n_ops_per_job_batch = []
        rewards = []
        dones = []

        any_participating = False

        for split_shared_obs_tensor in self.split_shared_obs_tensors:
            (participating,
                op_msk,
                prlvl_msk,
                active_job_msk,
                n_avail_workers,
                reward,
                done) = split_shared_obs_tensor

            participating_msk += [participating.item()]

            active_job_msk = active_job_msk.bool()
            
            if participating:
                any_participating = True

                op_msk = op_msk.reshape(self.n_job_arrivals, self.max_ops).bool()
                op_msk_batch += [op_msk[active_job_msk]]

                prlvl_msk = prlvl_msk.reshape(self.n_job_arrivals, self.n_workers+1).bool()
                prlvl_msk_batch += [prlvl_msk[active_job_msk]]

                active_job_ids_batch += [active_job_msk.nonzero().flatten()]
                n_avail_workers_batch += [n_avail_workers]
                n_ops_per_job_batch += [self.n_ops_per_job[active_job_msk]]

            rewards += [reward.item()]
            dones += [done.item()]

        if not any_participating:
            print('--- HERE ---')
            return None, None, None, dones

        self.participating_msk = torch.tensor(participating_msk, dtype=torch.bool)
        self.active_job_ids_batch = active_job_ids_batch
        self.n_active_jobs_batch = [len(active_job_ids) for active_job_ids in self.active_job_ids_batch]
        self.n_avail_workers_batch = torch.tensor(n_avail_workers_batch)
        self.n_ops_per_job_batch = torch.cat(n_ops_per_job_batch)

        op_msk_batch = torch.cat(op_msk_batch)
        prlvl_msk_batch = torch.cat(prlvl_msk_batch)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.bool)

        return op_msk_batch, prlvl_msk_batch, rewards, dones

    

    def _init_dag_batch(self, initial_timeline):
        data_list = [e.job.init_pyg_data() for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list * self.n)



    def _construct_subbatch(self):
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)

        participating_idxs = self.participating_msk.nonzero().flatten()
        for i, active_job_ids in zip(participating_idxs, self.active_job_ids_batch):
            mask[i * self.n_job_arrivals + active_job_ids] = True

        subbatch = construct_subbatch(self.dag_batch, mask)

        n_avail = torch.repeat_interleave(
            self.n_avail_workers_batch, 
            torch.tensor(self.n_active_jobs_batch)
        )
        n_avail = n_avail[subbatch.batch]
        subbatch.x[:, FeatureIdx.N_AVAIL_WORKERS] = n_avail

        return subbatch



    def find_job_indices(self, op_idx_batch):
        job_idx_batch = torch.zeros_like(op_idx_batch)
        op_id_batch = [None] * len(op_idx_batch)

        it = enumerate(zip(
            op_idx_batch, 
            self.active_job_ids_batch
        ))

        n_jobs_traversed = 0
        for i, (op_idx, active_job_ids) in it:
            n_ops_per_job = self.n_ops_per_job[active_job_ids]
            cum = torch.cumsum(n_ops_per_job, 0)
            j = (op_idx >= cum).sum()

            job_idx_batch[i] = n_jobs_traversed + j

            job_id = active_job_ids[j]
            op_id = op_idx - (cum[j-1] if j>0 else 0)
            op_id_batch[i] = (job_id.item(), op_id.item())

            n_jobs_traversed += len(active_job_ids)

        return job_idx_batch, op_id_batch

