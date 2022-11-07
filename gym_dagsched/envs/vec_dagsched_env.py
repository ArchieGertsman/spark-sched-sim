import sys
from copy import deepcopy as dcp
from copy import copy
from collections import defaultdict
from time import time
from torch.multiprocessing import Process, Pipe


import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen

from .dagsched_env import DagSchedEnv
from ..entities.operation import FeatureIdx

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
            proc = Process(target=self._env_step, args=(rank, self.datagen, conn_sub))
            self.procs += [proc]
            proc.start()



    def terminate(self):
        [conn.send(None) for conn in self.conns]
        [proc.join() for proc in self.procs]



    def _env_step(self, rank, datagen, conn):
        torch.manual_seed(rank)
        np.random.seed(rank)

        env = DagSchedEnv(rank)


        while header_data := conn.recv():
            header, data = header_data

            if header == 'reset':
                x_ptrs, shared_obs_tensor = data

                initial_timeline = datagen.initial_timeline(
                    n_job_arrivals=200, n_init_jobs=1, mjit=25000.)
                workers = datagen.workers(n_workers=50)

                env.reset(initial_timeline, workers, x_ptrs)

                conn.send(None)


            elif header == 'step':
                (job_id, op_id), prlvl = data

                t = time()

                reward, done = env.step(job_id, op_id, prlvl)

                active_jobs_msk = torch.zeros(env.n_total_jobs, dtype=torch.bool)
                active_jobs_msk[env.active_job_ids] = 1

                put_data = [
                    torch.tensor([env.are_actions_available]),
                    env.construct_op_msk().flatten(),
                    env.construct_prlvl_msk().flatten(), 
                    active_jobs_msk,
                    torch.tensor([len(env.avail_worker_ids)]),
                    torch.tensor([reward]), 
                    torch.tensor([done])
                ]

                t = torch.tensor([time() - t])
                put_data += [t]

                torch.cat(put_data, out=shared_obs_tensor)
                
                conn.send(None)


            else:
                raise Exception(f'proc {rank} received invalid data')



    def reset(self):
        t_reset = time()

        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals=200, n_init_jobs=1, mjit=25000.)
        workers = self.datagen.workers(n_workers=50)

        self.n_job_arrivals = len(initial_timeline.pq)
        self._init_dag_batch(initial_timeline)
        self.dag_batch.share_memory_()

        self.n_total_jobs = len(initial_timeline.pq)
        self.n_workers = len(workers)
        self.n_ops_per_job = torch.tensor([len(e.job.ops) for _,_,e in initial_timeline.pq])
        self.max_ops = self.n_ops_per_job.max()

        self.shared_obs_tensor_sizes = [ \
            1,                                      # participating
            self.n_total_jobs * self.max_ops,       # op mask
            self.n_total_jobs * self.n_workers,     # prlvl pask
            self.n_total_jobs,                      # active job mask
            1,  # num available workers
            1,  # reward
            1,  # done
            1   # time of step
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

            conn.send(('reset', (x_ptrs, shared_obs_tensor)))
            
        [conn.recv() for conn in self.conns]
        print('done resetting')

        self.participating_msk = torch.ones(self.n, dtype=torch.bool)

        self.t_reset = time() - t_reset
        self.t_substep = 0
        self.t_step = 0
        self.t_observe = [0,0,0]

        



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

        parsed = self._parse_results()
        
        if parsed is not None:
            op_msk_batch, prlvl_msk_batch, rewards, dones, t_substep = parsed
            self.t_substep += t_substep
            subbatch = self._construct_subbatch()
            obs = (subbatch, op_msk_batch, prlvl_msk_batch)
            return obs, rewards, dones
        else:
            # recursively keep trying until we see a valid observation
            return self.step(None, None)



    def _parse_results(self):
        participating_msk = []
        op_msk_batch = []
        prlvl_msk_batch = []
        active_job_ids_batch = []
        n_avail_workers_batch = []
        n_ops_per_job_batch = []
        rewards = []
        dones = []
        ts = []

        any_participating = False

        for split_shared_obs_tensor in self.split_shared_obs_tensors:
            (participating,
                op_msk,
                prlvl_msk,
                active_job_msk,
                n_avail_workers,
                reward,
                done,
                t) = split_shared_obs_tensor
            
            ts += [t]

            participating_msk += [participating.item()]

            active_job_msk = active_job_msk.bool()
            
            if participating:
                any_participating = True

                op_msk = op_msk.reshape(self.n_total_jobs, self.max_ops).bool()
                op_msk_batch += [op_msk[active_job_msk]]

                prlvl_msk = prlvl_msk.reshape(self.n_total_jobs, self.n_workers).bool()
                prlvl_msk_batch += [prlvl_msk[active_job_msk]]

                active_job_ids_batch += [active_job_msk.nonzero().flatten()]
                n_avail_workers_batch += [n_avail_workers]
                n_ops_per_job_batch += [self.n_ops_per_job[active_job_msk]]

            rewards += [reward.item()]
            dones += [done.item()]

        if not any_participating:
            print('--- HERE ---')
            return None

        self.participating_msk = torch.tensor(participating_msk, dtype=torch.bool)
        self.active_job_ids_batch = active_job_ids_batch
        self.n_active_jobs_batch = [len(active_job_ids) for active_job_ids in self.active_job_ids_batch]
        self.n_avail_workers_batch = torch.tensor(n_avail_workers_batch)
        self.n_ops_per_job_batch = torch.cat(n_ops_per_job_batch)

        op_msk_batch = torch.cat(op_msk_batch)
        prlvl_msk_batch = torch.cat(prlvl_msk_batch)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.bool)

        t_substep = max(ts)

        return op_msk_batch, prlvl_msk_batch, rewards, dones, t_substep

    

    def _init_dag_batch(self, initial_timeline):
        data_list = []

        for _,_,e in initial_timeline.pq:
            job = e.job
            data = from_networkx(job.dag)
            data.x = torch.tensor(
                job.init_feature_vectors(),
                dtype=torch.float32
            )
            data_list += [data]

        data_list_repeated = []
        for _ in range(self.n):
            data_list_repeated += dcp(data_list)

        self.dag_batch = Batch.from_data_list(data_list_repeated)



    def _get_x_ptr(self, env_idx, job_idx):
        mask = torch.zeros(self.n * self.n_job_arrivals, dtype=torch.bool)
        mask[env_idx * self.n_job_arrivals + job_idx] = True
        mask = mask[self.dag_batch.batch]
        idx = mask.nonzero().flatten()
        return self.dag_batch.x[idx[0] : idx[-1]+1]



    def _construct_subbatch(self):
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)

        participating_idxs = self.participating_msk.nonzero().flatten()
        for i, active_job_ids in zip(participating_idxs, self.active_job_ids_batch):
            mask[i * self.n_job_arrivals + active_job_ids] = True

        node_mask = mask[self.dag_batch.batch]

        subbatch = self.dag_batch.subgraph(node_mask)

        subbatch._num_graphs = mask.sum().item()

        assoc = torch.empty(self.dag_batch.num_graphs, dtype=torch.long)
        assoc[mask] = torch.arange(subbatch.num_graphs)
        subbatch.batch = assoc[self.dag_batch.batch][node_mask]

        ptr = self.dag_batch._slice_dict['x']
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        ptr = torch.cumsum(num_nodes_per_graph[mask], 0)
        ptr = torch.cat([torch.tensor([0]), ptr])
        subbatch.ptr = ptr

        edge_ptr = self.dag_batch._slice_dict['edge_index']
        num_edges_per_graph = edge_ptr[1:] - edge_ptr[:-1]
        edge_ptr = torch.cumsum(num_edges_per_graph[mask], 0)
        edge_ptr = torch.cat([torch.tensor([0]), edge_ptr])

        subbatch._inc_dict = defaultdict(dict, {
            'x': torch.zeros(subbatch.num_graphs, dtype=torch.long),
            'edge_index': ptr[:-1]
        })

        subbatch._slice_dict = defaultdict(dict, {
            'x': ptr,
            'edge_index': edge_ptr
        })

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

