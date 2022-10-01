from copy import deepcopy as dcp
from collections import defaultdict
from time import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx

from .dagsched_env import DagSchedEnv
from ..entities.operation import FeatureIdx

class VecDagSchedEnv:
    def __init__(self, n):
        self.n = n
        self.envs = [DagSchedEnv(rank) for rank in range(n)]


    def reset(self, initial_timeline, workers):
        self.timeline = initial_timeline
        self.n_job_arrivals = len(initial_timeline.pq)
        self._init_dag_batch()

        for i, env in enumerate(self.envs):
            x_ptrs = [self._get_x_ptr(i, j) for j in range(self.n_job_arrivals)]
            env.reset(initial_timeline, workers, x_ptrs)

        self.t_step = 0
        self.t_observe = [0,0,0]


    def step(self, op_vec, prlvl_vec):
        rewards = torch.zeros(self.n)
        dones = torch.zeros(self.n, dtype=torch.bool)

        t = time()
        j = 0
        for i, env in enumerate(self.envs):
            if env._are_actions_available() and prlvl_vec is not None:
                op, prlvl = op_vec[j], prlvl_vec[j].item()
                j += 1
            else:
                op, prlvl = None, None

            reward, done = env.step(op, prlvl)
            dones[i] = done
            rewards[i] = reward

        self.t_step += time() - t

        # t = time()
        obs = self._observe()
        if obs is None:
            return self.step(None, None)
        # self.t_observe += time() - t

        return obs, rewards, dones

    
    def _init_dag_batch(self):
        data_list = []

        for _,_,e in self.timeline.pq:
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
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)
        mask[env_idx * self.n_job_arrivals + job_idx] = True
        mask = mask[self.dag_batch.batch]
        idx = mask.nonzero().flatten()
        return self.dag_batch.x[idx[0] : idx[-1]+1]


    def _observe(self):
        if len([0 for env in self.envs if env._are_actions_available()]) == 0:
            return None

        t = time()
        subbatch = self._construct_subbatch()
        self.t_observe[0] += time() - t

        t = time()
        op_msk_batch = torch.cat([env._construct_op_msk() for env in self.envs if env._are_actions_available()])
        self.t_observe[1] += time() - t

        t = time()
        prlvl_msk_batch = torch.cat([env._construct_prlvl_msk() for env in self.envs if env._are_actions_available()])
        self.t_observe[2] += time() - t

        return subbatch, op_msk_batch, prlvl_msk_batch


    def _construct_subbatch(self):
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)

        # t = time()
        for i,env in enumerate(self.envs):
            if env._are_actions_available():
                mask[i * self.n_job_arrivals + torch.tensor(env.active_job_ids)] = True
        # self.t_observe[0] += time() - t

        node_mask = mask[self.dag_batch.batch]

        # t = time()
        subbatch = self.dag_batch.subgraph(node_mask)
        # self.t_observe[1] += time() - t



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

        subbatch._inc_dict = defaultdict(
            dict, {
                'x': torch.zeros(subbatch.num_graphs, dtype=torch.long),
                'edge_index': ptr[:-1]
            })

        subbatch._slice_dict = defaultdict(dict, {
            'x': ptr,
            'edge_index': edge_ptr
        })



        # t = time()

        n_avail = torch.tensor([len(env.avail_worker_ids) for env in self.envs if env._are_actions_available()])
        n_avail = torch.repeat_interleave(n_avail, self.num_jobs_per_participating_env())
        n_avail = n_avail[subbatch.batch]
        subbatch.x[:, FeatureIdx.N_AVAIL_WORKERS] = n_avail

        # self.t_observe[2] += time() - t

        return subbatch


    def find_op_batch(self, op_idx_batch):
        n_jobs_traversed = 0
        i = 0
        job_idx_batch = [None] * len(op_idx_batch)
        op_batch = [None] * len(op_idx_batch)

        for env in self.envs:
            if not env._are_actions_available():
                continue
            op_idx = op_idx_batch[i]
            n_ops_traversed = 0
            for j, job_id in enumerate(env.active_job_ids):
                job = env.jobs[job_id]
                if op_idx < n_ops_traversed + len(job.ops):
                    op_batch[i] = job.ops[op_idx - n_ops_traversed]
                    job_idx_batch[i] = n_jobs_traversed + j
                    break
                else:
                    n_ops_traversed += len(job.ops)
            n_jobs_traversed += env.n_active_jobs
            i += 1
            
        return op_batch, job_idx_batch



    def num_jobs_per_participating_env(self):
        return torch.tensor(
            [env.n_active_jobs for env in self.envs if env._are_actions_available()], 
            dtype=int)
