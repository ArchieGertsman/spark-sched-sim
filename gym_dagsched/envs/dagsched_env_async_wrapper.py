

import torch
from torch_geometric.data import Batch
import numpy as np

from .dagsched_env import DagSchedEnv
from ..data_generation.tpch_datagen import TPCHDataGen
from ..utils.pyg import construct_subbatch, chunk_feature_tensor



class DagSchedEnvAsyncWrapper:

    def __init__(self, rank, datagen_state):
        self.env = DagSchedEnv(rank)
        self.datagen = TPCHDataGen(datagen_state)



    @property
    def active_job_ids(self):
        return self.env.active_job_ids

    @property
    def completed_job_ids(self):
        return self.env.completed_job_ids

    @property
    def n_completed_jobs(self):
        return self.env.n_completed_jobs

    @property
    def jobs(self):
        return self.env.jobs

    @property
    def wall_time(self):
        return self.env.wall_time



    def reset(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals, n_init_jobs, mjit)
        workers = self.datagen.workers(n_workers)

        self.n_workers = n_workers

        self.num_total_jobs = n_init_jobs + n_job_arrivals

        self.global_op_idx_map = {}
        op_idx = 0
        for _,_,e in initial_timeline.pq:
            job = e.job
            self.global_op_idx_map[job.id_] = {}
            for op in job.ops:
                self.global_op_idx_map[job.id_][op.id_] = op_idx
                op_idx += 1

        self.num_total_ops = op_idx

        self.active_op_mask = None
        self.dag_subbatch = None

        self._reset_dag_batch(initial_timeline)
        # self.feature_tensor_chunks = \
        #     chunk_feature_tensor(self.dag_batch)

        obs = self.env.reset(initial_timeline, workers)

        return self._parse_obs(obs)



    def step(self, action):
        action = self._parse_action(action)
        obs, reward, done = self.env.step(action)
        obs = self._parse_obs(obs) if not done else None
        return obs, reward, done



    def _parse_action(self, action):
        (job_id, active_op_idx), prlvl = action
        job = self.env.jobs[job_id]
        active_ops = list(job.active_ops)
        op_id = active_ops[active_op_idx].id_
        action = ((job_id, op_id), prlvl)
        return action



    def _reset_dag_batch(self, initial_timeline):
        data_list = [
            e.job.init_pyg_data() 
            for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list)



    def _parse_obs(self, obs):
        (n_source_workers,
        source_job_id, 
        valid_ops, 
        active_jobs,
        wall_time) = obs

        (active_job_mask, 
        active_op_mask, 
        op_counts, 
        valid_ops_mask) = \
            self._bookkeep(active_jobs, valid_ops)

        # valid_prlvl_msk = np.zeros((len(active_jobs), self.n_workers), dtype=bool)
        # valid_prlvl_msk[:, :n_source_workers] = 1

        new_dag_subbatch = \
            self._update_dag_subbatch(
                active_op_mask,
                active_job_mask,
                op_counts,
                active_jobs
            )

        node_features = \
            self._update_node_features(
                active_jobs,
                source_job_id,
                n_source_workers
            )

        active_job_ids = np.array(self.env.active_job_ids)

        self.active_op_mask = active_op_mask

        return node_features, \
            new_dag_subbatch, \
            valid_ops_mask, \
            active_job_ids, \
            n_source_workers, \
            wall_time



    def _bookkeep(self, active_jobs, valid_ops):
        active_job_mask = np.zeros(self.num_total_jobs, dtype=bool)
        active_op_mask = np.zeros(self.num_total_ops, dtype=bool)
        op_counts = []
        valid_ops_mask = []

        for job in active_jobs:
            active_job_mask[job.id_] = 1
            op_counts += [job.num_active_ops]
            for op in job.active_ops:
                global_idx = self.global_op_idx_map[job.id_][op.id_]
                active_op_mask[global_idx] = 1
                valid_ops_mask += [1] if op in valid_ops else [0]

        op_counts = np.array(op_counts, dtype=int)

        valid_ops_mask = np.array(valid_ops_mask, dtype=bool)

        return active_job_mask, \
            active_op_mask, \
            op_counts, \
            valid_ops_mask



    def _update_node_features(self,
        active_jobs,
        source_job_id,
        n_source_workers
    ):
        all_node_features = np.zeros(self.dag_subbatch.x.shape, dtype=np.float32)

        ptr = self.dag_subbatch.ptr[1:].numpy()

        node_features_split = np.split(
            all_node_features,
            ptr)

        for job, node_features in zip(active_jobs, node_features_split):
            worker_count = job.total_worker_count
            is_source_job = (job.id_ == source_job_id)

            node_features[:, 0] = n_source_workers / 20
            node_features[:, 1] = int(is_source_job) * 4 - 2
            node_features[:, 2] = worker_count / 20

            for i, op in enumerate(iter(job.active_ops)):
                node_features[i, 3] = op.n_remaining_tasks / 200
                node_features[i, 4] = op.approx_remaining_work / 1e5

        all_node_features = torch.from_numpy(all_node_features)
        return all_node_features



    def _update_dag_subbatch(self,
        active_op_mask,
        active_job_mask,
        op_counts,
        active_jobs
    ):
        '''if the number of active ops has changed
        since the last `step`, then we need to re-
        construct `self.dag_subbatch` to reflact
        that.

        Returns: new subbatch object if the number
        of active ops has changed, otherwise `None`,
        indicating that cached subbatch should be
        reused.
        '''

        new_dag_subbatch = None

        if self.active_op_mask is None or \
            (self.active_op_mask.shape != active_op_mask.shape or \
                not (self.active_op_mask == active_op_mask).all()):

            self.dag_subbatch = construct_subbatch(
                self.dag_batch, 
                active_job_mask, 
                active_op_mask, 
                op_counts,
                len(active_jobs))

            new_dag_subbatch = self.dag_subbatch

        return new_dag_subbatch