

import torch
from torch_geometric.data import Batch
import numpy as np

from .dagsched_env import DagSchedEnv
from ..data_generation.tpch_datagen import TPCHDataGen
from ..utils.pyg import construct_subbatch



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



    def reset(self, 
              num_job_arrivals, 
              num_init_jobs, 
              job_arrival_rate, 
              num_workers):

        mean_job_interarrival_time = 1 / job_arrival_rate

        initial_timeline = \
            self.datagen.initial_timeline(num_job_arrivals, 
                                          num_init_jobs, 
                                          mean_job_interarrival_time)

        workers = self.datagen.workers(num_workers)

        self.num_workers = num_workers

        self.num_total_jobs = num_init_jobs + num_job_arrivals

        self._reset_global_op_idx_map(initial_timeline)

        self._reset_dag_batch(initial_timeline)

        self.active_op_mask = None
        self.dag_subbatch = None

        obs = self.env.reset(initial_timeline, workers)
        return self._parse_obs(obs)



    def step(self, action):
        action = self._parse_action(action)
        obs, reward, done = self.env.step(action)
        obs = self._parse_obs(obs) if not done else None
        return obs, reward, done



    def _parse_action(self, action):
        (job_id, active_op_idx), prlsm_lim = action

        # parse operation
        job = self.env.jobs[job_id]
        active_ops = list(job.active_ops)
        op_id = active_ops[active_op_idx].id_

        # parse parallelism limit
        _, num_source_workers, source_job_id, *_ = self.last_obs
        worker_count = prlsm_lim - job.total_worker_count
        if job.id_ == source_job_id:
            worker_count += num_source_workers
        
        assert worker_count >= 1

        action = ((job_id, op_id), worker_count)
        return action



    def _reset_dag_batch(self, initial_timeline):
        data_list = [e.job.init_pyg_data() 
                     for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list)


    def _reset_global_op_idx_map(self, initial_timeline):
        self.global_op_idx_map = {}
        op_idx = 0
        for _,_,e in initial_timeline.pq:
            job = e.job
            self.global_op_idx_map[job.id_] = {}
            for op in job.ops:
                self.global_op_idx_map[job.id_][op.id_] = op_idx
                op_idx += 1

        self.num_total_ops = op_idx



    def _parse_obs(self, obs):
        self.last_obs = obs

        (new_commitment_round,
         num_source_workers,
         source_job_id, 
         valid_ops, 
         active_jobs,
         wall_time) = obs

        (active_job_mask, 
         active_op_mask, 
         op_counts, 
         valid_ops_mask,
         valid_prlsm_lim_mask) = \
            self._bookkeep(active_jobs, 
                           valid_ops,
                           source_job_id,
                           num_source_workers)

        did_update = \
            self._update_dag_subbatch(active_op_mask,
                                      active_job_mask,
                                      op_counts,
                                      active_jobs)

        self._update_node_features(new_commitment_round,
                                   active_jobs,
                                   source_job_id,
                                   num_source_workers)

        active_job_ids = np.array(self.env.active_job_ids)

        self.active_op_mask = active_op_mask

        return did_update, \
            self.dag_subbatch, \
            valid_ops_mask, \
            valid_prlsm_lim_mask, \
            active_job_ids, \
            num_source_workers, \
            wall_time



    def _bookkeep(self, 
                  active_jobs, 
                  valid_ops,
                  source_job_id,
                  num_source_workers):
        active_job_mask = np.zeros(self.num_total_jobs, dtype=bool)
        active_op_mask = np.zeros(self.num_total_ops, dtype=bool)
        op_counts = []
        valid_ops_mask = []
        valid_prlsm_lim_mask = \
            np.zeros((len(active_jobs), self.num_workers), 
                     dtype=bool)

        for i, job in enumerate(active_jobs):
            active_job_mask[job.id_] = 1
            op_counts += [job.num_active_ops]

            # build parallelism limit mask
            min_prlsm_lim = job.total_worker_count + 1
            max_prlsm_lim = job.total_worker_count + num_source_workers
            if job.id_ == source_job_id:
                min_prlsm_lim -= num_source_workers
                max_prlsm_lim -= num_source_workers
            print('IS SOURCE JOB:', job.id_ == source_job_id)
            print(min_prlsm_lim, max_prlsm_lim)

            assert 0 < min_prlsm_lim
            assert     min_prlsm_lim <= self.num_workers + 1

            valid_prlsm_lim_mask[i, (min_prlsm_lim-1):max_prlsm_lim] = 1

            for op in job.active_ops:
                # build operation masks
                global_idx = self.global_op_idx_map[job.id_][op.id_]
                active_op_mask[global_idx] = 1
                if min_prlsm_lim <= self.num_workers and \
                   op in valid_ops:
                    valid_ops_mask += [1]
                else:
                    valid_ops_mask += [0]

        op_counts = np.array(op_counts, dtype=int)

        valid_ops_mask = np.array(valid_ops_mask, dtype=bool)

        return active_job_mask, \
               active_op_mask, \
               op_counts, \
               valid_ops_mask, \
               valid_prlsm_lim_mask



    def _update_node_features(self,
                              new_commitment_round,
                              active_jobs,
                              source_job_id,
                              n_source_workers):
        # make updates in an auxiliary numpy array 
        # instead of the tensor directly, because 
        # numpy is faster for cpu computations
        all_node_features = \
            np.zeros(self.dag_subbatch.x.shape, dtype=np.float32)

        ptr = self.dag_subbatch.ptr[1:].numpy()

        node_features_split = \
            np.split(all_node_features, ptr)

        all_node_features[:, 0] = n_source_workers / 20

        for job, node_features in zip(active_jobs, node_features_split):
            node_features[:, 2] = job.total_worker_count / 20

            # if not new_commitment_round:
            #     # if we are in the same commitment round
            #     # then none of the other features could
            #     # have changed.
            #     continue

            is_source_job = (job.id_ == source_job_id)
            node_features[:, 1] = int(is_source_job) * 4 - 2

            for i, op in enumerate(iter(job.active_ops)):
                node_features[i, 3] = op.n_remaining_tasks / 200
                node_features[i, 4] = op.approx_remaining_work / 1e5

        all_node_features = torch.from_numpy(all_node_features)
        self.dag_subbatch.x = all_node_features



    def _update_dag_subbatch(self,
                             active_op_mask,
                             active_job_mask,
                             op_counts,
                             active_jobs):
        '''if the number of active ops has changed
        since the last `step`, then we need to re-
        construct `self.dag_subbatch` to reflact
        that.

        Returns whether or not `self.dag_subbatch`
        was updated.
        '''

        did_update = False

        if self.active_op_mask is None or \
           (self.active_op_mask.shape != active_op_mask.shape or \
            not (self.active_op_mask == active_op_mask).all()):

            self.dag_subbatch = construct_subbatch(
                self.dag_batch, 
                active_job_mask, 
                active_op_mask, 
                op_counts,
                len(active_jobs))

            did_update = True

        return did_update