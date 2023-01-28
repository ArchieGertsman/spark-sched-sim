

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx

from .dagsched_env import DagSchedEnv
from ..data_generation.tpch_datagen import TPCHDataGen
from ..utils.pyg import construct_subbatch



class DagSchedEnvDecimaWrapper:
    '''Wrapper around `DagSchedEnv`, which parses
    observations from the env into model inputs,
    and parses actions from the agent for the env
    '''

    def __init__(self, env):
        self.env = env

    @property
    def active_job_ids(self):
        return self.env.active_job_ids

    @property
    def completed_job_ids(self):
        return self.env.completed_job_ids

    @property
    def num_completed_jobs(self):
        return self.env.num_completed_jobs

    @property
    def jobs(self):
        return self.env.jobs

    @property
    def wall_time(self):
        return self.env.wall_time



    def reset(self, 
              num_init_jobs, 
              num_job_arrivals, 
              job_arrival_rate, 
              num_workers,
              max_wall_time):

        obs = self.env.reset(num_init_jobs, 
                             num_job_arrivals, 
                             job_arrival_rate, 
                             num_workers,
                             max_wall_time)

        self.num_workers = num_workers

        self.num_total_jobs = num_init_jobs + num_job_arrivals

        self._reset_op_idx_map()

        self._reset_dag_batch()

        # induced subgraph of `self.dag_batch`, which
        # only contains nodes that correspond with
        # currently active operations. Only gets
        # reconstructed when set of active operations
        # changes.
        self.dag_subbatch = None

        # mask indicating which operations are currently
        # active. This gets recomputed during every step,
        # but the previous one is saved just to see if the 
        # set of nodes has changed
        self.prev_active_op_mask = None

        return self._parse_obs(obs)



    def step(self, action):
        action = self._parse_action(action)
        obs, reward, done = self.env.step(action)
        obs = self._parse_obs(obs)
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
        
        assert 1 <= worker_count
        # assert      worker_count <= num_source_workers

        worker_count = min(worker_count, num_source_workers)

        action = ((job_id, op_id), worker_count)
        return action



    def _reset_dag_batch(self):
        data_list = []
        for job in self.env.jobs.values():
            pyg_data = from_networkx(job.dag)
            pyg_data.x = torch.zeros((len(job.ops), 5))
            data_list += [pyg_data]

        self.dag_batch = Batch.from_data_list(data_list)



    def _reset_op_idx_map(self):
        '''maps the 'global' index of an operation (i.e. relative 
        to all operations in the entirety of the simulation) to
        the 'local' index of that operation (i.e. relative to its 
        job dag)
        '''
        self.op_idx_map = {}
        global_idx = 0
        for job in self.env.jobs.values():
            self.op_idx_map[job.id_] = {}
            for op in job.ops:
                self.op_idx_map[job.id_][op.id_] = global_idx
                global_idx += 1

        self.num_total_ops = global_idx



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

        active_job_ids = list(active_jobs.keys())

        self.prev_active_op_mask = active_op_mask

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

        for i, job in enumerate(active_jobs.values()):
            active_job_mask[job.id_] = 1
            op_counts += [job.num_active_ops]

            # build parallelism limit mask
            min_prlsm_lim = job.total_worker_count + 1
            # max_prlsm_lim = job.total_worker_count + num_source_workers
            if job.id_ == source_job_id:
                min_prlsm_lim -= num_source_workers
                # max_prlsm_lim -= num_source_workers

            assert 0 < min_prlsm_lim
            assert     min_prlsm_lim <= self.num_workers + 1

            valid_prlsm_lim_mask[i, (min_prlsm_lim-1):] = 1

            for op in iter(job.active_ops):
                # build operation masks
                global_idx = self.op_idx_map[job.id_][op.id_]
                active_op_mask[global_idx] = 1
                if min_prlsm_lim > self.num_workers:
                    assert op not in valid_ops
                valid_ops_mask += [1] if op in valid_ops else [0]

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
                              num_source_workers):
        # make updates in an auxiliary numpy array 
        # instead of the tensor directly, because 
        # numpy is faster for cpu computations
        all_node_features = \
            np.zeros(self.dag_subbatch.x.shape, dtype=np.float32)

        all_node_features[:, 0] = num_source_workers / self.num_workers

        ptr = self.dag_subbatch.ptr[1:].numpy()

        job_features_list = \
            np.split(all_node_features, ptr)

        for job, job_features in zip(active_jobs.values(), job_features_list):
            job_features[:, 2] = job.total_worker_count / self.num_workers

            # if not new_commitment_round:
            #     # if we are in the same commitment round
            #     # then none of the other features could
            #     # have changed.
            #     continue

            is_source_job = (job.id_ == source_job_id)
            job_features[:, 1] = int(is_source_job) * 4 - 2

            for i, op in enumerate(iter(job.active_ops)):
                job_features[i, 3] = op.num_remaining_tasks / 200
                job_features[i, 4] = op.approx_remaining_work * 1e-5

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

        if self.prev_active_op_mask is None or \
           (self.prev_active_op_mask.shape != active_op_mask.shape or \
            not (self.prev_active_op_mask == active_op_mask).all()):

            self.dag_subbatch = \
                construct_subbatch(self.dag_batch, 
                                   active_job_mask, 
                                   active_op_mask, 
                                   op_counts,
                                   len(active_jobs))

            did_update = True

        return did_update