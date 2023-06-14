from typing import NamedTuple

import numpy as np

from .scheduler import Scheduler



class HeuristicObs(NamedTuple):
    job_ptr: np.ndarray
    frontier_stages: set
    schedulable_stages: dict
    executor_counts: np.ndarray
    num_executors_to_schedule: int
    source_job_idx: int



class HeuristicScheduler(Scheduler):
    '''Base class for all heuristic schedulers'''

    @classmethod
    def preprocess_obs(cls, obs):
        frontier_mask = np.ones(obs['dag_batch'].nodes.shape[0], dtype=bool)
        dst_nodes = obs['dag_batch'].edge_links[:, 1]
        frontier_mask[dst_nodes] = False
        frontier_stages = set(frontier_mask.nonzero()[0])

        job_ptr = np.array(obs['dag_ptr'])
        stage_mask = obs['dag_batch'].nodes[:, 2].astype(bool)
        schedulable_stages = dict(
            zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum())))
        executor_counts = np.array(obs['executor_counts'])
        num_executors_to_schedule = obs['num_executors_to_schedule']
        source_job_idx = obs['source_job_idx']

        return HeuristicObs(
            job_ptr,
            frontier_stages,
            schedulable_stages,
            executor_counts,
            num_executors_to_schedule,
            source_job_idx
        )

    
    @classmethod
    def find_stage(cls, obs, job_idx):
        '''searches for a schedulable stage in a given job, prioritizing frontier stages'''
        stage_idx_start = obs.job_ptr[job_idx]
        stage_idx_end = obs.job_ptr[job_idx+1]

        selected_stage_idx = -1
        for node in range(stage_idx_start, stage_idx_end):
            try:
                i = obs.schedulable_stages[node]
            except:
                continue

            if i in obs.frontier_stages:
                return i

            if selected_stage_idx == -1:
                selected_stage_idx = i

        return selected_stage_idx