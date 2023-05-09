from typing import NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
from gymnasium.core import ObsType, ActType



class Scheduler(ABC):
    def __init__(self, name: str):
        self.name = name


    def __call__(self, obs: ObsType) -> ActType:
        return self.schedule(obs)
        

    @abstractmethod
    def schedule(self, obs: ObsType) -> ActType:
        '''must be implemented for every agent.
        Takes an observation of the environment,
        and returns an action.
        '''




class HeuristicObs(NamedTuple):
    G: nx.DiGraph
    job_ptr: np.ndarray
    frontier_stages: set
    schedulable_stages: set
    executor_counts: np.ndarray
    num_executors_to_schedule: int
    source_job_idx: int



class HeuristicScheduler(Scheduler):

    @classmethod
    def preprocess_obs(cls, obs):
        G, frontier_stages = cls._make_nx_graph(obs['dag_batch']['data'])
        job_ptr = np.array(obs['dag_batch']['ptr'])
        schedulable_stages = set(obs['dag_batch']['data'].nodes[:, 2].astype(bool).nonzero()[0])
        executor_counts = np.array(obs['executor_counts'])
        num_executors_to_schedule = obs['num_executors_to_schedule']
        source_job_idx = obs['source_job_idx']

        return HeuristicObs(
            G,
            job_ptr,
            frontier_stages,
            schedulable_stages,
            executor_counts,
            num_executors_to_schedule,
            source_job_idx
        )



    @classmethod
    def _make_nx_graph(cls, dag_batch_data):
        '''returns a mask where `mask[i]` is 1 if stage `i`
        is in its job's frontier (i.e. all its parents have completed
        and it's immediately runnable), and 0 otherwise.
        '''
        G = nx.DiGraph()

        G.add_nodes_from((
            (i, dict(num_remaining_tasks=node[0], most_recent_duration=node[1]))
            for i, node in enumerate(dag_batch_data.nodes)
        ))

        G.add_edges_from(dag_batch_data.edge_links)

        frontier_stages = set([(G.in_degree[i] == 0) for i in G.nodes])
        return G, frontier_stages

    

    @classmethod
    def find_stage(cls, obs, job_idx):
        '''searches for a schedulable stage in a given job, prioritizing frontier stages'''
        stage_idx_start = obs.job_ptr[job_idx]
        stage_idx_end = obs.job_ptr[job_idx+1]

        selected_stage_idx = -1
        for i in range(stage_idx_start, stage_idx_end):
            if not i in obs.schedulable_stages:
                continue

            if i in obs.frontier_stages:
                return i

            if selected_stage_idx == -1:
                selected_stage_idx = i

        return selected_stage_idx