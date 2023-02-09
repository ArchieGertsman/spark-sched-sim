from typing import NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
from gymnasium.core import ObsType, ActType



class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name


    def __call__(self, obs: ObsType) -> ActType:
        return self.predict(obs)
        

    @abstractmethod
    def predict(self, obs: ObsType) -> ActType:
        '''must be implemented for every agent.
        Takes an observation of the environment,
        and returns an action.
        '''




class HeuristicObs(NamedTuple):
    G: nx.DiGraph
    job_ptr: np.ndarray
    frontier_op_mask: np.ndarray
    schedulable_op_mask: np.ndarray
    worker_counts: np.ndarray
    num_workers_to_schedule: int
    source_job_idx: int



class HeuristicAgent(BaseAgent):

    @classmethod
    def preprocess_obs(cls, obs):
        G, frontier_op_mask = cls._make_nx_graph(obs['dag_batch']['data'])
        job_ptr = np.array(obs['dag_batch']['ptr'])
        schedulable_op_mask = np.array(obs['schedulable_op_mask'])
        worker_counts = np.array(obs['worker_counts'])
        num_workers_to_schedule = obs['num_workers_to_schedule']
        source_job_idx = obs['source_job_idx']

        return HeuristicObs(
            G,
            job_ptr,
            frontier_op_mask,
            schedulable_op_mask,
            worker_counts,
            num_workers_to_schedule,
            source_job_idx
        )



    @classmethod
    def _make_nx_graph(cls, dag_batch_data):
        '''returns a mask where `mask[i]` is 1 if operation `i`
        is in its job's frontier (i.e. all its parents have completed
        and it's immediately runnable), and 0 otherwise.
        '''
        G = nx.DiGraph()

        G.add_nodes_from((
            (i, dict(num_remaining_tasks=node[0], most_recent_duration=node[1]))
            for i, node in enumerate(dag_batch_data.nodes)
        ))

        G.add_edges_from(dag_batch_data.edge_links)

        frontier_op_mask = np.array([(G.in_degree[i] == 0) for i in G.nodes])
        return G, frontier_op_mask

    

    @classmethod
    def find_op(cls, obs, job_idx):
        '''searches for a schedulable operation in
        a given job, prioritizing frontier operations
        '''
        op_idx_start = obs.job_ptr[job_idx]
        op_idx_end = obs.job_ptr[job_idx+1]

        selected_op_idx = -1
        for i in range(op_idx_start, op_idx_end):
            if not obs.schedulable_op_mask[i]:
                continue

            if obs.frontier_op_mask[i]:
                return i

            if selected_op_idx == -1:
                selected_op_idx = i

        return selected_op_idx