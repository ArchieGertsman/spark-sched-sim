import numpy as np
import networkx as nx

from .base_agent import BaseAgent



class FifoAgent(BaseAgent):

    def __init__(self, num_workers, fair=True):
        name = 'FIFO ' + ('(fair)' if fair else '(greedy)')
        super().__init__(name)
        self.num_workers = num_workers
        self.fair = fair

    

    def predict(self, obs):
        self.job_ptr = obs['dag_batch']['ptr']
        self.schedulable_op_mask = obs['schedulable_op_mask']
        edge_links = obs['dag_batch']['data'].edge_links
        self.frontier_op_mask = self._get_frontier_op_mask(edge_links)

        worker_counts = obs['worker_counts']
        num_active_jobs = len(worker_counts)
        num_workers_to_schedule = obs['num_workers_to_schedule']
        source_job_idx = obs['source_job_idx']

        if self.fair:
            worker_cap = self.num_workers / max(1, len(num_active_jobs))
            worker_cap = int(np.ceil(worker_cap))
        else:
            worker_cap = self.num_workers

        if source_job_idx < num_active_jobs:
            selected_op_idx = self._find_op(source_job_idx)

            if selected_op_idx != -1:
                return {
                    'op_idx': selected_op_idx,
                    'prlsm_lim': worker_counts[source_job_idx]
                }

        for j in range(num_active_jobs):
            if worker_counts[j] >= worker_cap or \
               j == source_job_idx:
               continue

            selected_op_idx = self._find_op(j)
            if selected_op_idx == -1:
                continue

            prlsm_lim = worker_counts[j] + num_workers_to_schedule
            prlsm_lim = min(worker_cap, prlsm_lim)
            return {
                'op_idx': selected_op_idx,
                'prlsm_lim': prlsm_lim
            }

        # didn't find any ops to schedule
        return {
            'op_idx': -1,
            'prlsm_lim': num_workers_to_schedule
        }



    @classmethod
    def _get_frontier_op_mask(cls, edge_links):
        '''returns a mask where `mask[i]` is 1 if operation `i`
        is in its job's frontier (i.e. all its parents have completed
        and it's immediately runnable), and 0 otherwise.
        '''
        G = nx.from_edgelist(edge_links.T, create_using=nx.DiGraph)
        return np.array([(G.in_degree(i) == 0) for i in G.nodes])

    

    def _find_op(self, job_idx):
        '''searches for a schedulable operation in
        a given job, prioritizing frontier operations
        '''
        op_idx_start = self.job_ptr[job_idx]
        op_idx_end = self.job_ptr[job_idx+1]

        selected_op_idx = -1
        for i in range(op_idx_start, op_idx_end):
            if not self.schedulable_op_mask[i]:
                continue

            if self.frontier_op_mask[i]:
                return i

            if selected_op_idx == -1:
                selected_op_idx = i

        return selected_op_idx
