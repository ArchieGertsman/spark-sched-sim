from bisect import bisect_right

import numpy as np

from .base_agent import HeuristicAgent




class CPTAgent(HeuristicAgent):
    '''Critical-Path-Time agent. The critical path time of a 
    node is defined as the length of the longest path from that 
    node to any leaf of its dag, weighted by node durations. 
    This heuristic prioritizes operations based on their critical 
    path time.
    '''

    def __init__(self, 
                 num_workers, 
                 fair=True,
                 by_shortest=True):
        name = 'SCPT ' if by_shortest else 'LCPT '
        name += '(fair)' if fair else '(greedy)'
        super().__init__(name)
        self.num_workers = num_workers
        self.fair = fair
        self.by_shortest = by_shortest



    def predict(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.worker_counts)

        if obs.source_job_idx < num_active_jobs:
            selected_op_idx = self.find_op(obs, obs.source_job_idx)

            if selected_op_idx != -1:
                return {
                    'op_idx': selected_op_idx,
                    'prlsm_lim': obs.worker_counts[obs.source_job_idx]
                }

        if self.fair:
            worker_cap = self.num_workers / max(1, num_active_jobs)
            worker_cap = int(np.ceil(worker_cap))
        else:
            worker_cap = self.num_workers

        cp_lens = self.compute_cp_lengths(obs, worker_cap)

        if cp_lens == {}:
            # all jobs are saturated
            return {
                'op_idx': -1,
                'prlsm_lim': obs.num_workers_to_schedule
            }

        selected_op_idx = -1
        for op_idx in cp_lens.keys():
            if selected_op_idx == -1:
                selected_op_idx = op_idx

            if obs.frontier_op_mask[op_idx]:
                selected_op_idx = op_idx
                break

        if selected_op_idx == -1:
            return {
                'op_idx': -1,
                'prlsm_lim': obs.num_workers_to_schedule
            }

        job_idx = bisect_right(obs.job_ptr, selected_op_idx) - 1
        prlsm_lim = obs.worker_counts[job_idx] + obs.num_workers_to_schedule
        prlsm_lim = min(worker_cap, prlsm_lim)
        return {
            'op_idx': selected_op_idx,
            'prlsm_lim': prlsm_lim
        }



    def compute_cp_lengths(self, obs, worker_cap):
        '''for each active node in the job dag, computes
        the critical path length starting from that node
        '''
        def calc_cp_len(op_idx, cp_lens):
            cp_lens[op_idx] = \
                obs.G.nodes[op_idx]['most_recent_duration'] * \
                obs.G.nodes[op_idx]['num_remaining_tasks']

            if obs.G.out_degree[op_idx] == 0:
                return

            max_child_cpl = -1
            for child_op_idx in obs.G.successors(op_idx):
                if child_op_idx not in cp_lens:
                    calc_cp_len(child_op_idx, cp_lens)
                max_child_cpl = \
                    max(max_child_cpl, cp_lens[child_op_idx])
            
            cp_lens[op_idx] += max_child_cpl

        cp_lens = {}
        frontier_op_idxs = obs.frontier_op_mask.nonzero()[0]
        for op_idx in frontier_op_idxs:
            job_idx = bisect_right(obs.job_ptr, op_idx) - 1
            if obs.worker_counts[job_idx] < worker_cap:
                calc_cp_len(op_idx, cp_lens)

        if cp_lens == {}:
            return {}

        cp_lens = {op_idx: cp_len for op_idx, cp_len in cp_lens.items()
                   if obs.schedulable_op_mask[op_idx]}

        cp_lens = dict(sorted(cp_lens.items(), 
                              key=lambda item: item[1], 
                              reverse=(not self.by_shortest)))

        return cp_lens