from bisect import bisect_right

import numpy as np

from .base import HeuristicScheduler




class CPTScheduler(HeuristicScheduler):
    '''Critical-Path-Time agent. The critical path time of a 
    node is defined as the length of the longest path from that 
    node to any leaf of its dag, weighted by node durations. 
    This heuristic prioritizes stages based on their critical 
    path time.
    '''

    def __init__(
        self, 
        num_executors, 
        fair=True,
        by_shortest=True
    ):
        name = 'SCPT ' if by_shortest else 'LCPT '
        name += '(fair)' if fair else '(greedy)'
        super().__init__(name)
        self.num_executors = num_executors
        self.fair = fair
        self.by_shortest = by_shortest



    def schedule(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.executor_counts)

        # prioritize source job, if there is one
        if obs.source_job_idx < num_active_jobs:
            selected_stage_idx = self.find_stage(obs, obs.source_job_idx)

            if selected_stage_idx != -1:
                return {
                    'stage_idx': selected_stage_idx,
                    'prlsm_lim': obs.executor_counts[obs.source_job_idx]
                }

        if self.fair:
            executor_cap = self.num_executors / max(1, num_active_jobs)
            executor_cap = int(np.ceil(executor_cap))
        else:
            executor_cap = self.num_executors

        cp_lens = self.compute_cp_lengths(obs, executor_cap)

        if cp_lens == {}:
            # all jobs are saturated
            return {
                'stage_idx': -1,
                'prlsm_lim': obs.num_executors_to_schedule
            }

        selected_stage_idx = -1
        for stage_idx in cp_lens.keys():
            if selected_stage_idx == -1:
                selected_stage_idx = stage_idx

            if obs.frontier_stage_mask[stage_idx]:
                selected_stage_idx = stage_idx
                break

        if selected_stage_idx == -1:
            return {
                'stage_idx': -1,
                'prlsm_lim': obs.num_executors_to_schedule
            }

        job_idx = bisect_right(obs.job_ptr, selected_stage_idx) - 1
        prlsm_lim = obs.executor_counts[job_idx] + obs.num_executors_to_schedule
        prlsm_lim = min(executor_cap, prlsm_lim)
        return {
            'stage_idx': selected_stage_idx,
            'prlsm_lim': prlsm_lim
        }



    def compute_cp_lengths(self, obs, executor_cap):
        '''for each active node in the job dag, computes
        the critical path length starting from that node
        '''
        def calc_cp_len(stage_idx, cp_lens):
            cp_lens[stage_idx] = \
                obs.G.nodes[stage_idx]['most_recent_duration'] * \
                obs.G.nodes[stage_idx]['num_remaining_tasks']

            if obs.G.out_degree[stage_idx] == 0:
                return

            max_child_cpl = -1
            for child_stage_idx in obs.G.successors(stage_idx):
                if child_stage_idx not in cp_lens:
                    calc_cp_len(child_stage_idx, cp_lens)
                max_child_cpl = max(max_child_cpl, cp_lens[child_stage_idx])
            
            cp_lens[stage_idx] += max_child_cpl

        cp_lens = {}
        frontier_stage_idxs = obs.frontier_stage_mask.nonzero()[0]
        for stage_idx in frontier_stage_idxs:
            job_idx = bisect_right(obs.job_ptr, stage_idx) - 1
            if obs.executor_counts[job_idx] < executor_cap:
                calc_cp_len(stage_idx, cp_lens)

        if cp_lens == {}:
            return {}

        cp_lens = {
            stage_idx: cp_len 
            for stage_idx, cp_len in cp_lens.items()
            if obs.schedulable_stage_mask[stage_idx]
        }

        cp_lens = dict(sorted(cp_lens.items(), 
                              key=lambda item: item[1], 
                              reverse=(not self.by_shortest)))

        return cp_lens