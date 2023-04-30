import numpy as np

from .base import HeuristicScheduler



class FIFOScheduler(HeuristicScheduler):

    def __init__(self, num_executors, fair=True):
        name = 'Dynamic Partition' if fair else 'FIFO'
        super().__init__(name)
        self.num_executors = num_executors
        self.fair = fair

    

    def schedule(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.executor_counts)

        if self.fair:
            executor_cap = self.num_executors / max(1, num_active_jobs)
            executor_cap = int(np.ceil(executor_cap))
        else:
            executor_cap = self.num_executors

        if obs.source_job_idx < num_active_jobs:
            selected_stage_idx = self.find_stage(obs, obs.source_job_idx)

            if selected_stage_idx != -1:
                return {
                    'stage_idx': selected_stage_idx,
                    'prlsm_lim': obs.executor_counts[obs.source_job_idx]
                }

        for j in range(num_active_jobs):
            if obs.executor_counts[j] >= executor_cap or \
               j == obs.source_job_idx:
               continue

            selected_stage_idx = self.find_stage(obs, j)
            if selected_stage_idx == -1:
                continue

            prlsm_lim = obs.executor_counts[j] + obs.num_executors_to_schedule
            prlsm_lim = min(executor_cap, prlsm_lim)
            return {
                'stage_idx': selected_stage_idx,
                'prlsm_lim': prlsm_lim
            }

        # didn't find any stages to schedule
        return {
            'stage_idx': -1,
            'prlsm_lim': obs.num_executors_to_schedule
        }