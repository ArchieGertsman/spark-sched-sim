import numpy as np

from .base_agent import HeuristicAgent



class FIFOAgent(HeuristicAgent):

    def __init__(self, num_workers, fair=True):
        name = 'FIFO ' + ('(fair)' if fair else '(greedy)')
        super().__init__(name)
        self.num_workers = num_workers
        self.fair = fair

    

    def predict(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.worker_counts)

        if self.fair:
            worker_cap = self.num_workers / max(1, num_active_jobs)
            worker_cap = int(np.ceil(worker_cap))
        else:
            worker_cap = self.num_workers

        if obs.source_job_idx < num_active_jobs:
            selected_op_idx = self.find_op(obs, obs.source_job_idx)

            if selected_op_idx != -1:
                return {
                    'op_idx': selected_op_idx,
                    'prlsm_lim': obs.worker_counts[obs.source_job_idx]
                }

        for j in range(num_active_jobs):
            if obs.worker_counts[j] >= worker_cap or \
               j == obs.source_job_idx:
               continue

            selected_op_idx = self.find_op(obs, j)
            if selected_op_idx == -1:
                continue

            prlsm_lim = obs.worker_counts[j] + obs.num_workers_to_schedule
            prlsm_lim = min(worker_cap, prlsm_lim)
            return {
                'op_idx': selected_op_idx,
                'prlsm_lim': prlsm_lim
            }

        # didn't find any ops to schedule
        return {
            'op_idx': -1,
            'prlsm_lim': obs.num_workers_to_schedule
        }