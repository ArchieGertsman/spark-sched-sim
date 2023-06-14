from collections import deque
from itertools import chain

import numpy as np



class ReturnsCalculator:
    def __init__(self, buff_cap=int(2e5)):
        # estimate of the long-run average number of jobs in the system under the current policy
        self.avg_num_jobs = None

        # circular buffer used for computing the moving average
        self.buff = deque(maxlen=buff_cap)

        

    def __call__(self, rewards_list, times_list, resets_list):
        dt_list = [np.array(ts[1:]) - np.array(ts[:-1]) for ts in times_list]
        self._update_avg_num_jobs(dt_list, rewards_list)

        diff_returns_list = []
        for dt, rew, resets in zip(dt_list, rewards_list, resets_list):
            diff_returns = np.zeros(len(rew))
            dr = 0
            for k in reversed(range(dt.size)):
                job_time = -rew[k]
                expected_job_time = dt[k] * self.avg_num_jobs
                diff_rew = -(job_time - expected_job_time)
                if resets and k in resets:
                    dr = 0
                diff_returns[k] = dr = (diff_rew + dr)
            diff_returns_list += [diff_returns]
        return diff_returns_list
    

    def _update_avg_num_jobs(self, deltas_list, rewards_list):
        data = np.array(list(zip(chain(*deltas_list), chain(*rewards_list))))
        data = data[data[:,0] > 0] # filter out timesteps that have a duration of 0ms
        self.buff.extend(data)

        total_time, rew_sum = np.array(self.buff).sum(0)
        total_job_time = -rew_sum
        self.avg_num_jobs = total_job_time / total_time