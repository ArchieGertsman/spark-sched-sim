from typing import List, Optional
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
                if k in resets:
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
    


    # @classmethod
    # def _get_dt_list(cls, times_list):
    #     dt_list = []
    #     for ts in times_list:
    #         ts = np.concatenate([np.array([0.]), ts])
    #         dt = ts[1:] - ts[:-1]
    #         dt_list += [dt]
    #     return dt_list



    # def _update_buffers(self, rewards_list, deltas_list):
    #     for rewards, deltas in zip(rewards_list, deltas_list):
    #         for rew, dt in zip(rewards, deltas):
    #             if dt == 0:
    #                 continue

    #             work = -rew
    #             self.work_buff += [work]
    #             self.total_work += work

    #             self.dt_buff += [dt]
    #             self.total_time += dt

    #             if len(self.work_buff) > self.buff_cap:
    #                 # discard old data
    #                 self.total_work -= self.work_buff.pop(0)
    #                 self.total_time -= self.dt_buff.pop(0)

