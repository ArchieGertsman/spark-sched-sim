from typing import List, Optional

from scipy.signal import lfilter
import numpy as np



class ReturnsCalculator:
    def __init__(self):
        self.work_buff = []
        self.dt_buff = []
        self.total_work = 0
        self.total_time = 0
        self.buffer_cap = 100000
        # self.avg_num_jobs = 0.
        # self.alpha = 1.
        # self.min_alpha = .08
        # self.alpha_scale = .9
        

    def __call__(
        self, 
        rewards_list: List[np.ndarray], 
        times_list: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        return self.calculate(rewards_list, times_list)
    

    @property
    def avg_num_jobs(self):
        return self.total_work / self.total_time
    


    def calculate(self, 
        rewards_list: List[np.ndarray], 
        times_list: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:

        deltas_list = self._get_dt_list(times_list)

        for rewards, deltas in zip(rewards_list, deltas_list):
            for rew, dt in zip(rewards, deltas):
                if dt == 0:
                    continue
                self.work_buff += [rew]
                self.dt_buff += [dt]
                self.total_work += rew
                self.total_time += dt
                if len(self.work_buff) > self.buffer_cap:
                    self.total_work -= self.work_buff.pop(0)
                    self.total_time -= self.dt_buff.pop(0)

        diff_returns_list = []
        for work, deltas in zip(rewards_list, deltas_list):
            diff_returns = np.zeros_like(work)
            dr = 0
            for t in reversed(range(len(deltas))):
                dr = work[t] - deltas[t] * self.avg_num_jobs + dr
                diff_returns[t] = dr
            diff_returns_list += [diff_returns * 1e-5]

        return diff_returns_list




    # def calculate(self, 
    #     rewards_list: List[np.ndarray], 
    #     times_list: Optional[List[np.ndarray]] = None
    # ) -> List[np.ndarray]:
        
    #     rewards = np.hstack(rewards_list); rewards = rewards[rewards != 0]
    #     new_avg = rewards.mean()
    #     self.avg_num_jobs += self.alpha * (new_avg - self.avg_num_jobs)
    #     self.alpha = max(.9 * self.alpha, .05)

    #     diff_returns_list = []
    #     for num_jobs in rewards_list:
    #         diff_returns = np.zeros_like(num_jobs)
    #         dr = 0
    #         for t in reversed(range(len(num_jobs))):
    #             if num_jobs[t] == 0:
    #                 num_jobs[t] = num_jobs[t+1]
    #             dr = num_jobs[t] - self.avg_num_jobs + dr
    #             diff_returns[t] = dr
    #         diff_returns_list += [diff_returns]

    #     return diff_returns_list


    # def calculate(self, 
    #     rewards_list: List[np.ndarray], 
    #     times_list: Optional[List[np.ndarray]] = None
    # ) -> List[np.ndarray]:
        
    #     dt_list = self._get_dt_list(times_list)

    #     self._update_moving_avg(rewards_list, dt_list)

    #     gamma = .99
    #     tau = np.log(1/gamma)
    #     alpha = 1e-2

    #     diff_returns_list = []
    #     for work, dt in zip(rewards_list, dt_list):
    #         diff_rew = work # - dt * self.avg_num_jobs
    #         diff_ret = np.zeros_like(diff_rew)
    #         dr = 0
    #         for t in reversed(range(len(diff_rew))):
    #             dr = diff_rew[t] + np.exp(-tau * alpha * dt[t]) * dr
    #             diff_ret[t] = dr
    #         diff_returns_list += [diff_ret]

    #     return diff_returns_list
    

    # def _update_moving_avg(self, rewards_list, dt_list):
    #     N = len(rewards_list) # // 2
    #     rewards = np.hstack(rewards_list[:N]); rewards = rewards[rewards != 0]
    #     dt = np.hstack(dt_list[:N]); dt = dt[dt != 0]
    #     total_time = dt.sum()
    #     new_avg = rewards.sum() / total_time
    #     self.avg_num_jobs += self.alpha * (new_avg - self.avg_num_jobs)
    #     self.alpha = max(self.alpha_scale * self.alpha, self.min_alpha)


    @classmethod
    def _get_dt_list(cls, times_list):
        dt_list = []
        for ts in times_list:
            ts = np.concatenate([np.array([0.]), ts])
            dt = ts[1:] - ts[:-1]
            dt_list += [dt]
        return dt_list
