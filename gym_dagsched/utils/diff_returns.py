from scipy.signal import lfilter
import numpy as np


def compute_returns(rewards, discount):
    r = rewards[...,::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    y = y[...,::-1].copy()
    return y


class DifferentialReturnsCalculator(object):
    def __init__(self, discount, size=100000):
        self.discount = discount
        self.size = size
        self.count = 0
        self.reward_record = []
        self.time_record = []
        self.reward_sum = 0
        self.time_sum = 0



    def add(self, reward, time):
        if self.count >= self.size:
            stale_reward = self.reward_record.pop(0)
            stale_time = self.time_record.pop(0)
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time
        else:
            self.count += 1

        self.reward_record.append(reward)
        self.time_record.append(time)
        self.reward_sum += reward
        self.time_sum += time



    def add_list_filter_zero(self, rewards, times):
        assert len(rewards) == len(times)
        for i in range(len(rewards)):
            if times[i] != 0:
                self.add(rewards[i], times[i])
            else:
                assert rewards[i] == 0



    @property
    def avg_per_step_reward(self):
        return self.reward_sum / self.time_sum



    def calculate(self, times_list, rewards_list):
        diff_times_list = []
        for wall_times in times_list:
            diff_times = np.zeros_like(wall_times)
            diff_times[:-1] = \
                wall_times[1:] - wall_times[:-1]
            diff_times_list += [diff_times]

        for rewards, diff_times in zip(rewards_list, 
                                       diff_times_list):
            self.add_list_filter_zero(rewards, diff_times)

        diff_rewards_list = \
            [rewards - self.avg_per_step_reward * diff_times
             for rewards, diff_times in zip(rewards_list, 
                                            diff_times_list)]

        diff_returns_list = \
            [compute_returns(diff_rewards, self.discount) 
             for diff_rewards in diff_rewards_list]

        return diff_returns_list