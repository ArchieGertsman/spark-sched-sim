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



    def add(self, dt, reward):
        if self.count >= self.size:
            stale_reward = self.reward_record.pop(0)
            stale_time = self.time_record.pop(0)
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time
        else:
            self.count += 1

        self.reward_record.append(reward)
        self.time_record.append(dt)
        self.reward_sum += reward
        self.time_sum += dt



    def add_list_filter_zero(self, time_diffs, rewards):
        np.set_printoptions(precision=3, suppress=True)
        assert len(time_diffs) == len(rewards)
        for dt, reward in zip(time_diffs, rewards):
            print(dt, reward)
            if dt != 0:
                self.add(dt, reward)
            else:
                assert reward == 0



    @property
    def avg_per_step_reward(self):
        return self.reward_sum / self.time_sum



    def calculate(self, times_list, rewards_list):
        time_diffs_list = []
        for times in times_list:
            times = np.concatenate([np.array([0.]), times])
            time_diffs = times[1:] - times[:-1]
            time_diffs_list += [time_diffs]

        for time_diffs, rewards in zip(time_diffs_list,
                                       rewards_list):
            self.add_list_filter_zero(time_diffs, rewards)

        diff_rewards_list = \
            [rewards - self.avg_per_step_reward * time_diffs
             for rewards, time_diffs in zip(rewards_list, 
                                            time_diffs_list)]

        diff_returns_list = \
            [compute_returns(diff_rewards, self.discount) 
             for diff_rewards in diff_rewards_list]

        return diff_returns_list