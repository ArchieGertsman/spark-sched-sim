from scipy.signal import lfilter


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

    def add_list(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            self.add(list_reward[i], list_time[i])

    def add_list_filter_zero(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            if list_time[i] != 0:
                self.add(list_reward[i], list_time[i])
            else:
                assert list_reward[i] == 0

    def get_avg_per_step_reward(self):
        return float(self.reward_sum) / float(self.time_sum)


    def calculate(self,
        wall_times_list, 
        rewards_list,
    ):
        diff_times_list = [wall_times[1:] - wall_times[:-1] 
                        for wall_times in wall_times_list]

        for rewards, diff_times in zip(rewards_list, diff_times_list):
            self.add_list_filter_zero(
                list(rewards), 
                list(diff_times))

        avg_per_step_reward = \
            self.get_avg_per_step_reward()

        diff_rewards_list = \
            [rewards - avg_per_step_reward * diff_times
            for rewards, diff_times in zip(rewards_list, diff_times_list)]

        diff_returns_list = [compute_returns(diff_rewards, self.discount) 
                        for diff_rewards in diff_rewards_list]

        return diff_returns_list, avg_per_step_reward