from gymnasium.core import ObsType, ActType


class RolloutBuffer:
    def __init__(self):
        self.obsns: list[ObsType] = []
        self.wall_times: list[float] = []
        self.actions: list[ActType] = []
        self.rewards: list[float] = []

    def add(self, obs, wall_time, action, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]

    def __len__(self):
        return len(self.obsns)