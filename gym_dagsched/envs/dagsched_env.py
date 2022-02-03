import gym

class DagSchedEnv(gym.Env):
    def __init__(self):
        self.state_space = None
        self.action_space = None


    def reset(self):
        state = None
        return state


    def step(self, action):
        state, reward, done, info = (None) * 4
        return state, reward, done, info


    def seed(self):
        pass


    def render(self):
        pass


    def close(self):
        pass