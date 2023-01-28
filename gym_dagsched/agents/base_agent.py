

class BaseAgent:
    def __init__(self, name):
        self.name = name


    def __call__(self, obs):
        return self.invoke(obs)
        

    def invoke(self, obs):
        raise NotImplementedError