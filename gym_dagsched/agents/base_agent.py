class BaseAgent:
    def __init__(self, name):
        self.name = name


    def __call__(self, obs):
        return self.invoke(obs)
        

    def invoke(self, obs):
        '''must be implemented for every agent.
        Takes an observation of the environment,
        and returns an action.
        '''
        raise NotImplementedError