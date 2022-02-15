import gym
# import gym_dagsched

if __name__ == '__main__':
    env = gym.make('gym_dagsched/dagsched-v0')
    # from gym import envs
    # print(envs.registry.all())

    # print(env.observation_space.sample()['jobs'])