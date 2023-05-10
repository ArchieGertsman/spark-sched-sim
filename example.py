'''This is an example of how to run a job scheduling simulation
according to a reinforcement-learned scheduling policy.
'''
import os
import shutil
import sys

import torch
import gymnasium as gym
import numpy as np

from spark_sched_sim.schedulers import DecimaScheduler, RandomScheduler, FIFOScheduler
from spark_sched_sim.wrappers import DecimaObsWrapper, DecimaActWrapper
from train_algs.utils.hidden_prints import HiddenPrints
from spark_sched_sim import metrics



if __name__ == '__main__':
    # set rng seeds for reproducibility 
    env_seed = 0
    torch_seed = 42
    torch.manual_seed(torch_seed)

    # shutil.rmtree('ignore/log/proc1/', ignore_errors=True)
    # os.mkdir('ignore/log/proc1/')

    # sys.stdout = open(f'ignore/log/proc1/main.out', 'a')

    # select the number of simulated executors
    num_executors = 50

    # load learned agent
    model_dir = 'ignore/models/1000'
    model_name = 'model.pt'
    # decima_agent = \
    #     DecimaScheduler(
    #         num_executors,
    #         training_mode=False, 
    #         state_dict_path=f'{model_dir}/{model_name}'
    #     )
    random_agent = RandomScheduler()
    # fifo_agent = FIFOScheduler(num_executors)


    # same settings as in training
    env_kwargs = {
        'num_executors': num_executors,
        'num_init_jobs': 1,
        'num_job_arrivals': 200,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000. #,
        # 'render_mode': 'human' # visualize simulation
    }

    # setup gym environment
    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    # env = DecimaActWrapper(DecimaObsWrapper(base_env))
    env = base_env

    # run an episode

    random_agent.set_seed(0)
    # print('env seed:', env_seed, flush=True)
    # with HiddenPrints():
    obs, _ = env.reset(seed=env_seed, options=None)
    done = False
    
    while not done:
        # action, _, _ = decima_agent(obs)
        action = random_agent(obs)
        # action = fifo_agent(obs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = (terminated or truncated)

    avg_job_duration = int(metrics.avg_job_duration(env) * 1e-3)
    print(f'Average job duration: {avg_job_duration}s', flush=True)

    env_seed += 1
    
    # cleanup rendering
    env.close()