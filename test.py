import os
import shutil
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from gym_dagsched.envs.dagsched_env import DagSchedEnv
from gym_dagsched.envs.dagsched_env_decima_wrapper import DagSchedEnvDecimaWrapper
from gym_dagsched.agents.decima_agent import DecimaAgent
from gym_dagsched.agents.dynamic_partition_agent import DynamicPartitionAgent
from gym_dagsched.utils.metrics import avg_job_duration
from gym_dagsched.utils.hidden_prints import HiddenPrints
from gym_dagsched.utils.device import device



def main():
    setup()

    num_workers = 10

    heuristic_agent = DynamicPartitionAgent(num_workers)
    decima_agent = \
        DecimaAgent(num_workers,
                    mode='eval', 
                    state_dict_path='model_20batch.pt',
                    device=device)

    base_env = DagSchedEnv()
    wrapped_env = DagSchedEnvDecimaWrapper(base_env)

    num_tests = 10
    heuristic_results = test(base_env, heuristic_agent, num_tests, num_workers)
    decima_results = test(wrapped_env, decima_agent, num_tests, num_workers)

    plt.plot(np.arange(num_tests), heuristic_results, label=heuristic_agent.name)
    plt.plot(np.arange(num_tests), decima_results, label=decima_agent.name)
    plt.legend()
    plt.savefig('results.png')




def test(env, agent, num_tests, num_workers):
    avg_job_durations = []

    for i in range(num_tests):
        print(f'{agent.name}: iteration {i+1}', flush=True)
        np.random.seed(i)
        
        with HiddenPrints():
            run_episode(env,
                        agent,
                        num_job_arrivals=0,
                        num_init_jobs=20,
                        job_arrival_rate=0,
                        num_workers=num_workers,
                        max_wall_time=np.inf)

            avg_job_durations += [avg_job_duration(env)*1e-3]

    return avg_job_durations




def run_episode(env,
                agent,
                num_job_arrivals,
                num_init_jobs,
                job_arrival_rate,
                num_workers,
                max_wall_time):  

    obs = env.reset(num_init_jobs, 
                    num_job_arrivals, 
                    job_arrival_rate, 
                    num_workers,
                    max_wall_time)

    done = False
    rewards = []
    
    while not done:
        action = agent(obs)
        obs, reward, done = env.step(action)
        rewards += [reward]

    return rewards




def setup():
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    torch.manual_seed(0)
    np.random.seed(0)

    print('cuda available:', torch.cuda.is_available())




if __name__ == '__main__':
    main()