import os
import shutil
import sys


import pandas as pd
import numpy as np
import torch
from torch.multiprocessing import set_start_method, Pool
import gymnasium as gym
import matplotlib.pyplot as plt

from gym_dagsched.wrappers.decima_wrapper import DecimaWrapper
from gym_dagsched.utils.metrics import avg_job_duration
from gym_dagsched.utils.hidden_prints import HiddenPrints

from gym_dagsched.agents.decima_agent import DecimaAgent
from gym_dagsched.agents.fifo_agent import FIFOAgent
from gym_dagsched.agents.scpt_agent import SCPTAgent




def main():
    setup()

    num_tests = 100

    num_workers = 20

    # should be greater than the number of epochs the
    # model was trained on, so that the job sequences
    # are unseen
    base_seed = 500

    model_dir = 'gym_dagsched/models'

    fifo_agent = FIFOAgent(num_workers)
    scpt_agent = SCPTAgent(num_workers)
    dynamic_fifo_agent = FIFOAgent(num_workers, dynamic=True)
    dynamic_scpt_agent = SCPTAgent(num_workers, dynamic=True)
    decima_agent = \
        DecimaAgent(num_workers,
                    mode='eval', 
                    state_dict_path=\
                        f'{model_dir}/model_20b_10w_150ep.pt')

    base_env = gym.make('gym_dagsched:gym_dagsched/DagSchedEnv-v0')
    wrapped_env = DecimaWrapper(base_env)

    env_options = {
        'num_workers': num_workers,
        'num_init_jobs': 20,
        'num_job_arrivals': 0,
        'job_arrival_rate': 0.,
        'max_wall_time': np.inf,
        'moving_delay': 2000.,
        'reward_scale': 1e-5
    }

    test_instances = [
        (fifo_agent, base_env, num_tests, base_seed, env_options),
        (scpt_agent, base_env, num_tests, base_seed, env_options),
        (dynamic_fifo_agent, base_env, num_tests, base_seed, env_options),
        (dynamic_scpt_agent, base_env, num_tests, base_seed, env_options),
        (decima_agent, wrapped_env, num_tests, base_seed, env_options)
    ]

    # run tests in parallel using multiprocessing
    with Pool(len(test_instances)) as p:
        test_results = p.map(test, test_instances)

    agent_names = [agent.name for agent, *_ in test_instances]

    visualize_results('job_duration_cdf.png', 
                      agent_names, 
                      test_results, 
                      env_options)



def test(instance):
    sys.stdout = open(f'log/proc/main.out', 'a')
    torch.manual_seed(42)
    torch.set_num_threads(1)

    agent, env, num_tests, base_seed, env_options = instance

    avg_job_durations = []

    for i in range(num_tests):
        print(f'{agent.name}: iteration {i+1}', flush=True)
        
        with HiddenPrints():
            run_episode(env, agent, base_seed + i, env_options)
            avg_job_durations += [avg_job_duration(env)*1e-3]

    return np.array(avg_job_durations)




def compute_CDF(arr, num_bins=100):
    """
    usage: x, y = compute_CDF(arr):
           plt.plot(x, y)
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])




def run_episode(env, agent, seed, options):  
    obs, _ = env.reset(seed=seed, options=options)

    done = False
    rewards = []
    
    while not done:
        action = agent(obs)

        obs, reward, terminated, truncated, _ = \
            env.step(action)

        rewards += [reward]
        done = (terminated or truncated)

    return rewards



def visualize_results(out_fname, 
                      agent_names, 
                      test_results, 
                      env_options):

    # plot CDF's
    for agent_name, avg_job_durations in zip(agent_names, 
                                             test_results):
        x, y = compute_CDF(avg_job_durations)
        plt.plot(x, y, label=agent_name)

    # display environment options in a table
    plt.table(cellText=\
                [[key,val] 
                for key,val in env_options.items()],
            colWidths=[.25, .1],
            cellLoc='center', 
            rowLoc='center',
            loc='right')

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel('Average job duration (s)')
    plt.ylabel('CDF')
    num_tests = len(test_results[0])
    plt.title(f'CDF of avg. job duration over {num_tests} runs')
    plt.savefig(out_fname, bbox_inches='tight')




def setup():
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    set_start_method('forkserver')

    torch.manual_seed(42)




if __name__ == '__main__':
    main()