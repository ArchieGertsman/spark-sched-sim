import os
import shutil
import sys
import json

import numpy as np
import torch
from torch.multiprocessing import set_start_method, Pool
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from spark_sched_sim.wrappers.decima_wrappers import DecimaObsWrapper, DecimaActWrapper
from spark_sched_sim.metrics import avg_job_duration
from train_algs.utils.hidden_prints import HiddenPrints
from spark_sched_sim.schedulers import DecimaScheduler, FIFOScheduler, CPTScheduler


def main():
    setup()

    print('testing', flush=True)

    num_tests = 1

    num_executors = 50

    # should be greater than the number of epochs the
    # model was trained on, so that the job sequences
    # are unseen
    base_seed = 20000

    env_kwargs = {
        'num_executors': num_executors,
        'job_arrival_cap': 1000,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    wrapped_env = DecimaActWrapper(DecimaObsWrapper(base_env))

    test_instances = []

    fifo_scheduler = FIFOScheduler(num_executors)
    test_instances += [(fifo_scheduler, base_env, num_tests, base_seed)]

    # lcpt_scheduler = CPTScheduler(num_executors, by_shortest=False)
    # test_instances += [(lcpt_scheduler, base_env, num_tests, base_seed)]

    # LCPT  61.8
    # DynP  60.4
    # SCPT  60.2
    # 6200  60.1
    # 11000 59.7
    # 8100  59.1
    # 7100  58.4
    # 8300  57.7
    # 5700  57.5

    # 5700: 59

    model_name = 'model.pt'
    for i in range(4000, 5000 + 1, 100):
        model_dir = f'ignore/models/{i}'
        decima_scheduler = \
            DecimaScheduler(
                num_executors,
                training_mode=False, 
                state_dict_path=f'{model_dir}/{model_name}'
            )
        test_instances += [(decima_scheduler, wrapped_env, num_tests, base_seed)]

    # run tests in parallel using multiprocessing
    with Pool(len(test_instances)) as p:
        test_results = p.map(test, test_instances)

    sched_names = [sched.name for sched, *_ in test_instances]

    df = pd.DataFrame({k: v[0] for k, v in zip(sched_names, test_results)})
    df.to_csv('job_durations.csv')

    with open(f'num_active_jobs.json', 'w') as fp:
        d = {k: v[1] for k, v in zip(sched_names, test_results)}
        json.dump(d, fp)

    for sched_name, test_result in zip(sched_names, test_results):
        x, y = compute_CDF(test_result[0])
        plt.plot(x, y, label=sched_name)
    plt.legend()
    plt.ylabel('CDF')
    plt.xlabel('job duration (s)')
    plt.savefig('job_duration_cdf.png', bbox_inches='tight')

    # visualize_results(
    #     'job_duration_cdf.png', 
    #     sched_names, 
    #     test_results,
    #     env_kwargs
    # )



def test(instance):
    sys.stdout = open(f'ignore/log/proc_test/main.out', 'a')
    torch.set_num_threads(1)

    sched, env, num_tests, base_seed = instance

    # avg_job_durations = []

    for i in range(num_tests):
        torch.manual_seed(42)

        with HiddenPrints():
            data = run_episode(env, sched, base_seed + i)

        ajd = avg_job_duration(env) * 1e-3
        # avg_job_durations += [result]
        print(f'{sched.name}: test {i+1}, avj={ajd:.1f}s', flush=True)

    # return np.array(avg_job_durations)
    job_durations = [job.t_completed - job.t_arrival for job in env.jobs.values()]
    return 1e-3 * np.array(job_durations), data




def compute_CDF(arr, num_bins=100):
    """
    usage: x, y = compute_CDF(arr):
           plt.plot(x, y)
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])




def run_episode(env, sched, seed): 
    env_options = {'max_wall_time': np.inf} 
    obs, _ = env.reset(seed=seed, options=env_options)

    done = False
    data = []
    next_wall_time = 0
    
    while not done:
        wall_time = next_wall_time

        if isinstance(sched, DecimaScheduler):
            action, *_ = sched.schedule(obs)
        else:
            action = sched(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        next_wall_time = info['wall_time']

        duration = next_wall_time - wall_time
        if duration > 0:
            avg_num_jobs = -reward / duration
            data += [(wall_time, avg_num_jobs)]
        done = (terminated or truncated)

    return data



def visualize_results(
    out_fname, 
    sched_names, 
    test_results,
    env_kwargs
):
    # plot CDF's
    for sched_name, avg_job_durations in zip(sched_names, test_results):
        x, y = compute_CDF(avg_job_durations)
        plt.plot(x, y, label=sched_name)

    # display environment options in a table
    plt.table(
        cellText=[[key,val] for key,val in env_kwargs.items()],
        colWidths=[.25, .1],
        cellLoc='center', 
        rowLoc='center',
        loc='right'
    )

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel('Average job duration (s)')
    plt.ylabel('CDF')
    num_tests = len(test_results[0])
    plt.title(f'CDF of avg. job duration over {num_tests} runs')
    plt.savefig(out_fname, bbox_inches='tight')




def setup():
    shutil.rmtree('ignore/log/proc_test/', ignore_errors=True)
    os.mkdir('ignore/log/proc_test/')

    sys.stdout = open(f'ignore/log/proc_test/main.out', 'a')

    set_start_method('forkserver')

    torch.manual_seed(42)




if __name__ == '__main__':
    main()