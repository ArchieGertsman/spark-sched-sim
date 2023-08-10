'''Examples of how to run job scheduling simulations with different schedulers
'''
import os.path as osp
from pprint import pprint

import gymnasium as gym

from cfg_loader import load
from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from spark_sched_sim import metrics


def main():
    fair_example()
    decima_example()



def fair_example():
    env_kwargs = {
        'num_executors': 50,
        'job_arrival_cap': 200,
        'job_arrival_rate': 4.e-5,
        'moving_delay': 2000.,
        'warmup_delay': 1000.,
        'query_dir': 'data/tpch'
    }
    
    # Fair scheduler
    scheduler = RoundRobinScheduler(env_kwargs['num_executors'],
                                    dynamic_partition=True)
    
    print(f'Example: Fair Scheduler')
    print('Env settings:')
    pprint(env_kwargs)

    print('Running episode...')
    avg_job_duration = run_episode(env_kwargs, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)
    print()



def decima_example():
    cfg = load(filename=osp.join('config', 'decima_ppo_discrew.yaml'))

    env_cfg = cfg['env']
    env_cfg.pop('mean_time_limit')

    agent_cfg = cfg['agent'] \
        | {'num_executors': env_cfg['num_executors'],
           'state_dict_path': osp.join('models', 'decima', 'model.pt')}
    
    scheduler = make_scheduler(agent_cfg)

    print(f'Example: Decima')
    print('Env settings:')
    pprint(env_cfg)

    print('Running episode...')
    avg_job_duration = run_episode(env_cfg, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)



def run_episode(env_kwargs, scheduler, seed=1234):
    env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_kwargs)
    if isinstance(scheduler, NeuralScheduler):
        env = NeuralActWrapper(env)
        env = scheduler.obs_wrapper_cls(env)

    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False
    
    while not (terminated or truncated):
        if isinstance(scheduler, NeuralScheduler):
            action, *_ = scheduler(obs)
        else:
            action = scheduler(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    avg_job_duration = metrics.avg_job_duration(env) * 1e-3

    # cleanup rendering
    env.close()

    return avg_job_duration


if __name__ == '__main__':
    main()