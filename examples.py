'''Examples of how to run job scheduling simulations with different schedulers
'''
import os.path as osp
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib

from cfg_loader import load
from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from spark_sched_sim import metrics


ENV_KWARGS = {
    'num_executors': 10,
    'job_arrival_cap': 50,
    'job_arrival_rate': 4.e-5,
    'moving_delay': 2000.,
    'warmup_delay': 1000.,
    'dataset': 'tpch',
    'render_mode': 'human'
}


def main():
    # save final rendering to artifacts dir
    pathlib.Path('artifacts').mkdir(parents=True, exist_ok=True) 

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--sched',
        choices=['fair', 'decima'],
        dest='sched',
        help='which scheduler to run',
        required=True,
    )

    args = parser.parse_args()

    sched_map = {
        'fair': fair_example,
        'decima': decima_example
    }

    sched_map[args.sched]()



def fair_example():
    # Fair scheduler
    scheduler = RoundRobinScheduler(ENV_KWARGS['num_executors'],
                                    dynamic_partition=True)
    
    print(f'Example: Fair Scheduler')
    print('Env settings:')
    pprint(ENV_KWARGS)

    print('Running episode...')
    avg_job_duration = run_episode(ENV_KWARGS, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)
    print()



def decima_example():
    cfg = load(filename=osp.join('config', 'decima_tpch.yaml'))

    agent_cfg = cfg['agent'] \
        | {'num_executors': ENV_KWARGS['num_executors'],
           'state_dict_path': osp.join('models', 'decima', 'model.pt')}
    
    scheduler = make_scheduler(agent_cfg)

    print(f'Example: Decima')
    print('Env settings:')
    pprint(ENV_KWARGS)

    print('Running episode...')
    avg_job_duration = run_episode(ENV_KWARGS, scheduler)

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