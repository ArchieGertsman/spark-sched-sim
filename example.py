'''An example of how to run a job scheduling simulation'''

import gymnasium as gym

from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from trainers.utils import HiddenPrints
from spark_sched_sim import metrics



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
    # number of simulated executors
    num_executors = 50

    # Trained instance of the Decima scheduler
    scheduler = DecimaScheduler(
        num_executors, state_dict_path='results/models/decima.pt')
    
    # Fair scheduler
    # scheduler = RoundRobinScheduler(num_executors, True)

    # gym environment settings
    env_kwargs = {
        'num_executors': num_executors,
        'job_arrival_cap': 1000,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
        # 'render_mode': 'human' # visualize simulation
    }

    avg_job_duration = run_episode(env_kwargs, scheduler)

    print(f'Average job duration: {avg_job_duration:.1f}s', flush=True)
    