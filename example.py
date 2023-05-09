'''An example of how to run a job scheduling simulation'''

import gymnasium as gym

from spark_sched_sim.schedulers import RandomScheduler, RoundRobinScheduler
from trainers.utils import HiddenPrints
from spark_sched_sim import metrics



if __name__ == '__main__':
    # number of simulated executors
    num_executors = 50

    scheduler = RandomScheduler()
    # scheduler = RoundRobinScheduler(num_executors, dynamic_partition=True)

    # setup gym environment
    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    env_kwargs = {
        'num_executors': num_executors,
        'job_arrival_cap': 100,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000. #,
        # 'render_mode': 'human' # visualize simulation
    }
    env = gym.make(env_id, **env_kwargs)

    # run an episode
    with HiddenPrints():
        obs, _ = env.reset(seed=42, options=None)
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = scheduler(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

    avg_job_duration = int(metrics.avg_job_duration(env) * 1e-3)
    print(f'Average job duration: {avg_job_duration}s', flush=True)
    
    # cleanup rendering
    env.close()