'''An example of how to run a job scheduling simulation'''

import gymnasium as gym

from spark_sched_sim.schedulers import RandomScheduler
from train_algs.utils.hidden_prints import HiddenPrints
from spark_sched_sim import metrics



if __name__ == '__main__':
    # select the number of simulated executors
    random_scheduler = RandomScheduler()

    # same settings as in training
    env_kwargs = {
        'num_executors': 50,
        'num_init_jobs': 1,
        'num_job_arrivals': 20,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000. #,
        # 'render_mode': 'human' # visualize simulation
    }

    # setup gym environment
    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    env = gym.make(env_id, **env_kwargs)

    # run an episode
    with HiddenPrints():
        obs, _ = env.reset(seed=0, options=None)
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = random_scheduler(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

    avg_job_duration = int(metrics.avg_job_duration(env) * 1e-3)
    print(f'Average job duration: {avg_job_duration}s', flush=True)
    
    # cleanup rendering
    env.close()