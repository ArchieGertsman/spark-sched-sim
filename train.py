import numpy as np

from gym_dagsched.train_algs.reinforce import Reinforce
from gym_dagsched.train_algs.ppo import PPO



if __name__ == '__main__':
    env_kwargs = {
        'num_workers': 50,
        'num_init_jobs': 1,
        'num_job_arrivals': 200,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    Reinforce(
        env_kwargs,
        num_iterations=1,
        num_envs=8,
        log_dir='ignore/log/proc',
        # summary_writer_dir='ignore/log/train', 
        model_save_dir='ignore/models',
        optim_lr=5e-3,
        # entropy_weight_init=1e-4,
        # entropy_weight_decay=1e-5,
        # entropy_weight_min=1e-7,
        entropy_weight_init=0.,
        entropy_weight_decay=0.,
        entropy_weight_min=0.,
        max_time_mean_init=2000e3,
        max_time_mean_growth=10e3,
        max_time_mean_clip_range=1000e3,
        batch_size=256,
        num_epochs=4
    ).train()
