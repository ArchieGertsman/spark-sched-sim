import torch

from gym_dagsched.train_algs import *



if __name__ == '__main__':
    env_kwargs = {
        'num_workers': 50,
        'num_init_jobs': 1,
        'num_job_arrivals': 200,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    PPO(
        env_kwargs,
        num_iterations=12000,
        num_envs=10,
        log_dir='ignore/log/proc',
        summary_writer_dir='ignore/log/train/', 
        model_save_dir='ignore/models',
        optim_lr=3e-4,
        entropy_weight_init=.01,
        entropy_weight_decay=1.,
        entropy_weight_min=.01,
        max_time_mean_init=2e6,
        max_time_mean_growth=1.0008,
        max_time_mean_ceil=2e7,
        batch_size=12,
        num_epochs=2,
        seed=2147483647,
        target_kl=.01,
        gamma=1.,
        clip_range=.2
    ).train()
