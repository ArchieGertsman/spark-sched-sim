import torch

from gym_dagsched.train_algs import PPO, VPG



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
        num_iterations=5000,
        num_envs=8,
        log_dir='ignore/log/proc',
        summary_writer_dir='ignore/log/train', 
        model_save_dir='ignore/models',
        optim_lr=.001,
        entropy_weight_init=.01,
        entropy_weight_decay=1e-4,
        entropy_weight_min=0.,
        max_time_mean_init=2000e3,
        max_time_mean_growth=10e3,
        max_time_mean_clip_range=100e3,
        batch_size=1024,
        num_epochs=4,
        seed=2147483647,
        target_kl=.01,
    ).train()
