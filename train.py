from train_algs import *


if __name__ == '__main__':
    env_kwargs = {
        'num_executors': 50,
        'job_arrival_cap': 200,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    VPG(
        env_kwargs,
        num_iterations=10000,
        num_envs=16,
        log_dir='ignore/log/proc',
        summary_writer_dir='ignore/log/train/', 
        model_save_dir='ignore/models',
        optim_lr=.001,
        entropy_weight_init=.1,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        mean_time_limit_init=2e6,
        mean_time_limit_growth=1.0008,
        mean_time_limit_ceil=2e7,
        seed=2147483647,
        model_save_freq=100,
        max_grad_norm=2.
    ).train()
