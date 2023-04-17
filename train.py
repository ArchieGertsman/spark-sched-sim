from train_algs import *


if __name__ == '__main__':
    env_kwargs = {
        'num_executors': 50,
        'num_init_jobs': 1,
        'num_job_arrivals': 200,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    VPG(
        env_kwargs,
        num_iterations=5,
        num_envs=4,
        log_dir='ignore/log/proc',
        # summary_writer_dir='ignore/log/train/', 
        model_save_dir='ignore/models',
        optim_lr=.001,
        entropy_weight_init=1.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        max_time_mean_init=2e6,
        max_time_mean_growth=1.0008,
        max_time_mean_ceil=2e7,
        num_epochs=2,
        seed=2147483647,
        gamma=1.
    ).train()
