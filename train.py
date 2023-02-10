from gym_dagsched.train_algs.reinforce import Reinforce


if __name__ == '__main__':
    env_kwargs = {
        'num_workers': 10,
        'num_init_jobs': 1,
        'num_job_arrivals': 20,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    Reinforce(
        env_kwargs,
        num_iterations=500,
        num_envs=4,
        log_dir='ignore/log/proc',
        summary_writer_dir='ignore/log/train', 
        model_save_dir='ignore/models',
    ).train()
