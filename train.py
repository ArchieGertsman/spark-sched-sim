from gym_dagsched.train_algs.reinforce import Reinforce
from gym_dagsched.train_algs.ppo import PPO


if __name__ == '__main__':
    env_kwargs = {
        'num_workers': 10,
        'num_init_jobs': 1,
        'num_job_arrivals': 20,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.
    }

    PPO(
        env_kwargs,
        num_iterations=500,
        num_envs=4,
        log_dir='ignore/log/proc',
        summary_writer_dir='ignore/log/train', 
        model_save_dir='ignore/models',
        optim_lr=3e-3,
        entropy_weight_init=1e-3,
        entropy_weight_decay=0.,
        entropy_weight_min=0.
    ).train()
