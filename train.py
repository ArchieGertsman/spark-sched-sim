import torch
from torch import optim

from trainers import *
from spark_sched_sim.schedulers import *


if __name__ == '__main__':
    trainer = PPO(
        scheduler_cls = DecimaScheduler,

        device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu'),

        log_options = {
            'proc_dir': 'ignore/log/proc',
            # 'tensorboard_dir': 'ignore/log/train/'
        },

        checkpoint_options = {
            'dir': 'ignore/models',
            'freq': 50
        },

        env_kwargs = {
            'num_executors': 50,
            'job_arrival_cap': 200,
            'job_arrival_rate': 1/25000,
            'moving_delay': 2000.,
            'mean_time_limit': 2e7
        },

        model_kwargs = {
            'dim_embed': 16,
            'optim_class': optim.Adam,
            'optim_lr': .0005,
            'max_grad_norm': .5
        }
    )

    trainer.train(num_iterations=1, num_envs=16)
