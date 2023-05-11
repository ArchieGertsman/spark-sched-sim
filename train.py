from torch.optim import Adam

from trainers import VPG


if __name__ == '__main__':
    trainer = VPG(
        num_iterations=10000,
        num_envs=16,
        log_options={
            'proc_dir': 'ignore/log/proc',
            'tensorboard_dir': 'ignore/log/train/'
        },
        model_save_options={
            'dir': 'ignore/models',
            'freq': 100
        },
        time_limit_options={
            'init': 2e6,
            'factor': 1.0008,
            'ceil': 2e7
        },
        entropy_options={
            'init': 1.,
            'delta': 1e-3,
            'floor': 1e-4
        },
        env_kwargs={
            'num_executors': 50,
            'job_arrival_cap': 200,
            'job_arrival_rate': 1/25000,
            'moving_delay': 2000.
        },
        model_kwargs={
            'dim_embed': 8,
            'optim_class': Adam,
            'optim_lr': .001,
            'max_grad_norm': 2.
        }
    )

    trainer.train()
