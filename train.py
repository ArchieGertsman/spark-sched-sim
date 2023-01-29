import sys
import shutil
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.agents.decima_agent import DecimaAgent
from gym_dagsched.algs.reinforce import train


def main():
    setup()

    num_workers = 50

    decima_agent = DecimaAgent(num_workers)

    writer = SummaryWriter('log/train')
    # writer = None

    train(decima_agent,
          optim_class=torch.optim.Adam,
          optim_lr=.001,
          n_sequences=1,
          num_envs=4,
          discount=.99,
          entropy_weight_init=1.,
          entropy_weight_decay=1e-3,
          entropy_weight_min=1e-4,
          num_job_arrivals=0,
          num_init_jobs=20, 
          job_arrival_rate=1/25000,
          num_workers=num_workers,
          max_time_mean_init=2e6,
          max_time_mean_growth=1.6e3,
          max_time_mean_ceil=2e7,
          moving_delay=2000.,
          reward_scale=1e-5,
          writer=writer)

    if writer:
        writer.close()

    # torch.save(model.state_dict(), 'model.pt')




def setup():
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    set_start_method('forkserver')

    torch.manual_seed(42)
    np.random.seed(42)

    print('cuda available:', torch.cuda.is_available())



if __name__ == '__main__':
    main()