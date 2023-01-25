import sys
import shutil
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.reinforce.reinforce_async import train


def main():
    configure_main_process()

    num_workers = 50

    model = ActorNetwork(
        num_node_features=5, 
        num_dag_features=3,
        num_workers=num_workers)

    # model.load_state_dict(torch.load('model.pt'))

    # writer = SummaryWriter('log/train')
    writer = None

    train(model,
          optim_class=torch.optim.Adam,
          optim_lr=.001,
          n_sequences=500,
          num_envs=16,
          discount=.99,
          entropy_weight_init=1.,
          entropy_weight_decay=1e-3,
          entropy_weight_min=1e-4,
          num_job_arrivals=30,
          num_init_jobs=1, 
          job_arrival_rate=1/25000,
          num_workers=num_workers,
          max_time_mean_init=2e6,
          max_time_mean_growth=1.6e3,
          max_time_mean_ceil=2e7,
          writer=writer)

    if writer:
        writer.close()

    # torch.save(model.state_dict(), 'model.pt')




def configure_main_process():
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    # torch.autograd.set_detect_anomaly(True)
    set_start_method('forkserver')

    torch.manual_seed(69)
    np.random.seed(69)

    # torch.set_num_threads(1)

    print('cuda available:', torch.cuda.is_available())



if __name__ == '__main__':
    main()