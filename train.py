import sys
import shutil
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.agents.decima_agent import DecimaAgent
import gym_dagsched.train_algs.reinforce as reinforce


def main():
    setup()

    num_workers = 10

    decima_agent = \
        DecimaAgent(num_workers, 
                    device=torch.device('cpu'))

    writer = SummaryWriter('log/train')
    # writer = None

    reinforce.train(decima_agent, 
                    writer=writer, 
                    num_epochs=200,
                    num_init_jobs=1,
                    num_job_arrivals=20,
                    job_arrival_rate=1/25000,
                    entropy_weight_init=.1)

    if writer:
        writer.close()




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