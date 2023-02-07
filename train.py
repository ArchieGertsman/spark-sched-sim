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

    num_workers = 50

    # model_dir = 'gym_dagsched/data/models'
    # state_dict_path = f'{model_dir}/model_1b_20s_10w_200ep.pt'
    state_dict_path = None

    decima_agent = DecimaAgent(num_workers,
                               state_dict_path=state_dict_path)

    # writer = SummaryWriter('ignore/log/train/c')
    writer = None

    reinforce.train(
        decima_agent,
        writer=writer, 
        world_size=4,
        num_epochs=2,
        num_workers=num_workers,
        num_init_jobs=1,
        num_job_arrivals=20,
        job_arrival_rate=1/25000,
        entropy_weight_init=.1
    )

    if writer:
        writer.close()




def setup():
    shutil.rmtree('ignore/log/proc/', ignore_errors=True)
    os.mkdir('ignore/log/proc/')

    sys.stdout = open(f'ignore/log/proc/main.out', 'a')

    set_start_method('forkserver')

    torch.manual_seed(42)
    np.random.seed(42)

    print('cuda available:', torch.cuda.is_available())



if __name__ == '__main__':
    main()