import sys
import shutil
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.reinforce import reinforce_sync, reinforce_async


def main():
    configure_main_process()

    model = ActorNetwork(
        num_node_features=5, 
        num_dag_features=3)

    # writer = SummaryWriter('log/train')
    writer = None

    reinforce_sync.train(
        model,
        optim_type=torch.optim.Adam,
        optim_lr=.001,
        n_sequences=1,
        num_envs=16,
        discount=.99,
        entropy_weight_init=1.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_job_arrivals=200,
        n_init_jobs=1, 
        mjit=25000,
        n_workers=50,
        initial_mean_ep_len=2000,
        ep_len_growth=0,
        min_ep_len=0,
        writer=writer
    )

    # writer.close()

    # torch.save(agent.state_dict(), 'agent.pt')




def configure_main_process():
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    # torch.autograd.set_detect_anomaly(True)
    set_start_method('spawn')

    torch.manual_seed(69)
    np.random.seed(69)

    torch.set_num_threads(1)

    print('cuda available:', torch.cuda.is_available())



if __name__ == '__main__':
    main()