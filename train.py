import sys
import shutil
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.reinforce import reinforce_sync, reinforce_async


if __name__ == '__main__':
    shutil.rmtree('log/proc/', ignore_errors=True)
    os.mkdir('log/proc/')

    sys.stdout = open(f'log/proc/main.out', 'a')

    torch.autograd.set_detect_anomaly(True)
    set_start_method('spawn')

    print('cuda available:', torch.cuda.is_available())

    model = ActorNetwork(
        num_node_features=5, 
        num_dag_features=3)

    # optim = torch.optim.Adam(agent.parameters(), lr=.005)

    # writer = SummaryWriter('log/train')
    writer = None

    reinforce_async.train(
        model,
        optim_type=torch.optim.Adam,
        optim_lr=.005,
        n_sequences=1,
        num_envs=2,
        discount=.99,
        entropy_weight_init=1.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_job_arrivals=5,
        n_init_jobs=1, 
        mjit=25000,
        n_workers=50,
        initial_mean_ep_len=3500,
        ep_len_growth=0,
        min_ep_len=0,
        writer=writer
    )

    # writer.close()

    # torch.save(agent.state_dict(), 'agent.pt')