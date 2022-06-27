"""this script is called as follows:
    python train.py [processing_mode]
where [processing_mode] is either
    - s (serial), or
    - m (multiprocessing)
"""

import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from gym_dagsched.data_generation.random_datagen import RandomDataGen
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.utils.device import device
from gym_dagsched.reinforce import reinforce_mp
from gym_dagsched.reinforce import reinforce_serial


if __name__ == '__main__':
    assert len(sys.argv) == 2
    processing_mode = sys.argv[1]
    assert processing_mode in ['m', 's']

    if processing_mode == 'm':
        torch.set_num_threads(1)

    datagen = RandomDataGen(
        max_ops=8, # 20
        max_tasks=4, # 200
        mean_task_duration=2000.,
        n_worker_types=1)

    n_workers = 10

    policy = ActorNetwork(5, 8, n_workers)
    policy.to(device)

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    writer = SummaryWriter('tensorboard')

    train = reinforce_mp.train \
        if processing_mode == 'm' \
        else reinforce_serial.train

    train(
        datagen, 
        policy, 
        optim, 
        n_sequences=40,
        n_ep_per_seq=8,
        discount=.99,
        entropy_weight_init=.1,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_workers=n_workers,
        initial_mean_ep_len=250,
        ep_len_growth=25,
        min_ep_len=250,
        writer=writer
    )

    torch.save(policy.state_dict(), 'policy.pt')