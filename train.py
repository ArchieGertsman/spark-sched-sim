"""this script is called as follows:
    python train.py [processing_mode]
where [processing_mode] is either
    - s (serial), or
    - m (multiprocessing)
"""

import sys

import torch

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

    train = reinforce_mp.train \
        if processing_mode == 'm' \
        else reinforce_serial.train

    train(
        datagen, 
        policy, 
        optim, 
        n_sequences=20,
        n_ep_per_seq=4,
        discount=.99,
        n_workers=n_workers,
        initial_mean_ep_len=500,
        delta_ep_len=50,
        min_ep_len=250
    )