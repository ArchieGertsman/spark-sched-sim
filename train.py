"""this script is called as follows:
    python train.py [processing_mode]
where [processing_mode] is either
    - s (serial), or
    - m (multiprocessing)
"""

import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method

from gym_dagsched.data_generation.random_datagen import RandomDataGen
from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.utils.device import device
from gym_dagsched.reinforce import reinforce_mp
from gym_dagsched.reinforce import reinforce_serial


if __name__ == '__main__':
    assert len(sys.argv) in [2, 3]
    processing_mode = sys.argv[1]
    assert processing_mode in ['m', 's']

    # n_workers = 50
    n_workers = 10
    policy = ActorNetwork(5, 8, n_workers)

    if len(sys.argv) == 3:
        policy_path = sys.argv[2]
        policy.load_state_dict(torch.load(policy_path))

    print('cuda available:', torch.cuda.is_available())

    if processing_mode == 'm':
        set_start_method('spawn')

    datagen = RandomDataGen(
        max_ops=8, # 20
        max_tasks=4, # 200
        mean_task_duration=2000.,
        n_worker_types=1)

    # datagen = TPCHDataGen()

    writer = SummaryWriter('tensorboard')

    train = reinforce_mp.train \
        if processing_mode == 'm' \
        else reinforce_serial.train

    train(
        datagen, 
        policy,
        n_sequences=200,
        n_ep_per_seq=6,
        discount=.99,
        entropy_weight_init=5.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_workers=n_workers,
        # initial_mean_ep_len=5000, #50,
        # ep_len_growth=250, #10,
        # min_ep_len=1000, #50,
        initial_mean_ep_len=250,
        ep_len_growth=10,
        min_ep_len=100,
        writer=writer
    )

    torch.save(policy.state_dict(), 'policy.pt')