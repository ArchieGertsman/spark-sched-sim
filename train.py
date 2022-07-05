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
from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.utils.device import device
from gym_dagsched.reinforce import reinforce_mp
from gym_dagsched.reinforce import reinforce_serial


if __name__ == '__main__':
    assert len(sys.argv) == 2
    processing_mode = sys.argv[1]
    assert processing_mode in ['m', 's']

    print(torch.cuda.is_available())

    if processing_mode == 'm':
        assert not torch.cuda.is_available()
        torch.set_num_threads(1)

    datagen = RandomDataGen(
        max_ops=8, # 20
        max_tasks=4, # 200
        mean_task_duration=2000.,
        n_worker_types=1)

    # datagen = TPCHDataGen()

    n_workers = 100

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
        n_sequences=100,
        n_ep_per_seq=8,
        discount=.99,
        entropy_weight_init=.1,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_workers=n_workers,
        # initial_mean_ep_len=5000, #50,
        # ep_len_growth=250, #10,
        # min_ep_len=1000, #50,
        initial_mean_ep_len=50,
        ep_len_growth=10,
        min_ep_len=50,
        writer=writer
    )

    torch.save(policy.state_dict(), 'policy.pt')