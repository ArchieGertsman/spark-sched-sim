import torch
from torch.utils.tensorboard import SummaryWriter

from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.reinforce import reinforce


if __name__ == '__main__':
    policy = ActorNetwork()

    writer = SummaryWriter('tensorboard')

    reinforce.train(
        policy,
        n_sequences=10,
        n_ep_per_seq=16,
        discount=.99,
        entropy_weight_init=1.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_job_arrivals=200, 
        n_init_jobs=1, 
        mjit=25000,
        n_workers=50,
        # initial_mean_ep_len=1000,
        initial_mean_ep_len=1000,
        ep_len_growth=100,
        # min_ep_len=500,
        min_ep_len=100,
        writer=writer
    )

    torch.save(policy.state_dict(), 'policy.pt')