import torch
from torch.utils.tensorboard import SummaryWriter

from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.reinforce import reinforce


if __name__ == '__main__':
    agent = ActorNetwork(5, 3)
    optim = torch.optim.Adam(agent.parameters(), lr=.005)

    writer = SummaryWriter('tensorboard')

    torch.autograd.set_detect_anomaly(True)

    reinforce.train(
        agent,
        optim,
        n_sequences=1,
        n_ep_per_seq=1,
        discount=.99,
        entropy_weight_init=1.,
        entropy_weight_decay=1e-3,
        entropy_weight_min=1e-4,
        n_job_arrivals=50, 
        n_init_jobs=1, 
        mjit=25000,
        n_workers=50,
        initial_mean_ep_len=3500,
        ep_len_growth=0,
        min_ep_len=500,
        writer=writer
    )

    torch.save(agent.state_dict(), 'agent.pt')