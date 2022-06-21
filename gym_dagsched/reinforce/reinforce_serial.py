import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .reinforce_utils import *
from ..envs.dagsched_env import DagSchedEnv
from ..utils.metrics import avg_job_duration



def learn_from_trajectories(action_lgprobs_list, returns_list, optim):
    '''given a list of trajectories from multiple MDP episodes
    that were repeated on a fixed job arrival sequence, update the model 
    parameters using the REINFORCE algorithm as in the Decima paper.
    '''
    action_lgprobs_mat = torch.stack(action_lgprobs_list)
    returns_mat = torch.stack(returns_list)

    baselines = returns_mat.mean(axis=0)
    advantages_mat = returns_mat - baselines

    optim.zero_grad()

    loss_mat = -action_lgprobs_mat * advantages_mat
    loss = loss_mat.sum()
    loss.backward()

    optim.step()
    return loss.item()



def train(
    datagen, 
    policy, 
    optim, 
    n_sequences,
    n_ep_per_seq,
    discount,
    n_workers,
    initial_mean_ep_len,
    delta_ep_len,
    min_ep_len
):
    '''train the model on multiple different job arrival sequences'''

    env = DagSchedEnv()

    writer = SummaryWriter('tensorboard')

    mean_ep_len = initial_mean_ep_len

    for i in range(n_sequences):
        print(f'beginning training on sequence {i+1}')

        ep_len = np.random.geometric(1/mean_ep_len)
        ep_len = max(ep_len, min_ep_len)

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=1000.)
        workers = datagen.workers(n_workers=n_workers)

        # run multiple episodes on this fixed sequence
        action_lgprobs_list = []
        returns_list = []

        avg_job_durations = np.zeros(n_ep_per_seq)
        n_completed_jobs = np.zeros(n_ep_per_seq)

        for j in range(n_ep_per_seq):
            action_lgprobs, returns = \
                run_episode(
                    env,
                    initial_timeline,
                    workers,
                    ep_len,
                    policy,
                    discount
                )

            action_lgprobs_list += [action_lgprobs]
            returns_list += [returns]

            n_completed_jobs[j] = env.n_completed_jobs
            avg_job_durations[j] = avg_job_duration(env)

            print(f'episode {j+1} complete:', n_completed_jobs[j], avg_job_durations[j])

        loss = learn_from_trajectories(action_lgprobs_list, returns_list, optim)

        writer.add_scalar('episode length', ep_len, i)
        writer.add_scalar('loss', -loss / ep_len, i)
        writer.add_scalar('avg job duration', avg_job_durations.mean() / n_completed_jobs.mean(), i)
        writer.add_scalar('n completed jobs', n_completed_jobs.mean() / ep_len, i)

        mean_ep_len += delta_ep_len

    writer.close()



