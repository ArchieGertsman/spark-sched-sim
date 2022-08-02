import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')
from time import time

import numpy as np
import torch

from .reinforce_base import *
from ..envs.dagsched_env import DagSchedEnv
from ..utils.metrics import avg_job_duration



def learn_from_trajectories(
    optim,
    entropy_weight,
    action_lgprobs_list, 
    returns_list, 
    entropies_list
):
    '''given a list of trajectories from multiple MDP episodes
    that were repeated on a fixed job arrival sequence, update the model 
    parameters using the REINFORCE algorithm as in the Decima paper.
    '''
    action_lgprobs_mat = torch.stack(action_lgprobs_list).to(device=device)
    returns_mat = torch.stack(returns_list)
    entropies_mat = torch.stack(entropies_list).to(device=device)

    baselines = returns_mat.mean(axis=0)
    advantages_mat = returns_mat - baselines

    action_lgprobs = action_lgprobs_mat.flatten().to(device=device)
    advantages = advantages_mat.flatten().to(device=device)

    policy_loss  = -action_lgprobs @ advantages
    entropy_loss = entropy_weight * entropies_mat.sum()

    ep_len = baselines.numel()

    optim.zero_grad()
    loss = (policy_loss + entropy_loss) / ep_len
    loss.backward()
    optim.step()

    return loss.item()



def train(
    datagen, 
    policy, 
    n_sequences,
    n_ep_per_seq,
    discount,
    entropy_weight_init,
    entropy_weight_decay,
    entropy_weight_min,
    n_workers,
    initial_mean_ep_len,
    ep_len_growth,
    min_ep_len,
    writer
):
    '''train the model on multiple different job arrival sequences'''

    env = DagSchedEnv()

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    for epoch in range(n_sequences):
        print(f'beginning training on sequence {epoch+1}')

        # ep_len = np.random.geometric(1/mean_ep_len)
        # ep_len = max(ep_len, min_ep_len)
        ep_len = 8000

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=300, n_init_jobs=0, mjit=1000.)
        workers = datagen.workers(n_workers=n_workers)

        # run multiple episodes on this fixed sequence
        action_lgprobs_list = []
        returns_list = []
        entropies_list = []

        avg_job_durations = np.zeros(n_ep_per_seq)
        n_completed_jobs_list = np.zeros(n_ep_per_seq)

        # times = []
        t_start = time()

        for j in range(n_ep_per_seq):
            # t0 = time()

            t0 = time()
            action_lgprobs, returns, entropies = \
                run_episode(
                    env,
                    initial_timeline,
                    workers,
                    ep_len,
                    policy,
                    discount
                )
            t1 = time()
            print('t_ep:', t1-t0)

            action_lgprobs_list += [action_lgprobs]
            returns_list += [returns]
            entropies_list += [entropies]

            avg_job_durations[j] = 0 # avg_job_duration(env)
            n_completed_jobs_list[j] = env.n_completed_jobs

            # print(f'episode {j+1} complete:', n_completed_jobs_list[j], avg_job_durations[j])

            # t1 = time()
            # times += [t1-t0]

        # t_ep = np.mean(times)


        t0 = time()
        loss = learn_from_trajectories(
            optim, 
            entropy_weight,
            action_lgprobs_list, 
            returns_list, 
            entropies_list)
        t1 = time()
        t_learn = t1-t0
        print('t_learn:', t_learn)

        t_end = time()
        t_total = t_end - t_start
        print('t_total:', t_total)

        write_tensorboard(
            writer, 
            epoch, 
            ep_len, 
            loss, 
            avg_job_durations.mean(), 
            n_completed_jobs_list.mean()
        )

        mean_ep_len += ep_len_growth

        entropy_weight = max(
            entropy_weight - entropy_weight_decay, 
            entropy_weight_min)

        

        # print(f'ep: {t_ep/t_total*100.: 3f}%; learn: {t_learn/t_total*100.: 3f}%')

    writer.close()



