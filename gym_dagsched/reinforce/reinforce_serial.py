import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')
from time import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .reinforce_base import *
from ..envs.vec_dagsched_env import VecDagSchedEnv
from ..utils.metrics import avg_job_duration



def learn_from_trajectories(
    optim,
    entropy_weight,
    action_lgps_batch, 
    returns_batch, 
    entropies_batch
):
    '''given a list of trajectories from multiple MDP episodes
    that were repeated on a fixed job arrival sequence, update the model 
    parameters using the REINFORCE algorithm as in the Decima paper.
    '''
    action_lgps_batch = action_lgps_batch.to(device=device)
    entropies_batch = entropies_batch.to(device=device)

    baselines = returns_batch.mean(axis=0)
    advantages_batch = returns_batch - baselines

    action_lgprobs = action_lgps_batch.flatten().to(device=device)
    advantages = advantages_batch.flatten().to(device=device)

    policy_loss  = -action_lgprobs @ advantages
    entropy_loss = entropy_weight * entropies_batch.sum()

    ep_len = baselines.numel()

    optim.zero_grad()
    loss = (policy_loss + entropy_loss) / ep_len
    loss.backward()
    optim.step()

    return loss.item()





def invoke_policy(policy, obs_batch, num_jobs_per_env):
    dag_batch, op_msk_batch, prlvl_msk_batch = obs_batch 

    op_scores_batch, prlvl_scores_batch, num_ops_per_env = \
        policy(
            dag_batch.to(device=device), 
            num_jobs_per_env.to(device=device)
        )

    op_scores_batch, prlvl_scores_batch, num_ops_per_env = \
        op_scores_batch.cpu(), prlvl_scores_batch.cpu(), num_ops_per_env.cpu()

    op_scores_batch[(1-op_msk_batch).nonzero()] = torch.finfo(torch.float).min
    prlvl_scores_batch[(1-prlvl_msk_batch).nonzero()] = torch.finfo(torch.float).min

    op_scores_list = torch.split(op_scores_batch, num_ops_per_env.tolist())
    op_scores_batch = pad_sequence(op_scores_list, padding_value=torch.finfo(torch.float).min).t()

    return op_scores_batch, prlvl_scores_batch




def sample_action_batch(vec_env, op_scores_batch, prlvl_scores_batch):
    c_op = Categorical(logits=op_scores_batch)
    op_idx_batch = c_op.sample()
    op_idx_lgp_batch = c_op.log_prob(op_idx_batch)
    op_batch, job_idx_batch = vec_env.find_op_batch(op_idx_batch)

    prlvl_scores_batch = prlvl_scores_batch[job_idx_batch]
    c_prlvl = Categorical(logits=prlvl_scores_batch)
    prlvl_batch = c_prlvl.sample()
    prlvl_lgp_batch = c_prlvl.log_prob(prlvl_batch)

    action_lgp_batch = op_idx_lgp_batch + prlvl_lgp_batch

    entropy_batch = c_op.entropy() + c_prlvl.entropy()

    return op_batch, 1+prlvl_batch, action_lgp_batch, entropy_batch






def compute_returns_batch(rewards_batch, discount):
    rewards_batch = np.array(rewards_batch)
    r = rewards_batch[...,::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return torch.from_numpy(y[...,::-1].copy()).float()






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

    vec_env = VecDagSchedEnv(n=n_ep_per_seq)

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    policy.to(device)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    for epoch in range(n_sequences):
        print(f'beginning training on sequence {epoch+1}')

        # ep_len = np.random.geometric(1/mean_ep_len)
        # ep_len = max(ep_len, min_ep_len)
        ep_len = 100

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=1000.)
        workers = datagen.workers(n_workers=n_workers)

        # run multiple episodes on this fixed sequence

        avg_job_durations = np.zeros(n_ep_per_seq)
        n_completed_jobs_list = np.zeros(n_ep_per_seq)








        
        vec_env.reset(initial_timeline, workers)

        action_lgps_batch = torch.zeros((vec_env.n, ep_len))
        rewards_batch = torch.zeros((vec_env.n, ep_len))
        entropies_batch = torch.zeros((vec_env.n, ep_len))

        obs_batch, reward_batch, done_batch = \
            vec_env.step([None]*vec_env.n, [None]*vec_env.n)

        i = 0
        while i < ep_len and not done_batch.any().item():
            op_scores_batch, prlvl_scores_batch = \
                invoke_policy(
                    policy, 
                    obs_batch, 
                    vec_env.num_jobs_per_env())

            op_batch, prlvl_batch, action_lgp_batch, entropy_batch = \
                sample_action_batch(
                    vec_env, 
                    op_scores_batch, 
                    prlvl_scores_batch)

            obs_batch, reward_batch, done_batch = \
                vec_env.step(op_batch, prlvl_batch)

            action_lgps_batch[:,i] = action_lgp_batch
            rewards_batch[:,i] = reward_batch
            entropies_batch[:,i] = entropy_batch

            i += 1

            



        returns_batch = compute_returns_batch(rewards_batch.detach(), discount)

        loss = learn_from_trajectories(
            optim, 
            entropy_weight,
            action_lgps_batch, 
            returns_batch, 
            entropies_batch)

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

    

    writer.close()



