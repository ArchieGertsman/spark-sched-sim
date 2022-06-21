import sys
sys.path.append('../data_generation/tpch/')

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import lfilter

from ..utils.device import device



def sample_action(env, ops_probs, prlvl_probs):
    '''given probabilities for selecting the next operation
    and the parallelism level for that operation's job (returned
    by the neural network), returns a randomly sampled 
    action according to those probabilities conisting of
    - an operation `next_op`, and 
    - parallelism level `prlvl`, 
    plus the log probability of the sampled action `action_lgprob` 
    (which maintains a computational graph for learning)
    '''
    c = torch.distributions.Categorical(probs=ops_probs)
    next_op_idx = c.sample()
    next_op_idx_lgp = c.log_prob(next_op_idx)
    next_op, j = env.find_op(next_op_idx)

    c = torch.distributions.Categorical(probs=prlvl_probs[j])        
    prlvl = c.sample()
    prlvl_lgp = c.log_prob(prlvl)

    action_lgprob = next_op_idx_lgp + prlvl_lgp

    return next_op, prlvl.item(), action_lgprob.unsqueeze(-1)



def compute_returns(rewards, discount):
    ''' returs array `y` where `y[i] = rewards[i] + discount * y[i+1]`
    credit: https://stackoverflow.com/a/47971187/5361456
    '''
    rewards = np.array(rewards)
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return torch.from_numpy(y[::-1].copy()).float().to(device=device)



def pad_trajectory(ep_len, action_lgprobs, returns):
    '''if the episode ended in less than `ep_len` steps, then pads
    the trajectory with zeros at the end to have length `ep_len`'''
    diff = ep_len - len(action_lgprobs)
    if diff > 0:
        action_lgprobs = F.pad(action_lgprobs, pad=(0, diff))
        returns = F.pad(returns, pad=(0, diff))
    return action_lgprobs, returns



def run_episode(
    env,
    initial_timeline,
    workers,
    ep_len,
    policy,
    discount
):
    '''runs one MDP episode for `ep_len` iterations given 
    a job arrival sequence stored in `initial_timeline` and a 
    set of workers, and returns the trajectory of action log
    probabilities (which contain gradients) and returns, each 
    of length `ep_len`
    '''
    env.reset(initial_timeline, workers)

    action_lgprobs = []
    rewards = []

    done = False
    obs = None

    while len(action_lgprobs) < ep_len and not done:
        if obs is None or env.n_active_jobs == 0:
            next_op, prlvl = None, 0
        else:
            dag_batch, op_msk, prlvl_msk = obs
            
            ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)

            next_op, prlvl, action_lgprob = \
                sample_action(env, ops_probs, prlvl_probs)

            action_lgprobs += [action_lgprob]
            rewards += [reward]

        obs, reward, done = env.step(next_op, prlvl)
        
    action_lgprobs = torch.cat(action_lgprobs)
    returns = compute_returns(rewards, discount)

    return pad_trajectory(ep_len, action_lgprobs, returns)