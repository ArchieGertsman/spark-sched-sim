from memory_profiler import profile
import sys
sys.path.append('../data_generation/tpch/')

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.signal import lfilter

from ..utils.device import device



def calc_joint_entropy(op_probs, prlvl_probs):
    '''returns the joint entropy over the two distributions returned
    by the neural network'''
    joint_probs = torch.outer(op_probs, prlvl_probs).flatten()
    return Categorical(joint_probs).entropy()



def sample_action(env, op_probs, prlvl_probs):
    '''given probabilities for selecting the next operation
    and the parallelism level for that operation's job (returned
    by the neural network), returns a randomly sampled 
    action according to those probabilities conisting of
    - an operation `next_op`, and 
    - parallelism level `prlvl`, 
    plus the log probability of the sampled action `action_lgprob` 
    (which maintains a computational graph for learning) and the
    joint entropy of the distributions that came from the network
    '''
    c = Categorical(probs=op_probs)
    next_op_idx = c.sample()
    next_op_idx_lgp = c.log_prob(next_op_idx)
    next_op, j = env.find_op(next_op_idx)

    c = Categorical(probs=prlvl_probs[j])        
    prlvl = c.sample()
    prlvl_lgp = c.log_prob(prlvl)

    action_lgprob = next_op_idx_lgp + prlvl_lgp

    entropy = calc_joint_entropy(op_probs, prlvl_probs[j])

    return \
        next_op, \
        prlvl.item(), \
        action_lgprob.unsqueeze(-1), \
        entropy.unsqueeze(-1)



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



def pad_trajectory(ep_len, tensors):
    '''if the episode ended in less than `ep_len` steps, then pads
    the trajectory with zeros at the end to have length `ep_len`'''
    diff = ep_len - tensors.shape[1]

    if diff > 0:
        tensors = F.pad(tensors, pad=(0, diff))
            
    return tensors



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
    entropies = []

    done = False
    obs = None

    while len(action_lgprobs) < ep_len and not done:
        if obs is None or env.n_active_jobs == 0:
            next_op, prlvl = None, 0
        else:
            dag_batch, op_msk, prlvl_msk = obs
            
            ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)

            next_op, prlvl, action_lgprob, entropy = \
                sample_action(env, ops_probs, prlvl_probs)

            action_lgprobs += [action_lgprob]
            rewards += [reward]
            entropies += [entropy]

        obs, reward, done = env.step(next_op, prlvl)
        
    action_lgprobs = torch.cat(action_lgprobs)
    returns = compute_returns(rewards, discount)
    entropies = torch.cat(entropies)

    return pad_trajectory(
        ep_len, torch.stack([action_lgprobs, returns, entropies]))



def write_tensorboard(
    writer, 
    epoch, 
    ep_len, 
    loss, 
    avg_job_durations_mean, 
    n_completed_jobs_mean
):
    writer.add_scalar('episode length', ep_len, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('avg job duration', avg_job_durations_mean / ep_len, epoch)
    writer.add_scalar('n completed jobs', n_completed_jobs_mean / ep_len, epoch)