from memory_profiler import profile
import sys
sys.path.append('../data_generation/tpch/')
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.signal import lfilter
from pympler import tracker

from ..utils.device import device



def calc_joint_entropy(op_probs, prlvl_probs):
    '''returns the joint entropy over the two distributions returned
    by the neural network'''
    joint_probs = torch.outer(op_probs, prlvl_probs).flatten()
    c = Categorical(joint_probs)
    entropy = c.entropy()
    return entropy



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
    next_op, job_idx = env.find_op(next_op_idx)

    prlvl_probs = prlvl_probs[job_idx]
    c = Categorical(probs=prlvl_probs)        
    prlvl = c.sample()
    prlvl_lgp = c.log_prob(prlvl)

    action_lgprob = next_op_idx_lgp + prlvl_lgp

    entropy = calc_joint_entropy(op_probs, prlvl_probs)

    return \
        next_op, \
        prlvl.item(), \
        action_lgprob, \
        entropy



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

    trajectory = torch.zeros((ep_len, 3))

    # references to different parts of the trajectory tensor
    # for readability purposes
    action_lgprobs = trajectory[:,0]
    rewards = trajectory[:,1]
    entropies = trajectory[:,2]

    done = False
    obs = None

    # t_model = 0.
    # t_env = 0.

    # t_start = time()

    tr = tracker.SummaryTracker()

    i = 0
    
    while i < ep_len and not done:

        # if len(action_lgprobs) % 200 == 0:
        #     tr.print_diff()
        #     print(flush=True)  
        

        if obs is None or env.n_active_jobs == 0:
            next_op, prlvl = None, 0
        else:
            dag_batch, op_msk, prlvl_msk = obs

            # time the policy
            # t0 = time()

            # print('print 1')
            # tr.print_diff()

            ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)

            # print('print 2')
            # tr.print_diff()

            # t1 = time()
            # t_model += t1-t0


            # print('print 1')
            # tr.print_diff()

            next_op, prlvl, action_lgprob, entropy = \
                sample_action(env, ops_probs, prlvl_probs)

            # print('print 3')
            # tr.print_diff()

            action_lgprobs[i] = action_lgprob
            rewards[i] = reward
            entropies[i] = entropy

            # print('print 4')
            # tr.print_diff()

        
        # time the env
        # t0 = time()


    
        obs, reward, done = env.step(next_op, prlvl)

        # print('print 5')
        # tr.print_diff()

        # print(flush=True)

        i += 1

        # t1 = time()
        # t_env += t1-t0

    # t_end = time()
    # t_total = t_end - t_start


    # print(f'policy: {t_model/t_total*100.: 3f}%; obs: {env.total_time/t_total*100.: 3f}%')

    returns = compute_returns(rewards.detach(), discount)

    return action_lgprobs, returns, entropies



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