import sys
sys.path.append('../data_generation/tpch/')
from time import time
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.signal import lfilter
from pympler import tracker
# from torch.profiler import profile, record_function, ProfilerActivity
from gym_dagsched.policies.decima_agent import ActorNetwork

from ..utils.device import device



def calc_joint_entropy(op_probs, prlvl_probs):
    '''returns the joint entropy over the two distributions returned
    by the neural network'''
    joint_probs = torch.outer(op_probs, prlvl_probs).flatten()
    c = Categorical(joint_probs)
    entropy = c.entropy()
    return entropy



def sample_action(env, op_probs, prlvl_probs, op_msk, prlvl_msk):
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
    op_probs[(1-op_msk).nonzero()] = torch.finfo(torch.float).min
    c1 = Categorical(logits=op_probs)
    next_op_idx = c1.sample()
    next_op_idx_lgp = c1.log_prob(next_op_idx)
    next_op, job_idx = env.find_op(next_op_idx)

    prlvl_probs = prlvl_probs[job_idx]
    prlvl_probs[(1-prlvl_msk[job_idx]).nonzero()] = torch.finfo(torch.float).min
    c2 = Categorical(logits=prlvl_probs)        
    prlvl = c2.sample()
    prlvl_lgp = c2.log_prob(prlvl)

    action_lgprob = next_op_idx_lgp + prlvl_lgp

    # entropy = calc_joint_entropy(c1.probs, c2.probs)
    entropy = c1.entropy() + c2.entropy()

    return \
        next_op, \
        1+prlvl.item(), \
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
    return torch.from_numpy(y[::-1].copy()).float()



def invoke_policy(policy, obs):
    dag_batch, op_msk, prlvl_msk = obs

    num_ops_per_dag = dag_batch.num_ops_per_dag
    ops_probs, prlvl_probs = policy(
        dag_batch.to(device=device), 
        num_ops_per_dag.to(device=device),
        op_msk,
        prlvl_msk)

    return ops_probs.cpu(), prlvl_probs.cpu()



def print_time(name, t, total):
    print(f'{name}: {t/total*100:.2f}')


def run_episode(
    rank,
    env,
    initial_timeline,
    workers,
    ep_len,
    policy,
    discount,
    last_obs,
    last_reward,
    done
):
    '''runs one MDP episode for `ep_len` steps on an environment
    initialized with a job arrival sequence stored in `initial_timeline` 
    and a set of workers stored in `workers`. Returns the trajectory of 
    action log probabilities, returns, and entropies, each of length `ep_len`.
    '''
    if initial_timeline is not None:
        assert workers is not None
        env.reset(initial_timeline, workers)

    trajectory = torch.zeros((ep_len, 3))

    # references to different parts of the trajectory tensor
    # for readability purposes
    action_lgprobs = trajectory[:,0]
    rewards = trajectory[:,1]
    entropies = trajectory[:,2]

    obs = last_obs
    reward = last_reward

    i = 0
    while i < ep_len and not done:
        if obs is None or env.n_active_jobs == 0:
            next_op, prlvl = None, 0
        else:
            ops_probs, prlvl_probs = invoke_policy(policy, obs)

            next_op, prlvl, action_lgprob, entropy = \
                sample_action(env, ops_probs, prlvl_probs, obs[1], obs[2])

            action_lgprobs[i] = action_lgprob
            rewards[i] = reward
            entropies[i] = entropy

        obs, reward, done = env.step(next_op, prlvl)
        i += 1

    last_obs = obs
    last_reward = reward
    returns = compute_returns(rewards.detach(), discount)

    return last_obs, last_reward, done, action_lgprobs, returns, entropies



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
    writer.add_scalar('avg job duration', avg_job_durations_mean, epoch)
    writer.add_scalar('n completed jobs', n_completed_jobs_mean, epoch)