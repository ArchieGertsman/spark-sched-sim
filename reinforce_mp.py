import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')
import time
from copy import deepcopy as dcp

import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, SimpleQueue
from scipy.signal import lfilter

from gym_dagsched.envs.dagsched_env import DagSchedEnv
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.data_generation.random_datagen import RandomDataGen
from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
from gym_dagsched.utils.metrics import avg_job_duration
from gym_dagsched.utils.device import device



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



def episode_runner(worker_id, discount, optim, in_q, out_q):
    '''subprocess function which runs episodes and trains the model 
    by communicating with the parent process'''

    # IMPORTANT! ensures that the different child processes
    # don't all generate the same random numbers. Otherwise,
    # each process would produce an identical episode.
    torch.manual_seed(worker_id)

    # this subprocess's copy of the environment
    env = DagSchedEnv()

    while data := in_q.get():
        # receive episode data from parent process
        initial_timeline, workers, ep_length, policy = data

        action_lgprobs, returns = \
            run_episode(
                env,
                initial_timeline,
                workers,
                ep_length,
                policy,
                discount
            )        

        # send returns back to parent process
        out_q.put(returns)

        # receive baselines from parent process
        baselines = in_q.get()

        advantages = returns - baselines

        # update model parameters
        optim.zero_grad()
        loss = -action_lgprobs @ advantages
        loss.backward()
        optim.step()

        # signal end of parameter updates to parent
        out_q.put(None)



def launch_subprocesses(n_ep_per_seq, discount, optim):
    procs = []
    in_qs = []
    out_qs = []
    for i in range(n_ep_per_seq):
        in_q = SimpleQueue()
        in_qs += [in_q]
        out_q = SimpleQueue()
        out_qs += [out_q]
        proc = Process(
            target=episode_runner, 
            args=(i, discount, dcp(optim), in_q, out_q))
        proc.start()
    return procs, in_qs, out_qs



def terminate_subprocesses(in_qs, procs):
    for in_q in in_qs:
        in_q.put(None)

    for proc in procs:
        proc.join()



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
    '''trains the model on multiple different job arrival sequences, where
    `n_ep_per_seq` episodes are repeated on each sequence in parallel
    using multiprocessing'''

    procs, in_qs, out_qs = \
        launch_subprocesses(n_ep_per_seq, discount, optim)

    mean_ep_len = initial_mean_ep_len

    for i in range(n_sequences):
        print(f'beginning training on sequence {i+1}')

        ep_len = np.random.geometric(1/mean_ep_len)
        ep_len = max(ep_len, min_ep_len)

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=2000.)
        workers = datagen.workers(n_workers=n_workers)

        # send episode data to each of the subprocesses
        for in_q in in_qs:
            data = initial_timeline, workers, ep_len, policy
            in_q.put(data)

        # retreieve returns from each of the subprocesses
        returns_list = []
        for out_q in out_qs:
            returns = out_q.get()
            returns_list += [returns]

        # compute baselines
        returns_mat = torch.stack(returns_list)
        baselines = returns_mat.mean(axis=0)

        # send baselines back to each of the subprocesses
        for in_q in in_qs:
            in_q.put(baselines)

        # wait for each of the subprocesses to finish parameter updates
        for j,out_q in enumerate(out_qs):
            out_q.get()
            print(f'ep {j+1} complete')

        mean_ep_len += delta_ep_len

    terminate_subprocesses(in_qs, procs)



if __name__ == '__main__':
    torch.set_num_threads(1)

    datagen = RandomDataGen(
        max_ops=8, # 20
        max_tasks=4, # 200
        mean_task_duration=2000.,
        n_worker_types=1)

    n_workers = 5

    policy = ActorNetwork(5, 8, n_workers)
    policy.to(device)
    policy.share_memory()

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    train(
        datagen, 
        policy, 
        optim, 
        n_sequences=4,
        n_ep_per_seq=8,
        discount=.99,
        n_workers=n_workers,
        initial_mean_ep_len=500,
        delta_ep_len=50,
        min_ep_len=250
    )
