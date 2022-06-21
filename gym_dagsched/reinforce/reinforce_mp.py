"""NOTE: this only works with CPU, and moreover, CUDA devices 
must be made invisible to python by running
    export CUDA_VISIBLE_DEVICES=""
prior to running main.py with [processing_mode] set to 'm'
"""

import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')

import numpy as np
import torch
from torch.multiprocessing import Process, SimpleQueue
from scipy.signal import lfilter

from .reinforce_utils import *
from ..envs.dagsched_env import DagSchedEnv
from ..utils.metrics import avg_job_duration
from ..utils.device import device



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
            args=(i, discount, optim, in_q, out_q))
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

        # sample the length of the current episode
        ep_len = np.random.geometric(1/mean_ep_len)
        ep_len = max(ep_len, min_ep_len)

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=2000.)
        workers = datagen.workers(n_workers=n_workers)

        # send episode data to each of the subprocesses, 
        # which starts the episodes
        for in_q in in_qs:
            data = initial_timeline, workers, ep_len, policy
            in_q.put(data)

        # retreieve returns from each of the subprocesses
        returns_list = [out_q.get() for out_q in out_qs]

        # compute baselines
        returns_mat = torch.stack(returns_list)
        baselines = returns_mat.mean(axis=0)

        # send baselines back to each of the subprocesses
        for in_q in in_qs:
            in_q.put(baselines)

        # wait for each of the subprocesses to finish parameter updates
        for j,out_q in enumerate(out_qs):
            out_q.get()
            # print(f'ep {j+1} complete')

        # increase the mean episode length
        mean_ep_len += delta_ep_len

    terminate_subprocesses(in_qs, procs)

