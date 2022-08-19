"""NOTE: this only works with CPU, and moreover, CUDA devices 
must be made invisible to python by running
    export CUDA_VISIBLE_DEVICES=""
prior to running main.py with [processing_mode] set to 'm'
"""

import sys
import os
sys.path.append('./gym_dagsched/data_generation/tpch/')
from time import time
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
from copy import deepcopy

import numpy as np
import torch
from torch.multiprocessing import Process, SimpleQueue, set_start_method
import torch.distributed as dist
from pympler import tracker

from .reinforce_base import *
from ..envs.dagsched_env import DagSchedEnv
from ..utils.metrics import avg_job_duration
from ..utils.device import device



def send_ep_stats(out_q, loss, env):
    '''sends statistics back to main process for tensorboard 
    logging
    '''
    out_q.put((
        loss,
        avg_job_duration(env),
        env.n_completed_jobs))



def update_policy(policy, optim, loss):
    '''updates the policy by summing gradients from each
    of the subprocesses
    '''
    optim.zero_grad()
    loss.backward()
    for param in policy.parameters():
        dist.all_reduce(param.grad)
    optim.step()



def run_episodes(rank, n_ep_per_seq, policy, discount, in_q, out_q):
    '''subprocess function which runs episodes and trains the model 
    by communicating with the parent process'''

    if rank == 0:
        tr = tracker.SummaryTracker()

    # set up local model and optimizer
    policy = deepcopy(policy).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=.008)

    # IMPORTANT! ensures that the different child processes
    # don't all generate the same random numbers. Otherwise,
    # each process would produce an identical episode.
    torch.manual_seed(rank)
    np.random.seed(rank)

    # set up torch.distributed for IPC
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=n_ep_per_seq)

    # this subprocess's copy of the environment
    env = DagSchedEnv()

    while data := in_q.get():
        

        # receive episode data from parent process
        initial_timeline, workers, ep_len, entropy_weight = data

        n_remaining_steps = ep_len
        n_steps_per_window = 500
        last_obs = None
        last_reward = None
        done = False
        total_loss = 0.

        while n_remaining_steps > 0 and not done:
            # if rank == 0:
            #     print('next iter')
            #     tr.print_diff()

            n_steps = min(n_steps_per_window, n_remaining_steps)

            last_obs, last_reward, done, action_lgprobs, returns, entropies = \
                run_episode(
                    rank,
                    env,
                    initial_timeline,
                    workers,
                    n_steps,
                    policy,
                    discount,
                    last_obs,
                    last_reward,
                    done
                )

            # if rank == 0:
            #     tr.print_diff()
            #     print(flush=True)

            initial_timeline = None
            workers = None

            n_remaining_steps -= n_steps

            # compute advantages
            baselines = returns.clone()
            dist.all_reduce(baselines)
            baselines /= n_ep_per_seq
            advantages = returns - baselines

            policy_loss = -action_lgprobs @ advantages
            entropy_loss = entropy_weight * entropies.sum()
            loss = (policy_loss + entropy_loss) / ep_len

            update_policy(policy, optim, loss)

            total_loss += loss.item()

            done_t = torch.tensor(done, dtype=torch.int8)
            dist.all_reduce(done_t, op=dist.ReduceOp.BOR)
            done = bool(done_t.item())

            if rank == 0 and done == True:
                print('done, ', ep_len - n_remaining_steps)

        

        send_ep_stats(out_q, total_loss, env)



def launch_subprocesses(n_ep_per_seq, discount, policy):
    procs = []
    in_qs = []
    out_qs = []
    for rank in range(n_ep_per_seq):
        in_q = SimpleQueue()
        in_qs += [in_q]
        out_q = SimpleQueue()
        out_qs += [out_q]
        proc = Process(
            target=run_episodes, 
            args=(rank, n_ep_per_seq, policy, discount, in_q, out_q))
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
    '''trains the model on multiple different job arrival sequences, where
    `n_ep_per_seq` episodes are repeated on each sequence in parallel
    using multiprocessing'''

    # policy.to(device)
    # optim = torch.optim.SGD(policy.parameters(), lr=.005)

    procs, in_qs, out_qs = \
        launch_subprocesses(n_ep_per_seq, discount, policy)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    for epoch in range(n_sequences):
        # sample the length of the current episode
        ep_len = np.random.geometric(1/mean_ep_len)
        ep_len = max(ep_len, min_ep_len)
        # ep_len = min(ep_len, 4500)
        # ep_len = 250

        print(f'beginning training on sequence {epoch+1} with ep_len = {ep_len}', flush=True)

        # sample a job arrival sequence and worker types
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=2000.)
        workers = datagen.workers(n_workers=n_workers)

        # optim.zero_grad()

        # send episode data to each of the subprocesses, 
        # which starts the episodes
        for in_q in in_qs:
            data = initial_timeline, workers, ep_len, entropy_weight
            in_q.put(data)

        # wait for each of the subprocesses to finish parameter updates
        losses = np.empty(n_ep_per_seq)
        avg_job_durations = np.empty(n_ep_per_seq)
        n_completed_jobs_list = np.empty(n_ep_per_seq)
        for j,out_q in enumerate(out_qs):
            loss, avg_job_duration, n_completed_jobs = out_q.get()
            losses[j] = loss
            avg_job_durations[j] = avg_job_duration
            n_completed_jobs_list[j] = n_completed_jobs

        # optim.step()

        print(n_completed_jobs_list, n_completed_jobs_list.mean())

        write_tensorboard(
            writer, 
            epoch, 
            ep_len, 
            losses.sum(), 
            avg_job_durations.mean(), 
            n_completed_jobs_list.mean()
        )

        # increase the mean episode length
        mean_ep_len += ep_len_growth

        # decrease the entropy weight
        entropy_weight = max(
            entropy_weight - entropy_weight_decay, 
            entropy_weight_min)

        if (epoch+1) % 10 == 0:
            torch.save(policy.state_dict(), 'policy.pt')

    terminate_subprocesses(in_qs, procs)

