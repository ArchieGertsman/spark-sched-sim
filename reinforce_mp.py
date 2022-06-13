from re import L
import sys
from typing import List
sys.path.append('./gym_dagsched/data_generation/tpch/')
from dataclasses import dataclass
import time
from copy import deepcopy as dcp

import numpy as np
import torch
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from torch.multiprocessing import Process, Queue

from gym_dagsched.envs.dagsched_env import DagSchedEnv
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.data_generation.random_datagen import RandomDataGen
from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
from gym_dagsched.utils.metrics import avg_job_duration



@dataclass
class Trajectory:
    '''stores the trajectory of one MDP episode'''

    action_lgprobs: List

    returns: List




def find_op(jobs, op_idx):
    '''returns an Operation object corresponding to
    the `op_idx`th operation in the system'''
    i = 0
    op = None
    for job in jobs:
        if op_idx < i + len(job.ops):
            op = job.ops[op_idx - i]
            break
        else:
            i += len(job.ops)
    assert op is not None
    return op



def sample_action(jobs, ops_probs, prlvl_probs):
    '''given probabilities for selecting the next operation
    and the parallelism level for that operation's job (returned
    by the neural network), constructs categorical distributions 
    from those probabilities and then returns a randomly sampled 
    operation `next_op` (with index `next_op_idx`) and parallelism 
    level `prlvl` from those distributions.
    '''
    c = torch.distributions.Categorical(probs=ops_probs)
    next_op_idx = c.sample()
    next_op_idx_lgp = c.log_prob(next_op_idx)
    next_op = find_op(jobs, next_op_idx)

    c = torch.distributions.Categorical(probs=prlvl_probs[next_op.job_id])        
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
    return y[::-1]



def episode_runner(in_q, out_q):
    '''runs one MDP episode for `ep_length` iterations given 
    a job arrival sequence stored in `initial_timeline` and a 
    set of workers, and returns the history of actions, 
    observations, and rewards throughout the episode, each
    of length `ep_length`.
    '''

    while data := in_q.get():
        base_env, base_optim, policy, initial_timeline, workers, ep_length, discount = data
        if base_env is not None:
            env = base_env
        if base_optim is not None:
            optim = base_optim

        print('received data')

        env.reset(initial_timeline, workers)

        action_lgprobs = []
        rewards = []

        done = False
        obs = None

        while len(action_lgprobs) < ep_length and not done:

            if obs is None:
                next_op, prlvl = None, 0
            else:
                dag_batch, op_msk, prlvl_msk = obs
                ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)

                next_op, prlvl, action_lgprob = \
                    sample_action(env.jobs, ops_probs, prlvl_probs)

                action_lgprobs += [action_lgprob]
                rewards += [reward]

            obs, reward, done = env.step(next_op, prlvl)
        
        returns = compute_returns(rewards, discount)
        traj = Trajectory(action_lgprobs, returns)
        print('sending traj')
        out_q.put(traj)

        baselines = in_q.get()
        advantages = returns - baselines
        print('received baselines')

        optim.zero_grad()
        loss = -action_lgprobs @ advantages
        loss.backward()
        optim.step()

        out_q.put(None)








class ReinforceTrainer:
    '''Trains the Decima model'''


    def __init__(self, 
        env, 
        datagen, 
        policy, 
        optim, 
        n_sequences,
        n_ep_per_seq,
        discount,
        n_workers,
        initial_mean_ep_len,
        delta_ep_len
    ):
        # instance of dagsched_env
        self.env = env

        # data generator which provides that data
        # used for training
        self.datagen = datagen
        
        # neural network to be trained
        self.policy = policy
        
        # torch optimizer
        self.optim = optim
        
        # number of job arrival sequences to train on
        self.n_sequences = n_sequences
        
        # number of times to train on a fixed sequence
        self.n_ep_per_seq = n_ep_per_seq
        
        # discount factor for computing total rewards
        self.discount = discount
        
        # number of workers to include in the simulation
        self.n_workers = n_workers
        
        # (geometric distribution) mean number of environment 
        # steps in an episode; this quantity gradually increases
        # as a form of curriculum learning
        self.mean_ep_len = initial_mean_ep_len
        
        # amount by which the mean episode length increases
        # every time training is complete on some sequence
        self.delta_ep_len = delta_ep_len

        # minimum number of steps in an episode
        self.min_ep_len = 5


    
    def train(self):
        '''train the model on multiple different job arrival sequences'''


        procs = []
        in_qs = []
        out_qs = []
        for _ in range(self.n_ep_per_seq):
            in_q = Queue(1)
            in_qs += [in_q]
            out_q = Queue(1)
            out_qs += [out_q]
            proc = Process(target=episode_runner, args=(in_q,out_q))
            proc.start()



        # y = []

        for i in range(self.n_sequences):
            print(f'beginning training on sequence {i+1}')

            # ep_len = np.random.geometric(1/self.mean_ep_len)
            # ep_len = max(ep_len, self.min_ep_len)
            ep_len = self.mean_ep_len

            # sample a job arrival sequence and worker types
            initial_timeline = datagen.initial_timeline(
                n_job_arrivals=50, n_init_jobs=0, mjit=2000.)
            workers = datagen.workers(n_workers=self.n_workers)


            for in_q in in_qs:
                base_env = None if i>0 else dcp(self.env)
                base_optim = None if i>0 else dcp(self.optim)
                data = base_env, base_optim, self.policy, initial_timeline, workers, ep_len, self.discount
                in_q.put(data)

            trajectories = []
            for out_q in out_qs:
                traj = out_q.get()
                trajectories += [traj]
                print('received traj')

            returns_mat = torch.tensor([traj.returns for traj in trajectories])
            baselines = returns_mat.mean(axis=0)

            for in_q in in_qs:
                in_q.put(baselines)


            for j,out_q in enumerate(out_qs):
                out_q.get()
                print(f'ep {j+1} complete')

            # run multiple episodes on this fixed sequence

            # trajectories = []
            # avg_job_durations = np.zeros(self.n_ep_per_seq)
            # n_completed_jobs = np.zeros(self.n_ep_per_seq)
            # for j in range(self.n_ep_per_seq):
            #     traj = self._run_episode(ep_len, initial_timeline, workers)
            #     trajectories += [traj]
            #     n_completed_jobs[j] = self.env.n_completed_jobs
            #     avg_job_durations[j] = avg_job_duration(self.env)
            #     print(f'episode {j+1} complete:', n_completed_jobs[j], avg_job_durations[j])

            # y += [avg_job_durations.mean() / n_completed_jobs.mean()]

            self.mean_ep_len += self.delta_ep_len


        for in_q in in_qs:
            in_q.put(None)

        for proc in procs:
            proc.join()

        # y = np.array(y)
        # print(y)
        # plt.plot(np.arange(len(y)), y)
        # # plt.show()
        # plt.savefig('bruh.png')



if __name__ == '__main__':
    
    mean_ep_length = 20

    datagen = RandomDataGen(
        max_ops=8, # 20
        max_tasks=4, # 200
        mean_task_duration=2000.,
        n_worker_types=1)
    # datagen = TPCHDataGen()

    env = DagSchedEnv()

    n_workers = 5

    policy = ActorNetwork(5, 8, n_workers)
    policy.share_memory()
    # policy.cuda()

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    trainer = ReinforceTrainer(
        env, 
        datagen, 
        policy, 
        optim, 
        n_sequences=10,
        n_ep_per_seq=4,
        discount=1.,
        n_workers=n_workers,
        initial_mean_ep_len=5000,
        delta_ep_len=0)

    trainer.train()




