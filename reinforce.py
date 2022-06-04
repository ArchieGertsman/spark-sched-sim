import sys
from typing import List
sys.path.append('./gym_dagsched/data_generation/tpch/')

import numpy as np
import torch
from scipy.signal import lfilter

from gym_dagsched.envs.dagsched_env import DagSchedEnv
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.data_generation.random_datagen import RandomDataGen



class Trajectory:
    '''stores the trajectory of one MDP episode'''

    def __init__(self):
        self.action_lgprobs = []
        self.rewards = []


    def append(self, action_lgprob, reward):
        self.action_lgprobs += [action_lgprob]
        self.rewards += [reward]



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



    def _find_op(self, op_idx):
        '''returns an Operation object corresponding to
        the `op_idx`th operation in the system'''
        i = 0
        op = None
        for job in self.env.jobs:
            if op_idx < i + len(job.ops):
                op = job.ops[op_idx - i]
                break
            else:
                i += len(job.ops)
        assert op is not None
        return op



    def sample_action(self, ops_probs, prlvl_probs):
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
        next_op = self._find_op(next_op_idx)

        c = torch.distributions.Categorical(probs=prlvl_probs[next_op.job_id])        
        prlvl = c.sample()
        prlvl_lgp = c.log_prob(prlvl)

        action_lgprob = next_op_idx_lgp + prlvl_lgp

        return next_op, prlvl.item(), action_lgprob.unsqueeze(-1)



    def _run_episode(self, ep_length, initial_timeline, workers):
        '''runs one MDP episode for `ep_length` iterations given 
        a job arrival sequence stored in `initial_timeline` and a 
        set of workers, and returns the history of actions, 
        observations, and rewards throughout the episode, each
        of length `ep_length`.
        '''
        self.env.reset(initial_timeline, workers)

        traj = Trajectory()

        done = False
        obs = None

        while len(traj.action_lgprobs) < ep_length and not done:
            if obs is None:
                next_op, prlvl = None, 0
            else:
                dag_batch, op_msk, prlvl_msk = obs
                ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)
                next_op, prlvl, action_lgprob = \
                    self.sample_action(ops_probs, prlvl_probs)

                traj.append(action_lgprob, reward)

            obs, reward, done = self.env.step(next_op, prlvl)
        
        return traj



    def _run_episodes(self, initial_timeline, workers, ep_length):
        '''runs multiple episodes on the same sequence and returns
        a list containing each of their (action,observation,reward) 
        trajectories
        '''

        def compute_cum_discounted_rewards(rewards, discount):
            ''' returs array `y` where `y[i] = rewards[i] + discount * y[i+1]`
            credit: https://stackoverflow.com/a/47971187/5361456
            '''
            r = rewards[::-1]
            a = [1, -discount]
            b = [1]
            y = lfilter(b, a, x=r)
            return y[::-1]


        trajectories = []

        for _ in range(self.n_ep_per_seq):
            traj = self._run_episode(ep_length, initial_timeline, workers)
            rewards = np.array(traj.rewards)
            cum_disc_rewards = \
                compute_cum_discounted_rewards(rewards, self.discount)
            traj.rewards = cum_disc_rewards
            trajectories += [traj]

        return trajectories



    def _train_on_trajectories(self, trajectories):
        '''given a list of trajectories from multiple MDP episodes
        that were repeated on a fixed job arrival sequence, update the model 
        parameters using the REINFORCE algorithm as in the Decima paper.
        '''
        cum_disc_rewards_mat = torch.tensor([traj.rewards for traj in trajectories])

        action_lgprobs_mat = torch.stack([
            torch.cat(traj.action_lgprobs) 
            for traj in trajectories
        ])

        baselines = cum_disc_rewards_mat.mean(axis=0)
        advantages_mat = cum_disc_rewards_mat - baselines

        optim.zero_grad()

        loss_mat = -action_lgprobs_mat * advantages_mat
        loss = loss_mat.sum(axis=1).mean()
        loss.backward()

        optim.step()


    
    def train(self):
        '''train the model on multiple different job arrival sequences'''
        for _ in range(self.n_sequences):
            ep_len = np.random.geometric(1/self.mean_ep_len)
            ep_len = max(ep_len, self.min_ep_len)

            # sample a job arrival sequence and worker types
            initial_timeline = datagen.initial_timeline(
                n_job_arrivals=20, n_init_jobs=0, mjit=1000.)
            workers = datagen.workers(n_workers=self.n_workers)

            # run multiple episodes on this fixed sequence
            trajectories = self._run_episodes(initial_timeline, workers, ep_len)

            self._train_on_trajectories(trajectories)

            self.mean_ep_len += self.delta_ep_len



if __name__ == '__main__':
    
    mean_ep_length = 20

    datagen = RandomDataGen(
        max_ops=20,
        max_tasks=4, # 200 in Decima
        mean_task_duration=2000.,
        n_worker_types=1)

    env = DagSchedEnv()

    n_workers = 5

    policy = ActorNetwork(5, 8, n_workers)

    optim = torch.optim.Adam(policy.parameters(), lr=.005)

    trainer = ReinforceTrainer(
        env, 
        datagen, 
        policy, 
        optim, 
        n_sequences=10,
        n_ep_per_seq=4,
        discount=.99,
        n_workers=n_workers,
        initial_mean_ep_len=20,
        delta_ep_len=3)

    trainer.train()