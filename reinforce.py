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
    actions = []
    obsns = []
    rewards = []


class ReinforceTrainer:

    # # number of job arrival sequences to train on
    # N_SEQUENCES = 1

    # # number of times to train on a fixed sequence
    # N_EP_PER_SEQ = 1

    # # discount factor for computing total rewards
    # DISCOUNT = .99

    # # how much to increase mean episode length every time
    # # training on a sequence is complete
    # DELTA_EP_LENGTH = 3


    # N_WORKERS = 5


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
        self.env = env
        self.datagen = datagen
        self.policy = policy
        self.optim = optim
        self.n_sequences = n_sequences
        self.n_ep_per_seq = n_ep_per_seq
        self.discount = discount
        self.n_workers = n_workers
        self.mean_ep_len = initial_mean_ep_len
        self.delta_ep_len = delta_ep_len



    def find_op(self, op_idx):
        '''returns an Operation object corresponding to
        the `op_idx`th operation in the system'''
        i = 0
        for job in self.env.jobs:
            if op_idx < i + len(job.ops):
                op = job.ops[op_idx - i]
                break
            else:
                i += len(job.ops)
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
        next_op_idx = c.sample().item()
        next_op = self.find_op(next_op_idx)
        c = torch.distributions.Categorical(probs=prlvl_probs[next_op.job_id])        
        prlvl = c.sample().item()

        return next_op_idx, next_op, prlvl



    def compute_action_log_probs(self, ops_probs, prlvl_probs, next_op_idx, prlvl):
        '''returns the log probabilities of the policy sampling an operation with 
        index `next_op_idx` and a parallelism level `prlvl`, given action
        probabilities `ops_probs` and `prlvl_probs`.
        '''
        c = torch.distributions.Categorical(probs=ops_probs)
        next_op_idx_lgp = c.log_prob(torch.tensor(next_op_idx))
        
        next_op = self.find_op(next_op_idx)
        c = torch.distributions.Categorical(probs=prlvl_probs[next_op.job_id])  
        prlvl_lgp = c.log_prob(torch.tensor(prlvl))

        return next_op_idx_lgp, prlvl_lgp



    def run_episode(self, ep_length, initial_timeline, workers):
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

        while len(traj.actions) < ep_length and not done:
            if obs is None:
                next_op, prlvl = None, 0
            else:
                dag_batch, op_msk, prlvl_msk = obs
                ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)
                next_op_idx, next_op, prlvl = self.sample_action(ops_probs, prlvl_probs)
                
                traj.actions += [(next_op_idx, prlvl)]
                traj.obsns += [obs]
                traj.rewards += [reward]

            obs, reward, done = self.env.step(next_op, prlvl)
        
        return traj



    @staticmethod
    def compute_total_discounted_rewards(rewards, discount):
        ''' returs array `y` where `y[i] = rewards[i] + discount * y[i+1]`
        credit: https://stackoverflow.com/a/47971187/5361456
        '''
        r = rewards[::-1]
        a = [1, -discount]
        b = [1]
        y = lfilter(b, a, x=r)
        return y[::-1]



    def run_episodes(self, initial_timeline, workers, ep_length):
        '''run multiple episodes on the same sequence, and
        records the (action,observation,reward) trajectories 
        of each episode. Each of the following list/array
        objects will have shape (N_EP_PER_SEQ, ep_length)
        '''
        trajectories = []

        for _ in range(self.n_ep_per_seq):
            traj = self.run_episode(ep_length, initial_timeline, workers)
            rewards = np.array(traj.rewards)
            total_rewards = \
                ReinforceTrainer.compute_total_discounted_rewards(rewards, self.discount)
            traj.rewards = total_rewards
            trajectories += [traj]

        return trajectories


    def train_on_trajectories(self, trajectories):
        '''given a list of trajectories from multiple MDP episodes,
        update the model parameters using the REINFORCE algorithm
        as in the Decima paper.
        '''
        rewards_matrix = np.array([traj.rewards for traj in trajectories])
        baselines = rewards_matrix.mean(axis=0)

        ep_len = baselines.size
        for k in range(ep_len):
            baseline = baselines[k]
            for traj in trajectories:
                action, obs, reward = traj.actions[k], traj.obsns[k], traj.rewards[k]
                dag_batch, op_msk, prlvl_msk = obs
                next_op_idx, prlvl = action

                ops_probs, prlvl_probs = policy(dag_batch, op_msk, prlvl_msk)

                next_op_idx_lgp, prlvl_lgp = \
                    self.compute_action_log_probs(ops_probs, prlvl_probs, next_op_idx, prlvl)
                    
                loss = -(next_op_idx_lgp + prlvl_lgp) * (reward - baseline)

                optim.zero_grad()
                loss.backward()
                optim.step()


    
    def train(self):
        '''train the model on multiple different job arrival sequences'''
        for _ in range(self.n_sequences):
            ep_len = np.random.geometric(1/self.mean_ep_len)

            # sample a job arrival sequence and worker types
            initial_timeline = datagen.initial_timeline(
                n_job_arrivals=20, n_init_jobs=0, mjit=1000.)
            workers = datagen.workers(n_workers=self.n_workers)

            # run multiple episodes on this fixed sequence
            trajectories = self.run_episodes(initial_timeline, workers, ep_len)

            self.train_on_trajectories(trajectories)

            self.mean_ep_len += self.delta_ep_len



if __name__ == '__main__':
    # (geometric distribution) mean number of environment 
    # steps in an episode; this quantity gradually increases
    # as a form of curriculum learning
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
        n_sequences=1,
        n_ep_per_seq=1,
        discount=.99,
        n_workers=n_workers,
        initial_mean_ep_len=20,
        delta_ep_len=3)

    trainer.train()