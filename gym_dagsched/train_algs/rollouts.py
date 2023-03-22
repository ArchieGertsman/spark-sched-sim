from typing import Iterable, List, Tuple, Any
import sys
from multiprocessing.connection import Connection
from copy import deepcopy

import gymnasium as gym
from gymnasium.core import ObsType, ActType
from gymnasium.wrappers.normalize import NormalizeReward
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np

from ..wrappers.decima_wrappers import DecimaActWrapper, DecimaObsWrapper
from ..wrappers.avg_reward_wrapper import AverageReward
from ..agents.ac_decima_agent import DecimaAgent
from ..utils.graph import ObsBatch, collate_obsns
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.hidden_prints import HiddenPrints
from ..utils import metrics



class RolloutBuffer:
    def __init__(self):
        self.obsns: list[ObsType] = []
        self.wall_times: list[float] = []
        self.actions: list[ActType] = []
        self.lgprobs: list[float] = []
        self.rewards: list[float] = []


    def add(self, obs, wall_time, action, lgprob, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]
        self.lgprobs += [lgprob]
        self.last_obs = None
        self.values = None
        self.sd = None
        self.returns = None
        self.value_loss = None


    def __len__(self):
        return len(self.obsns)



## rollout workers

def setup_rollout_worker(rank: int, env_kwargs: dict, log_dir: str) -> None:
    # log each of the processes to separate files
    sys.stdout = open(f'{log_dir}/{rank}.out', 'a')

    # torch multiprocessing is very slow without this
    torch.set_num_threads(1)

    # IMPORTANT! Each worker needs to produce unique 
    # rollouts, which are determined by the rng seed
    torch.manual_seed(rank)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(env)
    env = DecimaObsWrapper(env)
    # env = AverageReward(env)

    agent = DecimaAgent(env_kwargs['num_workers'])
    agent.build(device=device)

    return env, agent



def rollout_worker(
    rank: int, 
    num_envs: int,
    conn: Connection, 
    env_kwargs: dict,
    log_dir: str
) -> None:
    '''collects rollouts and trains the model by communicating 
    with the main process and other workers
    '''
    env, agent = setup_rollout_worker(rank, env_kwargs, log_dir)
    
    while data := conn.recv():
        actor_sd, critic_sd, env_seed, env_options = data

        # load updated model parameters
        agent.actor.load_state_dict(actor_sd)
        agent.critic.load_state_dict(critic_sd)
        
        with Profiler(), HiddenPrints():
            rollout_buffer = \
                collect_rollout(
                    env, 
                    agent, 
                    seed=env_seed, 
                    options=env_options
                )
            

        # returns = _compute_returns(
        #     agent, 
        #     rollout_buffer.last_obs, 
        #     np.array(rollout_buffer.rewards)
        # )
        # rollout_buffer.returns = returns

        # sd_old = deepcopy(critic_sd)

        # value_losses = []

        # values_odd, sd1, losses = adapt_and_predict(
        #     agent,
        #     rollout_buffer.obsns[::2],
        #     returns[::2],
        #     rollout_buffer.obsns[1::2]
        # )
        # value_losses += losses

        # agent.critic.load_state_dict(sd_old)

        # values_even, sd2, losses = adapt_and_predict(
        #     agent,
        #     rollout_buffer.obsns[1::2],
        #     returns[1::2],
        #     rollout_buffer.obsns[::2]
        # )
        # value_losses += losses

        # values = torch.zeros_like(returns)
        # values[::2] = values_even
        # values[1::2] = values_odd
        # rollout_buffer.values = values

        # sd_avg = {name:
        #     (sd1[name] + sd2[name]) / 2
        #     for name in sd_old
        # }

        # rollout_buffer.sd = sd_avg
        # rollout_buffer.value_loss = np.mean(value_losses)

        # send rollout buffer and stats to center
        avg_job_duration = metrics.avg_job_duration(env) * 1e-3
        num_job_arrivals = env.num_completed_jobs + env.num_active_jobs
        conn.send((
            rollout_buffer, 
            avg_job_duration, 
            env.num_completed_jobs,
            num_job_arrivals
        ))

        

def collect_rollout(
    env, 
    agent: DecimaAgent, 
    seed: int, 
    options: dict
) -> RolloutBuffer:

    obs, info = env.reset(seed=seed, options=options)
    done = False

    rollout_buffer = RolloutBuffer()

    i = 0
    while not done:
        i += 1
        action, lgprob = agent(obs)

        new_obs, reward, terminated, truncated, info = env.step(action)

        done = (terminated or truncated)

        rollout_buffer.add(obs, info['wall_time'], action, lgprob, reward)

        obs = new_obs

    rollout_buffer.last_obs = obs

    return rollout_buffer


def adapt_and_predict(agent, obsns_adpt, returns_adpt, obsns_pred):
    obsns_adpt = {i: obs for i, obs in enumerate(obsns_adpt)}
    dataloader_adpt = DataLoader(
        ValueDataset(obsns_adpt, returns_adpt),
        batch_size=(len(obsns_adpt) // 8 + 1),
        shuffle=True,
        collate_fn=ValueDataset.collate
    )

    print('adapt')
    value_losses = []
    for _ in range(5):
        for obsns, value_targets in dataloader_adpt:
            values = agent.predict_values(obsns).flatten()
            print(values[0].item(), value_targets[0].item(), flush=True)
            value_loss = F.mse_loss(values, value_targets)
            # value_losses += [value_loss.item()]
            # agent.update_parameters(value_loss)
            agent.inner_opt.zero_grad()
            value_loss.backward()
            agent.inner_opt.step()
    value_losses = [value_loss.item()]

    with torch.no_grad():
        values_pred = agent.predict_values(collate_obsns(obsns_pred)).flatten()

    sd = deepcopy(agent.critic.state_dict())

    return values_pred, sd, value_losses

    


def _compute_returns(agent, last_obs, rewards):
    returns_list = []
    # with torch.no_grad():
    #     last_value = agent.predict_values(collate_obsns([last_obs])).item()
    returns = np.zeros_like(rewards)
    returns[-1] = rewards[-1] # + last_value # value bootstrap
    for i in reversed(range(len(rewards)-1)):
        returns[i] = rewards[i] + returns[i+1]
    returns_list += [returns]

    returns = torch.from_numpy(np.hstack(returns_list)).float()

    return returns




class RolloutDataset(Dataset):
    '''torch dataset created from rollout data, used for minibatching'''
    def __init__(
        self, 
        obsns: dict[int, ObsType],
        actions: dict[int, ActType],
        advantages: Tensor,
        old_lgprobs: Tensor
    ):
        '''
        Args:
            obsns: dict which maps indicies to observations for
                constant access time
            actions: dict which maps indices to actions for
                constant access time
            advantages: advantages tensor
            old_lgprobs: tensor of action log-probabilities prior
                to updating the model parameters
        '''
        super().__init__()
        self.obsns = obsns
        self.actions = actions
        self.advantages = advantages
        self.old_lgprobs = old_lgprobs


    def __getitem__(self, idx: int):
        return self.obsns[idx], \
               self.actions[idx], \
               self.advantages[idx], \
               self.old_lgprobs[idx]


    def __len__(self):
        return len(self.advantages)


    @classmethod
    def collate(cls, batch):
        '''
        Args:
            batch: list of (obs, action, advantage, old_lgprob) 
                tuples to be collated into a minibatch
        Returns:
            tuple of collated observations, actions, advantages, 
                and old log-probabilities
        '''
        obsns, actions, advantages, old_lgprobs = zip(*batch)
        obsns = collate_obsns(obsns)
        actions = torch.stack(actions)
        advantages = torch.stack(advantages)
        old_lgprobs = torch.stack(old_lgprobs)
        return obsns, actions, advantages, old_lgprobs
    



class ValueDataset(Dataset):
    def __init__(
        self, 
        obsns: dict[int, ObsType],
        value_targets: Tensor
    ):
        super().__init__()
        self.obsns = obsns
        self.value_targets = value_targets


    def __getitem__(self, idx: int):
        return self.obsns[idx], self.value_targets[idx]


    def __len__(self):
        return len(self.value_targets)


    @classmethod
    def collate(cls, batch):
        obsns, value_targets = zip(*batch)
        obsns = collate_obsns(obsns)
        value_targets = torch.stack(value_targets)
        return obsns, value_targets