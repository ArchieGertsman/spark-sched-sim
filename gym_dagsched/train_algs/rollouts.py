from typing import Iterable, List, Tuple, Any
import sys
from multiprocessing.connection import Connection

from gymnasium.core import ObsType, ActType
import gymnasium as gym
from torch.utils.data import Dataset
from torch import Tensor
import torch

from ..wrappers.decima_wrappers import DecimaActWrapper, DecimaObsWrapper
from ..agents.decima_agent import DecimaAgent
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
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.lgprobs: list[float] = []


    def add(
        self, 
        obs: ObsType, 
        wall_time: float, 
        action: ActType, 
        reward: float, 
        value: float, 
        lgprob: float
    ) -> None:
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.lgprobs += [lgprob]


    def __len__(self):
        return len(self.obsns)



## rollout workers

def setup_rollout_worker(rank: int, env_kwargs: dict) -> None:
    # log each of the processes to separate files
    sys.stdout = open(f'ignore/log/proc/{rank}.out', 'a')

    # torch multiprocessing is very slow without this
    torch.set_num_threads(1)

    # IMPORTANT! Each worker needs to produce unique 
    # rollouts, which are determined by the rng seed
    torch.manual_seed(rank)

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    agent = DecimaAgent(env_kwargs['num_workers'])
    agent.build(device=device)

    return env, agent



def rollout_worker(
    rank: int, 
    world_size: int,
    conn: Connection, 
    env_kwargs: dict
) -> None:
    '''collects rollouts and trains the model by communicating 
    with the main process and other workers
    '''
    env, agent = setup_rollout_worker(rank, env_kwargs)
    
    iteration = 0
    while data := conn.recv():
        state_dict, env_options = data

        # load updated model parameters
        agent.load_state_dict(state_dict)
        
        with Profiler(), HiddenPrints():
            rollout_buffer = \
                collect_rollout(
                    env, 
                    agent, 
                    seed=world_size*iteration + rank, 
                    options=env_options
                )

        # send rollout buffer and stats to center
        avg_job_duration = metrics.avg_job_duration(env) * 1e-3
        num_job_arrivals = env.num_completed_jobs + env.num_active_jobs
        conn.send((
            rollout_buffer, 
            avg_job_duration, 
            env.num_completed_jobs,
            num_job_arrivals
        ))

        iteration += 1

        

def collect_rollout(
    env, 
    agent: DecimaAgent, 
    seed: int, 
    options: dict
) -> RolloutBuffer:

    obs, info = env.reset(seed=seed, options=options)
    done = False

    rollout_buffer = RolloutBuffer()

    while not done:
        action, value, lgprob = agent(obs)

        new_obs, reward, terminated, truncated, info = env.step(action)

        done = (terminated or truncated)

        rollout_buffer.add(
            obs, 
            info['wall_time'], 
            action, 
            reward, 
            value, 
            lgprob
        )

        obs = new_obs

    return rollout_buffer




class RolloutDataset(Dataset):
    '''torch dataset created from rollout data, used for minibatching'''
    def __init__(
        self, 
        obsns: dict[int, ObsType],
        actions: dict[int, ActType],
        returns: Tensor,
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
        self.returns = returns
        self.advantages = advantages
        self.old_lgprobs = old_lgprobs


    def __getitem__(self, idx: int) -> tuple[ObsType, ActType, Tensor, Tensor]:
        return self.obsns[idx], \
               self.actions[idx], \
               self.returns[idx], \
               self.advantages[idx], \
               self.old_lgprobs[idx]


    def __len__(self):
        return len(self.advantages)


    @classmethod
    def collate(
        cls, 
        batch: Iterable[tuple[ObsType, ActType, Tensor, Tensor, Tensor]]
    ) -> tuple[ObsBatch, Tensor, Tensor, Tensor]:
        '''
        Args:
            batch: list of (obs, action, advantage, old_lgprob) 
                tuples to be collated into a minibatch
        Returns:
            tuple of collated observations, actions, advantages, 
                and old log-probabilities
        '''
        obsns, actions, returns, advantages, old_lgprobs = zip(*batch)
        obsns = collate_obsns(obsns)
        actions = Tensor([[int(a) for a in act.values()] for act in actions])
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        old_lgprobs = torch.stack(old_lgprobs)
        return obsns, actions, returns, advantages, old_lgprobs