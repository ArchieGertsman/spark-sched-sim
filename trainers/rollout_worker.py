import sys
from multiprocessing.connection import Connection

import gymnasium as gym
from gymnasium.core import ObsType, ActType
import torch

from spark_sched_sim.wrappers.decima_wrappers import DecimaActWrapper, DecimaObsWrapper
from spark_sched_sim.schedulers import DecimaScheduler
from spark_sched_sim import metrics
from .utils import Profiler, HiddenPrints



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


    def __len__(self):
        return len(self.obsns)



## rollout workers

def setup_rollout_worker(rank, env_kwargs, model_kwargs, log_dir):
    # log each of the processes to separate files
    sys.stdout = open(f'{log_dir}/{rank}.out', 'a')

    # torch multiprocessing is very slow without this
    torch.set_num_threads(1)

    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    env = gym.make(env_id, **env_kwargs)
    env = DecimaObsWrapper(DecimaActWrapper(env))
    agent = DecimaScheduler(env_kwargs['num_executors'], **model_kwargs)

    # IMPORTANT! Each worker needs to produce unique 
    # rollouts, which are determined by the rng seed
    torch.manual_seed(rank)

    return env, agent



def rollout_worker(rank, conn, env_kwargs, model_kwargs, log_dir):
    env, agent = setup_rollout_worker(rank, env_kwargs, model_kwargs, log_dir)
    
    while data := conn.recv():
        actor_sd, env_seed, env_options = data

        # load updated model parameters
        agent.actor.load_state_dict(actor_sd)
        obs, _ = env.reset(seed=env_seed, options=env_options)
        
        try:
            with Profiler(), HiddenPrints():
                rollout_buffer, obs = collect_rollout(env, agent, obs)
            avg_job_duration = metrics.avg_job_duration(env) * 1e-3
            num_completed_jobs = env.num_completed_jobs
            num_job_arrivals = env.num_completed_jobs + env.num_active_jobs
        except AssertionError as msg:
            print(msg, '\naborting rollout.', flush=True)
            rollout_buffer = avg_job_duration = num_completed_jobs = num_job_arrivals = None

        conn.send((
            rollout_buffer,
            avg_job_duration, 
            num_completed_jobs,
            num_job_arrivals
        ))

        

def collect_rollout(env, agent, obs):
    rollout_buffer = RolloutBuffer()
    
    wall_time = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, lgprob = agent(obs)

        new_obs, reward, terminated, truncated, info = env.step(action)
        next_wall_time = info['wall_time']

        rollout_buffer.add(obs, wall_time, list(action.values()), lgprob, reward)

        obs = new_obs
        wall_time = next_wall_time

    rollout_buffer.wall_times += [wall_time]

    return rollout_buffer, obs