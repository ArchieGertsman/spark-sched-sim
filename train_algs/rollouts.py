import sys

import gymnasium as gym
import torch

from spark_sched_sim.wrappers.decima_wrappers import DecimaActWrapper, DecimaObsWrapper
from spark_sched_sim.schedulers import DecimaScheduler
from .utils.profiler import Profiler
from .utils.hidden_prints import HiddenPrints
from spark_sched_sim import metrics



class RolloutBuffer:
    def __init__(self):
        self.obsns = []
        self.wall_times = []
        self.actions = []
        self.lgprobs = []
        self.rewards = []
        self.resets = None


    def add(self, obs, wall_time, action, lgprob, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]
        self.lgprobs += [lgprob]


    def __len__(self):
        return len(self.obsns)



def setup_rollout_worker(rank, env_kwargs, log_dir):
    # log each of the processes to separate files
    sys.stdout = open(f'{log_dir}/{rank}.out', 'a')

    # torch multiprocessing is very slow without this line
    torch.set_num_threads(1)

    env_id = 'spark_sched_sim:SparkSchedSimEnv-v0'
    env = gym.make(env_id, **env_kwargs)
    env = DecimaObsWrapper(DecimaActWrapper(env))
    agent = DecimaScheduler(env_kwargs['num_executors'])
    agent.build(device='cpu')

    # IMPORTANT! Each worker needs to produce unique 
    # rollouts, which are determined by the rng seed
    torch.manual_seed(rank)

    return env, agent



def rollout_worker(rank, conn, env_kwargs, log_dir):
    env, agent = setup_rollout_worker(rank, env_kwargs, log_dir)
    
    next_obs = None

    while data := conn.recv():
        actor_sd, env_options = data

        if not next_obs:
            next_obs, _ = env.reset(options=env_options)
            next_wall_time = 0

        # load updated model parameters
        agent.actor.load_state_dict(actor_sd)
        
        try:
            with Profiler(), HiddenPrints():
                rollout_buffer, next_obs, next_wall_time = \
                    collect_rollout(env, env_options, agent, next_obs, next_wall_time)
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

        

def collect_rollout(env, env_options, agent, next_obs, next_wall_time, rollout_duration=1e6):
    rollout_buffer = RolloutBuffer()

    elapsed_time = 0
    step = 0
    resets = set()
    while elapsed_time < rollout_duration:
        obs, wall_time = next_obs, next_wall_time

        action, lgprob = agent.schedule(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_wall_time = info['wall_time']

        rollout_buffer.add(obs, elapsed_time, list(action.values()), lgprob, reward)
        
        # add the duration of the this step to the total
        elapsed_time += next_wall_time - wall_time

        if terminated or truncated:
            # either all jobs complete or time limit has been reached, so reset
            next_obs, _ = env.reset(options=env_options)
            next_wall_time = 0
            resets.add(step)

        step += 1

    rollout_buffer.wall_times += [elapsed_time]
    rollout_buffer.resets = resets

    return rollout_buffer, next_obs, next_wall_time