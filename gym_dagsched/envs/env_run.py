import torch
import numpy as np

from .dagsched_env import DagSchedEnv
from ..utils import metrics



def env_run(rank, datagen, conn):
    torch.manual_seed(rank)
    np.random.seed(rank)

    env = DagSchedEnv(rank)
    shared_obs = None
    first_episode = True

    while header_data := conn.recv():
        header, data = header_data

        if header == 'reset':
            prev_episode_stats = _get_prev_episode_stats(env) \
                if not first_episode else None

            first_episode = False

            # parse data
            try:
                (n_job_arrivals, 
                n_init_jobs,
                mjit, 
                n_workers, 
                shared_obs) = data
            except:
                _raise_invalid_data(rank)

            # reset
            _env_reset(
                env, 
                datagen, 
                n_job_arrivals, 
                n_init_jobs, 
                mjit,
                n_workers, 
                shared_obs)

            # send back to main process
            conn.send(prev_episode_stats)
            
        elif header == 'step':
            if shared_obs is None:
                raise Exception(f'proc {rank} is trying to step before resetting')

            # parse data
            action = data

            if action is not None:
                try:
                    (_, _), _ = data
                except:
                    _raise_invalid_data(rank)

            # step
            env.step(action)
            
            # notify main process
            conn.send(None)

        else:
            raise Exception(f'proc {rank} received invalid data')



def _raise_invalid_data(rank):
    raise Exception(f'proc {rank} received invalid data')



def _get_prev_episode_stats(env):
    avg_job_duration = metrics.avg_job_duration(env)
    n_completed_jobs = env.n_completed_jobs
    return avg_job_duration, n_completed_jobs



def _env_reset(env, datagen, n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs):
    initial_timeline = datagen.initial_timeline(
        n_job_arrivals, n_init_jobs, mjit)
    workers = datagen.workers(n_workers)

    env.reset(initial_timeline, workers, shared_obs)



