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
            avg_job_duration, n_completed_jobs = \
                _get_prev_episode_stats(env, first_episode)
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
            conn.send((avg_job_duration, n_completed_jobs))
            
        elif header == 'step':
            if shared_obs is None:
                raise Exception(f'proc {rank} is trying to step before resetting')

            # parse data
            try:
                (job_id, op_id), prlvl = data
            except:
                _raise_invalid_data(rank)

            # step
            _env_step(env, job_id, op_id, prlvl)
            
            # notify main process
            conn.send(None)

        else:
            raise Exception(f'proc {rank} received invalid data')



def _raise_invalid_data(rank):
    raise Exception(f'proc {rank} received invalid data')



def _get_prev_episode_stats(env, first_episode):
    if not first_episode:
        avg_job_duration = metrics.avg_job_duration(env)
        n_completed_jobs = env.n_completed_jobs
    else:
        avg_job_duration, n_completed_jobs = None, None

    return avg_job_duration, n_completed_jobs



def _env_reset(env, datagen, n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs):
    initial_timeline = datagen.initial_timeline(
        n_job_arrivals, n_init_jobs, mjit)
    workers = datagen.workers(n_workers)

    env.reset(initial_timeline, workers, shared_obs)



def _env_step(env, job_id, op_id, prlvl):
    env.step(job_id, op_id, prlvl)



