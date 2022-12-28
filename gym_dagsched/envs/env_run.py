import sys
import os
from time import time

import torch
import numpy as np

from ..data_generation.tpch_datagen import TPCHDataGen
from .dagsched_env import DagSchedEnv
from ..utils import metrics



def env_run(rank, datagen_state, conn):
    torch.manual_seed(rank)
    np.random.seed(rank)
    sys.stdout = open(f'log/proc/{rank}.out', 'a')

    datagen = TPCHDataGen(datagen_state)
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
            obs = _env_reset(
                env, 
                datagen, 
                n_job_arrivals, 
                n_init_jobs, 
                mjit,
                n_workers)

            assert obs
            _update_shared_obs(shared_obs, obs)

            # send back to main process
            conn.send(prev_episode_stats)
            
        elif header == 'step':
            if shared_obs is None:
                raise Exception(
                    f'proc {rank} is trying to step before resetting')

            # parse data
            action = data

            if action is not None:
                try:
                    (_, _), _ = data
                except:
                    _raise_invalid_data(rank)

            # step
            obs, reward, done = env.step(action)
            _update_shared_obs(shared_obs, obs, reward, done)

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



def _env_reset(
    env, 
    datagen, 
    n_job_arrivals, 
    n_init_jobs, 
    mjit, 
    n_workers
):
    initial_timeline = datagen.initial_timeline(
        n_job_arrivals, n_init_jobs, mjit)
    workers = datagen.workers(n_workers)

    obs = env.reset(initial_timeline, workers)
    return obs



def _update_shared_obs(shared_obs, obs, reward=0, done=False):
    shared_obs.reward.copy_(torch.tensor(reward))
    shared_obs.done.copy_(torch.tensor(done))

    if not obs:
        return

    job_feature_tensors, op_masks, prlvl_msk = obs

    for job_id, job_feature_tensor in job_feature_tensors.items():
        chunk = shared_obs.feature_tensor_chunks[job_id]
        chunk.copy_(job_feature_tensor)

    shared_obs.op_msk.zero_()
    for job_id, op_msk in op_masks.items():
        shared_obs.op_msk[job_id, :len(op_msk)] = op_msk

    active_job_ids = list(job_feature_tensors.keys())

    shared_obs.prlvl_msk.zero_()
    shared_obs.prlvl_msk[active_job_ids] = prlvl_msk

    shared_obs.active_job_msk.zero_()
    shared_obs.active_job_msk[active_job_ids] = 1