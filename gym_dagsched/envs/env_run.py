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
            if not first_episode:
                avg_job_duration = metrics.avg_job_duration(env)
                n_completed_jobs = env.n_completed_jobs
            else:
                avg_job_duration, n_completed_jobs = None, None

            n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs = data
            # _env_reset(env, datagen, n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs)
            initial_timeline = datagen.initial_timeline(
                n_job_arrivals, n_init_jobs, mjit)
            workers = datagen.workers(n_workers)

            env.reset(initial_timeline, workers, shared_obs)

            first_episode = False
            conn.send((avg_job_duration, n_completed_jobs))
            
        elif header == 'step':
            if shared_obs is None:
                raise Exception(f'proc {rank} is trying to step before resetting')
            (job_id, op_id), prlvl = data
            # _env_step(rank, env, job_id, op_id, prlvl, shared_obs)
            env.step(job_id, op_id, prlvl)
            conn.send(None)

        else:
            raise Exception(f'proc {rank} received invalid data')



# def _env_reset(env, datagen, n_job_arrivals, n_init_jobs, mjit, n_workers, shared_obs):
#     initial_timeline = datagen.initial_timeline(
#         n_job_arrivals, n_init_jobs, mjit)
#     workers = datagen.workers(n_workers)

#     env.reset(initial_timeline, workers, shared_obs)



# def _env_step(
#     rank, 
#     env, 
#     job_id, 
#     op_id, 
#     prlvl, 
# ):
#     env.step(job_id, op_id, prlvl)

    # if not done:
    #     env.update_node_features()

    # active_job_msk = torch.zeros(env.n_job_arrivals, dtype=torch.bool)
    # active_job_msk[env.active_job_ids] = 1



    # put_data = [
    #     active_job_msk,
    #     env.construct_op_mask().flatten(),
    #     env.construct_prlvl_mask().flatten(), 
    #     # env.construct_node_feature_tensor().flatten(),
    #     # env.compute_worker_counts(),
    #     # env.get_source_job_mask(),
    #     # torch.tensor([env.n_source_workers]),
    #     torch.tensor([reward]), 
    #     torch.tensor([done])
    # ]

    # torch.cat(put_data, out=shared_obs_tensor)



