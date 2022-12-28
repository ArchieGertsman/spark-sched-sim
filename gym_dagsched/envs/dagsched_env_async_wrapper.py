

import torch
from torch_geometric.data import Batch

from .dagsched_env import DagSchedEnv
from ..data_generation.tpch_datagen import TPCHDataGen
from ..utils.pyg import construct_subbatch, chunk_feature_tensor



class DagSchedEnvAsyncWrapper:

    def __init__(self, rank, datagen_state):
        self.env = DagSchedEnv(rank)
        self.datagen = TPCHDataGen(datagen_state)



    @property
    def active_job_ids(self):
        return self.env.active_job_ids

    @property
    def completed_job_ids(self):
        return self.env.completed_job_ids

    @property
    def n_completed_jobs(self):
        return self.env.n_completed_jobs

    @property
    def jobs(self):
        return self.env.jobs



    def reset(self, n_job_arrivals, n_init_jobs, mjit, n_workers):
        initial_timeline = self.datagen.initial_timeline(
            n_job_arrivals, n_init_jobs, mjit)
        workers = self.datagen.workers(n_workers)

        self.op_counts = torch.tensor([
            len(e.job.ops) for _,_,e in initial_timeline.pq])

        self._reset_dag_batch(initial_timeline)
        self.feature_tensor_chunks = \
            chunk_feature_tensor(self.dag_batch)

        obs = self.env.reset(initial_timeline, workers)

        return self._observe(obs)



    def step(self, action):
        obs, reward, done = self.env.step(action)
        obs = self._observe(obs) if not done else None
        return obs, reward, done



    def _reset_dag_batch(self, initial_timeline):
        data_list = [
            e.job.init_pyg_data() 
            for _,_,e in initial_timeline.pq]
        self.dag_batch = Batch.from_data_list(data_list)



    def _observe(self, obs):
        job_feature_tensors, op_masks, prlvl_msk = obs

        active_job_ids = list(job_feature_tensors.keys())
        active_job_msk = torch.zeros(
            self.dag_batch.num_graphs, 
            dtype=torch.bool)
        active_job_msk[active_job_ids] = 1

        for job_id, job_feature_tensor in job_feature_tensors.items():
            self.feature_tensor_chunks[job_id].copy_(job_feature_tensor)
        
        dag_batch = construct_subbatch(
            self.dag_batch, active_job_msk)

        op_msk = torch.cat(list(op_masks.values()))

        prlvl_msk = prlvl_msk.unsqueeze(0) \
            .repeat_interleave(dag_batch.num_graphs, dim=0)

        return dag_batch, op_msk, prlvl_msk
