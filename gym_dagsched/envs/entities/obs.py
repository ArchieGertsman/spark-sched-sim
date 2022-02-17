import typing
from dataclasses import dataclass, fields

import numpy as np

from .job import Job
from .worker import Worker

@dataclass
class Obs:
    wall_time: np.ndarray

    job_count: int

    jobs: typing.Tuple[Job, ...]

    frontier_stages_mask: np.ndarray

    workers: typing.Tuple[Worker, ...]


    @property
    def max_stages(self):
        return len(self.jobs[0].stages)

    @property
    def max_jobs(self):
        return len(self.jobs)

    @property
    def n_workers(self):
        return len(self.workers)


    def add_stage_to_frontier(self, stage_idx):
        self.frontier_stages_mask[stage_idx] = 1


    def remove_stage_from_frontier(self, stage_idx):
        self.frontier_stages_mask[stage_idx] = 0


    def stage_in_frontier(self, stage_idx):
        return self.frontier_stages_mask[stage_idx]


    def get_frontier_stages(self):
        stage_indices = np.argwhere(self.frontier_stages_mask==1).flatten()
        stages = [self.get_stage_from_idx(stage_idx) for stage_idx in stage_indices]
        return stages


    def get_avail_workers_mask(self):
        avail_workers_mask = np.zeros(self.n_workers)
        for i,worker in enumerate(self.workers):
            if worker.job_id == -1:
                avail_workers_mask[i] = 1
        return avail_workers_mask


    def get_stage_idx(self, job_id, stage_id):
        return job_id * self.max_stages + stage_id


    def get_stage_from_idx(self, stage_idx):
        stage_id = stage_idx % self.max_stages
        job_id = (stage_idx - stage_id) // self.max_stages
        return self.jobs[job_id].stages[stage_id]


    def add_job(self, new_job):
        old_job = self.jobs[new_job.id_]
        for field in fields(old_job):
            setattr(old_job, field.name, getattr(new_job, field.name))
        
        self.add_src_nodes_to_frontier(new_job)

        self.job_count += 1


    def add_src_nodes_to_frontier(self, job):
        source_ids = job.find_src_nodes()
        source_ids = np.array(source_ids)
        indices = job.id_ * self.max_stages + source_ids
        self.frontier_stages_mask[indices] = 1


    def get_workers_from_mask(self, workers_mask):
        worker_indices = np.argwhere(workers_mask==1).flatten()
        workers = [self.workers[i] for i in worker_indices]
        return workers