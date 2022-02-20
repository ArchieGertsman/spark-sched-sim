import typing
from dataclasses import dataclass, fields

import numpy as np

from .job import Job
from .worker import Worker
from ..utils import invalid_time

@dataclass
class DagSchedState:
    wall_time: np.ndarray

    job_count: int

    jobs: typing.Tuple[Job, ...]

    workers: typing.Tuple[Worker, ...]

    frontier_stages_mask: np.ndarray

    saturated_stages_mask: np.ndarray



    @property
    def max_stages(self):
        return len(self.jobs[0].stages)

    @property
    def max_jobs(self):
        return len(self.jobs)

    @property
    def n_workers(self):
        return len(self.workers)


    def add_stages_to_frontier(self, stage_idxs):
        self.frontier_stages_mask[stage_idxs] = 1


    def remove_stage_from_frontier(self, stage_idx):
        self.frontier_stages_mask[stage_idx] = 0


    def stage_in_frontier(self, stage_idx):
        return self.frontier_stages_mask[stage_idx]


    # def saturate_stage(self, stage_idx):
    #     assert(self.stage_in_frontier(stage_idx))
    #     self.remove_stage_from_frontier(stage_idx)
    #     self.saturated_stages_mask[stage_idx] = 1


    # def complete_stage(self, stage_idx):
    #     self.frontier_stages_mask[stage_idx] = 0


    # def stage_in_frontier(self, stage_idx):
    #     return self.frontier_stages_mask[stage_idx]


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


    def get_stage_indices(self, job_id, stage_ids):
        return job_id * self.max_stages + np.array(stage_ids, dtype=int)

    
    def get_stage_idx(self, job_id, stage_id):
        return self.get_stage_indices(job_id, np.array([stage_id]))


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
        indices = self.get_stage_indices(job.id_, source_ids)
        self.add_stages_to_frontier(indices)


    def get_workers_from_mask(self, workers_mask):
        worker_indices = np.argwhere(workers_mask==1).flatten()
        workers = [self.workers[i] for i in worker_indices]
        return workers


    def check_action_validity(self, action):
        stage = self.jobs[action.job_id].stages[action.stage_id]

        workers = self.get_workers_from_mask(action.workers_mask)
        for worker in workers:
            if not worker.is_available or not worker.compatible_with(stage):
                # either one of the selected workers is currently busy,
                # or worker type is not suitible for stage
                return False

        stage_idx = self.get_stage_idx(action.job_id, action.stage_id)
        if not self.stage_in_frontier(stage_idx):
            return False

        return True


    def take_action(self, action):
        # assign worker to selected stage's job
        workers = self.get_workers_from_mask(action.workers_mask)
        for worker in workers:
            worker.job_id = action.job_id

        # retrieve selected stage object, update it
        stage = self.jobs[action.job_id].stages[action.stage_id]
        stage.n_workers = len(workers)
        stage.t_accepted = self.wall_time.copy()

        stage_idx = self.get_stage_idx(action.job_id, action.stage_id)
        if stage.n_workers == stage.n_tasks:
            self.remove_stage_from_frontier(stage_idx)

        # TODO: mark stage as saturated

        t_completion = stage.generate_completion_time()

        return t_completion[0], stage


    def process_stage_completion(self, stage):
        stage.t_completed = self.wall_time.copy()

        stage_idx = self.get_stage_idx(stage.job_id, stage.id_)
        self.remove_stage_from_frontier(stage_idx)

        job = self.jobs[stage.job_id]
        new_stages_ids = job.find_new_frontiers(stage)
        new_stages_idxs = \
            self.get_stage_indices(job.id_, new_stages_ids)
        self.add_stages_to_frontier(new_stages_idxs)

        # free the workers
        for worker in self.workers:
            if worker.job_id == stage.job_id:
                worker.job_id = -1


    def actions_available(self):
        frontier_stages = self.get_frontier_stages()
        avail_workers_mask = self.get_avail_workers_mask()
        avail_worker_idxs = \
            np.argwhere(avail_workers_mask==1).flatten()

        if len(avail_worker_idxs) == 0 or len(frontier_stages) == 0:
            return False
        
        avail_workers = [self.workers[i] for i in avail_worker_idxs]

        for stage in frontier_stages:
            for worker in avail_workers:
                if stage.worker_type == worker.type_:
                    return True

        return False