from dataclasses import dataclass

import numpy as np

from ..utils import mask_to_indices


@dataclass 
class Stage:
    # each stage has a unique id
    id_: int

    # id of job this stage belongs to
    job_id: int

    # number of identical tasks to complete 
    # within the stage
    n_tasks: int

    # number of tasks in this stage that 
    # have already been completed
    n_completed_tasks: int

    # expected completion time of a single 
    # task in this stage
    task_duration: np.ndarray

    # which types of worker are compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_types_mask: np.ndarray

    # worker_ids[i] is the id of the worker
    # assigned to task i. -1 if none
    worker_ids: np.ndarray

    # time at which a set of workers began
    # processing this stage
    t_accepted: np.ndarray

    # time at which this stage finished
    # being processed
    t_completed: np.ndarray


    @property
    def is_complete(self):
        return self.n_completed_tasks == self.n_tasks

    @property
    def next_task_id(self):
        assert self.n_completed_tasks <= self.n_tasks
        return self.n_completed_tasks


    def add_task_completion(self, task_id):
        assert self.n_completed_tasks < self.n_tasks
        self.n_completed_tasks += 1
        self.worker_ids[task_id] = -1


    def generate_task_duration(self):
        # TODO: do a more complex calculation given 
        # other properties of this stage
        return self.task_duration


    def compatible_worker_types(self):
        return mask_to_indices(self.worker_types_mask)


    def incompatible_worker_types(self):
        return mask_to_indices(1-self.worker_types_mask)


    def remove_worker(self, worker):
        assert worker.is_available
        assert worker.stage_id == self.id_
        self.worker_ids[worker.task_id] = -1


    def add_worker(self, worker):
        assert self.n_completed_tasks < self.n_tasks
        assert worker.compatible_with(self)
        task_id = self.next_task_id
        self.worker_ids[task_id] = worker.id_
        return task_id


    def is_saturated(self):
        n_active_workers = (self.worker_ids != -1).sum()
        n_saturated_tasks = self.n_completed_tasks + n_active_workers
        assert n_saturated_tasks <= self.n_tasks
        return n_saturated_tasks == self.n_tasks


    def complete(self, t_completion):
        assert self.n_completed_tasks == self.n_tasks
        self.t_completed = t_completion