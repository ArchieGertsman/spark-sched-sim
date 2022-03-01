from dataclasses import dataclass
import typing

import numpy as np

from .task import Task
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

    tasks: typing.Tuple[Task, ...]


    @property
    def is_complete(self):
        return self.n_completed_tasks == self.n_tasks

    @property
    def n_processing_tasks(self):
        return np.array([task.is_processing for task in self.tasks]).sum()

    @property 
    def n_saturated_tasks(self):
        return self.n_completed_tasks + self.n_processing_tasks

    @property
    def n_remaining_tasks(self):
        return self.n_tasks - self.n_saturated_tasks

    @property
    def saturated(self):
        assert self.n_saturated_tasks <= self.n_tasks
        return self.n_saturated_tasks == self.n_tasks

    @property
    def next_task_id(self):
        assert self.n_saturated_tasks <= self.n_tasks
        return self.n_saturated_tasks


    def add_task_completion(self, task_id, wall_time):
        assert self.n_completed_tasks < self.n_tasks
        self.n_completed_tasks += 1
        task = self.tasks[task_id]
        task.is_processing = 0
        task.t_completed = wall_time


    def generate_task_duration(self):
        # TODO: do a more complex calculation given 
        # other properties of this stage
        return self.task_duration


    def compatible_worker_types(self):
        return mask_to_indices(self.worker_types_mask)


    def incompatible_worker_types(self):
        return mask_to_indices(1-self.worker_types_mask)


    def add_worker(self, worker, wall_time):
        assert self.n_saturated_tasks < self.n_tasks
        assert worker.compatible_with(self)
        task_id = self.next_task_id
        task = self.tasks[task_id]
        task.worker_id = worker.id_
        task.is_processing = 1
        task.t_accepted = wall_time
        return task_id