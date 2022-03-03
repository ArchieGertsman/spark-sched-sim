from dataclasses import dataclass
import typing

import numpy as np

from args import args
from dagsched_utils import invalid_time, mask_to_indices
from .task import Task


@dataclass 
class Stage:
    INVALID_ID = args.max_stages


    # each stage has a unique id
    id_: int = INVALID_ID

    # id of job this stage belongs to
    job_id: int = args.n_jobs

    # number of identical tasks to complete 
    # within the stage
    n_tasks: int = 0

    # number of tasks in this stage that 
    # have already been completed
    n_completed_tasks: int = 0

    # expected completion time of a single 
    # task in this stage
    task_duration: np.ndarray = invalid_time()

    # which types of worker are compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_types_mask: np.ndarray = invalid_time()

    tasks: typing.Tuple[Task, ...] = \
        tuple([Task() for _ in range(args.max_tasks)])


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