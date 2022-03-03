from dataclasses import dataclass

import numpy as np

from args import args
from . import task


@dataclass
class Worker:
    INVALID_ID = args.n_workers

    INVALID_TYPE = args.n_worker_types


    id_: int = INVALID_ID

    # type of the worker (for heterogeneous
    # environments)
    type_: int = INVALID_TYPE

    # id of current job assigned to this worker
    job_id: int = args.n_jobs

    stage_id: int = args.max_stages

    task_id: int = args.max_tasks



    @property
    def available(self):
        return self.task_id == task.Task.INVALID_ID


    def make_available(self):
        self.task_id = task.Task.INVALID_ID


    def compatible_with(self, stage):
        return np.isin(self.type_, stage.compatible_worker_types())


    def can_assign(self, stage):
        return self.available and self.compatible_with(stage)


    def assign_new_stage(self, stage):
        assert self.available
        assert self.compatible_with(stage)
        self.job_id = stage.job_id
        self.stage_id = stage.id_
        self.task_id = stage.next_task_id