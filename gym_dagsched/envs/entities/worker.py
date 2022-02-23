from dataclasses import dataclass

import numpy as np

@dataclass
class Worker:
    id_: int

    # type of the worker (for heterogeneous
    # environments)
    type_: int

    # id of current job assigned to this worker
    job_id: int

    stage_id: int

    task_id: int


    @property
    def is_available(self):
        return self.task_id == -1


    def make_available(self):
        self.task_id = -1


    def compatible_with(self, stage):
        return np.isin(self.type_, stage.compatible_worker_types())


    def assign_new_stage(self, stage):
        assert self.is_available
        assert self.compatible_with(stage)
        self.job_id = stage.job_id
        self.stage_id = stage.id_
        self.task_id = stage.next_task_id