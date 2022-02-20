from dataclasses import dataclass

@dataclass
class Worker:
    # type of the worker (for heterogeneous
    # environments)
    type_: int

    # id of current job assigned to this worker
    job_id: int

    # stage_id: int

    # task_id: int


    @property
    def is_available(self):
        return self.job_id == -1

    def compatible_with(self, stage):
        return self.type_ == stage.worker_type