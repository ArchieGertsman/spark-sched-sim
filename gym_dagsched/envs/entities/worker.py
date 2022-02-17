from dataclasses import dataclass

@dataclass
class Worker:
    # type of the worker (for heterogeneous
    # environments)
    type_: int

    # id of current job assigned to this worker
    job_id: int