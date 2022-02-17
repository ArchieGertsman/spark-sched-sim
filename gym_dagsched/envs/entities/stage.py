from dataclasses import dataclass

import numpy as np


@dataclass 
class Stage:
    # each stage has a unique id
    id_: int

    # id of job this stage belongs to
    job_id: int

    # number of identical tasks to complete 
    # within the stage
    n_tasks: int

    # which type of worker is compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_type: int

    # expected completion time of this stage
    duration: np.ndarray

    # number of workers assigned to this task
    n_workers: int

    # time at which a set of workers began
    # processing this stage
    t_accepted: np.ndarray

    # time at which this stage finished
    # being processed
    t_completed: np.ndarray
