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

    # number of tasks in this stage that 
    # have already been completed
    n_completed_tasks: int

    # expected completion time of a single 
    # task in this stage
    task_duration: np.ndarray

    # which type of worker is compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_type: int

    # number of workers currently assigned to 
    # this task
    n_workers: int

    # time at which a set of workers began
    # processing this stage
    t_accepted: np.ndarray

    # time at which this stage finished
    # being processed
    t_completed: np.ndarray


    def generate_completion_time(self):
        # TODO: do a more complex calculation given 
        # other properties of this stage
        return self.t_accepted + self.n_tasks * self.task_duration
