from enum import Enum, auto
from time import time

import numpy as np

from .task import Task


class FeatureIdx:
    N_REMAINING_TASKS = 0
    N_PROCESSING_TASKS = 1
    MEAN_TASK_DURATION = 2
    N_AVAIL_WORKERS = 3
    N_AVAIL_LOCAL_WORKERS = 4
    N_FEATURES = 5



class Operation:

    def __init__(self, id, job_id, n_tasks, task_duration):
        self.id_ = id
        self.job_id = job_id
        self.task_duration = task_duration
        self.compatible_worker_types = self.find_compatible_worker_types()

        self.n_tasks = n_tasks
        tasks = [
            Task(id_=i, op_id=self.id_, job_id=self.job_id) 
            for i in range(n_tasks)
        ]
        self.remaining_tasks = set(tasks)
        self.processing_tasks = set()
        self.completed_tasks = set()
        self.remaining_time = np.inf


    def __hash__(self):
        return hash(self.__unique_id)
        

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__unique_id == other.__unique_id
        else:
            return False


    @property
    def __unique_id(self):
        return (self.job_id, self.id_)

    @property
    def is_complete(self):
        return len(self.completed_tasks) == self.n_tasks

    @property
    def n_saturated_tasks(self):
        return len(self.processing_tasks) + len(self.completed_tasks)

    @property
    def next_task_id(self):
        return self.n_saturated_tasks

    @property
    def saturated(self):
        assert self.n_saturated_tasks <= self.n_tasks
        return self.n_saturated_tasks == self.n_tasks

    @property
    def n_remaining_tasks(self):
        return self.n_tasks - self.n_saturated_tasks



    def find_compatible_worker_types(self):
        types = set()
        for worker_type in range(len(self.task_duration)):
            if self.task_duration[worker_type] < np.inf:
                types.add(worker_type)
        return types
        


    
    
