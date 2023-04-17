import heapq
import itertools
from typing import NamedTuple, Any

from .job import Job
from .stage import Stage
from .task import Task
from .executor import Executor


# scheduling event objects

class JobArrival(NamedTuple):
    job: Job
    

class TaskCompletion(NamedTuple):
    stage: Stage
    task: Task


class ExecutorArrival(NamedTuple):
    executor: Executor
    stage: Stage



# heap-based timeline

class Timeline:
    def __init__(self):
        # priority queue
        self.pq = []

        # tie breaker
        self.counter = itertools.count()


    def __len__(self):
        return len(self.pq)
    

    @property
    def empty(self):
        return len(self) == 0
    

    def peek(self):
        try:
            key, _, item = self.pq[0]
            return key, item
        except:
            return None, None


    def push(self, key, item):
        heapq.heappush(self.pq, (key, next(self.counter), item))
        

    def pop(self):
        if len(self.pq) > 0:
            key, _, item = heapq.heappop(self.pq)
            return key, item
        else:
            return None, None
        

    def reset(self):
        self.pq = []
        self.counter = itertools.count()