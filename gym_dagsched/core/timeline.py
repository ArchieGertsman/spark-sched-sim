import heapq
import itertools
from dataclasses import dataclass

from .entities.job import Job
from .entities.operation import Operation
from .entities.task import Task
from .entities.worker import Worker


@dataclass
class JobArrival(object):
    job: Job

@dataclass
class TaskCompletion(object):
    op: Operation
    task: Task

@dataclass
class WorkerArrival(object):
    worker: Worker
    op: Operation



class Timeline(object):
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
        if len(self.pq) > 0:
            (key, counter, item) = self.pq[0]
            return key, item
        else:
            return None, None

    def push(self, key, item):
        heapq.heappush(self.pq,
            (key, next(self.counter), item))

    def pop(self):
        if len(self.pq) > 0:
            (key, counter, item) = heapq.heappop(self.pq)
            return key, item
        else:
            return None, None

    def reset(self):
        self.pq = []
        self.counter = itertools.count()