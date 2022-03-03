import heapq
import itertools
from dataclasses import dataclass


@dataclass
class Event:
    obj: object

@dataclass
class JobArrival(Event):
    pass

@dataclass
class TaskCompletion(Event):
    task_id: int



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