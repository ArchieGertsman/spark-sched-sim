import heapq
import itertools
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class TimelineEvent:
    class Type(Enum):
        JOB_ARRIVAL = auto()
        TASK_COMPLETION = auto()
        EXECUTOR_ARRIVAL = auto()

    type: Type

    data: dict



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


    def events(self):
        return (event for *_, event in self.pq)