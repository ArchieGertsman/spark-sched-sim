import heapq
import itertools
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class Event:
    class Type(Enum):
        JOB_ARRIVAL = auto()
        TASK_FINISHED = auto()
        EXECUTOR_READY = auto()

    type: Type

    data: dict


class EventQueue:
    def __init__(self) -> None:
        # priority queue
        self._pq: list[tuple[float, int, Event]] = []

        # tie breaker
        self._counter = itertools.count()

    def reset(self) -> None:
        self._pq.clear()
        self._counter = itertools.count()

    def __bool__(self) -> bool:
        return bool(self._pq)

    def push(self, t: float, event: Event) -> None:
        heapq.heappush(self._pq, (t, next(self._counter), event))

    def top(self) -> tuple[float, Event] | None:
        if not self._pq:
            return None

        t, _, event = self._pq[0]
        return t, event

    def pop(self) -> tuple[float, Event] | None:
        if not self._pq:
            return None

        t, _, event = heapq.heappop(self._pq)
        return t, event
