__all__ = [
    "Scheduler",
    "TrainableScheduler",
    "DecimaScheduler",
    "RandomScheduler",
    "RoundRobinScheduler",
    "make_scheduler",
]

from copy import deepcopy

from .scheduler import Scheduler, TrainableScheduler
from .decima import DecimaScheduler
from .heuristics import RandomScheduler, RoundRobinScheduler


def make_scheduler(agent_cfg):
    glob = globals()
    agent_cls = agent_cfg["agent_cls"]
    assert agent_cls in glob, f"'{agent_cls}' is not a valid scheduler."
    return glob[agent_cls](**deepcopy(agent_cfg))
