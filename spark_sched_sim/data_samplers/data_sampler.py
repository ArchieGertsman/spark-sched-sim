from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from spark_sched_sim.components import Job, Stage, Task, Executor


class DataSampler(ABC):
    np_random: np.random.Generator | None

    def reset(self, np_random: np.random.Generator):
        self.np_random = np_random

    @abstractmethod
    def job_sequence(self, max_time: float) -> Iterable[tuple[float, Job]]:
        pass

    @abstractmethod
    def task_duration(
        self, job: Job, stage: Stage, task: Task, executor: Executor
    ) -> float:
        pass
