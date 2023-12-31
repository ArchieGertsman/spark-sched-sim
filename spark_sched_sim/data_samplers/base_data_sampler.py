from abc import ABC, abstractmethod


class BaseDataSampler(ABC):
    @abstractmethod
    def job_sequence(self, max_time):
        pass

    @abstractmethod
    def task_duration(self, job, stage, task, executor):
        pass
