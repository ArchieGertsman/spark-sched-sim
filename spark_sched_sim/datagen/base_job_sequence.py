from abc import ABC, abstractmethod

import numpy as np

from ..components.timeline import Timeline, TimelineEvent
from ..components import Job


class BaseJobSequenceGen(ABC):

    def __init__(self, job_arrival_rate: float, job_arrival_cap: int):
        self.mean_interarrival_time = 1 / job_arrival_rate
        self.job_arrival_cap = job_arrival_cap
        self.np_random = None



    def reset(self, np_random: np.random.RandomState):
        self.np_random = np_random



    def new_timeline(self, time_limit) -> Timeline:
        '''Fills timeline with job arrivals, which follow a Poisson process 
        parameterized by `job_arrival_rate`
        '''
        assert self.np_random
        timeline = Timeline()

        # wall time of current arrival
        t = 0
        job_idx = 0
        while t < time_limit and job_idx < self.job_arrival_cap:
            job = self.generate_job(job_idx, t)
            timeline.push(
                t, 
                TimelineEvent(
                    type = TimelineEvent.Type.JOB_ARRIVAL, 
                    data = {'job': job}
                )
            )

            # sample time in ms until next arrival
            t += self.np_random.exponential(self.mean_interarrival_time)
            job_idx += 1

        return timeline



    @abstractmethod
    def generate_job(self, job_id: int, t_arrival: float) -> Job:
        pass