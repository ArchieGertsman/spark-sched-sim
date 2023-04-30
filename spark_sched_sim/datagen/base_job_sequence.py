from abc import ABC, abstractmethod

import numpy as np

from ..components.timeline import Timeline, TimelineEvent
from ..components import Job


class BaseJobSequenceGen(ABC):

    def __init__(
        self, 
        num_init_jobs: int, 
        num_job_arrivals: int, 
        job_arrival_rate: float
    ):
        self.num_init_jobs = num_init_jobs
        self.num_job_arrivals = num_job_arrivals
        self.mean_interarrival_time = 1 / job_arrival_rate
        self.np_random = None



    def reset(self, np_random: np.random.RandomState):
        self.np_random = np_random



    def new_timeline(self) -> Timeline:
        '''Fills timeline with job arrivals, which follow a Poisson process 
        parameterized by `job_arrival_rate`
        '''
        assert self.np_random
        timeline = Timeline()

        # wall time of current arrival
        t = 0

        for job_id in range(self.num_init_jobs + self.num_job_arrivals):
            if job_id >= self.num_init_jobs:
                # sample time in ms until next arrival
                t += self.np_random.exponential(self.mean_interarrival_time)
                
            job = self.generate_job(job_id, t)
            timeline.push(
                t, 
                TimelineEvent(
                    type = TimelineEvent.Type.JOB_ARRIVAL, 
                    data = {'job': job}
                )
            )

        return timeline



    @abstractmethod
    def generate_job(self, job_id: int, t_arrival: float) -> Job:
        pass