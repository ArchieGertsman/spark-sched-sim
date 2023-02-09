from abc import ABC, abstractmethod

import numpy as np

from ..timeline import Timeline, JobArrival
from ..entities.job import Job


class BaseJobSequenceGen(ABC):

    def __init__(
        self, 
        num_init_jobs: int, 
        num_job_arrivals: int, 
        job_arrival_rate: float
    ):
        self.num_init_jobs = num_init_jobs
        self.num_job_arrivals = num_job_arrivals
        self.job_arrival_rate = job_arrival_rate



    def new_timeline(self, np_random: np.random.RandomState) -> Timeline:
        '''Fills timeline with job arrivals, which follow a
        Poisson process parameterized by `job_arrival_rate`
        '''
        self.np_random = np_random
        timeline = Timeline()

        # wall time of current arrival
        t = 0.

        for job_id in range(self.num_init_jobs + self.num_job_arrivals):
            if job_id >= self.num_init_jobs:
                # sample time until next arrival
                t += np_random.exponential(1 / self.job_arrival_rate)

            job = self.generate_job(job_id, t)
            timeline.push(t, JobArrival(job))
        
        return timeline



    @abstractmethod
    def generate_job(self, job_id: int, t_arrival: float) -> Job:
        pass