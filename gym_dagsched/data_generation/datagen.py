from abc import abstractmethod

import numpy as np

from ..entities.worker import Worker
from ..entities.timeline import Timeline, JobArrival


class DataGen:

    def __init__(self, n_worker_types=1):
        self.N_WORKER_TYPES = n_worker_types



    def initial_timeline(self, 
                         n_init_jobs, 
                         n_job_arrivals, 
                         job_arrival_rate):
        '''Fills timeline with job arrival events, which follow
        a Poisson process parameterized by mjit (mean job
        interarrival time)
        '''
        timeline = Timeline()

        t = 0.      # time of current arrival

        for id in range(n_init_jobs + n_job_arrivals):
            # generate a job and add its arrival to the timeline
            if id < n_init_jobs:
                job = self._job(id, t)
            else:
                # sample time until next arrival
                dt_interarrival = \
                    np.random.exponential(1/job_arrival_rate)
                t += dt_interarrival
                job = self._job(id, t)

            job.populate_remaining_times()
            timeline.push(t, JobArrival(job))
        
        return timeline



    def workers(self, n_workers):
        '''Initializes the workers with randomly generated attributes'''
        workers = [self._worker(i) for i in range(n_workers)]
        return workers



    def _worker(self, i):
        return Worker(id_=i)



    @abstractmethod
    def _job(self, id, t):
        pass