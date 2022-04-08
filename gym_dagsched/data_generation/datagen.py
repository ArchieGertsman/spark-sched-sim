from abc import abstractmethod
import numpy as np
import networkx as nx

from ..entities.job import Job
from ..entities.operation import Operation
from ..entities.worker import Worker
from ..utils.timeline import Timeline, JobArrival


class DataGen:

    def __init__(self, n_worker_types=1):
        self.N_WORKER_TYPES = n_worker_types



    def initial_timeline(self, n_job_arrivals, n_init_jobs, mjit):
        '''Fills timeline with job arrival events, which follow
        a Poisson process, parameterized by args.mjit (mean job
        interarrival time)
        '''
        timeline = Timeline()

        t = 0.      # time of current arrival

        for i in range(n_job_arrivals):
            # sample time until next arrival
            dt_interarrival = np.random.exponential(mjit)
            t += dt_interarrival

            # generate a job and add its arrival to the timeline
            if i < n_init_jobs:
                id = i
                job = self._initial_job(id, t)
            else:
                id = i + n_init_jobs
                job = self._streaming_job(id, t)

            timeline.push(t, JobArrival(job))
        
        return timeline



    def workers(self, n_workers):
        '''Initializes the workers with randomly generated attributes'''
        workers = [self._worker(i) for i in range(n_workers)]
        return workers



    def _worker(self, i):
        type_ = i if i < self.N_WORKER_TYPES \
            else np.random.randint(low=0, high=self.N_WORKER_TYPES)
        return Worker(id_=i, type_=type_)



    @abstractmethod
    def _initial_job(self, id, t):
        pass



    @abstractmethod
    def _streaming_job(self, id, t):
        pass