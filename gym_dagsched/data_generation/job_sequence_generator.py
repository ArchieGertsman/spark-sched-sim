
from ..core.timeline import Timeline, JobArrival


class JobSequenceGenerator:

    def new_timeline(self,
                     np_random,
                     num_init_jobs, 
                     num_job_arrivals, 
                     job_arrival_rate):
        '''Fills timeline with job arrivals, which follow a
        Poisson process parameterized by `job_arrival_rate`
        '''
        self.np_random = np_random
        timeline = Timeline()

        # wall time of current arrival
        t = 0.

        for job_id in range(num_init_jobs + num_job_arrivals):
            if job_id >= num_init_jobs:
                # sample time until next arrival
                t += self.np_random.exponential(1/job_arrival_rate)

            job = self.generate_job(job_id, t)
            timeline.push(t, JobArrival(job))
        
        return timeline



    def generate_job(self, job_id, t_arrival):
        raise NotImplementedError