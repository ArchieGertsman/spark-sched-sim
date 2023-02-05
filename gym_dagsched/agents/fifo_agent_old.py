import numpy as np

from .base_agent import BaseAgent




class FIFOAgent(BaseAgent):
    '''First-In-First-Out agent. This heuristic prioritizes
    operations whose jobs arrived earlier into the system.
    '''

    def __init__(self, 
                 num_workers, 
                 fair=True):
        name = 'FIFO ' + ('(fair)' if fair else '(greedy)')
        super().__init__(name)
        self.num_workers = num_workers
        self.fair = fair



    def predict(self, obs):
        (num_source_workers,
         source_job_id, 
         schedulable_ops, 
         active_jobs,
         _) = obs

        if self.fair:
            worker_cap = self.num_workers / max(1, len(active_jobs))
            worker_cap = int(np.ceil(worker_cap))
        else:
            worker_cap = self.num_workers

        # first, try to find an op within the source job
        if source_job_id in active_jobs.keys():
            job = active_jobs[source_job_id]

            # try to find a ready op
            for op in iter(job.frontier_ops):
                if op in schedulable_ops:
                    return op.pool_key, num_source_workers

            # try to find any schedulable op
            for op in iter(schedulable_ops):
                if op.job_id == source_job_id:
                    return op.pool_key, num_source_workers

        # `active_jobs` is always sorted by job arrival time,
        # so this loop effectively finds an op in a job with
        # earliest arrival time
        for job in active_jobs.values():
            if job.total_worker_count >= worker_cap:
                continue

            selected_op = None

            # try to find a ready op
            for op in iter(job.frontier_ops):
                if op in schedulable_ops:
                    selected_op = op
                    break

            # try to find any schedulable op
            if selected_op is None:
                for op in iter(schedulable_ops):
                    if op in job.active_ops:
                        selected_op = op
                        break

            if selected_op is None:
                continue

            # try to allocate workers up to the capacity
            num_workers = worker_cap - job.total_worker_count
            num_workers = min(num_workers, num_source_workers)

            return selected_op.pool_key, num_workers

        return None