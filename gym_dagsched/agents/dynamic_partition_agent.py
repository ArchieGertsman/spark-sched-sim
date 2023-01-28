
import numpy as np

from .base_agent import BaseAgent




class DynamicPartitionAgent(BaseAgent):

    def __init__(self, num_workers):
        super().__init__('Dynamic Partition')
        self.num_workers = num_workers



    def invoke(self, obs):
        (_,
         num_source_workers,
         source_job_id, 
         schedulable_ops, 
         active_jobs,
         _) = obs

        print('invoke', num_source_workers, len(schedulable_ops))

        worker_cap = self.num_workers / max(1, len(active_jobs))
        worker_cap = int(np.ceil(worker_cap))

        if source_job_id in active_jobs.keys():
            job = active_jobs[source_job_id]

            for op in iter(job.frontier_ops):
                if op in schedulable_ops:
                    return op.pool_key, num_source_workers

            for op in iter(schedulable_ops):
                if op.job_id == source_job_id:
                    return op.pool_key, num_source_workers

        for job in active_jobs.values():
            if job.total_worker_count >= worker_cap:
                continue

            selected_op = None

            for op in iter(job.frontier_ops):
                if op in schedulable_ops:
                    selected_op = op
                    break

            if selected_op is None:
                for op in iter(schedulable_ops):
                    if op in job.active_ops:
                        selected_op = op
                        break

            if selected_op is None:
                continue

            prlism_lim = worker_cap - job.total_worker_count
            prlism_lim = min(prlism_lim, num_source_workers)

            return selected_op.pool_key, prlism_lim

        return None