import numpy as np

from .base_agent import BaseAgent




class SCPTAgent(BaseAgent):
    '''Shortest-Critical-Path-Time agent. The critical path 
    time of a node is defined as the length of the longest path 
    from that node to any leaf, weighted by node durations. 
    This heuristic prioritizes operations with short critical 
    path time.
    '''

    def __init__(self, 
                 num_workers, 
                 dynamic=False):
        super().__init__(f'SCPT (dyanmic={dynamic})')
        self.num_workers = num_workers
        self.dynamic = dynamic



    def invoke(self, obs):
        (num_source_workers,
         source_job_id, 
         schedulable_ops, 
         active_jobs,
         _) = obs

        if self.dynamic:
            worker_cap = self.num_workers / max(1, len(active_jobs))
            worker_cap = int(np.ceil(worker_cap))
        else:
            worker_cap = self.num_workers

        # first, try to find an op within the source job
        if source_job_id in active_jobs.keys():
            job = active_jobs[source_job_id]

            for op in iter(job.frontier_ops):
                if op in schedulable_ops:
                    return op.pool_key, num_source_workers

            for op in iter(schedulable_ops):
                if op.job_id == source_job_id:
                    return op.pool_key, num_source_workers

        # compute critical path length for each remaining
        # operation over all unsaturated jobs
        cp_lens = {}
        for job in active_jobs.values():
            if job.total_worker_count >= worker_cap:
                continue
            cp_lens |= self.calculate_cp_lengths(job)

        if cp_lens == {}:
            # all jobs are saturated
            return None

        # filter out non-schedulable operations
        cp_lens = {op: cp_len for op, cp_len in cp_lens.items()
                   if op in schedulable_ops}

        # sort ops based on critical path length
        cp_lens = dict(sorted(cp_lens.items(), key=lambda item: item[1]))

        # select an op with shortest critical path length,
        # prioritizing ready ops
        selected_op = None
        for op in cp_lens.keys():
            if selected_op is None:
                selected_op = op

            if op in active_jobs[op.job_id].frontier_ops:
                # found a ready op
                break

        if selected_op is None:
            return None

        # try to allocate workers up to the capacity
        job = active_jobs[selected_op.job_id]
        num_workers = worker_cap - job.total_worker_count
        num_workers = min(num_workers, num_source_workers)

        return selected_op.pool_key, num_workers



    @classmethod
    def calculate_cp_lengths(cls, job):
        '''for each active node in the job dag, computes
        the critical path length starting from that node
        '''
        def calc_cp_len(op, cp_lens):
            cp_lens[op] = op.approx_remaining_work

            if job.dag.out_degree(op.id_) == 0:
                return

            max_child_cpl = -1
            for child_op_id in job.dag.successors(op.id_):
                child_op = job.ops[child_op_id]
                if child_op not in cp_lens:
                    calc_cp_len(child_op, cp_lens)
                max_child_cpl = \
                    max(max_child_cpl, cp_lens[child_op])
            
            cp_lens[op] += max_child_cpl

        cp_lens = {}
        for op in iter(job.frontier_ops):
            calc_cp_len(op, cp_lens)

        return cp_lens