import numpy as np
import networkx as nx

from ..entities.job import Job
from ..entities.operation import Operation
from ..entities.worker import Worker


class DataGenerator:

    def __init__(self, max_ops, max_tasks, n_worker_types):
        self.MAX_OPS = max_ops
        self.MAX_TASKS = max_tasks
        self.N_WORKER_TYPES = n_worker_types


    def job(self, id, t_arrival):
        ops = self.ops(id)
        return Job(
            id_=id, 
            ops=ops, 
            dag=self.dag(len(ops)), 
            t_arrival=t_arrival)


    def dag(self, n):
        upper_triangle = np.random.binomial(1, 2/n, n*(n-1)//2)
        adj_matrix = np.zeros((n,n))
        adj_matrix[np.triu_indices(n,1)] = upper_triangle
        dag = nx.convert_matrix.from_numpy_matrix(
            adj_matrix, create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(dag)
        return dag



    def ops(self, job_id):
        n_ops = np.random.randint(low=2, high=self.MAX_OPS+1)
        ops = []
        for i in range(n_ops):
            n_tasks = np.random.randint(low=1, high=self.MAX_TASKS+1)
            mask = self.compatible_worker_types_mask()
            durations = self.generate_task_duration_per_worker_type(mask)
            ops += [Operation(
                id=i,
                job_id=job_id,
                n_tasks=n_tasks,
                mean_task_duration=durations
            )]
        return ops


    def compatible_worker_types_mask(self):
        n_compatible_worker_types = np.random.randint(low=1, high=self.N_WORKER_TYPES+1)
        worker_types = np.arange(self.N_WORKER_TYPES)
        compatible_worker_types = \
            np.random.choice(worker_types, n_compatible_worker_types, replace=False)
        mask = np.zeros(self.N_WORKER_TYPES, dtype=bool)
        mask[compatible_worker_types] = True
        return mask


    def generate_task_duration_per_worker_type(self, compatible_worker_types_mask):
        # generate a baseline task duration
        baseline_duration = np.random.exponential(30.)

        # generate offsets from this baseline for each worker type
        # so that some workers will work slower than the baseline
        # while others will work faster than the baseline
        worker_types_offsets = np.random.normal(scale=10., size=self.N_WORKER_TYPES)

        # compute the expected durations from the baseline 
        # and offsets
        durations = baseline_duration * np.ones(self.N_WORKER_TYPES)
        durations += worker_types_offsets

        # ensure that no expected duration is too small
        durations = np.clip(durations, 10., None)

        # give incompatible worker types an expected duration
        # of infinity
        durations[~compatible_worker_types_mask] = np.inf
        return durations


    def worker(self, i):
        type_ = i if i < self.N_WORKER_TYPES \
            else np.random.randint(low=0, high=self.N_WORKER_TYPES)
        return Worker(id_=i, type_=type_)

    
    def task_duration(self, op, assigned_worker_type):
        return op.mean_task_duration[assigned_worker_type]