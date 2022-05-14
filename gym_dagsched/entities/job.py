from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np


@dataclass
class Job:

    id_: int
    ops: List
    dag: nx.DiGraph
    t_arrival: float
    n_completed_ops = 0
    t_completed = np.inf



    @property
    def is_complete(self):
        return self.n_completed_ops == len(self.ops)



    def add_op_completion(self):
        assert self.n_completed_ops < len(self.ops)
        self.n_completed_ops += 1



    def find_src_ops(self):
        sources = [self.ops[node] for node,in_deg in self.dag.in_degree() if in_deg==0]
        return set(sources)



    def find_new_frontiers(self, op):
        '''if `op` is completed, returns all of its
        successors whose other dependencies are also 
        completed, if any exists.
        '''
        if not op.is_complete:
            return set()

        new_frontiers = set()
        # search through successors of `stage`
        for suc_op_id in self.dag.successors(op.id_):
            # if all dependencies are completed, then
            # add this successor to the frontiers
            if self._check_dependencies(suc_op_id):
                new_frontiers.add(self.ops[suc_op_id])
        
        return new_frontiers



    def _check_dependencies(self, op_id):
        '''checks if all the dependencies of `stage_id`
        are completed.
        '''
        for dep_id in self.dag.predecessors(op_id):
            if not self.ops[dep_id].is_complete:
                return False

        return True



    def populate_remaining_times(self):
        def _populate_recursive(op):
            op.remaining_time = \
                op.task_duration[op.task_duration<np.inf].mean()

            if self.dag.out_degree(op.id_) == 0:
                return

            for child_op_id in self.dag.successors(op.id_):
                child_op = self.ops[child_op_id]
                _populate_recursive(child_op)
                op.remaining_time += child_op.remaining_time
            

        src_ops = self.find_src_ops()
        while len(src_ops) > 0:
            op = src_ops.pop()
            _populate_recursive(op)



    def update_feature_vectors(self, workers):
        n_avail, n_avail_local = self.n_workers(workers)
        feature_vectors = {
            i: self.form_feature_vector(
                self.ops[i], n_avail, n_avail_local) 
            for i in range(len(self.ops))
        }
        nx.set_node_attributes(self.dag, feature_vectors, 'x')



    def n_workers(self, workers):
        n_avail = 0
        n_avail_local = 0
        for worker in workers:
            if worker.available:
                n_avail += 1
                if worker.task is not None and worker.task.job_id == self.id_:
                    n_avail_local += 1
        return n_avail, n_avail_local



    def form_feature_vector(self, op, n_avail_workers, n_avail_local_workers):
        n_remaining_tasks = len(op.remaining_tasks)
        n_processing_tasks = len(op.processing_tasks)
        mean_task_duration = op.task_duration.mean()

        return np.array([
            n_remaining_tasks,
            n_processing_tasks,
            mean_task_duration,
            n_avail_workers,
            n_avail_local_workers
        ], dtype=np.float32)