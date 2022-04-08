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