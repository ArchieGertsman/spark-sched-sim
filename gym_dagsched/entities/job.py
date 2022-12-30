from dataclasses import dataclass
from typing import List
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx

from .operation import Features



@dataclass
class Job:
    '''An object representing a job in the system, containing
    a set of operations with interdependencies, stored as a dag.
    '''

    # unique identifier of this job
    id_: int

    # list of `Operation` objects
    ops: List

    # networkx dag storing the operations' interdependencies
    dag: nx.DiGraph

    # time that this job arrived into the system
    t_arrival: float

    # number of operations that have completed executing
    completed_ops_count = 0

    saturated_ops_count = 0

    # time that this job completed, i.e. when the last
    # operation completed executing
    t_completed = np.inf



    @property
    def completed(self):
        '''whether or not this job has completed'''
        return self.completed_ops_count == len(self.ops)



    @property
    def saturated(self):
        return self.saturated_ops_count == len(self.ops)


    @property
    def num_ops(self):
        return len(self.ops)



    def add_op_completion(self, op):
        '''increments the count of completed operations'''
        assert self.completed_ops_count < len(self.ops)
        self.completed_ops_count += 1

        self.frontier_ops.remove(op)

        new_ops = self.find_new_frontier_ops(op, 'completed')
        self.frontier_ops |= new_ops

        return len(new_ops) > 0



    def set_op_saturated(self, op, flag):
        op.saturated = flag

        if flag:
            assert self.saturated_ops_count < len(self.ops)
            self.saturated_ops_count += 1
        else:
            assert self.saturated_ops_count > 0
            self.saturated_ops_count -= 1
            


    def initialize_frontier(self):
        '''returns a set containing all the operations which are
        source nodes in the dag, i.e. which have no dependencies
        '''
        assert len(self.frontier_ops) == 0
        
        sources = self.source_ops()

        self.frontier_ops |= sources

        return sources


    def source_ops(self):
        return set(
            self.ops[node]
            for node, in_deg in self.dag.in_degree()
            if in_deg == 0
        )



    def children_ops(self, op):
        return set([
            self.ops[op_id] 
            for op_id in self.dag.successors(op.id_)
        ])


    def parent_ops(self, op):
        return set([
            self.ops[op_id] 
            for op_id in self.dag.predecessors(op.id_)
        ])  



    def find_new_frontier_ops(self, op, criterion):
        '''if `op` is completed, returns all of its
        successors whose other dependencies are also 
        completed, if any exist.
        '''
        assert criterion in ['saturated', 'completed']

        if not op.check_criterion(criterion):
            return set()

        new_ops = set()
        # search through op's children
        for suc_op_id in self.dag.successors(op.id_):
            # if all dependencies are satisfied, then
            # add this child to the frontiers
            new_op = self.ops[suc_op_id]
            if not new_op.check_criterion(criterion) and \
                self.check_dependencies(suc_op_id, criterion) \
                :
                new_ops.add(new_op)
        
        return new_ops



    def check_dependencies(self, op_id, criterion):
        '''searches to see if all the dependencies of operation 
        with id `op_id` are satisfied.
        '''
        for dep_id in self.dag.predecessors(op_id):
            if not self.ops[dep_id].check_criterion(criterion):
                return False

        return True



    def populate_remaining_times(self):
        '''populates the `remaining_time` field for each operation
        within this job via BFS. The remaining time of an operation
        is defined recursively as its expected duration plus the 
        remaining times of each of its children.
        '''
        # def _populate_recursive(op):
        #     op.remaining_time = \
        #         op.task_duration[op.task_duration<np.inf].mean()

        #     if self.dag.out_degree(op.id_) == 0:
        #         return

        #     for child_op_id in self.dag.successors(op.id_):
        #         child_op = self.ops[child_op_id]
        #         _populate_recursive(child_op)
        #         op.remaining_time += child_op.remaining_time
            

        # src_ops = self.find_src_ops()
        # # populate each connected component of the dag
        # while len(src_ops) > 0:
        #     op = src_ops.pop()
        #     _populate_recursive(op)
        pass



    def init_pyg_data(self):
        # feature_vectors = [self.init_feature_vector(op) for op in self.ops]
        pyg_data = from_networkx(self.dag)
        # pyg_data.x = torch.tensor(feature_vectors, dtype=torch.float32)
        pyg_data.x = torch.zeros((len(self.ops), 5))
        return pyg_data



    def add_local_worker(self, worker):
        self.local_workers.add(worker.id_)
        worker.job_id = self.id_



    def remove_local_worker(self, worker):
        self.local_workers.remove(worker.id_)
        worker.job_id = None



    def assign_worker(self, worker, op, wall_time):
        assert op.n_saturated_tasks < op.n_tasks

        task = op.remaining_tasks.pop()
        op.processing_tasks.add(task)
            
        worker.task = task
        task.worker_id = worker.id_
        task.t_accepted = wall_time
        return task



    def add_task_completion(self, op, task, worker, wall_time):
        assert not op.completed
        assert task in op.processing_tasks

        op.processing_tasks.remove(task)
        op.completed_tasks.add(task)

        worker.task = None
        task.t_completed = wall_time