from typing import List
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx

from .operation import Features



class Job:
    '''An object representing a job in the system, containing
    a set of operations with interdependencies, stored as a dag.
    '''

    def __init__(self, id_, ops, dag, t_arrival):
        # unique identifier of this job
        self.id_ = id_

        # list of `Operation` objects
        self.ops = ops
        self.active_ops = set(ops)

        # networkx dag storing the operations' interdependencies
        self.dag = dag

        # time that this job arrived into the system
        self.t_arrival = t_arrival

        # time that this job completed, i.e. when the last
        # operation completed executing
        self.t_completed = np.inf

        self.local_workers = set()
        self.frontier_ops = set()

        self.num_commitments = 0
        self.num_moving_workers = 0
        self.saturated_op_count = 0



    @property
    def completed(self):
        '''whether or not this job has completed'''
        # return self.completed_ops_count == len(self.ops)
        return self.num_active_ops == 0



    @property
    def saturated(self):
        return self.saturated_op_count == len(self.ops)


    @property
    def num_ops(self):
        return len(self.ops)



    @property
    def total_worker_count(self):
        return len(self.local_workers) + \
            self.num_commitments + \
            self.num_moving_workers


    @property
    def num_active_ops(self):
        return len(self.active_ops)



    def add_op_completion(self, op):
        '''increments the count of completed operations'''
        # assert self.completed_ops_count < len(self.ops)
        # self.completed_ops_count += 1

        self.active_ops.remove(op)

        self.frontier_ops.remove(op)

        new_ops = self.find_new_frontier_ops(op, 'completed')
        self.frontier_ops |= new_ops

        return len(new_ops) > 0
            


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
        return (
            self.ops[op_id] 
            for op_id in self.dag.successors(op.id_)
        )


    def parent_ops(self, op):
        return (
            self.ops[op_id] 
            for op_id in self.dag.predecessors(op.id_)
        )



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
        pyg_data = from_networkx(self.dag)
        pyg_data.x = torch.zeros((len(self.ops), 5))
        return pyg_data



    def add_commitments(self, n):
        self.num_commitments += n

    def remove_commitment(self):
        self.num_commitments -= 1
        assert self.num_commitments >= 0


    def add_moving_worker(self):
        self.num_moving_workers += 1

    def remove_moving_worker(self):
        self.num_moving_workers -= 1
        assert self.num_moving_workers >= 0



    def add_local_worker(self, worker):
        self.local_workers.add(worker.id_)
        worker.job_id = self.id_



    def remove_local_worker(self, worker):
        self.local_workers.remove(worker.id_)
        worker.job_id = None



    def assign_worker(self, worker, op, wall_time):
        assert op.num_saturated_tasks < op.num_tasks

        task = op.start_on_next_task()

        if op.num_remaining_tasks == 0:
            self.saturated_op_count += 1
            
        worker.task = task
        task.worker_id = worker.id_
        task.t_accepted = wall_time
        return task



    def add_task_completion(self, op, task, worker, wall_time):
        assert not op.completed
        # assert task in op.processing_tasks

        op.mark_task_completed(task)

        worker.task = None
        task.t_completed = wall_time