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

    local_workers = set()

    x_ptr = None

    n_avail_local = 0



    @property
    def completed(self):
        '''whether or not this job has completed'''
        return self.completed_ops_count == len(self.ops)



    @property
    def saturated(self):
        return self.saturated_ops_count == len(self.ops)



    def add_op_completion(self):
        '''increments the count of completed operations'''
        assert self.completed_ops_count < len(self.ops)
        self.completed_ops_count += 1


    
    def add_op_saturation(self):
        assert self.saturated_ops_count < len(self.ops)
        self.saturated_ops_count += 1



    def find_src_ops(self):
        '''returns a set containing all the operations which are
        source nodes in the dag, i.e. which have no dependencies
        '''
        sources = [self.ops[node] \
            for node,in_deg in self.dag.in_degree() \
                if in_deg==0]
        return set(sources)



    def find_new_frontier_ops(self, op):
        '''if `op` is completed, returns all of its
        successors whose other dependencies are also 
        completed, if any exist.
        '''
        if not op.completed:
            return set()

        new_ops = set()
        # search through successors of `stage`
        for suc_op_id in self.dag.successors(op.id_):
            # if all dependencies are completed, then
            # add this successor to the frontiers
            if self._check_dependencies(suc_op_id):
                new_op = self.ops[suc_op_id]
                # self.x_ptr[suc_op_id, Features.REMAINING_WORK] = \
                #     new_op.n_tasks * new_op.rough_duration
                new_ops.add(new_op)
        
        return new_ops



    def _check_dependencies(self, op_id):
        '''searches to see if all the dependencies of operation 
        with id `op_id` are satisfied.
        '''
        for dep_id in self.dag.predecessors(op_id):
            if not self.ops[dep_id].completed:
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



    # def init_feature_vector(self, op):
    #     '''returns a feature vector for a single node in the dag'''
    #     n_remaining_tasks = len(op.remaining_tasks)
    #     n_processing_tasks = len(op.processing_tasks)
    #     mean_task_duration = op.rough_duration

    #     return [
    #         n_remaining_tasks,
    #         n_processing_tasks,
    #         mean_task_duration,
    #         0, 0
    #     ] 



    # def update_n_avail_local(self, n):
    #     self.n_avail_local += n
    #     self.x_ptr[:, Features.N_LOCAL_WORKERS] += n
    #     # assert (self.x_ptr[:, FeatureIdx.N_AVAIL_LOCAL_WORKERS] >= 0).all()
    #     # assert (self.x_ptr[:, FeatureIdx.N_AVAIL_LOCAL_WORKERS] <= len(self.local_workers)).all()



    def add_local_worker(self, worker):
        self.local_workers.add(worker.id_)
        worker.job_id = self.id_
        # self.update_n_avail_local(1)



    def remove_local_worker(self, worker_id):
        self.local_workers.remove(worker_id)
        # self.update_n_avail_local(-1)



    def assign_worker(self, worker, op, wall_time):
        assert op.n_saturated_tasks < op.n_tasks

        task = op.remaining_tasks.pop()
        op.processing_tasks.add(task)

        # self.n_avail_local -= 1

        # self.x_ptr[op.id_, Features.N_REMAINING_TASKS] -= 1
        # self.x_ptr[op.id_, Features.REMAINING_WORK] -= op.rough_duration

        # self.update_x_ptr(
        #     op.id_, 
        #     n_remaining_tasks=-1, 
        #     n_processing_tasks=1, 
        #     n_avail_local_workers=-1)
            
        worker.task = task
        task.worker_id = worker.id_
        task.t_accepted = wall_time
        return task



    def add_task_completion(self, op, task, worker, wall_time):
        assert not op.completed
        assert task in op.processing_tasks

        op.processing_tasks.remove(task)
        op.completed_tasks.add(task)

        # self.n_avail_local += 1

        # self.update_x_ptr(
        #     op.id_, 
        #     n_processing_tasks=-1, 
        #     n_avail_local_workers=1)

        worker.task = None
        task.t_completed = wall_time



    # def update_x_ptr(
    #     self,
    #     op_id,
    #     n_remaining_tasks=0,
    #     n_processing_tasks=0,
    #     mean_task_duration=0,
    #     n_avail_workers=0,
    #     n_avail_local_workers=0
    # ):
    #     self.x_ptr[op_id] += torch.tensor([
    #         n_remaining_tasks,
    #         n_processing_tasks,
    #         mean_task_duration,
    #         n_avail_workers,
    #         n_avail_local_workers
    #     ])