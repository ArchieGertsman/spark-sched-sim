import numpy as np



class Job:
    '''An object representing a job in the system, containing
    a set of operations with interdependencies, stored as a dag.
    '''

    def __init__(self, id_, ops, dag, t_arrival):
        # unique identifier of this job
        self.id_ = id_

        # list of objects of all the operations
        # that belong to this job
        self.ops = ops

        # all incomplete operations
        self.active_ops = set(ops)
        
        # incomplete operations whose parents have completed
        self.frontier_ops = set()

        # networkx dag storing the operations' interdependencies
        self.dag = dag

        # time that this job arrived into the system
        self.t_arrival = t_arrival

        # time that this job completed, i.e. when the last
        # operation completed
        self.t_completed = np.inf

        # set of workers that are local to this job
        self.local_workers = set()

        self.saturated_op_count = 0

        self.total_worker_count = 0



    @property
    def pool_key(self):
        return (self.id_, None)



    @property
    def completed(self):
        '''whether or not this job has completed'''
        return self.num_active_ops == 0



    @property
    def saturated(self):
        return self.saturated_op_count == len(self.ops)



    @property
    def num_ops(self):
        return len(self.ops)



    @property
    def num_active_ops(self):
        return len(self.active_ops)



    def add_op_completion(self, op):
        '''increments the count of completed operations'''
        self.active_ops.remove(op)

        self.frontier_ops.remove(op)

        new_ops = self.find_new_frontier_ops(op)
        self.frontier_ops |= new_ops

        return len(new_ops) > 0
            


    def init_frontier(self):
        '''returns a set containing all the operations which are
        source nodes in the dag, i.e. which have no dependencies
        '''
        assert len(self.frontier_ops) == 0
        self.frontier_ops |= self.source_ops()



    def source_ops(self):
        return \
            set(self.ops[node]
                for node, in_deg in self.dag.in_degree()
                if in_deg == 0)



    def children_ops(self, op):
        return (self.ops[op_id] 
                for op_id in self.dag.successors(op.id_))



    def parent_ops(self, op):
        return (self.ops[op_id] 
                for op_id in self.dag.predecessors(op.id_))



    def find_new_frontier_ops(self, op):
        '''if `op` is completed, returns all of its
        successors whose other dependencies are also 
        completed, if any exist.
        '''

        if not op.completed:
            return set()

        new_ops = set()
        # search through op's children
        for suc_op_id in self.dag.successors(op.id_):
            # if all dependencies are satisfied, then
            # add this child to the frontiers
            new_op = self.ops[suc_op_id]
            if not new_op.completed and \
               self.check_dependencies(suc_op_id):
                new_ops.add(new_op)
        
        return new_ops



    def check_dependencies(self, op_id):
        '''searches to see if all the dependencies of operation 
        with id `op_id` are satisfied.
        '''
        for dep_id in self.dag.predecessors(op_id):
            if not self.ops[dep_id].completed:
                return False

        return True



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
        op.add_task_completion()

        worker.task = None
        task.t_completed = wall_time