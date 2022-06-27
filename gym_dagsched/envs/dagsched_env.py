from collections import defaultdict
from copy import deepcopy as dcp
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Batch

from ..utils.timeline import JobArrival, TaskCompletion
from ..utils.device import device


class DagSchedEnv:
    '''An OpenAI-Gym-style simulation environment for scheduling 
    streaming jobs consisting of interdependent operations. 
    
    What?
    "job": consists of "operations" that need to be completed by "workers"
    "streaming": arriving stochastically and continuously over time
    "scheduling": assigning workers to jobs
    "operation": can be futher split into "tasks" which are identical 
      and can be worked on in parallel. The number of tasks in a 
      operation is equal to the number of workers that can work on 
      the operation in parallel.
    "interdependent": some operations may depend on the results of others, 
      and therefore cannot begin execution until those dependencies are 
      satisfied. These dependencies are encoded in directed acyclic graphs 
      (dags) where an edge from operation (a) to operation (b) means that 
      (a) must complete before (b) can begin.

    Example: a cloud computing cluster is responsible for receiving workflows
        (i.e. jobs) and executing them using its resources. A  machine learning 
        workflow may consist of numerous operations, such as data prep, training/
        validation, hyperparameter tuning, testing, etc. and these operations 
        depend on each other, e.g. data prep comes before training. Data prep 
        can further be broken into tasks, where each task entails prepping a 
        subset of the data, and these tasks can easily be parallelized. 
        
    The process of assigning workers to jobs is crutial, as sophisticated 
    scheduling algorithms can significantly increase the system's efficiency.
    Yet, it turns out to be a very challenging problem.
    '''

    # multiplied with reward to control its magnitude
    REWARD_SCALE = 1e-4

    # expected time to move a worker between jobs
    # (mean of exponential distribution)
    MOVING_COST = 2000.


    @property
    def all_jobs_complete(self):
        '''whether or not all the jobs in the system
        have been completed
        '''
        return len(self.active_job_ids) == 0


    @property
    def n_completed_jobs(self):
        return len(self.completed_job_ids)



    @property
    def n_active_jobs(self):
        return len(self.active_job_ids)



    @property
    def n_seen_jobs(self):
        return self.n_completed_jobs + self.n_active_jobs



    def reset(self, initial_timeline, workers):
        '''resets the simulation. should be called before
        each run (including first). all state data is found here.
        '''

        # a priority queue containing scheduling 
        # events indexed by wall time of occurance
        self.timeline = dcp(initial_timeline)

        # list of worker objects which are to be scheduled
        # to complete tasks within the simulation
        self.workers = dcp(workers)

        # wall clock time, keeps increasing throughout
        # the simulation
        self.wall_time = 0.

        # list of job objects within the system.
        # jobs don't get removed from this list
        # after completing; they only get flagged.

        self.jobs = {}

        self.active_job_ids = []

        self.completed_job_ids = []

        # operations in the system which are ready
        # to be executed by a worker because their
        # dependencies are satisfied
        self.frontier_ops = set()

        # operations in the system which have not 
        # yet completed but have all the resources
        # they need assigned to them
        self.saturated_ops = set()

        self._init_dag_batch()

        self.total_time = 0.



    def step(self, op, n_workers):
        '''steps onto the next scheduling event on the timeline, 
        which can be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            even though neither (1) nor (2) have occurred, so 
            the policy should consider taking one of them
        '''
        # t0 = time.time() 

        if op in self.frontier_ops:
            tasks = self._take_action(op, n_workers)
            if len(tasks) > 0:
                self._push_task_completion_events(tasks)
        else:
            pass # an invalid action was taken

        # if there are still actions available after
        # processing the most recent one, then push 
        # a "nudge" event to notify the scheduling agent
        # that another action can immediately be taken
        if self._actions_available():
            self._push_nudge_event()

        # check if simulation is done
        if self.timeline.empty:
            assert self.all_jobs_complete
            return None, None, True
            
        # retreive the next scheduling event from the timeline
        t, event = self.timeline.pop()

        prev_time = self.wall_time

        # update the current wall time
        self.wall_time = t

        reward = self._calculate_reward(prev_time)
        
        self._process_scheduling_event(event)

        # t1 = time.time()
        # self.total_time += t1-t0
        
        return self._observe(), reward, False



    def find_op(self, op_idx):
        '''returns an Operation object corresponding to
        the `op_idx`th operation in the system'''
        i = 0
        op = None
        for j, job_id in enumerate(self.active_job_ids):
            job = self.jobs[job_id]
            if op_idx < i + len(job.ops):
                op = job.ops[op_idx - i]
                break
            else:
                i += len(job.ops)
        assert op is not None
        return op, j



    def _init_dag_batch(self):
        data_list = []
        for _,_,e in self.timeline.pq:
            job = e.obj
            data = from_networkx(job.dag)
            data.x = torch.tensor(
                job.form_feature_vectors(),
                dtype=torch.float32,
                device=device
            )
            data_list += [data]

        self.dag_batch = Batch.from_data_list(data_list).to(device)

        inc_dict = self.dag_batch._inc_dict['edge_index'] 
        num_ops_per_dag = inc_dict
        num_ops_per_dag = torch.roll(num_ops_per_dag, -1)
        num_ops_per_dag[-1] = self.dag_batch.num_nodes
        num_ops_per_dag -= inc_dict
        self.num_ops_per_dag = num_ops_per_dag.to(device)

        self.x_ptrs = {}
        for _,_,e in self.timeline.pq:
            job = e.obj
            self.x_ptrs[job.id_] = self._get_feature_vecs(job.id_)



    def _observe(self):
        '''Returns an observation of the state that can be
        directly passed into the model. This observation
        consists of `dag_batch, op_msk, prlvl_msk`, where
        - `dag_batch` is a mini-batch of PyG graphs, where
            each graph is a dag in the system. See the
            'Advanced Mini-Batching' section in PyG's docs
        - `op_msk` is a mask indicating which operations
            can be scheduled, i.e. op_msk[i] = 1 if the
            i'th operation is in the frontier, 0 otherwise
        - `prlvl_msk` is a mask indicating which parallelism
            levels are valid for each job dag, i.e. 
            prlvl_msk[i,l] = 1 if parallelism level `l` is
            valid for job `i`
        '''
        # t0 = time.time()
        
        subbatch = self._subbatch()

        op_msk = []
        for j in self.active_job_ids:
            # append this job's operations to the mask
            for op in self.jobs[j].ops:
                op_msk += [1] if op in self.frontier_ops else [0]

        op_msk = torch.tensor(op_msk, device=device)

        prlvl_msk = torch.ones((subbatch.num_graphs, len(self.workers)), device=device)

        # t1 = time.time()
        # self.total_time += t1-t0

        return subbatch, op_msk, prlvl_msk




    def n_workers(self, mask):
        '''returns a tuple `(n_avail, n_avail_local)` where
        `n_avail` is the total number of available workers in
        the system, and `n_avail_local` is the number of 
        those workers that are local to this job.
        '''
        n_avail = 0
        n_avail_local = torch.zeros(self.dag_batch.num_graphs)
        for worker in self.workers:
            if worker.available:
                n_avail += 1
                if worker.task is not None:
                    n_avail_local[worker.task.job_id] += 1
        return n_avail, n_avail_local[mask]
    


    def _get_feature_vecs(self, i):
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)
        mask[i] = True
        mask = mask[self.dag_batch.batch]
        idx = mask.nonzero().flatten()
        return self.dag_batch.x[idx[0] : idx[-1]+1]



    def _subbatch(self):
        mask = torch.zeros(self.dag_batch.num_graphs, dtype=torch.bool)
        mask[self.active_job_ids] = True


        node_mask = mask[self.dag_batch.batch]

        subbatch = self.dag_batch.subgraph(node_mask)

        subbatch._num_graphs = mask.sum().item()

        assoc = torch.empty(self.dag_batch.num_graphs, dtype=torch.long)
        assoc[mask] = torch.arange(subbatch.num_graphs)
        subbatch.batch = assoc[self.dag_batch.batch][node_mask]

        ptr = self.dag_batch._slice_dict['x']
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        ptr = torch.cumsum(num_nodes_per_graph[mask], 0)
        ptr = torch.cat([torch.tensor([0]), ptr])
        subbatch.ptr = ptr

        subbatch.num_ops_per_dag = num_nodes_per_graph[mask]

        edge_ptr = self.dag_batch._slice_dict['edge_index']
        num_edges_per_graph = edge_ptr[1:] - edge_ptr[:-1]
        edge_ptr = torch.cumsum(num_edges_per_graph[mask], 0)
        edge_ptr = torch.cat([torch.tensor([0]), edge_ptr])

        subbatch._inc_dict = defaultdict(
            dict, {
                'x': torch.zeros(subbatch.num_graphs, dtype=torch.long),
                'edge_index': ptr[:-1]
            })

        subbatch._slice_dict = defaultdict(dict, {
            'x': ptr,
            'edge_index': edge_ptr
        })


        # update feature vectors with new worker info
        n_avail, n_avail_local = self.n_workers(mask)
        n_avail_local = n_avail_local[subbatch.batch]
        # TODO: name indicies
        subbatch.x[:, 3] = n_avail
        subbatch.x[:, 4] = n_avail_local
        
        return subbatch



    def _push_task_completion_events(self, tasks):
        '''Given a set of task ids and the operation they belong to,
        pushes each of their completions as events to the timeline
        '''
        assert len(tasks) > 0

        task = tasks.pop()
        job_id = task.job_id
        op_id = task.op_id
        op = self.jobs[job_id].ops[op_id]

        while task is not None:
            self._push_task_completion_event(op, task)
            task = tasks.pop() if len(tasks) > 0 else None


    
    def _push_task_completion_event(self, op, task):
        '''pushes a single task completion event to the timeline'''
        assigned_worker_id = task.worker_id
        worker_type = self.workers[assigned_worker_id].type_
        t_completion = \
            task.t_accepted + op.task_duration[worker_type]
        event = TaskCompletion(op, task)
        self.timeline.push(t_completion, event)



    def _push_nudge_event(self):
        '''Pushes a "nudge" event to the timeline at the current
        wall time, so that the scheduling agent can immediately
        choose another action
        '''
        self.timeline.push(self.wall_time, None)



    def _process_scheduling_event(self, event):
        '''handles a scheduling event, which can be a job arrival,
        a task completion, or a nudge
        '''
        if isinstance(event, JobArrival):
            # job arrival event
            job = event.obj
            self._add_job(job)
        elif isinstance(event, TaskCompletion):
            # task completion event
            task = event.task
            self._process_task_completion(task)
        else:
            # nudge event
            pass 



    def _add_job(self, job):
        '''adds a new job to the list of jobs, and adds all of
        its source operations to the frontier
        '''
        self.jobs[job.id_] = job
        self.active_job_ids += [job.id_]
        src_ops = job.find_src_ops()
        self.frontier_ops |= src_ops



    def _process_task_completion(self, task):
        '''performs some bookkeeping when a task completes'''
        job = self.jobs[task.job_id]
        op = job.ops[task.op_id]
        op.add_task_completion(task, self.wall_time, self.x_ptrs[job.id_][op.id_]) #job.data.x[task.op_id])

        worker = self.workers[task.worker_id]
        worker.make_available()

        if op.is_complete:
            self._process_op_completion(op)
        
        job = self.jobs[op.job_id]
        if job.is_complete:
            self._process_job_completion(job)


        
    def _process_op_completion(self, op):
        '''performs some bookkeeping when an operation completes'''
        job = self.jobs[op.job_id]
        job.add_op_completion()
        
        self.saturated_ops.remove(op)

        # add stage's decendents to the frontier, if their
        # other dependencies are also satisfied
        new_ops = job.find_new_frontiers(op)
        self.frontier_ops |= new_ops



    def _process_job_completion(self, job):
        '''performs some bookkeeping when a job completes'''
        # print('job completion')
        assert job.id_ in self.jobs
        # self.active_jobs.pop(job.id_)
        # self.completed_jobs[job.id_] = job
        self.active_job_ids.remove(job.id_)
        self.completed_job_ids += [job.id_]
        job.t_completed = self.wall_time



    def _take_action(self, op, n_workers):
        '''updates the state of the environment based on the
        provided action = (op, n_workers)

        op: Operation object which shall receive work next
        n_workers: number of workers to _try_ assigning to `op`.
            in reality, `op` gets `min(n_workers, n_assignable_workers)`
            where `n_assignable_workers` is the number of workers
            which are both available and compatible with `op`

        Returns: a set of the Task objects which have been scheduled
        '''
        tasks = set()

        # find workers that are closest to this operation's job
        for worker_type in op.compatible_worker_types:
            if op.saturated:
                break
            n_remaining_requests = n_workers - len(tasks)
            for _ in range(n_remaining_requests):
                worker = self._find_closest_worker(op, worker_type)
                if worker is None:
                    break
                task = self._schedule_worker(worker, op)
                tasks.add(task)

        # check if stage is now saturated; if so, remove from frontier
        if op.saturated:
            self.frontier_ops.remove(op)
            self.saturated_ops.add(op)

        return tasks



    def _find_closest_worker(self, op, worker_type):
        '''chooses an available worker for a stage's 
        next task, according to the following priority:
        1. worker is already at stage
        2. worker is not at stage but is at stage's job
        3. any other available worker

        Returns: if the stage is already saturated, or if no 
        worker is found, then `None` is returned. Otherwise
        a Worker object is returned.
        '''
        if op.saturated:
            return None

        # try to find available worker already at the stage
        completed_tasks = list(op.completed_tasks)
        for task in completed_tasks:
            if task.worker_id == None:
                continue
            worker = self.workers[task.worker_id]
            if worker.type_ == worker_type and worker.available:
                return worker

        # try to find available worker at stage's job;
        # if none is found then return any available worker
        avail_worker = None
        for worker in self.workers:
            if worker.type_ == worker_type and worker.available:
                if worker.task is not None and worker.task.job_id == op.job_id:
                    return worker
                elif avail_worker == None:
                    avail_worker = worker
        return avail_worker



    def _schedule_worker(self, worker, op):
        '''sends a worker to an operation, taking into
        account a moving cost, should the worker move 
        between jobs
        '''
        old_job_id = worker.task.job_id \
            if worker.task is not None \
            else None
        new_job_id = op.job_id
        moving_cost = self._job_moving_cost(old_job_id, new_job_id)

        job = self.jobs[op.job_id]

        task = op.add_worker(
            worker, 
            self.wall_time, 
            moving_cost,
            self.x_ptrs[job.id_][op.id_])

        return task


    
    def _job_moving_cost(self, old_job_id, new_job_id):
        '''calculates a moving cost between jobs, which is
        either zero if the jobs are the same, or a sample
        from a fixed exponential distribution
        '''
        e = torch.distributions.exponential.Exponential(self.MOVING_COST)
        return 0. if new_job_id == old_job_id \
            else e.sample().item()



    def _actions_available(self):
        '''checks if there are any valid actions that can be
        taken by the scheduling agent.
        '''
        avail_workers = self._find_available_workers()

        if len(avail_workers) == 0 or len(self.frontier_ops) == 0:
            return False

        for op in self.frontier_ops:
            for worker in avail_workers:
                if worker.compatible_with(op):
                    return True

        return False



    def _find_available_workers(self):
        '''returns all the available workers in the system'''
        return [worker for worker in self.workers if worker.available]



    def _calculate_reward(self, prev_time):
        '''number of jobs in the system multiplied by the time
        that has passed since the previous scheduling event compleiton.
        minimizing this quantity is equivalent to minimizing the
        average job completion time, by Little's Law (see Decima paper)
        '''
        reward = 0.
        for job_id in self.active_job_ids:
            job = self.jobs[job_id]
            start = max(job.t_arrival, prev_time)
            end = min(job.t_completed, self.wall_time)
            reward -= (end - start)
        return reward * self.REWARD_SCALE