from copy import deepcopy as dcp

import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Batch

from ..utils.timeline import JobArrival, TaskCompletion


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
        return self.n_completed_jobs == len(self.jobs)


    @property 
    def n_processing_jobs(self):
        '''number of jobs in the system which have not
        been completed
        '''
        return len(self.jobs) - self.n_completed_jobs



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
        self.jobs = []

        # number of jobs that have completed
        self.n_completed_jobs = 0

        # operations in the system which are ready
        # to be executed by a worker because their
        # dependencies are satisfied
        self.frontier_ops = set()

        # operations in the system which have not 
        # yet completed but have all the resources
        # they need assigned to them
        self.saturated_ops = set()



    def step(self, op, n_workers):
        '''steps onto the next scheduling event on the timeline, 
        which can be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            even though neither (1) nor (2) have occurred, so 
            the policy should consider taking one of them
        '''

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

        reward = self._calculate_reward(t)

        self._process_scheduling_event(t, event)

        return self._observe(), reward, False



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
        dags = []
        op_msk = []
        for job in self.jobs:
            # convert this job into a PyG graph
            job.update_feature_vectors(self.workers)
            dags += [from_networkx(job.dag)]
            # append this job's operations to the mask
            for op in job.ops:
                op_msk += [1] if op in self.frontier_ops else [0]

        if len(dags) == 0:
            return None

        dag_batch = Batch.from_data_list(dags)
        op_msk = torch.tensor(op_msk)
        prlvl_msk = torch.ones((len(dags), len(self.workers)))
        
        return dag_batch, op_msk, prlvl_msk



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



    def _process_scheduling_event(self, t, event):
        '''handles a scheduling event, which can be a job arrival,
        a task completion, or a nudge
        '''
        # update the current wall time
        self.wall_time = t

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
        self.jobs += [job]
        src_ops = job.find_src_ops()
        self.frontier_ops |= src_ops



    def _process_task_completion(self, task):
        '''performs some bookkeeping when a task completes'''
        op = self.jobs[task.job_id].ops[task.op_id]
        op.add_task_completion(task, self.wall_time)

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
        self.n_completed_jobs += 1
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

        task = op.add_worker(
            worker, 
            self.wall_time, 
            moving_cost)

        return task


    
    def _job_moving_cost(self, old_job_id, new_job_id):
        '''calculates a moving cost between jobs, which is
        either zero if the jobs are the same, or a sample
        from a fixed exponential distribution
        '''
        return 0. if new_job_id == old_job_id \
            else np.random.exponential(self.MOVING_COST)



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



    def _calculate_reward(self, t):
        '''number of jobs in the system multiplied by the time
        that has passed since the previous scheduling event compleiton.
        minimizing this quantity is equivalent to minimizing the
        average job completion time, by Little's Law (see Decima paper)
        '''
        reward = -(t - self.wall_time) * self.n_processing_jobs
        return reward * self.REWARD_SCALE