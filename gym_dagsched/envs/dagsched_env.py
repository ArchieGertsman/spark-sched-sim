from collections import defaultdict
from copy import deepcopy as dcp
from copy import copy as cp
from time import time
from sys import getsizeof as sizeof
from enum import Enum, auto

import numpy as np
import torch
import networkx as nx

from .state import State, WorkerState
from ..entities.timeline import JobArrival, TaskCompletion, WorkerArrival
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
    REWARD_SCALE = 1e-5

    # expected time to move a worker between jobs
    # (mean of exponential distribution)
    MOVING_COST = 2000.


    def __init__(self, rank):
        self.rank = rank
        self.state = State()
        self.t_step = 0


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



    # @property
    # def are_actions_available(self):
    #     '''checks if there are any valid actions that can be
    #     taken by the scheduling agent.
    #     '''
    #     return self.n_avail_workers > 0 and len(self.frontier_ops) > 0



    def n_ops_per_job(self):
        return [len(self.jobs[j].ops) for j in self.active_job_ids]




    ## OpenAI Gym style interface - reset & step

    def reset(self, initial_timeline, workers, x_ptrs):
        '''resets the simulation. should be called before
        each run (including first). all state data is found here.
        '''

        # a priority queue containing scheduling 
        # events indexed by wall time of occurance
        self.timeline = initial_timeline
        self.n_job_arrivals = len(initial_timeline.pq)
        
        # list of worker objects which are to be scheduled
        # to complete tasks within the simulation
        self.workers = workers
        self.n_workers = len(workers)

        # wall clock time, keeps increasing throughout
        # the simulation
        self.wall_time = 0.

        # set of job objects within the system
        self.jobs = {}

        # list of ids of all active jobs
        self.active_job_ids = []

        # list of ids of all completed jobs
        self.completed_job_ids = []

        # operations in the system which are ready
        # to be executed by a worker because their
        # dependencies are satisfied
        self.frontier_ops = set()

        self.x_ptrs = x_ptrs

        self.max_ops = np.max([len(e.job.ops) for _,_,e in initial_timeline.pq])

        self.executor_interval_map = self._get_executor_interval_map()

        self.t_step = 0

        self.state.reset()

        self.selected_ops = set()

        self.done = False



    def step(self, job_id, op_id, n_workers):
        if self.done:
            return 0, True

        if job_id is not None:
            op = self.jobs[job_id].ops[op_id]
            assert op in (self.frontier_ops - self.selected_ops)
            # mark op as selected so that it doesn't get
            # selected again during this commitment round
            self.selected_ops.add(op)
            self.state.add_commitment(n_workers, job_id, op_id)
        
        if not self.state.all_source_workers_committed:
            # current commitment round is not over yet,
            # so consult the agent again
            return 0, False
            
        # commitment round has completed, i.e.
        # all the workers at the current source
        # have somewhere to go
        self.selected_ops.clear()
        self._fulfill_source_commitments()

        t_prev = self.wall_time

        while not self.timeline.empty and self.state.all_source_workers_committed:
            self.wall_time, event = self.timeline.pop()
            self._process_scheduling_event(event)

        reward = self._calculate_reward(t_prev)
        self.done = self.timeline.empty

        # if the episode isn't done, then start a new commitment 
        # round at the current worker source

        return reward, self.done




    ## Action masks

    def construct_op_msk(self):
        op_msk = torch.zeros((self.n_job_arrivals, self.max_ops), dtype=torch.bool)
        for j in self.active_job_ids:
            job = self.jobs[j]
            for i,op in enumerate(job.ops):
                if op in (self.frontier_ops - self.selected_ops):
                    op_msk[j, i] = 1
        return op_msk



    def construct_prlvl_msk(self):
        prlvl_msk = torch.zeros((self.n_job_arrivals, self.n_workers+1), dtype=torch.bool)
        # prlvl_msk[:, 1:len(self.avail_worker_ids)+1] = 1
        prlvl_msk[:, 1:] = 1
        return prlvl_msk




    ## Scheduling events

    def _process_scheduling_event(self, event):
        if isinstance(event, JobArrival):
            self._process_job_arrival(event.job)
        elif isinstance(event, WorkerArrival):
            self._process_worker_arrival(event.worker, event.op)
        elif isinstance(event, TaskCompletion):
            self._process_task_completion(event.op, event.task)
        else:
            print('invalid event')
            assert False




    ## Job arrivals

    def _process_job_arrival(self, job):
        job.x_ptr = self.x_ptrs[job.id_]
        self.jobs[job.id_] = job
        self.active_job_ids += [job.id_]
        self.state.add_job(job.id_)

        src_ops = job.find_src_ops()
        self.frontier_ops |= src_ops
        [self.state.add_op(job.id_, op.id_) for op in iter(src_ops)]

        if self.state.null_pool_has_workers:
            # if there are any workers that don't
            # belong to any job, then give the 
            # agent a chance to assign them to this 
            # new job by starting a new commitment 
            # round at the 'null' pool
            self.state.update_worker_source()
     



    ## Worker arrivals

    def _push_worker_arrival_event(self, worker, op):
        '''pushes the event of a worker arriving to a job
        to the timeline'''
        t_arrival = self.wall_time + self.MOVING_COST
        event = WorkerArrival(worker, op)
        self.timeline.push(t_arrival, event)



    def _process_worker_arrival(self, worker, op):
        '''performs some bookkeeping when a worker arrives'''
        job = self.jobs[op.job_id]

        if job.is_complete:
            self._try_backup_schedule(worker)
            return
        
        job.add_local_worker(worker)

        if op.saturated:
            self.state.move_worker_to_job_pool(worker.id_)
            return

        # TODO: set false here but only set true in state class
        self.state.set_worker_moving(worker.id_, False)
        self._work_on_op(worker, op)


    

    ## Task completions

    def _push_task_completion_event(self, op, task):
        '''pushes a single task completion event to the timeline'''
        worker = self.workers[task.worker_id]

        n_local_workers = len(self.jobs[op.job_id].local_workers)
        duration = op.sample_task_duration(
            task, 
            worker, 
            n_local_workers, 
            self.executor_interval_map)
        t_completion = task.t_accepted + duration

        event = TaskCompletion(op, task)
        self.timeline.push(t_completion, event)



    def _process_task_completion(self, op, task):
        '''performs some bookkeeping when a task completes'''
        worker = self.workers[task.worker_id]

        job = self.jobs[op.job_id]
        job.add_task_completion(op, task, worker, self.wall_time)
        
        if not op.saturated:
            # reassign the worker to keep working on this operation.
            self._work_on_op(worker, op)
        else:
            self._process_saturated_op(op, worker)




    ## Helper functions

    def _work_on_op(self, worker, op):
        assert op is not None
        assert not op.saturated
        assert worker.is_at_job(op.job_id)
        assert worker.available

        job = self.jobs[op.job_id]
        task = job.assign_worker(worker, op, self.wall_time)

        # op may have just become saturated
        # after this assignment
        if op.saturated:
            self.frontier_ops.remove(op)

        self._push_task_completion_event(op, task)



    def _send_worker(self, worker, op):
        assert op is not None
        assert worker.available

        if worker.job_id is not None:
            old_job = self.jobs[worker.job_id]
            old_job.remove_local_worker(worker.id_)

        self._push_worker_arrival_event(worker, op)
            


    def _process_saturated_op(self, op, worker):
        frontier_changed = False

        if op.is_complete:
            # record whether or not the completion of this
            # operation unlocked new operations within the job
            frontier_changed = self._process_op_completion(op)

        job = self.jobs[op.job_id]
        if job.is_complete:
            self._process_job_completion(job)

        # see if the worker is committed to some next operation
        commitment = self.state.peek_commitment(op.job_id, op.id_)
        if commitment is not None:
            # op has at least one commitment, so fulfill it
            job_id_committed, op_id_committed = commitment
            op_committed = self.jobs[job_id_committed].ops[op_id_committed]
            self._fulfill_commitment(worker, op_committed)
        elif frontier_changed:
            # no commitment, but frontier changed
            self.state.move_worker_to_job_pool(worker.id_)

        # see if current worker source needs to be updated
        if frontier_changed:
            # if any new operations were unlocked within this 
            # job, then give the agent a chance to assign
            # them to free workers from this job's pool
            # by starting a new commitment round at this
            # job's pool
            self.state.update_worker_source(job.id_)
        elif commitment is None:
            # if no new operations were unlocked and
            # the worker has nowhere to go, then, necessarily,
            # none of the workers at this operation have
            # been committed anywhere. Then start a new
            # commitment round at this operation
            self.state.update_worker_source(op.job_id, op.id_)
        
        

    def _process_op_completion(self, op):
        '''performs some bookkeeping when an operation completes'''
        self.state.mark_op_completed(op.job_id, op.id_)

        job = self.jobs[op.job_id]
        job.add_op_completion()

        # add stage's decendents to the frontier, if their
        # other dependencies are also satisfied
        new_ops = job.find_new_frontiers(op)
        self.frontier_ops |= new_ops
        [self.state.add_op(job.id_, op.id_) for op in iter(new_ops)]

        frontier_changed = (len(new_ops) > 0)
        return frontier_changed
        

        
    def _process_job_completion(self, job):
        '''performs some bookkeeping when a job completes'''
        assert job.id_ in self.jobs

        self.state.mark_job_completed(job.id_)
        
        self.active_job_ids.remove(job.id_)
        self.completed_job_ids += [job.id_]
        job.t_completed = self.wall_time



    def _fulfill_commitment(self, worker, op):
        assert op is not None

        worker_is_present = \
            self.state.fulfill_commitment(worker.id_, op.job_id, op.id_)

        if worker_is_present:
            self._work_on_op(worker, op)
        else:
            self._send_worker(worker, op)



    def _fulfill_source_commitments(self):
        free_workers = set((
            worker_id 
            for worker_id in self.state.get_source_workers() 
            if self.workers[worker_id].available
        ))

        commitments = self.state.get_source_commitments()

        for job_id, op_id, n_workers in commitments:
            assert n_workers > 0
            while n_workers > 0 and len(free_workers) > 0:
                worker = free_workers.pop()
                op = self.jobs[job_id].ops[op_id]
                self._fulfill_commitment(worker, op)
                n_workers -= 1



    def _try_backup_schedule(self, worker):
        backup_op = self._find_backup_op(worker)
        if backup_op is not None:
            self._reroute_worker(worker, backup_op)
        else:
            self.state.move_worker_to_job_pool(worker.id_)



    def _reroute_worker(self, worker, op):
        is_worker_present = \
            self.state.reroute_worker(worker.id_, op.job_id, op.id_)

        if is_worker_present:
            self._work_on_op(worker, op)
        else:
            self._send_worker(worker, op)



    def _find_backup_op(self, worker):
        if len(self.frontier_ops) == 0:
            return None

        backup_op = None
        for op in iter(self.frontier_ops):
            backup_op = op
            if op.job_id == worker.job_id:
                break

        return backup_op



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


    
    def _get_executor_interval_map(self):
        executor_interval_map = {}

        executor_data_point = [5, 10, 20, 40, 50, 60, 80, 100]
        exec_cap = self.n_workers

        # get the left most map
        for e in range(executor_data_point[0] + 1):
            executor_interval_map[e] = \
                (executor_data_point[0],
                 executor_data_point[0])

        # get the center map
        for i in range(len(executor_data_point) - 1):
            for e in range(executor_data_point[i] + 1,
                            executor_data_point[i + 1]):
                executor_interval_map[e] = \
                    (executor_data_point[i],
                     executor_data_point[i + 1])
            # at the data point
            e = executor_data_point[i + 1]
            executor_interval_map[e] = \
                (executor_data_point[i + 1],
                 executor_data_point[i + 1])

        # get the residual map
        if exec_cap > executor_data_point[-1]:
            for e in range(executor_data_point[-1] + 1,
                            exec_cap + 1):
                executor_interval_map[e] = \
                    (executor_data_point[-1],
                     executor_data_point[-1])

        return executor_interval_map