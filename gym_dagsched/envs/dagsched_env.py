import numpy as np

from .state import State
from ..entities.timeline import JobArrival, TaskCompletion, WorkerArrival



class DagSchedEnv:

    # multiplied with reward to control its magnitude
    REWARD_SCALE = 1e-5

    # expected time to move a worker between jobs
    # (mean of exponential distribution)
    MOVING_COST = 2000.


    def __init__(self, rank):
        self.rank = rank
        self.state = State()


    @property
    def all_jobs_complete(self):
        '''whether or not all the jobs in the system
        have been completed
        '''
        return self.n_completed_jobs == self.n_job_arrivals


    @property
    def n_completed_jobs(self):
        return len(self.completed_job_ids)



    @property
    def n_active_jobs(self):
        return len(self.active_job_ids)



    @property
    def n_seen_jobs(self):
        return self.n_completed_jobs + self.n_active_jobs



    def n_ops_per_job(self):
        return [len(self.jobs[j].ops) for j in iter(self.active_job_ids)]




    ## OpenAI Gym style interface - reset & step

    def reset(self, initial_timeline, workers, max_wall_time):
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
        self.num_workers = len(workers)

        self.max_wall_time = max_wall_time

        # wall clock time, keeps increasing throughout
        # the simulation
        self.wall_time = 0.

        # dict which maps job id to job object
        # for each job that has arrived into the
        # system
        self.jobs = {}

        # list of ids of all active jobs
        self.active_job_ids = set()

        # list of ids of all completed jobs
        self.completed_job_ids = set()

        # operations in the system which are ready
        # to be executed by a worker because their
        # dependencies are satisfied
        self.ready_ops = set()

        self.executor_interval_map = \
            self._make_executor_interval_map()

        self.state.reset(self.num_workers)

        self.selected_ops = set()

        self.done = False

        # load all initial jobs into the system
        # by stepping through the timeline
        while not self.timeline.empty:
            wall_time, event = self.timeline.peek()
            if wall_time > 0:
                break
            self.timeline.pop()
            self._process_scheduling_event(event)

        self.schedulable_ops = self.ready_ops
        return self._observe(True)



    def step(self, action):
        assert not self.done

        print('step',
              np.round(self.wall_time*1e-3, 1),
              self.state.get_source(), 
              self.state.num_uncommitted_source_workers(), 
              flush=True)

        self._take_action(action)

        self.schedulable_ops = self._find_schedulable_ops()
        
        if not self._is_commitment_round_complete():
            # current commitment round is not over yet,
            # so consult the agent again
            return self._observe(False), 0, False
            
        # commitment round has completed, i.e.
        # all the workers at the current source
        # have somewhere to go
        self._fulfill_source_commitments()

        self.selected_ops.clear()
        self.state.clear_worker_source()

        reward, self.done = self._step_through_timeline()

        # if the episode isn't done, then start a new commitment 
        # round at the current worker source
        return self._observe(True), \
               reward, \
               self.done



    def _take_action(self, action):
        (job_id, op_id), num_workers = action
        print('action:', (job_id, op_id), num_workers)

        assert num_workers <= self.state.num_uncommitted_source_workers()

        assert job_id in self.active_job_ids
        job = self.jobs[job_id]

        assert op_id < len(job.ops)
        op = job.ops[op_id]

        assert op in self.schedulable_ops

        num_workers_adjusted = \
            self.adjust_num_workers(num_workers, op)

        # commit `num_workers` workers from the current worker
        # source to the op with id (job_id, op_id)
        self.add_commitment(num_workers_adjusted, op)

        if self.check_op_saturated(op):
            self._process_op_saturation(op)

        # mark op as selected so that it doesn't get
        # selected again during this commitment round
        self.selected_ops.add(op)



    def _step_through_timeline(self):
        t_prev = self.wall_time

        schedulable_ops = set()

        while not self.timeline.empty:
            self.wall_time, event = self.timeline.pop()
            self._process_scheduling_event(event)

            if self.state.all_source_workers_committed():
                continue

            self.ready_ops = self._find_ready_ops()
            if len(self.ready_ops) > 0:
                schedulable_ops = self._find_schedulable_ops()
                if len(schedulable_ops) > 0:
                    break

            print('no new round', flush=True)
            self._move_free_uncommitted_source_workers()

            self.state.clear_worker_source()

        reward = self._calculate_reward(t_prev)

        done = self.all_jobs_complete or \
               self.wall_time >= self.max_wall_time

        if not done:
            assert len(schedulable_ops) > 0 and \
                   not self.state.all_source_workers_committed()
            print('starting new commitment round', flush=True)
        else:
            print(f'DONE AT {self.wall_time/1e3:.1f} s', 
                  flush=True)

        self.schedulable_ops = schedulable_ops

        return reward, done



    def _print_job(self, job):
        print('job id:', job.id_)
        print('frontier ops:', 
              [op.id_ for op in iter(job.frontier_ops)])
        print('edges:', list(job.dag.edges))
        print('[op_id] [saturated] [completed]:')
        for op in job.ops:
            print(op.id_, op.saturated, op.completed)
        print()



    ## Observations

    def _observe(self, new_commitment_round):
        if not self.done:
            assert len(self.schedulable_ops) > 0

        n_source_workers = \
            self.state.num_uncommitted_source_workers()

        source_job_id = self.state.source_job()

        active_jobs = [self.jobs[job_id] 
                       for job_id in iter(self.active_job_ids)]

        return new_commitment_round, \
               n_source_workers, \
               source_job_id, \
               self.schedulable_ops, \
               active_jobs, \
               self.wall_time




    ## Scheduling events

    def _process_scheduling_event(self, event):
        if isinstance(event, JobArrival):
            self._process_job_arrival(event.job)
        elif isinstance(event, WorkerArrival):
            self._process_worker_arrival(event.worker, event.op)
        elif isinstance(event, TaskCompletion):
            self._process_task_completion(event.op, event.task)
        else:
            raise Exception('invalid event')




    ## Job arrivals

    def _process_job_arrival(self, job):
        print('job arrival', np.round(self.wall_time*1e-3, 1))
        self._print_job(job)

        self.jobs[job.id_] = job
        self.active_job_ids.add(job.id_)
        self.state.add_job(job.id_)
        [self.state.add_op(job.id_, op.id_) for op in job.ops]

        src_ops = job.initialize_frontier()
        self.ready_ops |= src_ops

        if self.state.general_pool_has_workers():
            # if there are any workers that don't
            # belong to any job, then give the 
            # agent a chance to assign them to this 
            # new job by starting a new commitment 
            # round at the 'general' pool
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
        print('worker arrived to', (op.job_id, op.id_))
        job = self.jobs[op.job_id]

        job.remove_moving_worker()
        job.add_local_worker(worker)

        if op.num_remaining_tasks == 0:
            # either the job has completed or the op has 
            # become saturated by the time of this arrival, 
            # so try to greedily find a backup operation 
            # for the worker
            self._try_backup_schedule(worker)
        elif op not in job.frontier_ops:
            # op's parents haven't completed yet
            self._process_worker_at_unready_op(worker, op)
        else:
            # the op is runnable, as anticipated.
            self.state.mark_worker_present(worker.id_, (job.id_, op.id_))
            self._work_on_op(worker, op)
        

    

    ## Task completions

    def _push_task_completion_event(self, op, task):
        '''pushes a single task completion event to the timeline'''
        worker = self.workers[task.worker_id]

        num_local_workers = \
            len(self.jobs[op.job_id].local_workers)

        duration = \
            op.sample_task_duration(task, 
                                    worker, 
                                    num_local_workers, 
                                    self.executor_interval_map)

        t_completion = task.t_accepted + duration
        op.most_recent_duration = duration

        event = TaskCompletion(op, task)
        self.timeline.push(t_completion, event)



    def _process_task_completion(self, op, task):
        '''performs some bookkeeping when a task completes'''
        print('task completion', (op.job_id, op.id_))

        worker = self.workers[task.worker_id]

        job = self.jobs[op.job_id]
        job.add_task_completion(op, 
                                task, 
                                worker, 
                                self.wall_time)
        
        if op.num_remaining_tasks > 0:
            # reassign the worker to keep working on this operation
            # if there is more work to do
            self._work_on_op(worker, op, reassign=True)
            return

        did_job_frontier_change = False

        if op.completed:
            did_job_frontier_change = \
                self._process_op_completion(op)

        if job.completed:
            self._process_job_completion(job)

        # worker may have somewhere to be moved
        had_commitment = \
            self._move_worker(worker, 
                              op, 
                              did_job_frontier_change)

        # worker source may need to be updated
        self._update_worker_source(op, 
                                   had_commitment, 
                                   did_job_frontier_change)




    ## Other helper functions

    def _find_schedulable_ops(self):
        schedulable_ops = self.ready_ops - self.selected_ops

        src_job_id = self.state.source_job()

        for job_id in iter(self.active_job_ids):
            job = self.jobs[job_id]

            total_worker_count = job.total_worker_count
            if job_id == src_job_id:
                num_source_workers = \
                    self.state.num_uncommitted_source_workers()
                total_worker_count -= num_source_workers

            if total_worker_count >= self.num_workers:
                print('SATURATED JOB', job.id_)
                schedulable_ops -= job.active_ops

        return schedulable_ops



    def _find_ready_ops(self, job_ids=None):
        '''returns a union of schedulable operations
        over all the jobs specified in `job_ids`. If
        no job ids are provided, then all active
        jobs are searched.
        '''
        if not job_ids:
            job_ids = list(self.active_job_ids)

        ready_ops = set(
            op
            for job_id in job_ids 
            for op in iter(self.jobs[job_id].active_ops)
            if self._check_op_schedulable(op)
        )

        return ready_ops



    def _check_op_schedulable(self, op):
        '''an operation is schedulable if 
        - it is unsaturated, and
        - all of its parent operations are saturated
        '''
        if self.check_op_saturated(op):
            return False

        job = self.jobs[op.job_id]
        for parent_op in job.parent_ops(op):
            if not self.check_op_saturated(parent_op):
                return False

        return True



    def add_commitment(self, num_workers, op):
        '''commits `num_workers` workers from the current
        worker source to `op`
        '''
        print('add commitment', (op.job_id, op.id_), num_workers)

        self.state.add_commitment(num_workers, (op.job_id, op.id_))
        
        job_id_src, _ = self.state.get_source()
        if job_id_src != op.job_id:
            job = self.jobs[op.job_id]
            job.add_commitments(num_workers)



    def adjust_num_workers(self, num_workers, op):
        '''truncates the numer of worker assigned
        to `op` to the op's demand, if it's larger
        '''
        worker_demand = self.get_worker_demand(op)

        num_workers_adjusted = min(num_workers, worker_demand)
        assert num_workers_adjusted > 0

        print('num_workers adjustment:',
             f'{num_workers} -> {num_workers_adjusted}')

        return num_workers_adjusted



    def get_worker_demand(self, op):
        '''an operation's worker demand is the
        number of workers that it can accept
        in addition to the workers currently
        working on, committed to, and moving to 
        the operation. 
        Note: demand can be negative if more 
        resources were assigned to the operation 
        than needed.
        '''
        job_id, op_id = op.job_id, op.id_

        num_workers_moving = \
            self.state.num_workers_moving_to_op(job_id, op_id)
            
        num_commitments = \
            self.state.num_commitments_to_op(job_id, op_id)

        demand = op.num_remaining_tasks - num_workers_moving - num_commitments

        return demand



    def check_op_saturated(self, op):
        '''an operation is saturated if it
        doesn't need any more workers.
        '''
        return self.get_worker_demand(op) <= 0



    def _is_commitment_round_complete(self):
        '''a round of commitments is complete
        if either 
        - all the workers at the source
        were committed somewhere, or 
        - there are no more operations to schedule.
        '''
        return self.state.all_source_workers_committed() or \
               len(self.schedulable_ops) == 0



    # def _should_start_new_commitment_round(self):
    #     '''start a new commitment round at the current 
    #     source if 
    #     - it contains uncommitted workers, and 
    #     - there are schedulable operations in the system
    #     '''
    #     return not self.state.all_source_workers_committed() and \
    #         len(self.ready_ops) > 0
            


    def _work_on_op(self, worker, op, reassign=False):
        '''starts work on another one of `op`'s 
        tasks, assuming there are still tasks 
        remaining and the worker is local to the 
        operation
        '''
        assert op is not None
        assert op.num_remaining_tasks > 0
        assert worker.is_at_job(op.job_id)
        assert worker.available

        job = self.jobs[op.job_id]
        task = job.assign_worker(worker, op, self.wall_time)

        # if op in self.ready_ops and self.check_op_saturated(op):
        #     self._process_op_saturation(op)

        self._push_task_completion_event(op, task)



    def _send_worker(self, worker, op):
        '''sends a `worker` to `op`, assuming
        that the worker is currently at a
        different job
        '''
        assert op is not None
        assert worker.available
        assert worker.job_id != op.job_id

        if worker.job_id is not None:
            old_job = self.jobs[worker.job_id]
            old_job.remove_local_worker(worker)

        # if op in self.ready_ops and self.check_op_saturated(op):
        #     self._process_op_saturation(op)

        job = self.jobs[op.job_id]
        job.add_moving_worker()

        self._push_worker_arrival_event(worker, op)
            


    def _process_op_saturation(self, op):
        print('op saturation', (op.job_id, op.id_))
        assert self.check_op_saturated(op)
        assert op in self.ready_ops

        self.ready_ops.remove(op)

        job = self.jobs[op.job_id]

        # this saturation may have unlocked new 
        # operations within the job dag
        self.ready_ops |= set(
            child_op
            for child_op in job.children_ops(op) 
            if self._check_op_schedulable(child_op)
        )



    def _process_op_unsaturation(self, op):
        print('op unsaturation', (op.job_id, op.id_))
        assert not self.check_op_saturated(op)
        assert op not in self.ready_ops

        job = self.jobs[op.job_id]

        self.ready_ops -= set(job.ops)
        
        self.ready_ops |= self._find_ready_ops([op.job_id])
        


    def _move_worker(self, worker, op, did_job_frontier_change):
        '''called upon a task completion.
        
        if the op has a commitment, then fulfill it, unless the worker
        is not needed there anymore. In that case, try to find a backup op
        to work on.
        
        Otherwise, if `op` became saturated and unlocked new ops within the 
        job dag, then move the worker to the job's worker pool so that it can 
        be assigned to the new ops
        '''
        commitment = self.state.peek_commitment((op.job_id, op.id_))
        had_commitment = (commitment is not None)

        if commitment is not None:
            # op has at least one commitment, so fulfill it
            job_id_committed, op_id_committed = commitment
            op_committed = self.jobs[job_id_committed].ops[op_id_committed]
            if op_committed.num_remaining_tasks > 0:
                self._fulfill_commitment(worker, op_committed)
            else:
                self._try_backup_schedule(worker, commitment)
        elif did_job_frontier_change:
            # no commitment, but frontier changed
            self.state.move_worker_to_pool(worker.id_, (op.job_id, None))
        
        return had_commitment



    def _update_worker_source(self, op, had_commitment, did_job_frontier_change):
        '''called upon a task completion.
        
        if any new operations were unlocked within this 
        job upon the task completion, then start a new 
        commitment round at this job's pool so that free
        workers can be assigned to the new operations.

        Otherwise, if the worker has nowhere to go, then
        start a new commitment round at this operation's
        pool to give it somewhere to go.
        '''
        if did_job_frontier_change:
            self.state.update_worker_source(op.job_id)
        elif not had_commitment:
            self.state.update_worker_source(op.job_id, op.id_)



    def _process_op_completion(self, op):
        '''performs some bookkeeping when an operation completes'''
        print('op completion', (op.job_id, op.id_))
        # assert op not in self.ready_ops
        job = self.jobs[op.job_id]
        frontier_changed = job.add_op_completion(op)
        print('frontier_changed:', frontier_changed)
        return frontier_changed
        

    
    def _process_job_completion(self, job):
        '''performs some bookkeeping when a job completes'''
        assert job.id_ in self.jobs
        print('job completion', job.id_)
        
        self.active_job_ids.remove(job.id_)
        self.completed_job_ids.add(job.id_)
        job.t_completed = self.wall_time



    def _fulfill_commitment(self, worker, op):
        assert op.num_remaining_tasks > 0
        print('fulfilling commitment to', (op.job_id, op.id_))

        job = self.jobs[op.job_id]

        if worker.job_id != op.job_id:
            job.remove_commitment()

        self.state.remove_commitment(worker.id_, (op.job_id, op.id_))

        if not worker.is_at_job(op.job_id):
            # worker isn't local to the job, 
            # so send it over.
            self.state.move_worker_to_pool(
                worker.id_, (op.job_id, op.id_), send=True)
            self._send_worker(worker, op)
            return

        # worker is at the job

        self.state.move_worker_to_pool(
            worker.id_, (op.job_id, op.id_))

        if op in job.frontier_ops:
            # op's dependencies are satisfied, so 
            # start working on it.
            self._work_on_op(worker, op)
        else:
            # dependencies not satisfied; op not ready.
            self._process_worker_at_unready_op(worker, op)



    def _process_worker_at_unready_op(self, worker, op):
        # this op's parents are saturated but have not
        # completed, so we can't actually start working
        # on the op. Move the worker to the
        # job pool instead.

        self.state.move_worker_to_pool(worker.id_, (op.job_id, None))

        # if op not in self.ready_ops and \
        #     not self.check_op_saturated(op):
        #     # we may have stopped scheduling this
        #     # op because it became saturated,
        #     # but it is no longer saturated
        #     # so we need to start scheduling it
        #     # again
        #     self._process_op_unsaturation(op)


    def _get_free_source_workers(self):
        source_worker_ids = self.state.get_source_workers()

        free_worker_ids = \
            set((worker_id
                 for worker_id in iter(source_worker_ids)
                 if self.workers[worker_id].available))

        return free_worker_ids
        


    def _fulfill_source_commitments(self):
        '''called at the end of a commitment round'''
        print('fulfilling source commitments')

        # some of the source workers may not be
        # free right now; find the ones that are.
        free_worker_ids = self._get_free_source_workers()

        commitments = self.state.get_source_commitments()

        # for job_id, op_id, num_workers in commitments:
        for (job_id, op_id), num_workers in commitments.items():
            assert num_workers > 0
            while num_workers > 0 and len(free_worker_ids) > 0:
                worker_id = free_worker_ids.pop()
                worker = self.workers[worker_id]
                op = self.jobs[job_id].ops[op_id]
                self._fulfill_commitment(worker, op)
                num_workers -= 1

        if len(free_worker_ids) > 0:
            self._move_free_uncommitted_source_workers(free_worker_ids)



    def _move_free_uncommitted_source_workers(self, free_worker_ids=None):
        if free_worker_ids is None:
            free_worker_ids = self._get_free_source_workers()

        job_id, op_id = self.state.get_source()

        if job_id is None or \
           (op_id is None and not self.jobs[job_id].saturated):
            # source is either the general pool or an unsaturated job's pool
            return

        # source is either a saturated job's pool or an op pool
        job = self.jobs[job_id]
        dst_pool_key = (None, None) if job.saturated else (job_id, None)
        for worker_id in iter(free_worker_ids):
            self.state.move_worker_to_pool(worker_id, dst_pool_key)



    def _try_backup_schedule(self, worker, commitment=None):
        print('trying backup; old commitment =', commitment)

        if commitment:
            self.state.remove_commitment(worker.id_, commitment)

        backup_op = self._find_backup_op(worker)

        if backup_op:
            self._reroute_worker(worker, backup_op)
            return

        # no backup op found, so move worker to job or general
        # pool depending on whether or not the worker's job 
        # is completed
        job = self.jobs[worker.job_id]
        dst_pool_key = (None, None) if job.completed else (worker.job_id, None)
        self.state.move_worker_to_pool(worker.id_, dst_pool_key)



    def _reroute_worker(self, worker, op):
        print('rerouting worker to', (op.job_id, op.id_))
        assert op.num_remaining_tasks > 0

        if not worker.is_at_job(op.job_id):
            self.state.move_worker_to_pool(
                worker.id_, (op.job_id, op.id_), send=True)
            self._send_worker(worker, op)
            return

        self.state.move_worker_to_pool(
            worker.id_, (op.job_id, op.id_))

        if op in self.jobs[op.job_id].frontier_ops:
            # op's dependencies are satisfied, so 
            # start working on it.
            self._work_on_op(worker, op)
        else:
            # dependencies not satisfied; op not ready.
            self._process_worker_at_unready_op(worker, op)



    def _find_backup_op(self, worker):
        local_ops = self._find_ready_ops([worker.job_id])

        if len(local_ops) > 0:
            return local_ops.pop()

        other_job_ids = \
            [job_id for job_id in iter(self.active_job_ids)
             if job_id != worker.job_id]
        other_ops = \
            self._find_ready_ops(other_job_ids)

        if len(other_ops) > 0:
            return other_ops.pop()

        return None



    def _calculate_reward(self, prev_time):
        '''number of jobs in the system multiplied by the time
        that has passed since the previous scheduling event compleiton.
        minimizing this quantity is equivalent to minimizing the
        average job completion time, by Little's Law (see Decima paper)
        '''
        reward = 0.
        for job_id in iter(self.active_job_ids):
            job = self.jobs[job_id]
            start = max(job.t_arrival, prev_time)
            end = min(job.t_completed, self.wall_time)
            reward -= (end - start)
        return reward * self.REWARD_SCALE


    
    def _make_executor_interval_map(self):
        executor_interval_map = {}

        executor_data_point = [5, 10, 20, 40, 50, 60, 80, 100]
        exec_cap = self.num_workers

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