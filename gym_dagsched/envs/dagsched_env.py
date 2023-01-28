import numpy as np


from .worker_assignment_state import WorkerAssignmentState, GENERAL_POOL_KEY
from ..data_generation.tpch_datagen import TPCHDataGen
from ..entities.timeline import JobArrival, TaskCompletion, WorkerArrival



class DagSchedEnv:

    # multiplied with reward to control its magnitude
    REWARD_SCALE = 1e-5

    # time to move a worker between jobs
    MOVING_COST = 2000.


    def __init__(self):
        self._datagen = TPCHDataGen()
        self._state = WorkerAssignmentState()


    @property
    def all_jobs_complete(self):
        return self.num_completed_jobs == self.num_job_arrivals


    @property
    def num_completed_jobs(self):
        return len(self.completed_job_ids)




    ## OpenAI Gym style interface - reset & step

    def reset(self, 
              num_init_jobs, 
              num_job_arrivals, 
              job_arrival_rate, 
              num_workers,
              max_wall_time):

        self.timeline = \
            self._datagen.initial_timeline(num_init_jobs, 
                                           num_job_arrivals, 
                                           job_arrival_rate)

        self.workers = self._datagen.workers(num_workers)

        # a priority queue containing scheduling 
        # events indexed by wall time of occurance
        self.num_job_arrivals = len(self.timeline.pq)
        
        # list of worker objects which are to be scheduled
        # to complete tasks within the simulation
        self.num_workers = len(self.workers)

        self.max_wall_time = max_wall_time

        # wall clock time, keeps increasing throughout
        # the simulation
        self.wall_time = 0.

        # dict which maps job id to job object
        # for each job that has arrived into the
        # system
        self.jobs = {i: e.job for i, (*_, e) in enumerate(self.timeline.pq)}

        # list of ids of all active jobs
        self.active_job_ids = set()

        # list of ids of all completed jobs
        self.completed_job_ids = set()

        self.executor_interval_map = \
            self._make_executor_interval_map()

        self._state.reset(self.num_workers)

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

        schedulable_ops = self._find_schedulable_ops()
        obs = self._observe(schedulable_ops, True)
        return obs



    def step(self, action):
        assert not self.done

        print('step',
              np.round(self.wall_time*1e-3, 1),
              self._state.get_source(), 
              self._state.num_workers_to_schedule(), 
              flush=True)

        schedulable_ops = self._take_action(action)

        if self._state.num_workers_to_schedule() > 0 and \
           len(schedulable_ops) > 0:
            # there are still scheduling decisions to be made, 
            # so consult the agent again
            obs = self._observe(schedulable_ops, False)
            return obs, 0, False
            
        # scheduling round has completed
        num_uncommitted_workers = self._state.num_workers_to_schedule()
        if num_uncommitted_workers > 0:
            self._state.add_commitment(num_uncommitted_workers, 
                                       GENERAL_POOL_KEY)

        self._fulfill_commitments_from_source()
        self._state.clear_worker_source()
        self.selected_ops.clear()

        # save old attributes for computing reward
        old_wall_time = self.wall_time
        old_active_job_ids = self.active_job_ids.copy()

        schedulable_ops = self._run_simulation()

        reward = self._calculate_reward(old_wall_time,
                                        old_active_job_ids)
        self.done = self.all_jobs_complete or \
                    self.wall_time >= self.max_wall_time

        if not self.done:
            print('starting new scheduling round', flush=True)
            assert self._state.num_workers_to_schedule() > 0 and \
                   len(schedulable_ops) > 0
        else:
            print(f'done at {self.wall_time*1e-3:.1f}s', flush=True)

        # if the episode isn't done, then start a new scheduling 
        # round at the current worker source
        obs = self._observe(schedulable_ops, True)
        return obs, reward, self.done



    def _take_action(self, action):
        if action is None:
            num_source_workers = self._state.num_workers_to_schedule()
            self._state.add_commitment(num_source_workers,
                                       GENERAL_POOL_KEY)
            return self.schedulable_ops

        (job_id, op_id), num_workers = action
        print('action:', (job_id, op_id), num_workers)

        assert num_workers <= \
               self._state.num_workers_to_schedule()

        assert job_id in self.active_job_ids
        job = self.jobs[job_id]

        assert op_id < len(job.ops)
        op = job.ops[op_id]

        assert op in self.schedulable_ops

        # agent may have requested more workers than
        # are actually needed
        num_workers = \
            self.adjust_num_workers(num_workers, op)

        print(f'committing {num_workers} workers '
              f'to {op.pool_key}')
        self._state.add_commitment(num_workers, 
                                   op.pool_key)

        # mark op as selected so that it doesn't get
        # selected again during this scheduling round
        self.selected_ops.add(op)

        # find remaining schedulable operations
        schedulable_ops = self._find_schedulable_ops()
        return schedulable_ops



    def _run_simulation(self):
        '''runs the simulation until either there are
        new scheduling decisions to be made, or it's done.
        '''
        assert not self.timeline.empty
        schedulable_ops = set()

        while not self.timeline.empty:
            self.wall_time, event = self.timeline.pop()
            self._process_scheduling_event(event)

            src = self._state.get_source()
            print(src, self._state._pools[src])

            if self._state.num_workers_to_schedule() == 0:
                continue

            schedulable_ops = self._find_schedulable_ops()
            if len(schedulable_ops) > 0:
                break

            print('no ops to schedule')
            self._move_free_uncommitted_source_workers()
            self._state.clear_worker_source()

        return schedulable_ops



    ## Observations

    def _observe(self, schedulable_ops, new_commitment_round):
        if not self.done:
            assert len(schedulable_ops) > 0

        # save the set of schedulable ops to later varify
        # that the agent selected one of them
        self.schedulable_ops = schedulable_ops

        active_jobs = {job_id: self.jobs[job_id] 
                       for job_id in iter(self.active_job_ids)}

        for job_id in self.active_job_ids:
            active_jobs[job_id].total_worker_count = \
                self._state.total_worker_count(job_id)

        print('gen pool:', (len(self._state._pools[GENERAL_POOL_KEY]), 
                            self._state._total_worker_count[None]),
              {job_id: self.jobs[job_id].total_worker_count 
               for job_id in self.active_job_ids})

        print('POOLS', self._state._pools)

        return new_commitment_round, \
               self._state.num_workers_to_schedule(), \
               self._state.source_job_id(), \
               schedulable_ops, \
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
        print(f'job {job.id_} arrived at t={self.wall_time*1e-3:.1f}s')
        print('dag edges:', list(job.dag.edges))

        # self.jobs[job.id_] = job
        self.active_job_ids.add(job.id_)
        self._state.add_job(job.id_)
        [self._state.add_op(*op.pool_key) for op in job.ops]

        job.init_frontier()

        if self._state.general_pool_has_workers():
            # if there are any workers that don't
            # belong to any job, then the agent might
            # want to schedule them to this job,
            # so start a new round at the general pool
            self._state.update_worker_source(GENERAL_POOL_KEY)
     



    ## Worker arrivals

    def _push_worker_arrival_event(self, worker, op):
        '''pushes the event of a worker arriving to a job
        to the timeline'''
        t_arrival = self.wall_time + self.MOVING_COST
        event = WorkerArrival(worker, op)
        self.timeline.push(t_arrival, event)



    def _process_worker_arrival(self, worker, op):
        '''performs some bookkeeping when a worker arrives'''
        print('worker arrived to', op.pool_key)
        job = self.jobs[op.job_id]

        job.add_local_worker(worker)

        self._state.count_worker_arrival(op.pool_key)

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
            self._state.move_worker_to_pool(worker.id_, op.pool_key)
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
        print('task completion', op.pool_key)

        worker = self.workers[task.worker_id]

        job = self.jobs[op.job_id]
        job.add_task_completion(op, 
                                task, 
                                worker, 
                                self.wall_time)
        
        if op.num_remaining_tasks > 0:
            # reassign the worker to keep working on this operation
            # if there is more work to do
            self._work_on_op(worker, op)
            return

        did_job_frontier_change = False

        if op.completed:
            did_job_frontier_change = \
                self._process_op_completion(op)

        if job.completed:
            self._process_job_completion(job)

        # worker may have somewhere to be moved
        had_commitment = \
            self._move_worker(worker.id_, 
                              op, 
                              did_job_frontier_change)

        # worker source may need to be updated
        self._update_worker_source(op, 
                                   had_commitment, 
                                   did_job_frontier_change)




    ## Other helper functions

    def _find_schedulable_ops(self, 
                              job_ids=None, 
                              source_job_id=None):
        '''An operation is schedulable if it is ready
        (see `_is_op_ready()`), it hasn't been selected
        in the current scheduling round, and its job
        is not saturated with workers (i.e. can gain
        more workers).
        
        returns a union of schedulable operations
        over all the jobs specified in `job_ids`. If
        no job ids are provided, then all active
        jobs are searched.
        '''
        if job_ids is None:
            job_ids = list(self.active_job_ids)

        if source_job_id is None:
            source_job_id = self._state.source_job_id()

        # filter out saturated jobs. The source job is
        # never considered saturated, because it is not
        # gaining any new workers during scheduling
        job_ids = [job_id for job_id in job_ids
                   if job_id == source_job_id or \
                      (self._state.total_worker_count(job_id) < \
                            self.num_workers)]

        schedulable_ops = \
            set(op
                for job_id in iter(job_ids)
                for op in iter(self.jobs[job_id].active_ops)
                if op not in self.selected_ops and \
                   self._is_op_ready(op))

        return schedulable_ops



    def _is_op_ready(self, op):
        '''an operation is ready if 
        - it is unsaturated, and
        - all of its parent operations are saturated
        '''
        if self._is_op_saturated(op):
            return False

        job = self.jobs[op.job_id]
        for parent_op in job.parent_ops(op):
            if not self._is_op_saturated(parent_op):
                return False

        return True



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
        num_workers_moving = \
            self._state.num_workers_moving_to_op(op.pool_key)
            
        num_commitments = \
            self._state.num_commitments_to_op(op.pool_key)

        demand = op.num_remaining_tasks - \
                 (num_workers_moving + num_commitments)

        return demand



    def _is_op_saturated(self, op):
        '''an operation is saturated if it
        doesn't need any more workers.
        '''
        return self.get_worker_demand(op) <= 0
            


    def _work_on_op(self, worker, op):
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

        self._push_worker_arrival_event(worker, op)
        


    def _move_worker(self, worker_id, op, did_job_frontier_change):
        '''called upon a task completion.
        
        if the op has a commitment, then fulfill it, unless the worker
        is not needed there anymore. In that case, try to find a backup op
        to work on.
        
        Otherwise, if `op` became saturated and unlocked new ops within the 
        job dag, then move the worker to the job's worker pool so that it can 
        be assigned to the new ops
        '''
        commitment_pool_key = \
            self._state.peek_commitment(op.pool_key)

        if commitment_pool_key is not None:
            self._fulfill_commitment(worker_id,
                                     commitment_pool_key)
            return True
    
        if did_job_frontier_change:
            self._state.move_worker_to_pool(worker_id, 
                                            op.job_pool_key)

        return False

        


    def _update_worker_source(self, 
                              op, 
                              had_commitment, 
                              did_job_frontier_change):
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
            self._state.update_worker_source(op.job_pool_key)
        elif not had_commitment:
            self._state.update_worker_source(op.pool_key)



    def _process_op_completion(self, op):
        '''performs some bookkeeping when an operation completes'''
        print('op completion', op.pool_key)
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



    def _fulfill_commitment(self, worker_id, dst_pool_key):
        print('fulfilling commitment to', dst_pool_key)

        src_pool_key = \
            self._state.remove_commitment(worker_id, dst_pool_key)

        if dst_pool_key == GENERAL_POOL_KEY:
            # this worker is free and isn't commited to
            # any actual operation
            self._move_free_uncommitted_source_workers(src_pool_key, [worker_id])
            return

        job_id, op_id = dst_pool_key
        op = self.jobs[job_id].ops[op_id]
        worker = self.workers[worker_id]

        if op.num_remaining_tasks == 0:
            # operation is saturated
            self._try_backup_schedule(worker)
            return

        if not worker.is_at_job(op.job_id):
            # worker isn't local to the job, 
            # so send it over.
            self._state.move_worker_to_pool(worker.id_, 
                                            op.pool_key, 
                                            send=True)
            self._send_worker(worker, op)
            return

        # worker is at the job
        self._state.move_worker_to_pool(worker.id_, op.pool_key)

        job = self.jobs[op.job_id]

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

        print(f'op {op.pool_key} is not ready yet')
        self._state.move_worker_to_pool(worker.id_, op.job_pool_key)



    def _get_free_source_workers(self):
        source_worker_ids = self._state.get_source_workers()

        free_worker_ids = \
            set((worker_id
                 for worker_id in iter(source_worker_ids)
                 if self.workers[worker_id].available))

        return free_worker_ids
        


    def _fulfill_commitments_from_source(self):
        '''called at the end of a scheduling round'''
        print('fulfilling source commitments')

        # some of the source workers may not be
        # free right now; find the ones that are.
        free_worker_ids = self._get_free_source_workers()
        commitments = self._state.get_source_commitments()

        for dst_pool_key, num_workers in commitments.items():
            assert num_workers > 0
            while num_workers > 0 and len(free_worker_ids) > 0:
                worker_id = free_worker_ids.pop()
                self._fulfill_commitment(worker_id, dst_pool_key)
                num_workers -= 1

        assert len(free_worker_ids) == 0



    def _move_free_uncommitted_source_workers(self, 
                                              src_pool_key=None, 
                                              worker_ids=None):
        '''A scheduling round may end with some workers still not 
        committed anywhere, because there were no more operations left
        to schedule. When this happens, we may need to move the free 
        workers of that bunch to a different pool.
        '''
        if src_pool_key is None:
            src_pool_key = self._state.get_source()
        assert src_pool_key is not None

        if worker_ids is None:
            worker_ids = list(self._get_free_source_workers())
        assert len(worker_ids) > 0

        if src_pool_key == GENERAL_POOL_KEY:
            return # don't move workers

        job_id, op_id = src_pool_key
        is_job_saturated = self.jobs[job_id].saturated

        if op_id is None and not is_job_saturated:
            # source is an unsaturated job's pool
            return # don't move workers

        # if the source is a saturated job's pool, then move it to the 
        # general pool. If it's an operation's pool, then move it to 
        # the job's pool.
        dst_pool_key = GENERAL_POOL_KEY if is_job_saturated \
                       else (job_id, None)

        print(f'moving free uncommited workers from {src_pool_key} to {dst_pool_key}')

        for worker_id in worker_ids:
            self._state.move_worker_to_pool(worker_id, 
                                            dst_pool_key)



    def _try_backup_schedule(self, worker):
        print('trying backup')

        backup_op = self._find_backup_op(worker)

        if backup_op is not None:
            # found a backup
            self._reroute_worker(worker, backup_op)
            return

        # no backup op found, so move worker to job or general
        # pool depending on whether or not the worker's job 
        # is completed
        job = self.jobs[worker.job_id]
        dst_pool_key = GENERAL_POOL_KEY if job.completed else (worker.job_id, None)
        self._state.move_worker_to_pool(worker.id_, dst_pool_key)



    def _reroute_worker(self, worker, op):
        print('rerouting worker to', op.pool_key)
        assert op.num_remaining_tasks > 0

        if not worker.is_at_job(op.job_id):
            self._state.move_worker_to_pool(
                worker.id_, op.pool_key, send=True)
            self._send_worker(worker, op)
            return

        self._state.move_worker_to_pool(
            worker.id_, op.pool_key)

        if op in self.jobs[op.job_id].frontier_ops:
            # op's dependencies are satisfied, so 
            # start working on it.
            self._work_on_op(worker, op)
        else:
            # dependencies not satisfied; op not ready.
            self._process_worker_at_unready_op(worker, op)



    def _find_backup_op(self, worker):
        # first, try searching within the same job
        local_ops = \
            self._find_schedulable_ops(job_ids=[worker.job_id], 
                                       source_job_id=worker.job_id)
        if len(local_ops) > 0:
            return local_ops.pop()

        # now, try searching all other jobs
        other_job_ids = \
            [job_id for job_id in iter(self.active_job_ids)
             if job_id != worker.job_id]
        other_ops = \
            self._find_schedulable_ops(job_ids=other_job_ids,
                                       source_job_id=worker.job_id)

        if len(other_ops) > 0:
            return other_ops.pop()

        # out of luck
        return None



    def _calculate_reward(self, 
                          old_wall_time, 
                          old_active_job_ids):
        reward = 0.

        # include jobs that completed and arrived
        # during the most recent simulation run
        job_ids = old_active_job_ids | self.active_job_ids

        print('calc reward', len(job_ids), old_wall_time, self.wall_time)

        for job_id in iter(job_ids):
            job = self.jobs[job_id]
            start = max(job.t_arrival, old_wall_time)
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