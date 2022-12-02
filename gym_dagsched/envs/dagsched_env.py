from bisect import bisect_left

import numpy as np
import torch

from .state import State, OpNode
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



    def n_ops_per_job(self):
        return [len(self.jobs[j].ops) for j in self.active_job_ids]




    ## OpenAI Gym style interface - reset & step

    def reset(self, initial_timeline, workers, shared_obs):
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

        # self.x_ptrs = x_ptrs
        self.shared_obs = shared_obs

        self.max_ops = max(len(e.job.ops) for _,_,e in initial_timeline.pq)

        self.executor_interval_map = self._get_executor_interval_map()

        self.t_step = 0

        self.state.reset()

        self.selected_ops = set()

        self.done = False

        self.first_step = True

        # take a step in the timeline so that
        # we can observe the environment
        self.wall_time, event = self.timeline.pop()
        self._process_scheduling_event(event)

        self._update_shared_obs(0, False)



    def step(self, action):
        reward, done = self._step(action)
        self._update_shared_obs(reward, done)



    def _update_shared_obs(self, reward, done):
        self._update_node_features()
        self._update_active_job_mask()
        self._update_op_mask()
        self._update_prlvl_mask()
        self.shared_obs.reward.copy_(torch.tensor(reward))
        self.shared_obs.done.copy_(torch.tensor(done))



    def _step(self, action):
        if self.done:
            return 0, True

        # take action
        (job_id, op_id), n_workers = action

        op = self.jobs[job_id].ops[op_id]
        assert op in (self.frontier_ops - self.selected_ops)

        # commit `n_workers` workers from the current worker
        # source to the op with id (job_id, op_id)
        self.state.add_commitment(n_workers, job_id, op_id)

        # mark op as selected so that it doesn't get
        # selected again during this commitment round
        self.selected_ops.add(op)
        
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

        while not (self.timeline.empty or self._should_start_new_commitment_round()):
            self.wall_time, event = self.timeline.pop()
            self._process_scheduling_event(event)

        reward = self._calculate_reward(t_prev)
        self.done = self.timeline.empty

        # if the episode isn't done, then start a new commitment 
        # round at the current worker source

        return reward, self.done




    ## Observations

    def _update_node_features(self):
        n_source_workers = self.state.num_uncommitted_source_workers
        source_job_id = self.state.source_job

        for job_id in self.active_job_ids:
            job = self.jobs[job_id]
            worker_count = self._count_workers(job_id)
            is_source_job = (job_id == source_job_id)

            job_feature_tensor = self.shared_obs.feature_tensor_chunks[job_id]

            # update job-level features
            job_feature_tensor[:, :3] = torch.tensor([
                n_source_workers,
                is_source_job,
                worker_count
            ])

            # update node-level features
            job_feature_tensor[:, 3:] = torch.stack([
                torch.tensor([
                    op.n_remaining_tasks,
                    op.approx_remaining_work
                ])
                for op in job.ops
            ])



    def _update_active_job_mask(self):
        self.shared_obs.active_job_msk.zero_()
        self.shared_obs.active_job_msk[self.active_job_ids] = 1



    def _update_op_mask(self):
        self.shared_obs.op_msk.zero_()

        # get (job_id, op_id) pairs for each operation
        # that is in the frontier but hasn't been selected
        # yet during this committment round
        id_pairs = (
            (op.job_id, op.id_) 
            for op in iter(self.frontier_ops - self.selected_ops)
        )

        # split pairs into two lists
        job_ids, op_ids = list(zip(*id_pairs))

        self.shared_obs.op_msk[job_ids, op_ids] = 1



    def _update_prlvl_mask(self):
        self.shared_obs.prlvl_msk.zero_()
        n_source_workers = self.state.num_uncommitted_source_workers
        self.shared_obs.prlvl_msk[:, :n_source_workers] = 1



    def _count_workers(self, job_id):
        '''for each active job, computes the total count
        of workers associated with that job. Includes:
        - workers sitting at the job's pool
        - workers sitting or moving to operations within the job
        - workers committed to operations within the job
        '''
        job = self.jobs[job_id]
        count = self.state.n_workers_at(job_id) + sum(
            self.state.n_workers_at(job_id, op_id) +
            self.state.n_commitments_to(job_id, op_id)
            for op_id in range(len(job.ops))
        )
        return count




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

        if job.completed or op.saturated:
            # if the job has completed or the op has 
            # become saturated by the time the worker 
            # arrives, then try to greedily find 
            # a backup operation for the worker
            self._try_backup_schedule(worker)
            return
        
        self.state.set_worker_moving(worker.id_, False)
        job.add_local_worker(worker)
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
        op.most_recent_duration = duration

        event = TaskCompletion(op, task)
        self.timeline.push(t_completion, event)



    def _process_task_completion(self, op, task):
        '''performs some bookkeeping when a task completes'''
        worker = self.workers[task.worker_id]

        job = self.jobs[op.job_id]
        job.add_task_completion(op, task, worker, self.wall_time)
        
        if not op.saturated:
            # reassign the worker to keep working on this operation
            # if there is more work to do
            self._work_on_op(worker, op)
        else:
            self._process_op_saturation(op, worker)




    ## Helper functions

    def _should_start_new_commitment_round(self):
        uncommitted_source_workers = not self.state.all_source_workers_committed

        if uncommitted_source_workers and len(self.frontier_ops) > 0:
            # there are source workers that aren't
            # committed anywhere, and there are frontier
            # operations which need more workers,
            # so start a new commitment round at
            # this source.
            return True

        if uncommitted_source_workers:
            # there are source workers that aren't
            # committed anywhere, but there also aren't
            # any frontier nodes they can be 
            # committed to, so move them somewhere
            # according to a fixed rule.
            self._move_all_source_workers()

        return False



    def _move_all_source_workers(self):
        '''current worker source must be an operation.
        moves all the workers at the op to its job's pool
        if the the job contains unsaturated ops, otherwise
        moves them to the null pool
        '''
        job_id = self.state.source_job
        assert job_id is not None
        job = self.jobs[job_id]
        self.state.move_all_source_workers(job.saturated)



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
        assert worker.job_id != op.job_id

        if worker.job_id is not None:
            old_job = self.jobs[worker.job_id]
            old_job.remove_local_worker(worker.id_)

        self._push_worker_arrival_event(worker, op)
            


    def _process_op_saturation(self, op, worker):
        job = self.jobs[op.job_id]
        job.add_op_saturation()
        self.frontier_ops.remove(op)

        frontier_changed = False

        if op.completed:
            # record whether or not the completion of this
            # operation unlocked new operations within the job
            frontier_changed = self._process_op_completion(op)

        if job.completed:
            self._process_job_completion(job)

        # worker may have somewhere to be moved
        commitment = self._move_worker(worker, op, frontier_changed)

        # worker source may need to be updated
        self._update_worker_source(op, frontier_changed, commitment)
        


    def _move_worker(self, worker, op, frontier_changed):
        '''if the worker has a commitment, then fulfill it. Otherwise,
        if `op` completed and unlocked new ops within the job dag, then 
        move the worker to the job's worker pool so that it can be assigned 
        to the new ops
        '''
        commitment = self.state.peek_commitment(op.job_id, op.id_)
        if commitment is not None:
            # op has at least one commitment, so fulfill it
            job_id_committed, op_id_committed = commitment
            op_committed = self.jobs[job_id_committed].ops[op_id_committed]
            self._fulfill_commitment(worker, op_committed)
        elif frontier_changed:
            # no commitment, but frontier changed
            self.state.move_worker_to_job_pool(worker.id_)
        return commitment



    def _update_worker_source(self, op, frontier_changed, commitment):
        if frontier_changed:
            # if any new operations were unlocked within this 
            # job, then give the agent a chance to assign
            # them to free workers from this job's pool
            # by starting a new commitment round at this
            # job's pool
            self.state.update_worker_source(op.job_id)
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
        new_ops = job.find_new_frontier_ops(op)
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
        assert not op.saturated

        worker_is_present = \
            self.state.fulfill_commitment(worker.id_, op.job_id, op.id_)

        if worker_is_present:
            self._work_on_op(worker, op)
        else:
            self._send_worker(worker, op)



    def _fulfill_source_commitments(self):
        '''called at the end of a commitment round
        '''
        assert self.state.all_source_workers_committed

        # some of the source workers may not be
        # free right now; find the ones that are.
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