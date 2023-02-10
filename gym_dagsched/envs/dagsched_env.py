from typing import Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import (
    Discrete,
    MultiBinary,
    Dict,
    Box,
    Sequence,
    Graph,
    GraphInstance
)

from ..core.worker_assignment_state import (
    WorkerAssignmentState, 
    GENERAL_POOL_KEY
)
from ..core.timeline import (
    JobArrival, 
    TaskCompletion, 
    WorkerArrival
)
from ..core.entities.worker import Worker
from ..core.datagen.tpch_job_sequence import TPCHJobSequenceGen
from ..utils.graph import subgraph
from ..utils import metrics
try:
    from ..utils.render import Renderer
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False



class DagSchedEnv(Env):
    '''Gymnasium environment that simulates job dag scheduling'''

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self,
        num_workers: int,
        num_init_jobs: int,
        num_job_arrivals: int,
        job_arrival_rate: float,
        moving_delay: float,
        render_mode: Optional[str] = None
    ):
        '''
        Args:
            num_workers (int): number of simulated workers. More workers
                means a higher possible level of parallelism.
            num_init_jobs (int): number of jobs in the system at time t=0
            num_job_arrivals: (int): number of jobs that arrive throughout
                the simulation, according to a Poisson process
            job_arrival_rate (float): non-negative number that controls how
                quickly new jobs arrive into the system. This is the parameter
                of an exponential distributions, and so its inverse is the
                mean job inter-arrival time in ms.
            moving_delay (float): time in ms it takes for a worker to move
                between jobs
            render_mode (optional str): if set to 'human', then a visualization
                of the simulation is rendred in real time
        '''

        num_total_jobs = num_init_jobs + num_job_arrivals

        self.num_workers = num_workers
        self.num_total_jobs = num_total_jobs
        self.moving_cost = moving_delay

        self._datagen = \
            TPCHJobSequenceGen(num_init_jobs,
                               num_job_arrivals,
                               job_arrival_rate)

        self._state = WorkerAssignmentState(num_workers)

        self.render_mode = render_mode
        if render_mode == 'human':
            if not PYGAME_AVAILABLE:
                raise Exception('pygame is unavailable')
            self._renderer = \
                Renderer(self.num_workers, 
                         self.num_total_jobs, 
                         render_fps=self.metadata['render_fps'])
        else:
            self._renderer = None

        self.action_space = Dict({
            # operation selection
            # NOTE: upper bound of this space is dynamic, equal to 
            # the number of active operations. Initialized to 1.
            'op_idx': Discrete(1),

            # parallelism limit selection
            'prlsm_lim': Discrete(num_workers, start=1)
        })

        self.observation_space = Dict({
            'dag_batch': Dict({
                # shape: (num active ops) x (2 features)
                # op features: num remaining tasks, most recent task duration
                # edge features: none
                'data': Graph(node_space=Box(0, np.inf, (2,)), 
                              edge_space=Discrete(1)),

                # length: num active jobs
                # `ptr[job_idx]` returns the index of the first operation
                # associated with that job. E.g., the range of operation
                # indices for a job is given by 
                # `ptr[job_idx], ..., (ptr[job_idx+1]-1)`
                # NOTE: upper bound of this space is dynamic, equal to 
                # the number of active operations. Initialized to 1.
                'ptr': Sequence(Discrete(1)),
            }),

            # length: num active ops
            # `mask[i]` = 1 if op `i` is schedulable, 0 otherwise
            'schedulable_op_mask': Sequence(Discrete(2)),

            # shape: (num active jobs, num workers)
            # `mask[job_idx][l]` = 1 if parallism limit `l` is valid
            # for that job
            'valid_prlsm_lim_mask': Sequence(MultiBinary(self.num_workers)),

            # integer that represents how many workers need to be scheduled
            'num_workers_to_schedule': Discrete(num_workers+1),

            # index of job who is releasing workers, if any.
            # set to `self.num_total_jobs` if source is general pool.
            'source_job_idx': Discrete(num_total_jobs+1),

            # length: num active jobs
            # count of workers associated with each active job,
            # including moving workers and commitments from other jobs
            'worker_counts': Sequence(Discrete(2*num_workers))
        })



    @property
    def all_jobs_complete(self):
        return self.num_completed_jobs == self.num_total_jobs



    @property
    def num_completed_jobs(self):
        return len(self.completed_job_ids)



    @property
    def num_active_jobs(self):
        return len(self.active_job_ids)



    @property
    def done(self):
        return self.terminated or self.truncated



    @property
    def info(self):
        return {'wall_time': self.wall_time}



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            self.max_wall_time = options['max_wall_time']
        except:
            self.max_wall_time = np.inf

        # simulation wall time in ms
        self.wall_time = 0.

        self.terminated = False

        self.truncated = False
        
        # priority queue of scheduling events, indexed by wall time
        self.timeline = self._datagen.new_timeline(self.np_random)

        self.workers = [Worker(i) for i in range(self.num_workers)]

        self._state.reset()

        # timeline is initially filled with all the job arrival
        # events, so extract all the job objects from there
        self.jobs = {
            i: e.job 
            for i, (*_, e) in enumerate(self.timeline.pq)
        }

        # the fastest way to obtain the edge links for an
        # observation is to start out with all of them in
        # a big array, and then to induce a subgraph based
        # on the current set of active nodes
        self.all_edge_links, self.all_job_ptr = \
            self._reset_edge_links()

        self.num_total_ops = self.all_job_ptr[-1]

        # must be ordered
        # TODO: use ordered set
        self.active_job_ids = []

        self.completed_job_ids = set()

        self.executor_interval_map = \
            self._make_executor_interval_map()

        # during a scheduling round, maintains the operations
        # that have already been selected so that they don't
        # get selected again
        self.selected_ops = set()

        # active op index -> op object
        # used to trace an action to its corresponding operation
        self.op_selection_map = {}

        self._load_initial_jobs()

        return self._observe(), self.info



    def step(self, action):
        assert not self.done

        print('step',
              np.round(self.wall_time*1e-3, 1),
              self._state.get_source(), 
              self._state.num_workers_to_schedule(), 
              flush=True)

        self._take_action(action)

        if self._state.num_workers_to_schedule() > 0 and \
           len(self.schedulable_ops) > 0:
            # there are still scheduling decisions to be made, 
            # so consult the agent again
            return self._observe(), 0, False, False, self.info
            
        # scheduling round has completed
        self._commit_remaining_workers()
        self._fulfill_commitments_from_source()
        self._state.clear_worker_source()
        self.selected_ops.clear()

        # save old attributes for computing reward
        old_wall_time = self.wall_time
        old_active_job_ids = self.active_job_ids.copy()

        self._resume_simulation()

        reward = self._calculate_reward(old_wall_time,
                                        old_active_job_ids)
        self.terminated = self.all_jobs_complete
        self.truncated = (self.wall_time >= self.max_wall_time)

        if not self.done:
            print('starting new scheduling round', flush=True)
            assert self._state.num_workers_to_schedule() > 0 and \
                   len(self.schedulable_ops) > 0
        else:
            print(f'done at {self.wall_time*1e-3:.1f}s', flush=True)

        if self.render_mode == 'human':
            self._render_frame()

        # if the episode isn't done, then start a new scheduling 
        # round at the current worker source
        return self._observe(), \
               reward, \
               self.terminated, \
               self.truncated, \
               self.info



    def close(self):
        self._renderer.close()





    ## internal methods

    def _render_frame(self):
        worker_histories = (worker.history for worker in self.workers)
        job_completion_times = (self.jobs[job_id].t_completed 
                                for job_id in iter(self.completed_job_ids))
        average_job_duration = int(metrics.avg_job_duration(self) * 1e-3)

        self._renderer.render_frame(
            worker_histories, 
            job_completion_times,
            self.wall_time,
            average_job_duration,
            self.num_active_jobs,
            self.num_completed_jobs
        )



    def _reset_edge_links(self):
        edge_links = []
        job_ptr = [0]
        for job in self.jobs.values():
            base_op_idx = job_ptr[-1]
            edge_links += [base_op_idx + np.vstack(job.dag.edges)]
            job_ptr += [base_op_idx + job.num_ops]
        return np.vstack(edge_links), np.array(job_ptr)



    def _load_initial_jobs(self):
        while not self.timeline.empty:
            wall_time, event = self.timeline.peek()
            if wall_time > 0:
                break
            self.timeline.pop()
            self._process_scheduling_event(event)

        self.schedulable_ops = self._find_schedulable_ops()



    def _take_action(self, action):
        print('raw action', action)
        
        if not self.action_space.contains(action):
            print('invalid action')
            self._commit_remaining_workers()
            return

        op = self.op_selection_map[action['op_idx']]
        assert op in self.schedulable_ops

        job_worker_count = self._state.total_worker_count(op.job_id)
        num_workers_to_schedule = self._state.num_workers_to_schedule()
        source_job_id = self._state.source_job_id()

        num_workers = action['prlsm_lim'] - job_worker_count
        if op.job_id == source_job_id:
            num_workers += num_workers_to_schedule
        assert num_workers >= 1

        print('action:', op.pool_key, num_workers)

        # agent may have requested more workers than
        # are actually needed or available
        num_workers = \
            self._adjust_num_workers(num_workers, op)

        print(f'committing {num_workers} workers '
              f'to {op.pool_key}')
        self._state.add_commitment(num_workers, op.pool_key)

        # mark op as selected so that it doesn't get
        # selected again during this scheduling round
        self.selected_ops.add(op)

        # find remaining schedulable operations
        job = self.jobs[op.job_id]
        self.schedulable_ops -= set(job.active_ops)
        self.schedulable_ops |= \
            self._find_schedulable_ops([op.job_id])



    def _resume_simulation(self):
        '''resumes the simulation until either there are
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
            self._move_free_workers()
            self._state.clear_worker_source()

        self.schedulable_ops = schedulable_ops



    def _observe(self):
        self.op_selection_map.clear()
        
        nodes = []
        ptr = [0]
        schedulable_op_mask = []
        active_op_mask = np.zeros(self.num_total_ops, dtype=bool)
        worker_counts = []
        source_job_idx = len(self.active_job_ids)

        for op in iter(self.schedulable_ops):
            op.schedulable = True

        for i, job_id in enumerate(self.active_job_ids):
            job = self.jobs[job_id]

            if job_id == self._state.source_job_id():
                source_job_idx = i

            worker_counts += [self._state.total_worker_count(job_id)]

            for op in iter(job.active_ops):
                self.op_selection_map[len(nodes)] = op
                
                nodes += [(op.num_remaining_tasks, op.most_recent_duration)]
                
                schedulable_op_mask += [1] if op.schedulable else [0]
                op.schedulable = False

                active_op_mask[self.all_job_ptr[job_id] + op.id_] = 1

            ptr += [len(nodes)]

        try:
            nodes = np.vstack(nodes).astype(np.float32)
        except:
            # no nodes
            nodes = np.zeros((0,2), dtype=np.float32)

        edge_links = subgraph(self.all_edge_links, active_op_mask)

        # we aren't using any edge data, so this array is always zeros
        edges = np.zeros(len(edge_links), dtype=int)

        num_workers_to_schedule = self._state.num_workers_to_schedule()

        valid_prlsm_lim_masks = []
        for i, worker_count in enumerate(worker_counts):
            min_prlsm_lim = worker_count + 1
            if i == source_job_idx:
                min_prlsm_lim -= num_workers_to_schedule

            assert 0 < min_prlsm_lim
            assert     min_prlsm_lim <= self.num_workers + 1

            valid_prlsm_lim_mask = np.zeros(self.num_workers, dtype=bool)
            valid_prlsm_lim_mask[(min_prlsm_lim-1):] = 1
            valid_prlsm_lim_masks += [valid_prlsm_lim_mask]

        obs = {
            'dag_batch': {
                'data': GraphInstance(nodes, edges, edge_links),
                'ptr': ptr
            },
            'schedulable_op_mask': schedulable_op_mask,
            'valid_prlsm_lim_mask': valid_prlsm_lim_masks,
            'num_workers_to_schedule': num_workers_to_schedule,
            'source_job_idx': source_job_idx,
            'worker_counts': worker_counts
        }

        # update op action space to reflect the current number of active ops
        self.observation_space['dag_batch']['ptr'].feature_space.n = len(nodes)+1
        self.action_space['op_idx'].n = len(nodes)

        return obs




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

        # self.active_job_ids.add(job.id_)
        self.active_job_ids += [job.id_]
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
        t_arrival = self.wall_time + self.moving_cost
        event = WorkerArrival(worker, op)
        self.timeline.push(t_arrival, event)



    def _process_worker_arrival(self, worker, op):
        '''performs some bookkeeping when a worker arrives'''
        print('worker arrived to', op.pool_key)
        job = self.jobs[op.job_id]

        job.add_local_worker(worker)
        worker.add_history(self.wall_time, job.id_)

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
            op.task_duration_gen.sample(task, 
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
            self._handle_released_worker(worker.id_, 
                                         op, 
                                         did_job_frontier_change)

        # worker source may need to be updated
        self._update_worker_source(op, 
                                   had_commitment, 
                                   did_job_frontier_change)




    ## Other helper functions

    def _commit_remaining_workers(self):
        '''There may be workers at the current source pool that
        weren't committed anywhere, e.g. because there were
        no more operations to schedule, or because the agent
        chose not to schedule all of them.
        
        This function explicitly commits those remaining workers 
        to the general pool. When those workers get released, 
        they either move to the job pool or the general pool, 
        depending on whether the job is saturated at that time. 
        
        It is important to do this, or else the agent could
        go in a loop, under-committing workers from the same
        source pool.
        '''
        num_uncommitted_workers = \
            self._state.num_workers_to_schedule()

        if num_uncommitted_workers > 0:
            self._state.add_commitment(num_uncommitted_workers, 
                                       GENERAL_POOL_KEY)



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



    def _adjust_num_workers(self, num_workers, op):
        '''truncates the numer of worker assigned
        to `op` to the op's demand, if it's larger
        '''
        worker_demand = self._get_worker_demand(op)
        num_source_workers = self._state.num_workers_to_schedule()

        num_workers_adjusted = \
            min(num_workers, worker_demand, num_source_workers)

        assert num_workers_adjusted > 0

        print('num_workers adjustment:',
             f'{num_workers} -> {num_workers_adjusted}')

        return num_workers_adjusted



    def _get_worker_demand(self, op):
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
        return self._get_worker_demand(op) <= 0
            


    def _work_on_op(self, worker, op):
        '''starts work on another one of `op`'s 
        tasks, assuming there are still tasks 
        remaining and the worker is local to the 
        operation
        '''
        assert op is not None
        assert op.num_remaining_tasks > 0
        assert worker.is_at_job(op.job_id)
        assert worker.is_free

        job = self.jobs[op.job_id]
        task = job.assign_worker(worker, op, self.wall_time)

        self._push_task_completion_event(op, task)



    def _send_worker(self, worker, op):
        '''sends a `worker` to `op`, assuming
        that the worker is currently at a
        different job
        '''
        assert op is not None
        assert worker.is_free
        assert worker.job_id != op.job_id

        if worker.job_id is not None:
            old_job = self.jobs[worker.job_id]
            old_job.remove_local_worker(worker)

        self._push_worker_arrival_event(worker, op)
        


    def _handle_released_worker(self, 
                                worker_id, 
                                op, 
                                did_job_frontier_change):
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
        print(f'job {job.id_} completed at time {self.wall_time*1e-3:.1f}s')
        
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
            self._move_free_workers(src_pool_key, [worker_id])
            return

        job_id, op_id = dst_pool_key
        op = self.jobs[job_id].ops[op_id]
        worker = self.workers[worker_id]

        if op.num_remaining_tasks == 0:
            # operation is saturated
            self._try_backup_schedule(worker)
            return

        self._move_worker(worker, op)



    def _process_worker_at_unready_op(self, worker, op):
        '''This op's parents are saturated but have not
        completed, so we can't actually start working
        on the op. Move the worker to the job pool instead.
        '''
        print(f'op {op.pool_key} is not ready yet')
        self._state.move_worker_to_pool(worker.id_, op.job_pool_key)



    def _get_free_source_workers(self):
        source_worker_ids = self._state.get_source_workers()

        free_worker_ids = \
            set((worker_id
                 for worker_id in iter(source_worker_ids)
                 if self.workers[worker_id].is_free))

        return free_worker_ids
        


    def _fulfill_commitments_from_source(self):
        print('fulfilling source commitments')

        # only consider the free workers
        free_worker_ids = self._get_free_source_workers()
        commitments = self._state.get_source_commitments()

        for dst_pool_key, num_workers in commitments.items():
            assert num_workers > 0
            while num_workers > 0 and len(free_worker_ids) > 0:
                worker_id = free_worker_ids.pop()
                self._fulfill_commitment(worker_id, dst_pool_key)
                num_workers -= 1

        assert len(free_worker_ids) == 0



    def _move_free_workers(self, 
                           src_pool_key=None, 
                           worker_ids=None):
        '''There may be free workers that were never committed
        anywhere. Such workers are either moved to to the job
        pool or general pool, depending on whether the job is
        saturated.
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

        print('moving free uncommited workers from'
              f'{src_pool_key} to {dst_pool_key}')

        for worker_id in worker_ids:
            worker = self.workers[worker_id]
            if dst_pool_key == GENERAL_POOL_KEY:
                worker.add_history(self.wall_time, -1)
            self._state.move_worker_to_pool(worker_id, 
                                            dst_pool_key)



    def _try_backup_schedule(self, worker):
        '''If a worker arrives to an operation that no
        longer needs any workers, then greedily try to
        find a backup operation.
        '''
        print('trying backup')

        backup_op = self._find_backup_op(worker)

        if backup_op is not None:
            # found a backup
            self._move_worker(worker, backup_op)
            return

        # no backup op found, so move worker to job or general
        # pool depending on whether or not the worker's job 
        # is completed
        job = self.jobs[worker.job_id]
        dst_pool_key = GENERAL_POOL_KEY if job.completed else (worker.job_id, None)
        self._state.move_worker_to_pool(worker.id_, dst_pool_key)

        if dst_pool_key == GENERAL_POOL_KEY:
            worker.add_history(self.wall_time, -1)



    def _move_worker(self, worker, op):
        print('moving worker to', op.pool_key)
        assert op.num_remaining_tasks > 0

        if not worker.is_at_job(op.job_id):
            self._state.move_worker_to_pool(
                worker.id_, op.pool_key, send=True)
            self._send_worker(worker, op)
            return

        self._state.move_worker_to_pool(
            worker.id_, op.pool_key)

        job = self.jobs[op.job_id]
        if op in job.frontier_ops:
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
        job_ids = set(old_active_job_ids) | set(self.active_job_ids)

        print('calc reward', len(job_ids), old_wall_time, self.wall_time)

        for job_id in iter(job_ids):
            job = self.jobs[job_id]
            start = max(job.t_arrival, old_wall_time)
            end = min(job.t_completed, self.wall_time)
            reward -= (end - start)
        return reward


    
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