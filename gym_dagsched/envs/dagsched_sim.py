import numpy as np

from ..utils.timeline import Timeline, JobArrival, TaskCompletion
from ..utils.data_generator import DataGenerator


class DagSchedSim:

    def __init__(self, 
        n_job_arrivals, 
        n_workers, 
        mjit, 
        max_ops, 
        max_tasks, 
        n_worker_types
    ):
        self.N_JOB_ARRIVALS = n_job_arrivals
        self.N_WORKERS = n_workers
        self.MJIT = mjit
        self.data_gen = DataGenerator(max_ops, max_tasks, n_worker_types)
        self.reset()



    @property
    def all_jobs_complete(self):
        return self.n_completed_jobs == len(self.jobs)



    def reset(self):
        self.wall_time = 0.
        self.jobs = []
        self.n_completed_jobs = 0
        self.workers = []
        self.frontier_ops = set()
        self.saturated_ops = set()
        self._init_timeline()
        self._init_workers()



    def step(self, op, n_workers):
        '''steps into the next scheduling event on the timeline, 
        which can be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            even though neither (1) nor (2) have occurred, so 
            the policy should consider taking one of them
        '''
        if op in self.frontier_ops:
            tasks = self.take_action(op, n_workers)
            self._push_task_completion_events(tasks)
        else:
            print('invalid action')

        # if there are still actions available after
        # processing the most recent one, then push 
        # a "nudge" event to notify the scheduling agent
        # that another action can immediately be taken
        if self.actions_available():
            print('pushing nudge')
            self._push_nudge_event()

        # check if simulation is done
        if self.timeline.empty:
            assert self.all_jobs_complete
            print('all jobs completed!')
            return True
            
        # retreive the next scheduling event from the timeline
        t, event = self.timeline.pop()

        self._process_scheduling_event(t, event)

        return False



    def _init_timeline(self):
        '''Fills timeline with job arrival events, which follow
        a Poisson process, parameterized by args.mjit (mean job
        interarrival time)
        '''
        self.timeline = Timeline()

        # time of current arrival
        t = 0.

        for id_ in range(self.N_JOB_ARRIVALS):
            # sample time until next arrival
            dt_interarrival = np.random.exponential(self.MJIT)
            t += dt_interarrival

            # generate a job and add its arrival to the timeline
            job = self.data_gen.job(id_, t)
            self.timeline.push(t, JobArrival(job))



    def _init_workers(self):
        '''Initializes the workers with randomly generated attributes'''
        for i in range(self.N_WORKERS):
            worker = self.data_gen.worker(i)
            self.workers += [worker]



    def _push_task_completion_events(self, tasks):
        '''Given a list of task ids and the stage they belong to,
        pushes their completions as events to the timeline
        '''
        assert len(tasks) > 0

        task = tasks.pop()
        job_id = task.job_id
        op_id = task.op_id
        op = self.jobs[job_id].ops[op_id]

        while task is not None:
            assigned_worker_id = task.worker_id
            worker_type = self.workers[assigned_worker_id].type_
            t_completion = \
                task.t_accepted + self.data_gen.task_duration(op, worker_type)
            event = TaskCompletion(op, task)
            self.timeline.push(t_completion, event)
            task = tasks.pop() if len(tasks) > 0 else None



    def _push_nudge_event(self):
        '''Pushes a "nudge" event to the timeline at the current
        wall time, so that the scheduling agent can immediately
        choose another action
        '''
        self.timeline.push(self.wall_time, None)



    def _process_scheduling_event(self, t, event):
        # update the current wall time
        self.wall_time = t

        if isinstance(event, JobArrival):
            job = event.obj
            print(f'{t}: job arrival')
            self.add_job(job)
        elif isinstance(event, TaskCompletion):
            task = event.task
            print(f'{t}: task completion', f'({task.job_id},{task.op_id},{task.id_})')
            self.process_task_completion(task)
        else:
            print(f'{t}: nudge')



    def add_job(self, job):
        self.jobs += [job]
        src_ops = job.find_src_ops()
        self.frontier_ops |= src_ops



    def process_task_completion(self, task):
        op = self.jobs[task.job_id].ops[task.op_id]
        op.add_task_completion(task, self.wall_time)

        worker = self.workers[task.worker_id]
        worker.make_available()

        if op.is_complete:
            print('operation completed')
            self.process_op_completion(op)
        
        job = self.jobs[op.job_id]
        if job.is_complete:
            print('job completed')
            self.process_job_completion(job)


        
    def process_op_completion(self, op):
        job = self.jobs[op.job_id]
        job.add_op_completion()
        
        self.saturated_ops.remove(op)

        # add stage's decendents to the frontier, if their
        # other dependencies are also satisfied
        new_ops = job.find_new_frontiers(op)
        self.frontier_ops |= new_ops



    def process_job_completion(self, job):
        self.n_completed_jobs += 1
        job.t_completed = self.wall_time



    def take_action(self, op, n_workers):
        tasks = set()

        # find workers that are closest to this operation's job
        for worker_type in op.compatible_worker_types:
            if op.saturated:
                break
            n_remaining_requests = n_workers - len(tasks)
            for _ in range(n_remaining_requests):
                worker = self.find_closest_worker(op, worker_type)
                if worker is None:
                    break
                task = self.schedule_worker(worker, op)
                tasks.add(task)

        print(f'scheduled {len(tasks)} tasks')

        # check if stage is now saturated; if so, remove from frontier
        if op.saturated:
            self.frontier_ops.remove(op)
            self.saturated_ops.add(op)

        return tasks



    def find_closest_worker(self, op, worker_type):
        '''chooses an available worker for a stage's 
        next task, according to the following priority:
        1. worker is already at stage
        2. worker is not at stage but is at stage's job
        3. any other available worker
        if the stage is already saturated, or if no 
        worker is found, then `None` is returned
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



    def schedule_worker(self, worker, op):
        old_job_id = worker.task.job_id if worker.task is not None else None
        new_job_id = op.job_id
        moving_cost = self.job_moving_cost(old_job_id, new_job_id)

        task = op.add_worker(
            worker, 
            self.wall_time, 
            moving_cost)

        return task


    
    def job_moving_cost(self, old_job_id, new_job_id):
        return 0. if new_job_id == old_job_id else np.random.exponential(10.)



    def actions_available(self):
        avail_workers = self.find_available_workers()

        if len(avail_workers) == 0 or len(self.frontier_ops) == 0:
            return False

        for op in self.frontier_ops:
            for worker in avail_workers:
                if worker.compatible_with(op):
                    return True

        return False



    def find_available_workers(self):
        return [worker for worker in self.workers if worker.available]