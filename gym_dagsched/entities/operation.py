import numpy as np

from .task import Task


class Features:
    IS_SOURCE = 0
    N_SOURCE_WORKERS = 1
    N_LOCAL_WORKERS = 2
    N_REMAINING_TASKS = 3
    RAMINING_WORK = 4



class Operation:

    def __init__(self, id, job_id, n_tasks, task_duration, rough_duration):
        self.id_ = id
        self.job_id = job_id
        self.task_duration = task_duration
        self.rough_duration = rough_duration
        self.most_recent_duration = rough_duration

        self.n_tasks = n_tasks
        tasks = [
            Task(id_=i, op_id=self.id_, job_id=self.job_id) 
            for i in range(n_tasks)
        ]
        self.remaining_tasks = set(tasks)
        self.processing_tasks = set()
        self.completed_tasks = set()
        self.remaining_time = np.inf


    def __hash__(self):
        return hash(self.__unique_id)
        

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__unique_id == other.__unique_id
        else:
            return False


    @property
    def __unique_id(self):
        return (self.job_id, self.id_)

    @property
    def completed(self):
        return len(self.completed_tasks) == self.n_tasks

    @property
    def n_saturated_tasks(self):
        return len(self.processing_tasks) + len(self.completed_tasks)

    @property
    def next_task_id(self):
        return self.n_saturated_tasks

    @property
    def saturated(self):
        assert self.n_saturated_tasks <= self.n_tasks
        return self.n_saturated_tasks == self.n_tasks

    @property
    def n_remaining_tasks(self):
        return self.n_tasks - self.n_saturated_tasks

    @property
    def approx_remaining_work(self):
        return self.most_recent_duration * self.n_remaining_tasks
        


    def sample_executor_key(self, num_executors, executor_interval_map):
        (left_exec, right_exec) = \
            executor_interval_map[num_executors]

        executor_key = None

        if left_exec == right_exec:
            executor_key = left_exec

        else:
            rand_pt = np.random.randint(1, right_exec - left_exec + 1)
            if rand_pt <= num_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration['first_wave']:
            # more executors than number of tasks in the job
            largest_key = 0
            for e in self.task_duration['first_wave']:
                if e > largest_key:
                    largest_key = e
            executor_key = largest_key

        return executor_key



    def sample_task_duration(self, task, worker, n_local_workers, executor_interval_map):

        # task duration is determined by wave
        assert n_local_workers > 0

        # sample an executor point in the data
        executor_key = self.sample_executor_key(n_local_workers, executor_interval_map)

        if worker.task is None or \
            worker.job_id != task.job_id:
            # the executor never runs a task in this job
            # fresh executor incurrs a warmup delay
            if len(self.task_duration['fresh_durations'][executor_key]) > 0:
                # (1) try to directly retrieve the warmup delay from data
                fresh_durations = \
                    self.task_duration['fresh_durations'][executor_key]
                i = np.random.randint(len(fresh_durations))
                duration = fresh_durations[i]
            else:
                # (2) use first wave but deliberately add in a warmup delay
                first_wave = \
                    self.task_duration['first_wave'][executor_key]
                i = np.random.randint(len(first_wave))
                duration = first_wave[i] + 1000 # args.warmup_delay

        elif worker.task is not None and \
                worker.task.op_id == task.op_id and \
                len(self.task_duration['rest_wave'][executor_key]) > 0:
            # executor was working on this node
            # the task duration should be retrieved from rest wave
            rest_wave = self.task_duration['rest_wave'][executor_key]
            i = np.random.randint(len(rest_wave))
            duration = rest_wave[i]
        else:
            # executor is fresh to this node, use first wave
            if len(self.task_duration['first_wave'][executor_key]) > 0:
                # (1) try to retrieve first wave from data
                first_wave = \
                    self.task_duration['first_wave'][executor_key]
                i = np.random.randint(len(first_wave))
                duration = first_wave[i]
            else:
                # (2) first wave doesn't exist, use fresh durations instead
                # (should happen very rarely)
                fresh_durations = \
                    self.task_duration['fresh_durations'][executor_key]
                i = np.random.randint(len(fresh_durations))
                duration = fresh_durations[i]

        return duration
        


    
    
