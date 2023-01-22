import numpy as np

from .task import Task


class Features:
    IS_SOURCE = 0
    N_SOURCE_WORKERS = 1
    N_LOCAL_WORKERS = 2
    N_REMAINING_TASKS = 3
    RAMINING_WORK = 4



class Operation:

    def __init__(self, id, job_id, num_tasks, task_duration, rough_duration):
        self.id_ = id
        self.job_id = job_id
        self.task_duration = task_duration
        self.rough_duration = rough_duration
        self.most_recent_duration = rough_duration

        self.num_tasks = num_tasks
        tasks = [
            Task(id_=i, op_id=self.id_, job_id=self.job_id) 
            for i in range(num_tasks)
        ]
        self.remaining_tasks = set(tasks)
        self.num_remaining_tasks = num_tasks

        # self.processing_tasks = set()
        self.num_processing_tasks = 0

        # self.completed_tasks = set()
        self.num_completed_tasks = 0

        self.saturated = False


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
        return self.num_completed_tasks == self.num_tasks

    @property
    def num_saturated_tasks(self):
        return self.num_processing_tasks + self.num_completed_tasks

    @property
    def next_task_id(self):
        return self.num_saturated_tasks

    @property
    def approx_remaining_work(self):
        return self.most_recent_duration * self.num_remaining_tasks


    def start_on_next_task(self):
        task = self.remaining_tasks.pop()
        self.num_remaining_tasks -= 1

        # self.processing_tasks.add(task)
        self.num_processing_tasks += 1

        return task


    def mark_task_completed(self, task):
        # self.processing_tasks.remove(task)
        self.num_processing_tasks -= 1

        # self.completed_tasks.add(task)
        self.num_completed_tasks += 1
        


    def check_criterion(self, criterion):
        if criterion == 'saturated':
            return self.saturated
        elif criterion == 'completed':
            return self.completed
        else:
            raise Exception('Operation.check_criterion: invalid criterion')



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
        


    
    
