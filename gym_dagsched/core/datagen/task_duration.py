import numpy as np



class TaskDurationGen:

    def __init__(self,
                 task_duration_data,
                 np_random,
                 warmup_delay=1000):
        self.task_duration_data = task_duration_data
        self.np_random = np_random
        self.warmup_delay = warmup_delay



    def sample(self, 
               task, 
               worker, 
               num_local_workers, 
               executor_interval_map):

        # task duration is determined by wave
        assert num_local_workers > 0

        # sample an executor point in the data
        executor_key = \
            self._sample_executor_key(num_local_workers, 
                                      executor_interval_map)

        if worker.task is None or worker.job_id != task.job_id:
            # the executor never runs a task in this job
            # fresh executor incurrs a warmup delay
            try:
                return self._sample('fresh_durations', executor_key)
            except:
                return self._sample('first_wave', executor_key, warmup=True)
        

        if worker.task.op_id == task.op_id:
            # executor was working on this node
            # the task duration should be retrieved from rest wave
            try:
                return self._sample('rest_wave', executor_key)
            except:
                pass

        # executor is fresh to this node, use first wave
        try:
            return self._sample('first_wave', executor_key)
        except:
            return self._sample('fresh_durations', executor_key)



    def _sample(self, 
                wave, 
                executor_key, 
                warmup=False):
        '''raises an exception if `executor_key` is not
        found in the durations from `wave`
        '''
        durations = \
            self.task_duration_data[wave][executor_key]
        duration = self.np_random.choice(durations)
        if warmup:
            duration += self.warmup_delay
        return duration




    def _sample_executor_key(self, 
                             num_executors, 
                             executor_interval_map):
        (left_exec, right_exec) = \
            executor_interval_map[num_executors]

        executor_key = None

        if left_exec == right_exec:
            executor_key = left_exec

        else:
            rand_pt = self.np_random.integers(1, right_exec - left_exec + 1)
            if rand_pt <= num_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration_data['first_wave']:
            # more executors than number of tasks in the job
            largest_key = 0
            for e in self.task_duration_data['first_wave']:
                if e > largest_key:
                    largest_key = e
            executor_key = largest_key

        return executor_key