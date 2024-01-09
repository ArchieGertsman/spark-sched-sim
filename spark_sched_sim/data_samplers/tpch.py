import os.path as osp
import pathlib
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
import networkx as nx

from .data_sampler import DataSampler
from ..components import Job, Stage

TPCH_URL = "https://bit.ly/3F1Go8t"
QUERY_SIZES = ["2g", "5g", "10g", "20g", "50g", "80g", "100g"]
NUM_QUERIES = 22


class TPCHDataSampler(DataSampler):
    def __init__(
        self,
        job_arrival_rate: float,
        job_arrival_cap: int,
        num_executors: int,
        warmup_delay: int,
        **kwargs,
    ):
        """
        job_arrival_rate (float): non-negative number that controls how
            quickly new jobs arrive into the system. This is the parameter
            of an exponential distributions, and so its inverse is the
            mean job inter-arrival time in ms.
        job_arrival_cap: (optional int): limit on the number of jobs that
            arrive throughout the simulation. If set to `None`, then the
            episode ends when a time limit is reached.
        num_executors (int): number of simulated executors. More executors
            means a higher possible level of parallelism.
        warmup_delay (int): an executor is slower on its first task from
            a stage if it was previously idle or moving jobs, which is
            caputred by adding a warmup delay (ms) to the task duration
        """
        self.job_arrival_cap = job_arrival_cap
        self.mean_interarrival_time = 1 / job_arrival_rate
        self.warmup_delay = warmup_delay

        self.np_random = None
        self._init_executor_intervals(num_executors)

        if not osp.isdir("data/tpch"):
            self._download_tpch_dataset()

    def reset(self, np_random: np.random.Generator):
        self.np_random = np_random

    def job_sequence(self, max_time):
        """generates a sequence of job arrivals over time, which follow a
        Poisson process parameterized by `self.job_arrival_rate`
        """
        assert self.np_random
        job_sequence = []

        t = 0
        job_idx = 0
        while t < max_time and (
            not self.job_arrival_cap or job_idx < self.job_arrival_cap
        ):
            job = self._sample_job(job_idx, t)
            job_sequence.append((t, job))

            # sample time in ms until next arrival
            t += self.np_random.exponential(self.mean_interarrival_time)
            job_idx += 1

        return job_sequence

    def task_duration(self, job, stage, task, executor):
        num_local_executors = len(job.local_executors)

        assert num_local_executors > 0
        assert self.np_random

        data = stage.task_duration_data

        # sample an executor point in the data
        executor_key = self._sample_executor_key(data, num_local_executors)

        if executor.is_idle:
            # the executor was just sitting idly or moving between jobs, so it needs time to warm up
            try:
                return self._sample_task_duration(data, "fresh_durations", executor_key)
            except (ValueError, KeyError):
                return self._sample_task_duration(
                    data, "first_wave", executor_key, warmup=True
                )

        if executor.task.stage_id == task.stage_id:
            # the executor is continuing work on the same stage, which is relatively fast
            try:
                return self._sample_task_duration(data, "rest_wave", executor_key)
            except (ValueError, KeyError):
                pass

        # the executor is new to this stage (or 'rest_wave' data was not available)
        try:
            return self._sample_task_duration(data, "first_wave", executor_key)
        except (ValueError, KeyError):
            return self._sample_task_duration(data, "fresh_durations", executor_key)

    @classmethod
    def _download_tpch_dataset(cls):
        print("Downloading the TPC-H dataset...", flush=True)
        pathlib.Path("data").mkdir(parents=True, exist_ok=True)
        with urlopen(TPCH_URL) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall("data")
        print("Done.", flush=True)

    @classmethod
    def _load_query(cls, query_num, query_size):
        query_path = osp.join("data/tpch", str(query_size))

        adj_matrix = np.load(
            osp.join(query_path, f"adj_mat_{query_num}.npy"), allow_pickle=True
        )

        task_duration_data = np.load(
            osp.join(query_path, f"task_duration_{query_num}.npy"), allow_pickle=True
        ).item()

        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        assert adj_matrix.shape[0] == len(task_duration_data)

        return adj_matrix, task_duration_data

    @classmethod
    def _pre_process_task_duration(cls, task_duration):
        # remove fresh durations from first wave
        clean_first_wave = {}
        for e in task_duration["first_wave"]:
            clean_first_wave[e] = []
            fresh_durations = MultiSet()
            # O(1) access
            for d in task_duration["fresh_durations"][e]:
                fresh_durations.add(d)
            for d in task_duration["first_wave"][e]:
                if d not in fresh_durations:
                    clean_first_wave[e].append(d)
                else:
                    # prevent duplicated fresh duration blocking first wave
                    fresh_durations.remove(d)

        # fill in nearest neighour first wave
        last_first_wave = []
        for e in sorted(clean_first_wave.keys()):
            if len(clean_first_wave[e]) == 0:
                clean_first_wave[e] = last_first_wave
            last_first_wave = clean_first_wave[e]

        # swap the first wave with fresh durations removed
        task_duration["first_wave"] = clean_first_wave

    @classmethod
    def _rough_task_duration(cls, task_duration_data):
        def durations(key):
            durations = task_duration_data[key].values()
            durations = [t for ts in durations for t in ts]
            return durations

        all_durations = (
            durations("fresh_durations")
            + durations("first_wave")
            + durations("rest_wave")
        )

        return np.mean(all_durations)

    def _sample_job(self, job_id, t_arrival):
        query_num = 1 + self.np_random.integers(NUM_QUERIES)
        query_size = self.np_random.choice(QUERY_SIZES)
        adj_mat, task_duration_data = self._load_query(query_num, query_size)

        num_stages = adj_mat.shape[0]
        stages = []
        for stage_id in range(num_stages):
            data = task_duration_data[stage_id]
            e = next(iter(data["first_wave"]))

            num_tasks = len(data["first_wave"][e]) + len(data["rest_wave"][e])

            # remove fresh duration from first wave duration
            # drag nearest neighbor first wave duration to empty spots
            self._pre_process_task_duration(data)

            # generate a node
            stage = Stage(stage_id, job_id, num_tasks, self._rough_task_duration(data))
            stage.task_duration_data = data
            stages += [stage]

        # generate DAG
        dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        for _, _, d in dag.edges(data=True):
            d.clear()

        job = Job(job_id, stages, dag, t_arrival)
        job.query_num = query_num
        job.query_size = query_size
        return job

    def _sample_task_duration(self, data, wave, executor_key, warmup=False):
        """raises an exception if `executor_key` is not found in the durations from `wave`"""
        durations = data[wave][executor_key]
        duration = self.np_random.choice(durations)
        if warmup:
            duration += self.warmup_delay
        return duration

    def _sample_executor_key(self, data, num_local_executors):
        left_exec, right_exec = self.executor_intervals[num_local_executors]

        executor_key = None

        if left_exec == right_exec:
            executor_key = left_exec
        else:
            # faster than random.randint
            rand_pt = 1 + int(self.np_random.random() * (right_exec - left_exec))
            if rand_pt <= num_local_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in data["first_wave"]:
            # more executors than number of tasks in the job
            executor_key = max(data["first_wave"])

        return executor_key

    def _init_executor_intervals(self, exec_cap):
        exec_levels = [5, 10, 20, 40, 50, 60, 80, 100]

        intervals = np.zeros((exec_cap + 1, 2))

        # get the left most map
        intervals[: exec_levels[0] + 1] = exec_levels[0]

        # get the center map
        for i in range(len(exec_levels) - 1):
            intervals[exec_levels[i] + 1 : exec_levels[i + 1]] = (
                exec_levels[i],
                exec_levels[i + 1],
            )

            if exec_levels[i + 1] > exec_cap:
                break

            # at the data point
            intervals[exec_levels[i + 1]] = exec_levels[i + 1]

        # get the residual map
        if exec_cap > exec_levels[-1]:
            intervals[exec_levels[-1] + 1 : exec_cap] = exec_levels[-1]

        self.executor_intervals = intervals


class MultiSet:
    """
    allow duplication in set
    """

    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]
