import numpy as np

from ..args import args
from ..entities.job import Job
from ..entities.stage import Stage
from ..entities.task import Task
from ..utils.misc import invalid_time, mask_to_indices, to_wall_time
from ..utils.spaces import dag_space


def generate_worker(worker, i):
    worker.id_ = i
    # ensure there is at least one worker of each type
    worker.type_ = i if i < args.n_worker_types \
        else np.random.randint(low=0, high=args.n_worker_types)


def generate_job(id_, t_arrival):
    stages, n_stages = generate_stages(id_)
    dag = dag_space.sample()
    # max_workers = np.random.randint(low=1, high=args.n_workers)
    max_workers = 2
    job = Job(
        id_=id_,
        dag=dag,
        t_arrival=t_arrival,
        t_completed=invalid_time(),
        stages=stages,
        n_stages=n_stages,
        n_completed_stages=0,
        max_workers=max_workers,
        n_workers=0
    )
    return job


def generate_stages(job_id):
    n_stages = np.random.randint(low=2, high=args.max_stages+1)
    stages = []
    for i in range(n_stages):
        n_tasks = np.random.randint(low=1, high=args.max_tasks+1)
        # duration = np.random.normal(loc=30., scale=15.)
        # TODO: duration for incompatible type should be inf
        worker_types_mask = generate_worker_types_mask()
        incompatible_worker_types = mask_to_indices(1-worker_types_mask)
        durations = generate_task_duration_per_worker_type(incompatible_worker_types)
        stages += [Stage(
            id_=i,
            job_id=job_id,
            n_tasks=n_tasks,
            n_completed_tasks=0,
            task_duration_per_worker_type=durations,
            tasks=tuple([Task() for _ in range(args.max_tasks)])
        )]

    stages += [Stage() for _ in range(args.max_stages-n_stages)]
    assert len(stages) == args.max_stages
    stages = tuple(stages)
    return stages, n_stages


def generate_task_duration_per_worker_type(incompatible_worker_types):
    # generate a baseline task duration
    baseline_duration = np.random.exponential(30.)

    # generate offsets from this baseline for each worker type
    # so that some workers will work slower than the baseline
    # while others will work faster than the baseline
    worker_types_offsets = np.random.normal(scale=10., size=args.n_worker_types)

    # compute the expected durations from the baseline 
    # and offsets
    durations = baseline_duration * np.ones(args.n_worker_types)
    durations += worker_types_offsets

    # ensure that no expected duration is too small
    durations = np.clip(durations, 10., None)

    # give incompatible worker types an expected duration
    # of infinity
    durations[incompatible_worker_types] = invalid_time()
    return durations.astype(np.float32)


def generate_worker_types_mask():
    n_types = np.random.randint(low=1, high=args.n_worker_types+1)
    worker_types = n_types*[1] + (args.n_worker_types-n_types)*[0]
    worker_types = np.array(worker_types, dtype=np.int8)
    np.random.shuffle(worker_types)
    return worker_types


def generate_task_duration(stage, assigned_worker_type):
        # TODO: do a more complex calculation given 
        # other properties of this stage
        return stage.task_duration_per_worker_type[assigned_worker_type]