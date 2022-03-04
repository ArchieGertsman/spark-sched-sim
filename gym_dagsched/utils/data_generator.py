import numpy as np

from ..args import args
from ..entities.job import Job
from ..entities.stage import Stage
from ..entities.task import Task
from ..utils.misc import invalid_time, to_wall_time
from ..utils.spaces import dag_space


def generate_worker(worker, i):
    worker.id_ = i
    # ensure there is at least one worker of each type
    worker.type_ = i if i < args.n_worker_types \
        else np.random.randint(low=0, high=args.n_worker_types)


def generate_job(id_, t_arrival):
    stages, n_stages = generate_stages(id_)
    dag = dag_space.sample()
    job = Job(
        id_=id_,
        dag=dag,
        t_arrival=t_arrival,
        t_completed=invalid_time(),
        stages=stages,
        n_stages=n_stages,
        n_completed_stages=0
    )
    return job


def generate_stages(job_id):
    n_stages = np.random.randint(low=2, high=args.max_stages+1)
    stages = []
    for i in range(n_stages):
        n_tasks = np.random.randint(low=1, high=args.max_tasks+1)
        duration = np.random.normal(loc=30., scale=10.)
        duration = np.clip(duration, 0, None)
        stages += [Stage(
            id_=i,
            job_id=job_id,
            n_tasks=n_tasks,
            n_completed_tasks=0,
            task_duration=to_wall_time(duration),
            worker_types_mask=generate_worker_types_mask(), 
            tasks=tuple([Task() for _ in range(args.max_tasks)])
        )]

    stages += [Stage() for _ in range(args.max_stages-n_stages)]
    assert len(stages) == args.max_stages
    stages = tuple(stages)
    return stages, n_stages


def generate_worker_types_mask():
    n_types = np.random.randint(low=1, high=args.n_worker_types+1)
    worker_types = np.array(n_types*[1] + (args.n_worker_types-n_types)*[0])
    np.random.shuffle(worker_types)
    return worker_types


def generate_task_duration(stage):
        # TODO: do a more complex calculation given 
        # other properties of this stage
        return stage.task_duration