from dataclasses import dataclass
import numpy as np
import typing


@dataclass 
class Stage:
    # each stage has a unique id
    id_: int

    # id of job this stage belongs to
    job_id: int

    # number of identical tasks to complete 
    # within the stage
    n_tasks: int

    # which type of worker is compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_type: int

    # expected completion time of this stage
    duration: np.ndarray

    # time at which a set of workers began
    # processing this stage
    t_accepted: np.ndarray

    # number of workers assigned to this task
    n_workers: int


@dataclass
class Worker:
    # type of the worker (for heterogeneous
    # environments)
    type_: int

    # id of current job assigned to this worker
    job_id: int


@dataclass
class Job:
    id_: int

    # lower triangle of the dag's adgacency 
    # matrix stored as a flattened array
    dag: np.ndarray

    # arrival time of this job
    t_arrival: np.ndarray

    # tuple of stages that make up the
    # nodes of the dag
    stages: typing.Tuple[Stage, ...]

    # number of stages this job consists of
    n_stages: int

        
@dataclass
class Obs:
    wall_time: np.ndarray
    jobs: typing.Tuple[Job, ...]
    job_count: int
    frontier_stages_mask: np.ndarray
    workers: typing.Tuple[Worker, ...]


@dataclass
class Action:
    job_id: int

    # which stage to execute next
    stage_id: int

    # which workers to assign to the stage's job
    workers_mask: np.ndarray