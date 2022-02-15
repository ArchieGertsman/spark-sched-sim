from dataclasses import dataclass
import numpy as np
import typing


@dataclass 
class Stage:
    # each stage has a unique id
    id_: int

    # number of identical tasks to complete 
    # within the stage
    n_tasks: int

    # which type of worker is compaitble with
    # this type of stage (for heterogeneous
    # environments)
    worker_type: int

    # expected completion time of this stage
    t_completion: np.ndarray

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

    # current job assigned to this worker
    job: int


@dataclass
class Job:
    # upper triangle of the dag's adgacency 
    # matrix stored as a flattened array
    dag: np.ndarray

    # arrival time of this job
    t_arrival: np.ndarray

    # tuple of stages that make up the
    # nodes of the dag
    stages: typing.Tuple[Stage, ...]

    n_stages: int

        
@dataclass
class Obs:
    frontier_stages: np.ndarray
    jobs: typing.Tuple[Job, ...]
    workers: typing.Tuple[Worker, ...]


@dataclass
class Action:
    # which stage to execute next
    stage: int

    # which workers to assign to the stage's job
    workers: np.ndarray