# from dataclasses import 
from copy import deepcopy as dcp

import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box, MultiDiscrete
import numpy as np
from dacite import from_dict

from .entities.action import Action
from .entities.dagsched_state import DagSchedState
from .entities.job import Job
from .entities.stage import Stage
from .entities.worker import Worker
from .utils import invalid_time, to_wall_time, triangle
from .timeline import Timeline, JobArrival, TaskCompletion

class DagSchedEnv(gym.Env):

    def __init__(self,
        max_jobs, 
        max_stages, 
        max_tasks, 
        n_worker_types, 
        n_workers,
        mean_job_interarrival_time=10.
    ):
        # fix a maximum state size
        self.max_jobs = Job.invalid_id = max_jobs

        self.max_stages = Stage.invalid_id = max_stages

        self.max_tasks = Stage.invalid_task_id = max_tasks

        # fixed workers
        self.n_workers = Worker.invalid_id = n_workers

        self.n_worker_types = Worker.invalid_type = n_worker_types

        self.mean_job_interarrival_time = \
            mean_job_interarrival_time

        self._construct_null_entities()

        self._init_spaces()


    """
    abstract method implementations
    """

    def reset(self):
        self.state = dcp(self.null_state)
        self._init_timeline()
        return dcp(self.state)


    def step(self, action_dict):
        '''steps onto the next scheduling event, which can
        be one of the following:
        (1) new job arrival
        (2) task completion
        '''
        action = from_dict(Action, action_dict)
        if self.state.check_action_validity(action):
            stage, task_ids = self.state.take_action(action)
            self._push_tasks(stage, task_ids)
        else:
            print('invalid action')

        if self.timeline.empty:
            return None, None, True, None
            
        t, event = self.timeline.pop()
        self.state.wall_time = to_wall_time(t)

        if isinstance(event, JobArrival):
            print(f'{t}: job arrival')
            job = event.obj
            self.state.add_job(job)
        elif isinstance(event, TaskCompletion):
            # TODO: problem is that worker_id for a task is reset 
            # after task completion, so then that same task is 
            # reassigned a worker later. Need to keep better track
            # of completed tasks.
            print(f'{t}: task completion')
            stage, task_id = event.obj, event.task_id
            print(f'({stage.job_id},{stage.id_},{task_id})')
            self.state.process_task_completion(stage, task_id)
            if stage.is_complete:
                print(f'{t}: stage completion')
                self.state.process_stage_completion(stage)

        obs = dcp(self.state)
        return obs, None, False, None



    """
    helper functions
    """


    def _init_spaces(self):
        # time can be any non-negative real number
        self.time_space = Box(low=0., high=np.inf, shape=(1,))

        # exclusive discrete space 0 <= k < n, with
        # additional invalid state := n
        self.discrete_x = lambda n: Discrete(n+1)

        # inclusive discrete space 0 <= k <= n
        self.discrete_i = lambda n: Discrete(n+1)

        # lower triangle of the dag's adgacency matrix stored 
        # as a flattened array
        self.dag_space = MultiBinary(triangle(self.max_stages))

        self.stages_mask_space = MultiBinary(self.max_jobs * self.max_stages)

        self._init_observation_space()

        self._init_action_space()


    def _init_observation_space(self):
        # see entities folder for details about the attributes in
        # the following spaces

        self.worker_space = Dict({
            'id_': self.discrete_x(self.n_workers),
            'type_': self.discrete_x(self.n_worker_types),
            'job_id': self.discrete_x(self.max_jobs),
            'stage_id': self.discrete_x(self.max_stages),
            'task_id': self.discrete_x(self.max_tasks)
        })

        self.stage_space = Dict({
            'id_': self.discrete_x(self.max_stages),
            'job_id': self.discrete_x(self.max_jobs),
            'n_tasks': self.discrete_i(self.max_tasks),
            'n_completed_tasks': self.discrete_i(self.max_tasks),
            'task_duration': self.time_space,
            'worker_types_mask': MultiBinary(self.n_worker_types),
            'worker_ids': MultiDiscrete(self.max_tasks * [self.n_workers+1]),
            't_accepted': self.time_space,
            't_completed': self.time_space
        })

        self.job_space = Dict({
            'id_': self.discrete_x(self.max_jobs),
            'dag': self.dag_space,
            't_arrival': self.time_space,
            'stages': Tuple(self.max_stages * [self.stage_space]),
            'n_stages': self.discrete_i(self.max_stages)
        })

        self.observation_space = Dict({
            'wall_time': self.time_space,
            'jobs': Tuple(self.max_jobs * [self.job_space]),
            'job_count': self.discrete_i(self.max_jobs),
            'workers': Tuple(self.n_workers * [self.worker_space]),
            'frontier_stages_mask': self.stages_mask_space,
            'saturated_stages_mask': self.stages_mask_space
        })
    

    def _init_action_space(self):
        self.action_space = Dict({
            'job_id': self.discrete_x(self.max_jobs),
            'stage_id': self.discrete_x(self.max_stages),
            'worker_type_counts': MultiDiscrete(
                self.n_worker_types * [self.n_workers])
        })


    def _construct_null_entities(self):
        '''returns a 'null' observation where
        - exclusive discrete attributes are set to largest (invalid) value
        - invlusive discrete attributes are set to zero
        - time attributes are set to infinity
        - multi-binary attributes are zeroed out
        in object form. convert to dict using `asdict`
        '''
        self.null_worker = Worker(
            id_=Worker.invalid_id, 
            type_=Worker.invalid_type, 
            job_id=Job.invalid_id, 
            stage_id=Stage.invalid_id, 
            task_id=Stage.invalid_task_id)

        self.null_stage = Stage(
            id_=Stage.invalid_id,
            job_id=Job.invalid_id,
            n_tasks=0, 
            n_completed_tasks=0,
            task_duration=invalid_time(),
            worker_types_mask=np.array(self.n_worker_types*[0]), 
            worker_ids=np.array(self.max_tasks*[Worker.invalid_id]),
            t_accepted=invalid_time(),
            t_completed=invalid_time()
        )        

        self.null_job = Job(
            id_=Job.invalid_id,
            dag=np.zeros(triangle(self.max_stages)), 
            t_arrival=invalid_time(),
            stages=tuple([dcp(self.null_stage) for _ in range(self.max_stages)]),
            n_stages=0
        )

        null_jobs = [dcp(self.null_job) for _ in range(self.max_jobs)]
        null_workers = [dcp(self.null_worker) for _ in range(self.n_workers)]
        null_stages_mask = np.zeros(self.max_jobs * self.max_stages)

        self.null_state = DagSchedState(
            wall_time=to_wall_time(0),
            jobs=tuple(null_jobs),
            job_count=0,
            workers=tuple(null_workers),
            frontier_stages_mask=null_stages_mask.copy(),
            saturated_stages_mask=null_stages_mask.copy(),
        )


    def _init_timeline(self):
        self.timeline = Timeline()
        t = 0.
        for id_ in range(self.max_jobs):
            t += np.random.exponential(self.mean_job_interarrival_time)
            job = self._generate_job(id_, to_wall_time(t))
            self.timeline.push(t, JobArrival(job))

        for i, worker in enumerate(self.state.workers):
            worker.id_ = i
            worker.type_ = np.random.randint(low=0, high=self.n_worker_types)


    def _generate_job(self, id_, t_arrival):
        stages, n_stages = self._generate_stages(id_)
        dag = self.dag_space.sample()
        job = Job(
            id_=id_,
            dag=dag,
            t_arrival=t_arrival,
            stages=stages,
            n_stages=n_stages
        )
        return job


    def _generate_stages(self, job_id):
        n_stages = np.random.randint(low=2, high=self.max_stages+1)
        stages = []
        for i in range(n_stages):
            n_tasks = np.random.randint(low=1, high=self.max_tasks+1)
            duration = np.random.normal(loc=20., scale=2.)
            stages += [Stage(
                id_=i,
                job_id=job_id,
                n_tasks=n_tasks,
                n_completed_tasks=0,
                task_duration=to_wall_time(duration),
                worker_types_mask=self._generate_worker_types_mask(), 
                worker_ids=np.array(self.max_tasks*[Worker.invalid_id]),
                t_accepted=invalid_time(),
                t_completed=invalid_time()
            )]

        stages += (self.max_stages-n_stages) * [self.null_stage]
        assert len(stages) == self.max_stages
        stages = tuple(stages)
        return stages, n_stages


    def _generate_worker_types_mask(self):
        n_types = np.random.randint(low=1, high=self.n_worker_types+1)
        worker_types = np.array(n_types*[1] + (self.n_worker_types-n_types)*[0])
        np.random.shuffle(worker_types)
        return worker_types


    def _push_tasks(self, stage, task_ids):
        for task_id in task_ids:
            t_completion = \
                self.state.wall_time + stage.generate_task_duration()
            event = TaskCompletion(stage, task_id)
            self.timeline.push(t_completion[0], event)

    