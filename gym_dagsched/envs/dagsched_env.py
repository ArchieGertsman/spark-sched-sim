# from dataclasses import 
from copy import deepcopy as dcp


import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box
import numpy as np
from dacite import from_dict

from .entities.action import Action
from .entities.dagsched_state import DagSchedState
from .entities.job import Job
from .entities.stage import Stage
from .entities.worker import Worker
from .utils import invalid_time, to_wall_time, triangle
from .timeline import Timeline

class DagSchedEnv(gym.Env):
    def __init__(self,
        max_jobs, max_stages, max_tasks, n_worker_types, n_workers,
        mean_job_interarrival_time=10.):
        # fix a maximum state size
        self.max_jobs = max_jobs
        self.max_stages = max_stages
        self.max_tasks = max_tasks

        # fixed workers
        self.n_workers = n_workers
        self.n_worker_types = n_worker_types

        self.mean_job_interarrival_time = mean_job_interarrival_time

        self._construct_null_entities()

        self._init_observation_space()
        self._init_action_space()

        self.prev_t = 0




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
            - add stages to frontier
        (2) stage execution completed
            - remove stage from frontier
        wall time is updated
        '''
        action = from_dict(Action, action_dict)
        if self.state.check_action_validity(action):
            t_completion, stage = self.state.take_action(action)
            self.timeline.push(t_completion, stage)

        if self.state.actions_available():
            # TODO: add time it takes for agent to take
            # the action to `t` using real time
            self.timeline.push(self.prev_t, None)

        if self.timeline.empty:
            return None, None, True, None
            
        t, obj = self.timeline.pop()
        self.state.wall_time = to_wall_time(t)

        if isinstance(obj, Job):
            print(f'{t}: job arrival')
            job = obj
            self.state.add_job(job)
        elif isinstance(obj, Stage):
            print(f'{t}: task completion')
            stage = obj
            # worker = self.state.get_worker(task.worker_id)
            # worker.avail = 1
            self.state.process_stage_completion(stage)
        else:
            print(f'{t}: actions available')

        self.prev_t = t

        obs = dcp(self.state)
        return obs, None, False, None



    """
    helper functions
    """


    def _init_observation_space(self):
        # time can be any non-negative real number
        self.time_space = Box(low=0., high=np.inf, shape=(1,))

        # discrete space of size `n` with an additional 
        # invalid state encoded as -1
        self.discrete_inv_space = lambda n: Discrete(n+1, start=-1)

        # lower triangle of the dag's adgacency matrix stored 
        # as a flattened array
        self.dag_space = MultiBinary(triangle(self.max_stages))

        # frontier_stages[i] = 1[stage `i` is in the frontier]
        stages_mask_space = MultiBinary(self.max_jobs * self.max_stages)

        # see entities.py for details about the attributes in
        # the following spaces

        self.worker_space = Dict({
            'type_': self.discrete_inv_space(self.n_worker_types),
            'job_id': self.discrete_inv_space(self.max_jobs)
        })

        self.stage_space = Dict({
            'id_': self.discrete_inv_space(self.max_stages),
            'job_id': self.discrete_inv_space(self.max_jobs),
            'n_tasks': self.discrete_inv_space(self.max_tasks),
            'n_completed_tasks': self.discrete_inv_space(self.max_tasks),
            'task_duration': self.time_space,
            'worker_type': self.discrete_inv_space(self.n_workers),
            'n_workers': self.discrete_inv_space(self.n_workers),
            't_accepted': self.time_space,
            't_completed': self.time_space
        })

        self.job_space = Dict({
            'id_': self.discrete_inv_space(self.max_jobs),
            'dag': self.dag_space,
            't_arrival': self.time_space,
            'stages': Tuple(self.max_stages * [self.stage_space]),
            'n_stages': self.discrete_inv_space(self.max_stages)
        })

        self.observation_space = Dict({
            'wall_time': self.time_space,
            'jobs': Tuple(self.max_jobs * [self.job_space]),
            'job_count': Discrete(self.max_jobs),
            'workers': Tuple(self.n_workers * [self.worker_space]),
            'frontier_stages_mask': stages_mask_space,
            'saturated_stages_mask': stages_mask_space
        })
    

    def _init_action_space(self):
        self.action_space = gym.spaces.Dict({
            'job_id': self.discrete_inv_space(self.max_jobs),
            'stage_id': self.discrete_inv_space(self.max_stages),
            'workers_mask': MultiBinary(self.n_workers)
        })


    def _construct_null_entities(self):
        '''returns a 'null' observation where
        - all discrete attributes are set to -1
        - all time attributes are set to infinity
        - all multi-binary attributes are zeroed out
        in object form. convert to dict using `asdict`
        '''
        self.null_worker = Worker(type_=-1, job_id=-1)

        self.null_stage = Stage(
            id_=-1,
            job_id=-1,
            n_tasks=-1, 
            n_completed_tasks=-1,
            task_duration=invalid_time(),
            worker_type=-1, 
            n_workers=-1,
            t_accepted=invalid_time(),
            t_completed=invalid_time()
        )        

        self.null_job = Job(
            id_=-1,
            dag=np.zeros(triangle(self.max_stages)), 
            t_arrival=invalid_time(),
            stages=tuple([dcp(self.null_stage) for _ in range(self.max_stages)]),
            n_stages=-1
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
            self.timeline.push(t, job)

        for worker in self.state.workers:
            worker.type_ = np.random.randint(low=0, high=self.n_worker_types)
            # self.timeline.push


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
            worker_type = np.random.randint(low=0, high=self.n_worker_types)
            duration = np.random.normal(loc=20., scale=2.)
            stages += [Stage(
                id_=i,
                job_id=job_id,
                n_tasks=n_tasks, 
                n_completed_tasks=0,
                task_duration=to_wall_time(duration),
                worker_type=worker_type, 
                n_workers=0,
                t_accepted=invalid_time(),
                t_completed=invalid_time()
            )]

        stages += (self.max_stages-n_stages) * [self.null_stage]
        assert len(stages) == self.max_stages
        stages = tuple(stages)
        return stages, n_stages

        
    def _push_avail_workers(self, t):
        for worker in self.state.workers:
            if worker.is_available:
                self.timeline.push(t, worker)


    