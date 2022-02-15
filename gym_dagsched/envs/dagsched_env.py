import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box
import numpy as np
from dataclasses import asdict
import dacite
import copy
from .entities import *

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

        self.triangle = lambda n: n*(n-1)//2
        self.inv_time = np.array([np.inf], dtype=np.float32)

        self._init_observation_space()
        self._init_action_space()

        self._construct_null_entities()

        self.wall_time = 0.


    def _init_observation_space(self):
        # time can be any non-negative real number
        time_space = Box(low=np.array([0]), high=np.array([np.inf]))

        # discrete space of size `n` with an additional 
        # invalid state encoded as -1
        discrete_inv_space = lambda n: Discrete(n+1, start=-1)

        # upper triangle of the dag's adgacency matrix stored 
        # as a flattened array
        dag_space = MultiBinary(self.triangle(self.max_stages))

        # frontier_stages[i] = 1[stage `i` is in the frontier]
        frontier_stages_space = MultiBinary(self.max_jobs * self.max_stages)

        # see entities.py for details about the attributes in
        # the following spaces

        self.worker_space = Dict({
            'type_': discrete_inv_space(self.n_worker_types),
            'job': discrete_inv_space(self.max_jobs)
        })

        self.stage_space = Dict({
            'id_': discrete_inv_space(self.max_stages),
            'n_tasks': discrete_inv_space(self.max_tasks),
            'worker_type': discrete_inv_space(self.n_workers),
            't_completion': time_space,
            't_accepted': time_space,
            'n_workers': discrete_inv_space(self.n_workers)
        })

        self.job_space = Dict({
            'dag': dag_space,
            't_arrival': time_space,
            'stages': Tuple(self.max_stages * [self.stage_space]),
            'n_stages': discrete_inv_space(self.max_stages)
        })

        self.observation_space = Dict({
            'jobs': Tuple(self.max_jobs * [self.job_space]),
            'frontier_stages': frontier_stages_space,
            'workers': Tuple(self.n_workers * [self.worker_space])
        })
    

    def _init_action_space(self):
        self.action_space = gym.spaces.Dict({
            'stage': Discrete(self.max_stages),
            'workers': MultiBinary(self.n_workers)
        })


    def _construct_null_entities(self):
        '''returns an 'null' observation where
        - all discrete attributes are set to -1
        - all time attributes are set to infinity
        - all multi-binary attributes are zeroed out
        in object form. convert to dict using `asdict`
        '''
        self.null_worker = Worker(type_=-1, job=-1)

        self.null_stage = Stage(
            id_=-1, 
            n_tasks=-1, 
            worker_type=-1, 
            t_completion=self.inv_time,
            t_accepted=self.inv_time,
            n_workers=-1
        )        

        self.null_job = Job(
            dag=np.zeros(self.triangle(self.max_stages)), 
            t_arrival=self.inv_time,
            stages=tuple(self.max_stages*[self.null_stage]),
            n_stages=-1
        )

        self.null_obs = Obs(
            jobs=tuple(self.max_jobs * [self.null_job]),
            frontier_stages=np.zeros(self.max_jobs * self.max_stages),
            workers=tuple(self.n_workers * [self.null_worker])
        )


    def generate_job(self, t_arrival):
        stages, n_stages = self.generate_stages()
        # job = Job(
        #     dag=dag,
        #     t_arrival=t_arrival,
        #     stages=stages,
        #     n_stages=n_stages
        # )


    def generate_stages(self):
        n_stages = np.random.randint(low=2, high=self.max_stages+1)
        stages = []
        for i in range(n_stages):
            n_tasks = np.random.randint(low=1, high=self.max_tasks)
            worker_type = np.random.randint(low=1, high=self.n_worker_types)
            t_completion = np.random.normal(loc=8., scale=4.)
            stages += [Stage(
                id_=i, 
                n_tasks=n_tasks, 
                worker_type=worker_type, 
                t_completion=t_completion,
                t_accepted=self.inv_time,
                n_workers=0
            )]

        stages += (self.max_stages-n_stages) * [self.null_stage]
        assert(len(stages) == self.max_stages)
        stages = tuple(stages)
        return stages, n_stages


    def reset(self):
        self.current_state = copy.deepcopy(self.null_obs)
        return self.current_state


    def step(self, action):
        '''steps onto the next scheduling event, which can
        be one of the following:
        (1) new job arrival
            - add stages to frontier
        (2) stage execution completed
            - remove stage from frontier
        wall time is updated
        '''
        obs, reward, done, info = [None] * 4


        # sample time until next job arrival
        dt = np.random.exponential(self.mean_job_interarrival_time)

        # check if event (2) occurs before next arrival
        for stage in self.current_state.frontier_stages:
            # if this stage hasn't even started processing then move on
            if stage.t_accepted == self.inv_time:
                continue

            if (stage.t_accepted + stage.t_completion - self.wall_time) < dt:
                pass

        # event (2) doesn't occur, so generate a new job

        job = self.generate_job()

        # self.wall_time += dt

        return obs, reward, done, info
