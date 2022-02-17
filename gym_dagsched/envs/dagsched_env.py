from dataclasses import asdict
from copy import deepcopy as dcp

import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box
import numpy as np
from dacite import from_dict

from .entities.action import Action
from .entities.obs import Obs
from .entities.job import Job
from .entities.stage import Stage
from .entities.worker import Worker
from .utils import invalid_time, to_wall_time

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

        # time can be any non-negative real number
        self.time_space = Box(low=np.array([0]), high=np.array([np.inf]))

        # discrete space of size `n` with an additional 
        # invalid state encoded as -1
        self.discrete_inv_space = lambda n: Discrete(n+1, start=-1)

        self._init_observation_space()
        self._init_action_space()

        self._construct_null_entities()



    """
    abstract method implementations
    """

    def reset(self):
        self.state = dcp(self.null_obs)
        self._generate_workers()
        return asdict(self.state)


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
        if self.check_action_validity(action):
            self.take_action(action)

        info = {}

        # sample time until next job arrival
        if self.state.job_count < self.max_jobs:
            dt = np.random.exponential(self.mean_job_interarrival_time)
            dt = to_wall_time(dt)
        else:
            dt = invalid_time()

        # check if event (2) occurs before next arrival
        soonest_completed_stage, soonest_t_completion = \
            self.get_soonest_stage_completion(dt)

        if soonest_completed_stage is not None:
            self.state.wall_time = soonest_t_completion
            self.process_stage_completion(soonest_completed_stage)
            info['event'] = 'stage completion'
        elif dt != invalid_time():
            self.state.wall_time += dt
            job = self.generate_job()
            self.state.add_job(job)
            info['event'] = 'new job arrival'
        else:
            info['event'] = 'bruh'

        # obs = asdict(self.state)
        obs = self.state
        return obs, None, None, info



    """
    helper functions
    """


    def _init_observation_space(self):
        # lower triangle of the dag's adgacency matrix stored 
        # as a flattened array
        self.dag_space = MultiBinary(self.triangle(self.max_stages))

        # frontier_stages[i] = 1[stage `i` is in the frontier]
        frontier_stages_space = MultiBinary(self.max_jobs * self.max_stages)

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
            'worker_type': self.discrete_inv_space(self.n_workers),
            'duration': self.time_space,
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
            'frontier_stages_mask': frontier_stages_space,
            'workers': Tuple(self.n_workers * [self.worker_space])
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
            worker_type=-1, 
            duration=invalid_time(),
            n_workers=-1,
            t_accepted=invalid_time(),
            t_completed=invalid_time()
        )        

        self.null_job = Job(
            id_=-1,
            dag=np.zeros(self.triangle(self.max_stages)), 
            t_arrival=invalid_time(),
            stages=tuple([dcp(self.null_stage) for _ in range(self.max_stages)]),
            n_stages=-1
        )

        null_jobs = [dcp(self.null_job) for _ in range(self.max_jobs)]
        null_workers = [dcp(self.null_worker) for _ in range(self.n_workers)]

        self.null_obs = Obs(
            wall_time=to_wall_time(0),
            jobs=tuple(null_jobs),
            job_count=0,
            frontier_stages_mask=np.zeros(self.max_jobs * self.max_stages),
            workers=tuple(null_workers)
        )


    def _generate_workers(self):
        for worker in self.state.workers:
            worker.type_ = np.random.randint(low=1, high=self.n_worker_types+1)


    def generate_job(self):
        id_ = self.state.job_count
        stages, n_stages = self.generate_stages(id_)
        dag = self.dag_space.sample()
        t_arrival = self.state.wall_time.copy()
        job = Job(
            id_=id_,
            dag=dag,
            t_arrival=t_arrival,
            stages=stages,
            n_stages=n_stages
        )
        return job


    def generate_stages(self, job_id):
        n_stages = np.random.randint(low=2, high=self.max_stages+1)
        stages = []
        for i in range(n_stages):
            n_tasks = np.random.randint(low=1, high=self.max_tasks+1)
            worker_type = np.random.randint(low=1, high=self.n_worker_types+1)
            duration = np.random.normal(loc=8., scale=2.)
            stages += [Stage(
                id_=i,
                job_id=job_id,
                n_tasks=n_tasks, 
                worker_type=worker_type, 
                duration=to_wall_time(duration),
                n_workers=0,
                t_accepted=invalid_time(),
                t_completed=invalid_time()
            )]

        stages += (self.max_stages-n_stages) * [self.null_stage]
        assert(len(stages) == self.max_stages)
        stages = tuple(stages)
        return stages, n_stages


    def check_action_validity(self, action):
        stage = self.state.jobs[action.job_id].stages[action.stage_id]

        workers = self.state.get_workers_from_mask(action.workers_mask)
        for worker in workers:
            if worker.job_id != -1 or worker.type_ != stage.worker_type:
                # either one of the selected workers is currently busy,
                # or worker type is not suitible for stage
                return False

        stage_idx = self.state.get_stage_idx(action.job_id, action.stage_id)
        if not self.state.stage_in_frontier(stage_idx):
            return False

        return True


    def take_action(self, action):
        workers = self.state.get_workers_from_mask(action.workers_mask)
        for worker in workers:
            worker.job_id = action.job_id

        stage_idx = self.state.get_stage_idx(action.job_id, action.stage_id)
        self.state.add_stage_to_frontier(stage_idx)

        stage = self.state.jobs[action.job_id].stages[action.stage_id]
        stage.n_workers = len(workers)
        stage.t_accepted = self.state.wall_time.copy()


    def process_stage_completion(self, stage):
        stage.t_completed = self.state.wall_time.copy()

        stage_idx = self.state.get_stage_idx(stage.job_id, stage.id_)
        self.state.remove_stage_from_frontier(stage_idx)

        job = self.state.jobs[stage.job_id]
        new_frontier_stages = job.find_new_frontiers(stage)
        for new_stage in new_frontier_stages:
            self.state.add_stage_to_frontier(new_stage)

        # free the workers
        for worker in self.state.workers:
            pass
        


    def get_soonest_stage_completion(self, dt):
        '''if there are any stage completions within the time 
        interval `dt`, then return the soonest one, along with
        its time of completion. Otherwise, return (None, invalid time)
        '''
        soonest_completed_stage, soonest_t_completion = \
            None, invalid_time()

        frontier_stages = self.state.get_frontier_stages()

        for stage in frontier_stages:
            # if this stage hasn't even started processing then move on
            if stage.t_accepted == invalid_time():
                continue

            # expected completion time of this task
            t_completion = stage.t_accepted + stage.duration

            # search for stage with soonest completion time, if
            # such a stage exists.
            if (t_completion - self.state.wall_time) < dt:
                # stage has completed within the `dt` interval
                if t_completion < soonest_t_completion:
                    soonest_completed_stage = stage
                    soonest_t_completion = t_completion

        return soonest_completed_stage, soonest_t_completion