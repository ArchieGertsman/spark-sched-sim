from dataclasses import asdict
import copy

import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box
import numpy as np
from dacite import from_dict
import networkx as nx

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
        self.current_state = copy.deepcopy(self.null_obs)
        return asdict(self.current_state)


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

        # sample time until next job arrival
        dt = np.random.exponential(self.mean_job_interarrival_time)
        dt = self.to_wall_time(dt)

        # check if event (2) occurs before next arrival
        
        soonest_completed_stage, soonest_t_completion = \
            self.get_soonest_stage_completion(dt)

        if soonest_completed_stage is not None:
            self.process_stage_completion(soonest_completed_stage)
            self.current_state.wall_time = soonest_t_completion
        else:
            job = self.generate_job()
            self.add_src_nodes_to_frontier(job)
            self.current_state.wall_time += dt

        obs = asdict(self.current_state)
        return obs, None, None, None



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
            't_accepted': self.time_space,
            'n_workers': self.discrete_inv_space(self.n_workers)
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
            duration=self.invalid_time(),
            t_accepted=self.invalid_time(),
            n_workers=-1
        )        

        self.null_job = Job(
            id_=-1,
            dag=np.zeros(self.triangle(self.max_stages)), 
            t_arrival=self.invalid_time(),
            stages=tuple(self.max_stages*[self.null_stage]),
            n_stages=-1
        )

        self.null_obs = Obs(
            wall_time=self.to_wall_time(0),
            jobs=tuple(self.max_jobs * [self.null_job]),
            job_count=0,
            frontier_stages_mask=np.zeros(self.max_jobs * self.max_stages),
            workers=tuple(self.n_workers * [self.null_worker])
        )


    def to_wall_time(self, t):
        """converts a float to a singleton array to
        comply with the Box space"""
        assert(t >= 0)
        return np.array([t], dtype=np.float32)

    def invalid_time(self):
        '''invalid time is defined to be infinity.'''
        return self.to_wall_time(np.inf)


    def get_stage_idx(self, job_id, stage_id):
        return job_id * self.max_stages + stage_id


    def get_stage_from_idx(self, stage_idx):
        stage_id = stage_idx % self.max_stages
        job_id = (stage_idx - stage_id) // self.max_stages
        return self.current_state.jobs[job_id].stages[stage_id]


    def generate_job(self, t_arrival):
        id_ = self.current_state.job_count
        stages, n_stages = self.generate_stages(id_)
        dag = self.dag_space.sample()
        job = Job(
            id_=id_,
            dag=dag,
            t_arrival=t_arrival,
            stages=stages,
            n_stages=n_stages
        )
        self.current_state.job_count += 1
        return job


    def generate_stages(self, job_id):
        n_stages = np.random.randint(low=2, high=self.max_stages+1)
        stages = []
        for i in range(n_stages):
            n_tasks = np.random.randint(low=1, high=self.max_tasks)
            worker_type = np.random.randint(low=1, high=self.n_worker_types)
            duration = np.random.normal(loc=8., scale=4., size=(1,))
            stages += [Stage(
                id_=i,
                job_id=job_id,
                n_tasks=n_tasks, 
                worker_type=worker_type, 
                duration=duration,
                t_accepted=self.invalid_time(),
                n_workers=0
            )]

        stages += (self.max_stages-n_stages) * [self.null_stage]
        assert(len(stages) == self.max_stages)
        stages = tuple(stages)
        return stages, n_stages

    def job_dag_to_nx(self, job):
        # construct adjacency matrix from flattend
        # lower triangle array
        n = self.max_stages
        T = np.zeros((n,n))
        T[np.tril_indices(n,-1)] = job.dag

        # truncate adjacency matrix to only include valid nodes
        n = job.n_stages
        T = T[:n,:n]

        G = nx.convert_matrix.from_numpy_matrix(T, create_using=nx.DiGraph)
        assert(nx.is_directed_acyclic_graph(G))
        return G


    def find_src_nodes(self, job):
        '''`dag` is a flattened lower triangle'''
        G = self.job_dag_to_nx(job)
        sources = [node for node,in_deg in G.in_degree() if in_deg==0]
        return sources


    def add_src_nodes_to_frontier(self, job):
        source_ids = self.find_src_nodes(job)
        source_ids = np.array(source_ids)
        indices = job.id_ * self.max_stages + source_ids
        self.current_state.frontier_stages[indices] = 1


    def check_action_validity(self, action):
        stage = self.current_state.jobs[action.job_id].stages[action.stage_id]

        worker_indices = (action.workers==1)
        for i in worker_indices:
            worker = self.current_state.workers[i]
            if worker.job != -1 or worker.type_ != stage.worker_type:
                # either one of the selected workers is currently busy,
                # or worker type is not suitible for stage
                return False

        stage_idx = self.get_stage_idx(action.job_id, action.stage_id)
        if not self.stage_in_frontier(stage_idx):
            return False

        return True


    def add_stage_to_frontier(self, stage_idx):
        self.current_state.frontier_stages_mask[stage_idx] = 1

    def remove_stage_from_frontier(self, stage_idx):
        self.current_state.frontier_stages_mask[stage_idx] = 0

    def stage_in_frontier(self, stage_idx):
        return self.current_state.frontier_stages[stage_idx]


    def take_action(self, action):
        worker_indices = (action.workers_mask==1)
        for i in worker_indices:
            worker = self.current_state.workers[i]
            worker.job = action.job

        stage_idx = self.get_stage_idx(action.job_id, action.stage_id)
        self.add_stage_to_frontier(stage_idx)

        stage = self.current_state.jobs[action.job_id].stages[action.stage_id]
        stage.n_workers = worker_indices.size


    def process_stage_completion(self, stage):
        # TODO
        pass


    def get_soonest_stage_completion(self, dt):
        '''if there are any stage completions within the time 
        interval `dt`, then return the soonest one, along with
        its time of completion. Otherwise, return (None, invalid time)
        '''
        soonest_completed_stage, soonest_t_completion = \
            None, self.invalid_time()

        msk = self.current_state.frontier_stages_mask
        stage_indices = msk[msk==1]

        for stage_idx in stage_indices:
            stage = self.get_stage_from_idx(stage_idx)

            # if this stage hasn't even started processing then move on
            if stage.t_accepted == self.invalid_time():
                continue

            # expected completion time of this task
            t_completion = stage.t_accepted + stage.duration

            # search for stage with soonest completion time, if
            # such a stage exists.
            if (t_completion - self.current_state.wall_time) < dt:
                # stage has completed within the `dt` interval
                if t_completion < soonest_t_completion:
                    soonest_completed_stage = stage
                    soonest_t_completion = t_completion

        return soonest_completed_stage, soonest_t_completion