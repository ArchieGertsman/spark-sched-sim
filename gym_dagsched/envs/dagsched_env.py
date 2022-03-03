from copy import deepcopy as dcp

import gym
from gym.spaces import Dict, Tuple, MultiBinary, Discrete, Box, MultiDiscrete
import numpy as np
from dacite import from_dict

from args import args
from entities.action import Action
from entities.dagsched_state import DagSchedState
from dagsched_utils import to_wall_time, triangle
from timeline import Timeline, JobArrival, TaskCompletion
import data_generator as gen


class DagSchedEnv(gym.Env):

    def __init__(self):
        self._init_spaces()


    """
    abstract method implementations
    """

    def reset(self):
        self.state = DagSchedState()
        self._init_timeline()
        self._init_workers()
        return dcp(self.state)


    def step(self, action_dict):
        '''steps onto the next scheduling event, which can
        be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            and so the policy should consider taking one of them
        '''
        action = from_dict(Action, action_dict)
        if self.state.is_action_valid(action):
            stage, task_ids = self.state.take_action(action)
            self._push_tasks(stage, task_ids)
        else:
            print('invalid action')

        if self.state.actions_available():
            self._push_nudge()

        if self.timeline.empty:
            if self.state.all_jobs_complete:
                print('all jobs completed!')
                return None, None, True, None
            else:
                assert self.state.actions_available()
                self._push_nudge()
            
        t, event = self.timeline.pop()
        self.state.wall_time = to_wall_time(t)

        if isinstance(event, JobArrival):
            job = event.obj
            print(f'{t}: job arrival')
            self.state.add_job(job)
        elif isinstance(event, TaskCompletion):
            stage, task_id = event.obj, event.task_id
            print(f'{t}: task completion', f'({stage.job_id},{stage.id_},{task_id})')
            self.state.process_task_completion(stage, task_id)

            if stage.is_complete:
                print('stage completion')
                self.state.process_stage_completion(stage)
            
            job = self.state.jobs[stage.job_id]
            if job.is_complete:
                print('job completion')
                self.state.process_job_completion(job)
        else:
            print(f'{t}: nudge')

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
        self.dag_space = MultiBinary(triangle(args.max_stages))

        self.stages_mask_space = MultiBinary(args.n_jobs * args.max_stages)

        self._init_observation_space()

        self._init_action_space()


    def _init_observation_space(self):
        # see entities folder for details about the attributes in
        # the following spaces

        self.worker_space = Dict({
            'id_': self.discrete_x(args.n_workers),
            'type_': self.discrete_x(args.n_worker_types),
            'job_id': self.discrete_x(args.n_jobs),
            'stage_id': self.discrete_x(args.max_stages),
            'task_id': self.discrete_x(args.max_tasks)
        })

        self.task_space = Dict({
            'worker_id': self.discrete_x(args.n_workers),
            'is_processing': self.discrete_i(1),
            't_accepted': self.time_space,
            't_completed': self.time_space
        })

        self.stage_space = Dict({
            'id_': self.discrete_x(args.max_stages),
            'job_id': self.discrete_x(args.n_jobs),
            'n_tasks': self.discrete_i(args.max_tasks),
            'n_completed_tasks': self.discrete_i(args.max_tasks),
            'task_duration': self.time_space,
            'worker_types_mask': MultiBinary(args.n_worker_types),
            'tasks': Tuple(args.max_tasks * [self.task_space])
        })

        self.job_space = Dict({
            'id_': self.discrete_x(args.n_jobs),
            'dag': self.dag_space,
            't_arrival': self.time_space,
            't_completed': self.time_space,
            'stages': Tuple(args.max_stages * [self.stage_space]),
            'n_stages': self.discrete_i(args.max_stages),
            'n_completed_stages': self.discrete_i(args.max_stages)
        })

        self.observation_space = Dict({
            'wall_time': self.time_space,
            'jobs': Tuple(args.n_jobs * [self.job_space]),
            'n_jobs': self.discrete_i(args.n_jobs),
            'n_completed_jobs': self.discrete_i(args.n_jobs),
            'workers': Tuple(args.n_workers * [self.worker_space]),
            'frontier_stages_mask': self.stages_mask_space,
            'saturated_stages_mask': self.stages_mask_space
        })
    

    def _init_action_space(self):
        self.action_space = Dict({
            'job_id': self.discrete_x(args.n_jobs),
            'stage_id': self.discrete_x(args.max_stages),
            'worker_type_counts': MultiDiscrete(
                args.n_worker_types * [args.n_workers])
        })


    def _init_timeline(self):
        self.timeline = Timeline()
        t = 0.
        for id_ in range(args.n_jobs):
            t += np.random.exponential(args.mjit)
            job = gen.generate_job(id_, to_wall_time(t), self.dag_space)
            self.timeline.push(t, JobArrival(job))


    def _init_workers(self):
        for i, worker in enumerate(self.state.workers):
            gen.generate_worker(worker, i)


    def _push_tasks(self, stage, task_ids):
        for task_id in task_ids:
            t_completion = \
                self.state.wall_time + gen.generate_task_duration(stage)
            event = TaskCompletion(stage, task_id)
            self.timeline.push(t_completion[0], event)


    def _push_nudge(self):
        self.timeline.push(self.state.wall_time[0], None)

    