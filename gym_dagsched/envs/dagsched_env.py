from dataclasses import asdict

from gym import Env
import numpy as np
from dacite import from_dict

from ..args import args
from ..entities.sys_state import SysState, sys_state_space
from ..entities.action import Action, action_space
from ..utils.misc import to_wall_time
from ..utils.timeline import Timeline, JobArrival, TaskCompletion
from ..utils import data_generator as gen


class DagSchedEnv(Env):
    '''An environment for scheduling streaming jobs with interdependent stages.'''

    def __init__(self):
        self.observation_space = sys_state_space
        self.action_space = action_space


    """
    abstract method implementations
    """

    def reset(self):
        self.state = SysState()
        self._init_timeline()
        self._init_workers()
        return asdict(self.state)


    def step(self, action):
        '''steps into the next scheduling event on the timeline, 
        which can be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            even though neither (1) nor (2) have occurred, so 
            the policy should consider taking one of them
        '''
        if self.state.is_action_valid(action):
            stage, task_ids = self.state.take_action(action)
            self._push_task_completion_events(stage, task_ids)
        else:
            print('invalid action')

        # if there are still actions available after
        # processing the most recent one, then push 
        # a "nudge" event to notify the scheduling agent
        # that another action can immediately be taken
        if self.state.actions_available():
            print('pushing nudge')
            self._push_nudge_event()

        # check if simulation is done
        if self.timeline.empty:
            assert self.state.all_jobs_complete
            print('all jobs completed!')
            return self.state, None, True, None
            
        # retreive the next scheduling event from the timeline
        t, event = self.timeline.pop()

        self._process_scheduling_event(t, event)

        return self.state, None, False, None



    """
    helper functions
    """


    def _init_timeline(self):
        '''Fills timeline with job arrival events, which follow
        a Poisson process, parameterized by args.mjit (mean job
        interarrival time)
        '''
        self.timeline = Timeline()

        # time of current arrival
        t = 0. 

        for id_ in range(args.n_jobs):
            # sample time until next arrival
            dt_interarrival = np.random.exponential(args.mjit)
            t += dt_interarrival

            # generate a job and add its arrival to the timeline
            job = gen.generate_job(id_, to_wall_time(t))
            self.timeline.push(t, JobArrival(job))


    def _init_workers(self):
        '''Initializes the workers with randomly generated attributes'''
        for i, worker in enumerate(self.state.workers):
            gen.generate_worker(worker, i)


    def _push_task_completion_events(self, stage, task_ids):
        '''Given a list of task ids and the stage they belong to,
        pushes their completions as events to the timeline
        '''
        for task_id in task_ids:
            task = stage.tasks[task_id]
            assigned_worker_id = task.worker_id
            worker_type = self.state.workers[assigned_worker_id].type_
            t_completion = \
                task.t_accepted + gen.generate_task_duration(stage, worker_type)
            t_completion = t_completion[0]
            event = TaskCompletion(stage, task_id)
            self.timeline.push(t_completion, event)


    def _push_nudge_event(self):
        '''Pushes a "nudge" event to the timeline at the current
        wall time, so that the scheduling agent can immediately
        choose another action
        '''
        self.timeline.push(self.state.wall_time[0], None)


    def _process_scheduling_event(self, t, event):
        # update the current wall time
        self.state.wall_time = to_wall_time(t)

        if isinstance(event, JobArrival):
            job = event.obj
            print(f'{t}: job arrival')
            self.state.add_job(job)
        elif isinstance(event, TaskCompletion):
            stage, task_id = event.obj, event.task_id
            print(f'{t}: task completion', f'({stage.job_id},{stage.id_},{task_id})')
            self.state.process_task_completion(stage, task_id)
        else:
            print(f'{t}: nudge')