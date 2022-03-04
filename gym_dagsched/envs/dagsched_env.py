from copy import deepcopy as dcp

from gym import Env
import numpy as np
from dacite import from_dict

from ..args import args
from ..entities.action import Action, action_space
from ..entities.dagsched_state import DagSchedState, dagsched_state_space
from ..utils.misc import to_wall_time
from ..utils.timeline import Timeline, JobArrival, TaskCompletion
from ..utils import data_generator as gen


class DagSchedEnv(Env):

    def __init__(self):
        self.observation_space = dagsched_state_space
        self.action_space = action_space


    """
    abstract method implementations
    """

    def reset(self):
        self.state = DagSchedState()
        self._init_timeline()
        self._init_workers()
        return dcp(self.state)


    def step(self, action_dict):
        '''steps into the next scheduling event, which can
        be one of the following:
        (1) new job arrival
        (2) task completion
        (3) "nudge," meaning that there are available actions,
            even though neither (1) nor (2) have occurred, so 
            the policy should consider taking one of them
        '''
        action = from_dict(Action, action_dict)
        self._process_action(action)

        # if there are still actions available after
        # processing the most recent once, then push 
        # a "nudge" event to notify the scheduling agent
        # that another action can immediately be taken
        if self.state.actions_available():
            self._push_nudge()

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
        self.timeline = Timeline()
        t = 0.
        for id_ in range(args.n_jobs):
            t += np.random.exponential(args.mjit)
            job = gen.generate_job(id_, to_wall_time(t))
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


    def _process_action(self, action):
        if self.state.is_action_valid(action):
            stage, task_ids = self.state.take_action(action)
            self._push_tasks(stage, task_ids)
        else:
            print('invalid action')


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