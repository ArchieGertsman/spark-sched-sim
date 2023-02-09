from typing import Optional, List

from .task import Task


class Worker:
    
    def __init__(self, id_: int):
        # index of this operation within its operation
        self.id_ = id_

        # task object that this worker is currently executing, if any
        self.task: Optional[Task] = None

        # id of current job that this worker is local to, if any
        self.job_id: Optional[int] = None

        # id of current operation this worker is executing, if any
        self.op_id: Optional[int] = None

        # list of pairs [t, job_id], where
        # `t` is the wall time that this worker
        # was released from job with id `job_id`,
        # or `None` if it has not been released
        # yet. `job_id` is -1 if the worker is
        # at the general pool.
        # NOTE: only used for rendering
        self.history: List[list] = [[None, -1]]


    def add_history(self, wall_time, job_id):
        '''should be called whenever this worker is
        released from a job
        '''
        if self.history is None:
            self.history = []

        if len(self.history) > 0:
            # add release time to most recent history
            self.history[-1][0] = wall_time
        
        # add new history
        self.history += [[None, job_id]]


    @property
    def is_free(self):
        return self.task == None

    
    @property
    def pool_key(self):
        return (self.job_id, self.op_id)

    
    def is_at_job(self, job_id):
        return self.job_id == job_id

    

