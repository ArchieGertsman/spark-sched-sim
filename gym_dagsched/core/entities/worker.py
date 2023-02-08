from dataclasses import dataclass


@dataclass
class Worker:
    id_: int
    task = None
    job_id = None
    op_id = None
    history = None


    def add_history(self, wall_time, job_id):
        if self.history is None:
            self.history = []

        if len(self.history) > 0:
            self.history[-1][0] = wall_time
        
        self.history += [[None, job_id]]


    @property
    def is_free(self):
        return self.task == None

    
    @property
    def pool_key(self):
        return (self.job_id, self.op_id)

    
    def is_at_job(self, job_id):
        return self.job_id == job_id

    

