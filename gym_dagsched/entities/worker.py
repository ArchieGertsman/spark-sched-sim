from dataclasses import dataclass


@dataclass
class Worker:
    id_: int
    task = None
    job_id = None
    op_id = None


    @property
    def available(self):
        return self.task == None

    
    @property
    def pool_key(self):
        return (self.job_id, self.op_id)

    
    def is_at_job(self, job_id):
        return self.job_id == job_id

    

