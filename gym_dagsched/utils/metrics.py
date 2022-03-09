import numpy as np

def avg_job_duration(sys_state):
    durations = np.array([
        (job.t_completed - job.t_arrival)
        for job in sys_state.jobs
    ])
    return durations.mean()


def makespan(sys_state):
    assert sys_state.n_completed_jobs > 0
    completion_times = np.array([
        job.t_completed[0] 
        for job in sys_state.jobs
    ])
    completion_times = completion_times[completion_times<np.inf]
    return completion_times.max()