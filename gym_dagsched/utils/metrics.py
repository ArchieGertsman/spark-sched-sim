import numpy as np

def avg_job_duration(env):
    durations = np.array([
        (job.t_completed - job.t_arrival)
        for job in env.jobs if job.t_completed < np.inf
    ])
    return durations.mean()


def makespan(env):
    assert env.n_completed_jobs > 0
    completion_times = np.array([
        job.t_completed
        for job in env.jobs
    ])
    completion_times = completion_times[completion_times<np.inf]
    return completion_times.max()