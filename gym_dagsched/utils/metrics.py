import numpy as np

def avg_job_duration(env):
    durations = np.array([
        (env.jobs[j].t_completed - env.jobs[j].t_arrival)
        for j in env.completed_job_ids
    ])
    return durations.mean()


def makespan(env):
    assert env.n_completed_jobs > 0
    completion_times = np.array([
        env.jobs[j].t_completed
        for j in env.completed_job_ids
    ])
    completion_times = completion_times[completion_times<np.inf]
    return completion_times.max()