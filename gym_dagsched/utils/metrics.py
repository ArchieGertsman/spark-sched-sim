import numpy as np

def avg_job_duration(env):
    durations = []
    for job in env.jobs.values():
        t_completed = min(job.t_completed, env.wall_time)
        duration = t_completed - job.t_arrival
        durations += [duration]
    return np.mean(durations)


def makespan(env):
    assert env.n_completed_jobs > 0
    completion_times = np.array([
        env.jobs[j].t_completed
        for j in env.completed_job_ids
    ])
    completion_times = completion_times[completion_times<np.inf]
    return completion_times.max()