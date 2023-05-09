import numpy as np


def job_durations(env):
    durations = []
    for job_id in env.active_job_ids + list(env.completed_job_ids):
        job = env.jobs[job_id]
        t_end = min(job.t_completed, env.wall_time)
        durations += [t_end - job.t_arrival]
    return durations


def avg_job_duration(env):
    return np.mean(job_durations(env))


def avg_num_jobs(env):
    return sum(job_durations(env)) / env.wall_time