import numpy as np


def job_durations(env):
    durations = []
    for job_id in env.unwrapped.active_job_ids + list(env.unwrapped.completed_job_ids):
        job = env.unwrapped.jobs[job_id]
        t_end = min(job.t_completed, env.unwrapped.wall_time)
        durations += [t_end - job.t_arrival]
    return durations


def avg_job_duration(env):
    return np.mean(job_durations(env))


def avg_num_jobs(env):
    return sum(job_durations(env)) / env.unwrapped.wall_time


def job_duration_percentiles(env):
    jd = job_durations(env)
    return np.percentile(jd, [25, 50, 75, 100])
