import numpy as np




def avg_job_duration(env):
    durations = []
    for job_id in env.active_job_ids + list(env.completed_job_ids):
        job = env.jobs[job_id]
        t_end = min(job.t_completed, env.wall_time)
        durations += [t_end - job.t_arrival]
    return np.mean(durations)


def avg_num_jobs(env):
    total_work = 0
    for job_id in env.active_job_ids + list(env.completed_job_ids):
        job = env.jobs[job_id]
        t_end = min(job.t_completed, env.wall_time)
        total_work += t_end - job.t_arrival
    return total_work / env.wall_time


def makespan(env):
    assert env.num_completed_jobs > 0

    jobs = (env.jobs[job_id] 
            for job_id in env.completed_job_ids)

    return max(job.t_completed for job in jobs)