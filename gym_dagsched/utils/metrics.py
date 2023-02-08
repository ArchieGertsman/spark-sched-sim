import numpy as np




def avg_job_duration(env):
    # assert env.num_completed_jobs > 0

    durations = []
    for job_id in env.active_job_ids + list(env.completed_job_ids):
        job = env.jobs[job_id]
        t_end = min(job.t_completed, env.wall_time)
        durations += [t_end - job.t_arrival]

    return np.mean(durations)


    # jobs = (env.jobs[job_id] 
    #         for job_id in env.completed_job_ids)

    # return np.mean([job.t_completed - job.t_arrival 
    #                 for job in jobs])



def makespan(env):
    assert env.num_completed_jobs > 0

    jobs = (env.jobs[job_id] 
            for job_id in env.completed_job_ids)

    return max(job.t_completed for job in jobs)