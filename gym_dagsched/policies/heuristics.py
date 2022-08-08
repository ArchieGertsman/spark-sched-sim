import numpy as np


def _pick_first(env, key=None, reverse=False):
    '''sorts the frontier stages by `key`, if provided,
    then selects the first stage in the sorted frontier 
    for which there is at least one available, compatible
    worker.
    '''
    frontier_ops = list(env.frontier_ops)
    if key is not None:
        frontier_ops.sort(key=key, reverse=reverse)

    avail_workers = env._find_available_workers()
    
    first_op = None

    for op in frontier_ops:
        if first_op is not None:
            break
        for worker in avail_workers:
            if worker.compatible_with(op):
                first_op = op
                break


    if first_op is not None:
        # job = env.jobs[first_op.job_id]
        # n_avail_local = sum([int(env.workers[worker_id].task is None) for worker_id in list(job.local_workers)])
        # n_workers = len(job.local_workers) + max(0, int(first_op.n_remaining_tasks) - n_avail_local)
        n_workers = len(env.workers)
    else:
        n_workers = 0
    
    return first_op, n_workers


def fcfs(obs):
    '''selects a frontier stage whose job arrived first'''
    return _pick_first(obs)


def shortest_task_first(obs):
    '''selects a frontier stage whose task duration is shortest'''
    def key(stage):
        durations = stage.task_duration_per_worker_type
        durations = durations[durations<np.inf]
        return durations.mean()
    return _pick_first(obs, key)


def longest_task_first(obs):
    '''selects a frontier stage whose task duration is longest'''
    def key(stage):
        durations = stage.task_duration_per_worker_type
        durations = durations[durations<np.inf]
        return durations.mean()
    return _pick_first(obs, key, reverse=True)


def max_children(env):
    def key(op):
        job = env.jobs[op.job_id]
        return len(list(job.dag.successors(op.id_)))
    return _pick_first(env, key, reverse=True)


def srt(env):
    def key(op):
        return op.remaining_time
    return _pick_first(env, key)


def lrt(env):
    def key(op):
        return op.remaining_time
    return _pick_first(env, key, reverse=True)