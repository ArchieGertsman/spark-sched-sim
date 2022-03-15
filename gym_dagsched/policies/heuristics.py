import numpy as np

from ..args import args
from ..entities.action import Action


def _pick_first(obs, key=None, reverse=False):
    '''sorts the frontier stages by `key`, if provided,
    then selects the first stage in the sorted frontier 
    for which there is at least one available, compatible
    worker.
    '''
    frontier_stages = obs.get_frontier_stages()
    if key is not None:
        frontier_stages.sort(key=key, reverse=reverse)

    avail_workers = obs.find_available_workers()
    
    first_stage = None

    for stage in frontier_stages:
        if first_stage is not None:
            break
        for worker in avail_workers:
            if worker.compatible_with(stage):
                first_stage = stage
                break

    action = Action(
        job_id=first_stage.job_id,
        stage_id=first_stage.id_,
        n_workers=int(first_stage.n_remaining_tasks)
    ) if first_stage is not None else Action()
    
    return action


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