import numpy as np

from ..scheduler import Scheduler
from .utils import preprocess_obs, find_stage


class RandomScheduler(Scheduler):
    def __init__(self, seed=42):
        self.name = "Random"
        self.env_wrapper_cls = None
        self.set_seed(seed)

    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def schedule(self, obs: dict) -> tuple[dict, dict]:
        preprocess_obs(obs)
        num_active_jobs = len(obs["exec_supplies"])

        job_idxs = list(range(num_active_jobs))
        stage_idx = -1
        while len(job_idxs) > 0:
            j = self.np_random.choice(job_idxs)
            stage_idx = find_stage(obs, j)
            if stage_idx != -1:
                break
            else:
                job_idxs.remove(j)

        num_exec = self.np_random.randint(1, obs["num_committable_execs"] + 1)

        return {"stage_idx": stage_idx, "num_exec": num_exec}, {}
