from dataclasses import dataclass

import numpy as np
from gym.spaces import Dict

from ..args import args
from gym_dagsched.utils.spaces import discrete_x, discrete_i


action_space = Dict({
    'job_id': discrete_x(args.n_jobs),
    'stage_id': discrete_x(args.max_stages),
    'n_workers': discrete_i(args.n_workers)
})


@dataclass
class Action:
    job_id: int = args.n_jobs

    stage_id: int = args.max_stages

    n_workers: int = 0