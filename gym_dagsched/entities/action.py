from dataclasses import dataclass

import numpy as np
from gym.spaces import Dict, MultiDiscrete

from ..args import args
from gym_dagsched.utils.spaces import discrete_x


action_space = Dict({
    'job_id': discrete_x(args.n_jobs),
    'stage_id': discrete_x(args.max_stages),
    'worker_type_counts': MultiDiscrete(
        args.n_worker_types * [args.n_workers])
})


@dataclass
class Action:
    job_id: int

    stage_id: int

    worker_type_counts: np.ndarray