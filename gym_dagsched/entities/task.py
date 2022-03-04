from dataclasses import dataclass

import numpy as np
from gym.spaces import Dict

from ..args import args
from ..utils.misc import invalid_time
from ..utils.spaces import discrete_x, discrete_i, time_space
from .worker import Worker


task_space = Dict({
    'worker_id': discrete_x(args.n_workers),
    'is_processing': discrete_i(1),
    't_accepted': time_space,
    't_completed': time_space
})


@dataclass
class Task:
    INVALID_ID = args.max_tasks


    worker_id: int = Worker.INVALID_ID

    is_processing: bool = False

    t_accepted: np.ndarray = invalid_time()

    t_completed: np.ndarray = invalid_time()