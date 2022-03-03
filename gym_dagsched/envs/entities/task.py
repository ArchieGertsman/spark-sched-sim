from dataclasses import dataclass

import numpy as np

from args import args
from dagsched_utils import invalid_time
from .worker import Worker


@dataclass
class Task:
    INVALID_ID = args.max_tasks


    worker_id: int = Worker.INVALID_ID

    is_processing: bool = False

    t_accepted: np.ndarray = invalid_time()

    t_completed: np.ndarray = invalid_time()