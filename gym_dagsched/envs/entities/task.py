from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    worker_id: int

    is_processing: bool

    t_accepted: np.ndarray

    t_completed: np.ndarray