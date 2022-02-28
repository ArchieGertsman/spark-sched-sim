from dataclasses import dataclass

import numpy as np


@dataclass
class Action:
    job_id: int

    stage_id: int

    worker_type_counts: np.ndarray