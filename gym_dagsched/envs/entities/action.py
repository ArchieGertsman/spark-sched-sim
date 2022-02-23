from dataclasses import dataclass

import numpy as np


# @dataclass
# class Action:
#     job_id: int

#     # which stage to execute next
#     stage_id: int

#     # which workers to assign to the stage's job
#     workers_mask: np.ndarray


@dataclass
class Action:
    job_id: int
    
    stage_id: int

    worker_type_counts: np.ndarray