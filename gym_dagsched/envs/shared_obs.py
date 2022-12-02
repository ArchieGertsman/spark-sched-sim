from dataclasses import dataclass
from typing import List

import torch




@dataclass
class SharedObs:
    '''contains all of the shared memory tensors
    that each environment needs to communicate
    its observations with the main process,
    without having to send data through pipes. 
    Each environment gets its own instance of 
    this dataclass.
    '''

    # list of feature tensors for each job
    # in the environment. Rows correspond
    # to operations within the job, and columns
    # are the features.
    # shape[i]: torch.Size([num_ops_per_job[i], num_features])
    feature_tensor_chunks: List[torch.Tensor]

    # mask that indicated which jobs are
    # currently active
    # shape: torch.Size([num_job_arrivals])
    active_job_msk: torch.BoolTensor

    # mask that indicates which operations
    # are valid selections in each job.
    # shape: torch.Size([num_job_arrivals, max_ops_in_a_job])
    op_msk: torch.BoolTensor

    # mask that indicates which parallelism
    # limits are valid for each job
    # shape: torch.Size([num_job_arrivals, num_workers])
    prlvl_msk: torch.BoolTensor

    # reward signal
    # shape: torch.Size([])
    reward: torch.Tensor

    # whether or not the episode is done
    # shape: torch.Size([])
    done: torch.BoolTensor