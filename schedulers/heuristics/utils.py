from typing import Any
import numpy as np


def preprocess_obs(obs: dict[str, Any]) -> None:
    frontier_mask = np.ones(obs["dag_batch"].nodes.shape[0], dtype=bool)
    dst_nodes = obs["dag_batch"].edge_links[:, 1]
    frontier_mask[dst_nodes] = False
    stage_mask = obs["dag_batch"].nodes[:, 2].astype(bool)

    obs["frontier_stages"] = set(frontier_mask.nonzero()[0])
    obs["schedulable_stages"] = dict(
        zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum()))
    )


def find_stage(obs: dict[str, Any], job_idx: int) -> int:
    """searches for a schedulable stage in a given job, prioritizing
    frontier stages
    """
    stage_idx_start = obs["dag_ptr"][job_idx]
    stage_idx_end = obs["dag_ptr"][job_idx + 1]

    selected_stage_idx = -1
    for node in range(stage_idx_start, stage_idx_end):
        if node in obs["schedulable_stages"]:
            i = obs["schedulable_stages"][node]
        else:
            continue

        if node in obs["frontier_stages"]:
            return i

        if selected_stage_idx == -1:
            selected_stage_idx = i

    return selected_stage_idx
