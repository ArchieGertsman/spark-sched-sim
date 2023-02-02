from dataclasses import dataclass

import torch
from torch_geometric.data import Batch

from ..utils.pyg import add_adj



@dataclass
class Observation:
    dag_batch: object
    valid_ops_mask: object
    valid_prlsm_lim_mask: object
    num_jobs: int
    num_source_workers: int
    wall_time: float
    action: object
    reward: float



def generate_rollout(env, agent, seed, options):  
    obs, _ = env.reset(seed, options)

    done = False

    # save rollout from each step
    # of the episode, to later be used
    # in leaning
    rollout = []
    
    while not done:
        # unpack the current observation
        (_,
         dag_batch,
         valid_ops_mask,
         valid_prlsm_lim_mask,
         active_job_ids, 
         num_source_workers,
         _) = obs

        env_action, raw_action = agent(obs)
        obs, reward, terminated, truncated, _ = \
            env.step(env_action)

        *_, wall_time = obs

        rollout += \
            [Observation(dag_batch.clone(),
                         valid_ops_mask,
                         valid_prlsm_lim_mask,
                         len(active_job_ids),
                         num_source_workers,
                         wall_time,
                         raw_action,
                         reward)]

        done = (terminated or truncated)

    return rollout



def construct_nested_dag_batch(dag_batch_list, 
                               num_dags_per_obs):
    '''extracts the inputs to the model from each
    step of the episode, then stacks them into a
    large batch.
    '''
    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    num_nodes_per_dag = nested_dag_batch.batch.bincount()
    ptr = num_nodes_per_dag.cumsum(dim=0)
    nested_dag_batch.ptr = \
        torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = sum(num_dags_per_obs)
    add_adj(nested_dag_batch)

    return nested_dag_batch, num_nodes_per_dag