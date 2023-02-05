from typing import NamedTuple

import torch
from torch_geometric.data import Data, Batch
import numpy as np

from ..utils.graph import make_adj



class Experience(NamedTuple):
    obs: object
    wall_time: float
    action: object
    reward: float



def collect_rollout(env, agent, seed, options):  
    obs, info = env.reset(seed=seed, options=options)
    done = False

    rollout_buffer = []

    while not done:
        action = agent(obs)

        new_obs, reward, terminated, truncated, info = \
            env.step(action)

        done = (terminated or truncated)

        exp = Experience(obs, info['wall_time'], action, reward)
        rollout_buffer += [exp]

        obs = new_obs
    
    return rollout_buffer



def construct_nested_dag_batch(dag_batches):
    '''extracts the inputs to the model from each
    step of the episode, then stacks them into a
    large batch.
    '''
    to_pyg = lambda raw_data: \
        Data(x=torch.from_numpy(raw_data.nodes), 
             edge_index=torch.from_numpy(raw_data.edge_links.T))

    def get_num_nodes_per_dag(ptr):
        ptr = np.array(ptr)
        return ptr[1:] - ptr[:-1]

    data_list, num_dags_per_obs, num_nodes_per_dag = \
        zip(*((to_pyg(dag_batch['data']), 
               len(dag_batch['ptr'])-1,
               get_num_nodes_per_dag(dag_batch['ptr']))
              for dag_batch in dag_batches))

    num_dags_per_obs = torch.tensor(num_dags_per_obs)
    num_nodes_per_dag = torch.from_numpy(np.concatenate(num_nodes_per_dag))
    total_num_dags = num_dags_per_obs.sum()

    nested_dag_batch = Batch.from_data_list(data_list)
    nested_dag_batch.batch = \
        torch.arange(total_num_dags) \
             .repeat_interleave(num_nodes_per_dag)
    ptr = num_nodes_per_dag.cumsum(dim=0)
    nested_dag_batch.ptr = \
        torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = total_num_dags
    nested_dag_batch.adj = \
        make_adj(nested_dag_batch.edge_index, 
                 nested_dag_batch.x.shape[0])

    return nested_dag_batch, num_dags_per_obs, num_nodes_per_dag



def stack_obsns(obsns):
    (dag_batches,
     valid_ops_masks, 
     valid_prlsm_lim_masks) = \
        zip(*((obs['dag_batch'], 
               obs['schedulable_op_mask'], 
               obs['valid_prlsm_lim_mask']) 
              for obs in obsns))

    nested_dag_batch, num_dags_per_obs, num_nodes_per_dag = \
        construct_nested_dag_batch(dag_batches)

    valid_ops_masks = np.concatenate(valid_ops_masks).astype(bool)
    valid_prlsm_lim_masks = torch.from_numpy(np.vstack(valid_prlsm_lim_masks))

    return nested_dag_batch, \
           num_dags_per_obs, \
           num_nodes_per_dag, \
           valid_ops_masks, \
           valid_prlsm_lim_masks