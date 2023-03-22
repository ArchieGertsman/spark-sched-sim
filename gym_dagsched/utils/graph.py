from typing import NamedTuple
from torch import Tensor
from gymnasium.core import ObsType
from numpy import ndarray

import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch



def make_adj(
    edge_index: Tensor, 
    num_nodes: Tensor
) -> SparseTensor:
    '''returns a sparse representation of the edges'''
    return SparseTensor(
        row=edge_index[0], 
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
        trust_data=True,
        is_sorted=True
    )



def subgraph(
    edge_index: ndarray, 
    node_mask: ndarray
) -> ndarray:
    '''
    Args:
        edge_index: array of edges of shape (num_edges, 2),
            following the convention of gymnasium Graph space
        node_mask: indicates which nodes should be used for
            inducing the subgraph
    '''
    edge_mask = node_mask[edge_index[:,0]] & node_mask[edge_index[:,1]]
    edge_index = edge_index[edge_mask]

    node_idx = np.zeros(node_mask.size, dtype=int)
    node_idx[node_mask] = np.arange(node_mask.sum())
    edge_index = node_idx[edge_index]

    return edge_index



def obs_to_pyg(obs_dag_batch: ObsType) -> Batch:
    '''construct PyG `Batch` object from `obs['dag_batch']`'''
    ptr = np.array(obs_dag_batch['ptr'])
    num_nodes_per_dag = ptr[1:] - ptr[:-1]
    num_active_nodes = obs_dag_batch['data'].nodes.shape[0]
    num_active_jobs = len(num_nodes_per_dag)

    x = obs_dag_batch['data'].nodes
    edge_index = torch.from_numpy(obs_dag_batch['data'].edge_links.T)
    adj = make_adj(edge_index, num_active_nodes)
    batch = np.repeat(np.arange(num_active_jobs), num_nodes_per_dag)
    dag_batch = Batch(x=torch.from_numpy(x), 
                        edge_index=edge_index, 
                        adj=adj,
                        batch=torch.from_numpy(batch), 
                        ptr=torch.from_numpy(ptr))
    dag_batch._num_graphs = num_active_jobs

    return dag_batch



class ObsBatch(NamedTuple):
    nested_dag_batch: Batch
    num_dags_per_obs: Tensor
    num_nodes_per_dag: Tensor
    schedulable_op_masks: Tensor
    valid_prlsm_lim_masks: Tensor



def collate_obsns(obsns: list[ObsType]) -> ObsBatch:
    (dag_batches,
     schedulable_ops_masks, 
     valid_prlsm_lim_masks) = \
        zip(*((obs['dag_batch'], 
               obs['schedulable_op_mask'], 
               obs['valid_prlsm_lim_mask']) 
              for obs in obsns))

    nested_dag_batch, num_dags_per_obs, num_nodes_per_dag = \
        collate_dag_batches(dag_batches)

    schedulable_ops_masks = \
        torch.from_numpy(np.concatenate(schedulable_ops_masks).astype(bool))

    valid_prlsm_lim_masks = \
        torch.from_numpy(np.vstack(valid_prlsm_lim_masks))

    return ObsBatch(
        nested_dag_batch, 
        num_dags_per_obs, 
        num_nodes_per_dag, 
        schedulable_ops_masks, 
        valid_prlsm_lim_masks
    )



def collate_dag_batches(
    dag_batches: list[ObsType]
) -> tuple[Batch, Tensor, Tensor]:
    '''collates the dag batches from each observation into 
    one large dag batch
    '''
    _to_pyg = lambda raw_data: \
        Data(x=torch.from_numpy(raw_data.nodes), 
                edge_index=torch.from_numpy(raw_data.edge_links.T))

    def _get_num_nodes_per_dag(ptr):
        ptr = np.array(ptr)
        return ptr[1:] - ptr[:-1]

    data_list, num_dags_per_obs, num_nodes_per_dag = \
        zip(*((_to_pyg(dag_batch['data']), 
                len(dag_batch['ptr'])-1,
                _get_num_nodes_per_dag(dag_batch['ptr']))
                for dag_batch in dag_batches))

    num_dags_per_obs = torch.tensor(num_dags_per_obs)
    num_nodes_per_dag = torch.from_numpy(np.concatenate(num_nodes_per_dag))
    total_num_dags = num_dags_per_obs.sum()

    nested_dag_batch = Batch.from_data_list(data_list)
    nested_dag_batch.batch = \
        torch.arange(total_num_dags) \
             .repeat_interleave(num_nodes_per_dag)
    ptr = num_nodes_per_dag.cumsum(dim=0)
    nested_dag_batch.ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = total_num_dags
    nested_dag_batch.adj = \
        make_adj(nested_dag_batch.edge_index, 
                 nested_dag_batch.x.shape[0])

    return nested_dag_batch, num_dags_per_obs, num_nodes_per_dag