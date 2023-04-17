from typing import NamedTuple
from torch import Tensor
from gymnasium.core import ObsType
from numpy import ndarray

import numpy as np
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.utils as pyg_utils
from torch_scatter import segment_csr
import networkx as nx



def make_adj(
    edge_index: Tensor, 
    num_nodes: Tensor
):
    '''returns a sparse representation of the edges'''
    # return SparseTensor(
    #     row=edge_index[0], 
    #     col=edge_index[1],
    #     sparse_sizes=(num_nodes, num_nodes),
    #     trust_data=True,
    #     is_sorted=True
    # )
    return torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.shape[1], device=edge_index.device), 
        size=(num_nodes, num_nodes)
    )



def subgraph(
    edge_index: ndarray, 
    node_mask: ndarray
) -> ndarray:
    '''
    Simpler numpy version of PyG's subgraph utility function
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
    # num_active_nodes = obs_dag_batch['data'].nodes.shape[0]
    num_active_jobs = len(num_nodes_per_dag)

    x = obs_dag_batch['data'].nodes
    edge_index = torch.from_numpy(obs_dag_batch['data'].edge_links.T)
    # adj = make_adj(edge_index, num_active_nodes)
    batch = np.repeat(np.arange(num_active_jobs), num_nodes_per_dag)
    dag_batch = Batch(
        x=torch.from_numpy(x), 
        edge_index=edge_index, 
        # adj=adj,
        batch=torch.from_numpy(batch), 
        ptr=torch.from_numpy(ptr)
    )
    dag_batch._num_graphs = num_active_jobs

    return dag_batch



class ObsBatch(NamedTuple):
    dag_batch: Batch
    schedulable_stage_masks: Tensor
    valid_prlsm_lim_masks: Tensor



def collate_obsns(obsns: list[ObsType]) -> ObsBatch:
    dag_batches, schedulable_stage_masks, valid_prlsm_lim_masks = zip(*(
        (
            obs['dag_batch'], 
            obs['schedulable_stage_mask'], 
            obs['valid_prlsm_lim_mask']
        ) 
        for obs in obsns
    ))

    dag_batch = collate_dag_batches(dag_batches)
    schedulable_stage_masks = torch.from_numpy(np.concatenate(schedulable_stage_masks).astype(bool))
    valid_prlsm_lim_masks = torch.from_numpy(np.vstack(valid_prlsm_lim_masks))

    return ObsBatch(
        dag_batch, 
        schedulable_stage_masks, 
        valid_prlsm_lim_masks
    )



def collate_dag_batches(
    dag_batches: list[ObsType]
) -> tuple[Batch, Tensor, Tensor]:
    '''collates the dag batches from each observation into 
    one large dag batch
    '''
    _to_pyg = lambda raw_data: \
        Data(
            x=torch.from_numpy(raw_data.nodes), 
            edge_index=torch.from_numpy(raw_data.edge_links.T)
        )

    def _get_num_nodes_per_dag(ptr):
        ptr = np.array(ptr)
        return ptr[1:] - ptr[:-1]

    data_list, num_dags_per_obs, num_nodes_per_dag = zip(*(
        (
            _to_pyg(dag_batch['data']), 
            len(dag_batch['ptr'])-1,
            _get_num_nodes_per_dag(dag_batch['ptr'])
        )
        for dag_batch in dag_batches
    ))

    dag_batch = Batch.from_data_list(data_list)
    dag_batch.batch = None
    # dag_batch.batch = \
    #     torch.arange(total_num_dags) \
    #          .repeat_interleave(num_nodes_per_dag)
    
    dag_batch['num_dags_per_obs'] = torch.tensor(num_dags_per_obs)
    dag_batch['num_nodes_per_dag'] = torch.from_numpy(np.concatenate(num_nodes_per_dag))
    dag_batch['obs_ptr'] = make_ptr(dag_batch['num_dags_per_obs'])
    dag_batch['num_nodes_per_obs'] = segment_csr(dag_batch['num_nodes_per_dag'], dag_batch['obs_ptr'])

    dag_batch.ptr = make_ptr(dag_batch['num_nodes_per_dag'])
    dag_batch._num_graphs = dag_batch['num_dags_per_obs'].sum()

    return dag_batch



def make_ptr(x):
    ptr = x.cumsum(dim=0)
    ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
    return ptr



def construct_message_passing_levels(dag_batch):
    edge_index = dag_batch.edge_index.cpu()
    num_nodes = dag_batch.x.shape[0]
    subgraph_mask = torch.zeros(num_nodes, dtype=bool)
    level_mask = torch.zeros(num_nodes, dtype=bool)

    G = nx.DiGraph()
    G.add_nodes_from(range(dag_batch.x.shape[0]))
    G.add_edges_from(edge_index.T.numpy())
    node_levels = list(nx.topological_generations(G))

    if len(node_levels) <= 1:
        # no message passing to do
        return None, None

    level_edge_index_list = []
    level_mask_list = []
    for l in reversed(range(1, len(node_levels))):
        subgraph_mask.zero_()
        subgraph_mask[node_levels[l-1] + node_levels[l]] = True
        level_edge_index = pyg_utils.subgraph(subgraph_mask, edge_index, num_nodes=num_nodes)[0]
        level_edge_index_list += [level_edge_index]

        level_mask.zero_()
        level_mask[level_edge_index[0]] = True
        level_mask_list += [level_mask.clone()]
    
    # collate lists for O(1) number of device transfers
    num_edges_per_level = [level_edge_index.shape[1] for level_edge_index in level_edge_index_list]
    level_edge_index_batch = torch.cat(level_edge_index_list, dim=1).to(dag_batch.x.device)
    level_mask_batch = torch.stack(level_mask_list).to(dag_batch.x.device)

    level_edge_index_list = torch.split(level_edge_index_batch, num_edges_per_level, dim=1)
    return level_edge_index_list, level_mask_batch
