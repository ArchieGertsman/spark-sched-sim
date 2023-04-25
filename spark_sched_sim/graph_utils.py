from typing import NamedTuple
from torch import Tensor
from gymnasium.core import ObsType
from numpy import ndarray

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.collate import collate
from torch_scatter import segment_csr
import networkx as nx



def make_adj(edge_index, num_nodes):
    '''returns a sparse COO adjacency matrix'''
    return torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.shape[1], device=edge_index.device), 
        size=(num_nodes, num_nodes)
    )



def make_edge_mask(edge_links, node_mask):
    return node_mask[edge_links[:,0]] & node_mask[edge_links[:,1]]



def subgraph(edge_links: ndarray, node_mask: ndarray) -> ndarray:
    '''
    Minimal numpy version of PyG's subgraph utility function
    Args:
        edge_links: array of edges of shape (num_edges, 2),
            following the convention of gymnasium Graph space
        node_mask: indicates which nodes should be used for
            inducing the subgraph
    '''
    edge_mask = make_edge_mask(edge_links, node_mask)
    edge_links = edge_links[edge_mask]

    # relabel the nodes
    node_idx = np.zeros(node_mask.size, dtype=int)
    node_idx[node_mask] = np.arange(node_mask.sum())
    edge_links = node_idx[edge_links]

    return edge_links



def obs_to_pyg(obs: ObsType) -> Batch:
    '''converts an env observation to a PyG `Batch` object'''
    obs_dag_batch = obs['dag_batch']
    ptr = np.array(obs_dag_batch['ptr'])
    num_nodes_per_dag = ptr[1:] - ptr[:-1]
    num_active_jobs = len(num_nodes_per_dag)

    # NOTE: `node_to_dag_map` is exactly the `batch` attribute in PyG `Batch`
    # objects, but that attribute is not needed in the forward pass, so it's
    # left out of the `Batch` object and named more descriptively.
    node_to_dag_map = np.repeat(np.arange(num_active_jobs), num_nodes_per_dag)

    x = obs_dag_batch['data'].nodes
    edge_links = obs_dag_batch['data'].edge_links
    dag_batch = Batch(
        x=torch.from_numpy(x), 
        edge_index=torch.from_numpy(edge_links.T),
        ptr=torch.from_numpy(ptr),
        _num_graphs=num_active_jobs
    )
    dag_batch['edge_mask_batch'] = torch.from_numpy(obs['edge_mask_batch'])

    return dag_batch, node_to_dag_map



class ObsBatch(NamedTuple):
    dag_batch: Batch
    schedulable_stage_masks: Tensor
    valid_prlsm_lim_masks: Tensor



def collate_obsns(obsns: list[ObsType]) -> ObsBatch:
    dag_batches, schedulable_stage_masks, valid_prlsm_lim_masks, edge_mask_batches = zip(*(
        (
            obs['dag_batch'], 
            obs['schedulable_stage_mask'], 
            obs['valid_prlsm_lim_mask'],
            obs['edge_mask_batch']
        ) 
        for obs in obsns
    ))

    dag_batch = collate_dag_batches(dag_batches)
    dag_batch['edge_mask_batch'] = \
        collate_edge_mask_batches(edge_mask_batches, dag_batch.edge_index.shape[1])

    schedulable_stage_masks = torch.from_numpy(np.concatenate(schedulable_stage_masks).astype(bool))
    valid_prlsm_lim_masks = torch.from_numpy(np.vstack(valid_prlsm_lim_masks))

    return ObsBatch(
        dag_batch, 
        schedulable_stage_masks, 
        valid_prlsm_lim_masks
    )



def collate_edge_mask_batches(edge_mask_batches, total_num_edges):
    '''collates list of edge mask batches from each message passing path. Since the
    message passing depth varies between observations, edge mask batches are padded
    to the maximum depth.'''
    max_depth = max(edge_mask_batch.shape[0] for edge_mask_batch in edge_mask_batches)
    edge_mask_batch = np.zeros((max_depth, total_num_edges), dtype=bool)
    i = 0
    for mask_batch in edge_mask_batches:
        depth, num_edges = mask_batch.shape
        if depth > 0:
            edge_mask_batch[:depth, i : i+num_edges] = mask_batch
        i += num_edges
    return edge_mask_batch



def collate_dag_batches(
    dag_batches: list[ObsType]
) -> tuple[Batch, Tensor, Tensor]:
    '''collates the dag batches from each observation into one large dag batch'''
    _to_pyg = lambda raw_data: \
        Data(
            x=torch.from_numpy(raw_data.nodes), 
            edge_index=torch.from_numpy(raw_data.edge_links.T)
        )

    def _ptr_to_counts(ptr):
        ptr = np.array(ptr)
        return ptr[1:] - ptr[:-1]

    def _counts_to_ptr(x):
        ptr = x.cumsum(0)
        ptr = torch.cat([torch.tensor([0]), ptr], 0)
        return ptr

    data_list, num_dags_per_obs, num_nodes_per_dag = zip(*(
        (
            _to_pyg(dag_batch['data']), 
            len(dag_batch['ptr'])-1,
            _ptr_to_counts(dag_batch['ptr'])
        )
        for dag_batch in dag_batches
    ))

    # NOTE: `batch` attribute is not needed, but can be obtained as follows:
    # dag_batch.batch = torch.arange(total_num_dags).repeat_interleave(num_nodes_per_dag)

    dag_batch = collate(Batch, data_list, add_batch=False)[0]
    
    # add some custom attributes for bookkeeping
    dag_batch['num_dags_per_obs'] = torch.tensor(num_dags_per_obs)
    dag_batch['num_nodes_per_dag'] = torch.from_numpy(np.concatenate(num_nodes_per_dag))
    dag_batch['obs_ptr'] = _counts_to_ptr(dag_batch['num_dags_per_obs'])
    dag_batch['num_nodes_per_obs'] = segment_csr(dag_batch['num_nodes_per_dag'], dag_batch['obs_ptr'])

    dag_batch.ptr = _counts_to_ptr(dag_batch['num_nodes_per_dag'])
    dag_batch._num_graphs = dag_batch['num_dags_per_obs'].sum()

    return dag_batch



def construct_message_passing_masks(edge_links, num_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_links)
    node_levels = list(nx.topological_generations(G))

    if len(node_levels) <= 1:
        # no message passing to do
        return np.zeros((0, edge_links.shape[0]), dtype=bool)
    
    node_mask = np.zeros(num_nodes, dtype=bool)

    edge_mask_list = []
    for node_level in reversed(node_levels[:-1]):
        succ = set.union(*[set(G.successors(n)) for n in node_level])
        node_mask[:] = 0
        node_mask[node_level + list(succ)] = True
        edge_mask = make_edge_mask(edge_links, node_mask)
        edge_mask_list += [edge_mask]

    edge_mask_batch = np.stack(edge_mask_list)
    return edge_mask_batch