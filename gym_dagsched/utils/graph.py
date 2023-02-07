import numpy as np
from torch_sparse import SparseTensor



def make_adj(edge_index, num_nodes):
    '''returns a sparse representation of the edges'''
    return SparseTensor(
        row=edge_index[0], 
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
        trust_data=True,
        is_sorted=True
    )



def subgraph(edge_index, node_mask):
    '''numpy version of `torch_geometric.utils.subgraph`, omitting 
    features that are unneeded in this project. Note: assumes
    that `edge_index` is of shape (num_edges, 2)
    '''
    edge_mask = node_mask[edge_index[:,0]] & node_mask[edge_index[:,1]]
    edge_index = edge_index[edge_mask]

    node_idx = np.zeros(node_mask.size, dtype=int)
    node_idx[node_mask] = np.arange(node_mask.sum())
    edge_index = node_idx[edge_index]

    return edge_index