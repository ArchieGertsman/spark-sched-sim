import numpy as np
import torch
from torch_sparse import SparseTensor



def construct_subbatch(data_batch, 
                       graph_mask, 
                       node_mask, 
                       num_nodes_per_graph, 
                       num_graphs):
    node_mask = torch.from_numpy(node_mask)
    subbatch = data_batch.subgraph(node_mask)

    subbatch._num_graphs = num_graphs

    assoc = np.zeros(data_batch.num_graphs, dtype=int)
    assoc[graph_mask] = np.arange(subbatch.num_graphs)
    batch = assoc[data_batch.batch][node_mask]
    subbatch.batch = torch.from_numpy(batch)

    ptr = np.zeros(num_graphs + 1, dtype=int)
    np.cumsum(num_nodes_per_graph, out=ptr[1:])
    subbatch.ptr = torch.from_numpy(ptr)

    add_adj(subbatch)

    return subbatch



def add_adj(data):
    '''adds an attribute `adj` to a PyG graph which 
    stores a sparse representation of the edges
    '''
    num_nodes = data.x.shape[0]
    data.adj = SparseTensor(
        row=data.edge_index[0], 
        col=data.edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
        trust_data=True,
        is_sorted=True)



def subgraph(edge_index, node_mask):
    '''numpy version of `torch_geometric.utils.subgraph`, with 
    features unneeded in this project omitted. Note: assumes
    that `edge_index` is of shape (num_nodes, 2)
    '''
    edge_mask = node_mask[edge_index[:,0]] & node_mask[edge_index[:,1]]
    edge_index = edge_index[edge_mask]

    node_idx = np.zeros(node_mask.size, dtype=int)
    node_idx[node_mask] = np.arange(node_mask.sum())
    edge_index = node_idx[edge_index]

    return edge_index