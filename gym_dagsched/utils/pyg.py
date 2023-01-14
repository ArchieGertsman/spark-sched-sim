from collections import defaultdict

import numpy as np
import torch
from torch_sparse import SparseTensor



def construct_subbatch(data_batch, graph_mask, node_mask, num_nodes_per_graph, num_graphs):
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



def add_adj(dag_batch):
    num_nodes = dag_batch.x.shape[0]
    dag_batch.adj = SparseTensor(
        row=dag_batch.edge_index[0], 
        col=dag_batch.edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
        trust_data=True,
        is_sorted=True)




def chunk_feature_tensor(dag_batch):
    '''returns a list of chunks of the feature tensor,
    where there is one chunk per job, per environment,
    i.e. the list has length `self.n * self.n_job_arrivals`.
    each chunk has as many rows as there are operations
    in the corresponding job.
    '''
    ptr = dag_batch._slice_dict['x']
    num_nodes_per_graph = (ptr[1:] - ptr[:-1]).tolist()
    feature_tensor_chunks = torch.split(
        dag_batch.x, num_nodes_per_graph)
    return feature_tensor_chunks