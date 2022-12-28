from collections import defaultdict

import torch



def construct_subbatch(data_batch, mask):
    node_mask = mask[data_batch.batch]

    subbatch = data_batch.subgraph(node_mask)

    subbatch._num_graphs = mask.sum().item()

    assoc = torch.empty(data_batch.num_graphs, dtype=torch.long)
    assoc[mask] = torch.arange(subbatch.num_graphs)
    subbatch.batch = assoc[data_batch.batch][node_mask]

    ptr = data_batch._slice_dict['x']
    num_nodes_per_graph = ptr[1:] - ptr[:-1]
    ptr = torch.cumsum(num_nodes_per_graph[mask], 0)
    ptr = torch.cat([torch.tensor([0]), ptr])
    subbatch.ptr = ptr

    edge_ptr = data_batch._slice_dict['edge_index']
    num_edges_per_graph = edge_ptr[1:] - edge_ptr[:-1]
    edge_ptr = torch.cumsum(num_edges_per_graph[mask], 0)
    edge_ptr = torch.cat([torch.tensor([0]), edge_ptr])

    subbatch._inc_dict = defaultdict(dict, {
        'x': torch.zeros(subbatch.num_graphs, dtype=torch.long),
        'edge_index': ptr[:-1]
    })

    subbatch._slice_dict = defaultdict(dict, {
        'x': ptr,
        'edge_index': edge_ptr
    })

    return subbatch




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