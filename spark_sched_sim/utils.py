import numpy as np
from numpy import ndarray


def subgraph(edge_links: ndarray, node_mask: ndarray):
    """
    Minimal numpy version of PyG's subgraph utility function
    Args:
        edge_links: array of edges of shape (num_edges, 2),
            following the convention of gymnasium Graph space
        node_mask: indicates which nodes should be used for
            inducing the subgraph
    """
    edge_mask = node_mask[edge_links[:, 0]] & node_mask[edge_links[:, 1]]
    edge_links = edge_links[edge_mask]

    # relabel the nodes
    node_idx = np.zeros(node_mask.size, dtype=int)
    node_idx[node_mask] = np.arange(node_mask.sum())
    edge_links = node_idx[edge_links]

    return edge_links
