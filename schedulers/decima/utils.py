from collections.abc import Iterable
from typing import Any
from torch import Tensor
from numpy import ndarray

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_sparse import SparseTensor
from torch_scatter import segment_csr
import networkx as nx
from torch.distributions.utils import clamp_probs
import numpy as np


def sample(logits: Tensor) -> tuple[int, float]:
    pi = F.softmax(logits, 0).numpy()
    idx = random.choices(np.arange(pi.size), pi)[0]
    lgprob = np.log(pi[idx])
    return idx, lgprob


def evaluate(
    scores: Tensor, counts: Tensor, selections: Tensor
) -> tuple[Tensor, Tensor]:
    """
    scores: scores that the model assigned to each action at each step
    counts: count of available actions at each step
    selections: actions that the scheduler sampled at each step
    """
    ptr = counts.cumsum(0)
    ptr = torch.cat([torch.tensor([0]), ptr], 0)
    selections += ptr[:-1]
    probs = pyg.utils.softmax(scores, ptr=ptr)
    probs = clamp_probs(probs)
    log_probs = probs.log()
    selection_log_probs = log_probs[selections]
    entropies = -segment_csr(log_probs * probs, ptr)
    return selection_log_probs, entropies


def make_mlp(
    input_dim: int,
    hid_dims: list[int],
    output_dim: int,
    act_cls: str,
    act_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    act_clss = getattr(torch.nn.modules.activation, act_cls)

    mlp = nn.Sequential()
    prev_dim = input_dim
    hid_dims = hid_dims + [output_dim]
    for i, dim in enumerate(hid_dims):
        mlp.append(nn.Linear(prev_dim, dim))
        if i == len(hid_dims) - 1:
            break
        act_fn = act_clss(**(act_kwargs or {}))
        mlp.append(act_fn)
        prev_dim = dim
    return mlp


def make_adj(edge_index: Tensor, num_nodes: int) -> SparseTensor:
    """returns a sparse COO adjacency matrix"""
    return SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
        is_sorted=True,
        trust_data=True,
    )


def ptr_to_counts(ptr):
    return ptr[1:] - ptr[:-1]


def counts_to_ptr(x: Tensor) -> Tensor:
    ptr = x.cumsum(0)
    ptr = torch.cat([torch.tensor([0]), ptr], 0)
    return ptr


def obs_to_pyg(obs: dict[str, Any]) -> pyg.data.Batch:
    """converts an env observation into a PyG `Batch` object"""
    obs_dag_batch = obs["dag_batch"]
    ptr = np.array(obs["dag_ptr"])
    num_nodes_per_dag = ptr_to_counts(ptr)
    num_active_jobs = len(num_nodes_per_dag)

    dag_batch = pyg.data.Batch(
        x=torch.from_numpy(obs_dag_batch.nodes),
        edge_index=torch.from_numpy(obs_dag_batch.edge_links.T),
        ptr=torch.from_numpy(ptr),
        batch=torch.from_numpy(
            np.repeat(np.arange(num_active_jobs), num_nodes_per_dag)
        ),
        _num_graphs=num_active_jobs,
    )

    dag_batch["stage_mask"] = torch.tensor(obs["stage_mask"], dtype=torch.bool)
    dag_batch["exec_mask"] = torch.from_numpy(obs["exec_mask"])
    dag_batch["num_nodes_per_dag"] = ptr_to_counts(dag_batch.ptr)

    if "edge_masks" in obs:
        dag_batch["edge_masks"] = torch.from_numpy(obs["edge_masks"])

    if "node_depth" in obs:
        dag_batch["node_depth"] = torch.from_numpy(obs["node_depth"]).float()

    return dag_batch


def collate_obsns(obsns: Iterable[dict[str, Any]]) -> pyg.data.Batch:
    keys = ["dag_batch", "dag_ptr", "stage_mask", "exec_mask"]
    dag_batches, dag_ptrs, stage_masks, exec_masks = zip(
        *([obs[key] for key in keys] for obs in obsns)
    )

    dag_batch = collate_dag_batches(dag_batches, dag_ptrs)

    dag_batch["stage_mask"] = torch.from_numpy(np.concatenate(stage_masks).astype(bool))

    dag_batch["exec_mask"] = torch.from_numpy(np.vstack(exec_masks))

    # number of available stage actions at each step
    dag_batch["num_stage_acts"] = torch.tensor([msk.sum() for msk in stage_masks])

    # number of available exec actions at each step
    dag_batch["num_exec_acts"] = dag_batch["exec_mask"].sum(-1)

    if "edge_masks" in next(iter(obsns)):
        edge_masks_list = [obs["edge_masks"] for obs in obsns]
        total_num_edges = dag_batch.edge_index.shape[1]
        dag_batch["edge_masks"] = collate_edge_masks(edge_masks_list, total_num_edges)

    if "node_depth" in next(iter(obsns)):
        node_depth_list = [obs["node_depth"] for obs in obsns]
        dag_batch["node_depth"] = torch.from_numpy(
            np.concatenate(node_depth_list)
        ).float()

    return dag_batch


def collate_edge_masks(
    edge_masks_list: Iterable[ndarray], total_num_edges: int
) -> ndarray:
    """collates list of edge mask batches from each message passing path. Since the
    message passing depth varies between observations, edge mask batches are padded
    to the maximum depth."""
    max_depth = max(edge_masks.shape[0] for edge_masks in edge_masks_list)

    # array that will be populated with the masks from all the observations
    edge_masks = np.zeros((max_depth, total_num_edges), dtype=bool)

    i = 0
    for masks in edge_masks_list:
        # copy the data from these masks into the output array
        depth, num_edges = masks.shape
        if depth > 0:
            edge_masks[:depth, i : (i + num_edges)] = masks
        i += num_edges

    return edge_masks


def collate_dag_batches(
    dag_batches: Iterable[pyg.data.Batch], dag_ptrs: Iterable[ndarray]
) -> pyg.data.Batch:
    """collates the dag batches from each observation into one large dag batch"""
    num_dags_per_obs_tup, num_nodes_per_dag_tup = zip(
        *(
            (len(dag_ptr) - 1, ptr_to_counts(torch.tensor(dag_ptr)))
            for dag_ptr in dag_ptrs
        )
    )
    num_dags_per_obs = torch.tensor(num_dags_per_obs_tup)
    num_nodes_per_dag = torch.cat(num_nodes_per_dag_tup)
    obs_ptr = counts_to_ptr(num_dags_per_obs)
    num_nodes_per_obs = segment_csr(num_nodes_per_dag, obs_ptr)
    num_graphs = num_dags_per_obs.sum().item()

    x = torch.from_numpy(np.concatenate([dag_batch.nodes for dag_batch in dag_batches]))

    dag_batch = pyg.data.Batch(
        x=x,
        edge_index=collate_edges(dag_batches, num_nodes_per_obs),
        ptr=counts_to_ptr(num_nodes_per_dag),
        batch=torch.arange(num_graphs).repeat_interleave(
            num_nodes_per_dag, output_size=x.shape[0]
        ),
        _num_graphs=num_graphs,
    )

    # store bookkeeping attributes
    dag_batch["num_dags_per_obs"] = num_dags_per_obs
    dag_batch["num_nodes_per_dag"] = num_nodes_per_dag
    dag_batch["num_nodes_per_obs"] = num_nodes_per_obs
    dag_batch["obs_ptr"] = obs_ptr

    return dag_batch


def collate_edges(
    dag_batches: Iterable[pyg.data.Batch], num_nodes_per_obs: Tensor
) -> Tensor:
    edge_counts_tup, edge_links_tup = zip(
        *(
            (dag_batch.edge_links.shape[0], dag_batch.edge_links)
            for dag_batch in dag_batches
        )
    )
    edge_counts = torch.tensor(edge_counts_tup)
    edge_links = np.concatenate(edge_links_tup)

    edge_index = torch.from_numpy(edge_links.T)

    # relabel the edges
    node_ptr = counts_to_ptr(num_nodes_per_obs)
    edge_index += (
        node_ptr[:-1]
        .repeat_interleave(edge_counts, output_size=edge_index.shape[1])
        .unsqueeze(0)
    )

    return edge_index


def make_edge_mask(edge_links: ndarray, node_mask: ndarray) -> ndarray:
    return node_mask[edge_links[:, 0]] & node_mask[edge_links[:, 1]]


def make_dag_layer_edge_masks(
    graph_or_data: nx.DiGraph | tuple[ndarray, int]
) -> ndarray:
    """returns a batch of edge masks of shape (msg passing depth, num edges),
    where the i'th mask indicates which edges participate in the i'th root-to-leaf
    message passing step.
    """
    if isinstance(graph_or_data, nx.DiGraph):
        G = graph_or_data
    else:
        edge_links, num_nodes = graph_or_data
        G = np_to_nx(edge_links, num_nodes)

    node_levels = list(nx.topological_generations(G))

    if len(node_levels) <= 1:
        # no message passing to do
        return np.zeros((0, edge_links.shape[0]), dtype=bool)

    node_mask = np.zeros(len(G), dtype=bool)

    edge_masks = []
    for node_level in node_levels[:-1]:
        succ = set.union(*[set(G.successors(n)) for n in node_level])
        node_mask[:] = 0
        node_mask[node_level + list(succ)] = True
        edge_mask = make_edge_mask(edge_links, node_mask)
        edge_masks += [edge_mask]

    return np.stack(edge_masks)


def np_to_nx(edge_links: ndarray, num_nodes: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_links)
    return G
