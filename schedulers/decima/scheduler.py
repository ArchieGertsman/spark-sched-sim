from collections.abc import Iterable
from typing import Any
from torch import Tensor

import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric as pyg
import torch_sparse

from ..scheduler import TrainableScheduler
from .env_wrapper import DecimaEnvWrapper
from . import utils


class DecimaScheduler(TrainableScheduler):
    """Original Decima architecture, which uses asynchronous message passing
    as in DAGNN.
    Paper: https://dl.acm.org/doi/abs/10.1145/3341302.3342080
    """

    def __init__(
        self,
        num_executors: int,
        embed_dim: int,
        gnn_mlp_kwargs: dict[str, Any],
        policy_mlp_kwargs: dict[str, Any],
        state_dict_path: str | None = None,
        opt_cls: str | None = None,
        opt_kwargs: dict[str, Any] | None = None,
        max_grad_norm: float | None = None,
        num_node_features: int = 5,
        num_dag_features: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.name = "Decima"
        self.env_wrapper_cls = DecimaEnvWrapper
        self.max_grad_norm = max_grad_norm
        self.num_executors = num_executors

        self.encoder = EncoderNetwork(num_node_features, embed_dim, gnn_mlp_kwargs)

        emb_dims = {"node": embed_dim, "dag": embed_dim, "glob": embed_dim}

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, policy_mlp_kwargs
        )

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, policy_mlp_kwargs
        )

        self._reset_biases()

        if state_dict_path:
            self.name += f":{state_dict_path}"
            self.load_state_dict(torch.load(state_dict_path))

        if opt_cls:
            self.optim = getattr(torch.optim, opt_cls)(
                self.parameters(), **(opt_kwargs or {})
            )

    def _reset_biases(self) -> None:
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()

    @torch.no_grad()
    def schedule(self, obs: dict) -> tuple[dict, dict]:
        dag_batch = utils.obs_to_pyg(obs)
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch["stage_mask"]

        dag_batch.to(self.device, non_blocking=True)

        # 1. compute node, dag, and global representations
        h_dict = self.encoder(dag_batch)

        # 2. select a schedulable stage
        stage_scores = self.stage_policy_network(dag_batch, h_dict)
        stage_idx, stage_lgprob = utils.sample(stage_scores)

        # retrieve index of selected stage's job
        stage_idx_glob = pyg.utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob].item()

        # 3. select the number of executors to add to that stage, conditioned
        # on that stage's job
        exec_scores = self.exec_policy_network(dag_batch, h_dict, job_idx)
        num_exec, exec_lgprob = utils.sample(exec_scores)

        action = {"stage_idx": stage_idx, "job_idx": job_idx, "num_exec": num_exec}

        lgprob = stage_lgprob + exec_lgprob

        return action, {"lgprob": lgprob}

    def evaluate_actions(
        self, obsns: Iterable[dict], actions: Iterable[tuple]
    ) -> dict[str, Tensor]:
        dag_batch = utils.collate_obsns(obsns)
        actions_ten = torch.tensor(actions)

        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_indices, exec_selections = [
            col.clone() for col in actions_ten.T
        ]

        num_stage_acts = dag_batch["num_stage_acts"]
        num_exec_acts = dag_batch["num_exec_acts"]
        num_nodes_per_obs = dag_batch["num_nodes_per_obs"]
        obs_ptr = dag_batch["obs_ptr"]
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model with grads enabled
        dag_batch.to(self.device)
        h_dict = self.encoder(dag_batch)
        stage_scores = self.stage_policy_network(dag_batch, h_dict)
        exec_scores = self.exec_policy_network(dag_batch, h_dict, job_indices)

        stage_lgprobs, stage_entropies = utils.evaluate(
            stage_scores.cpu(), num_stage_acts, stage_selections
        )

        exec_lgprobs, exec_entropies = utils.evaluate(
            exec_scores.cpu(), num_exec_acts[job_indices], exec_selections
        )

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + exec_lgprobs

        action_entropies = stage_entropies + exec_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return {"lgprobs": action_lgprobs, "entropies": action_entropies}


class EncoderNetwork(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()

        self.node_encoder = NodeEncoder(num_node_features, embed_dim, mlp_kwargs)
        self.dag_encoder = DagEncoder(num_node_features, embed_dim, mlp_kwargs)
        self.global_encoder = GlobalEncoder(embed_dim, mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch) -> dict[str, Tensor]:
        """
        Returns:
            a dict of representations at three different levels:
            node, dag, and global.
        """
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        if "obs_ptr" in dag_batch:
            # batch of obsns
            obs_ptr = dag_batch["obs_ptr"]
            h_glob = self.global_encoder(h_dag, obs_ptr)
        else:
            # single obs
            h_glob = self.global_encoder(h_dag)

        return {"node": h_node, "dag": h_dag, "glob": h_glob}


class NodeEncoder(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        embed_dim: int,
        mlp_kwargs: dict[str, Any],
        reverse_flow: bool = True,
    ) -> None:
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = utils.make_mlp(
            num_node_features, output_dim=embed_dim, **mlp_kwargs
        )
        self.mlp_msg = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)
        self.mlp_update = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch) -> Tensor:
        """returns a tensor of shape [num_nodes, embed_dim]"""

        edge_masks = dag_batch["edge_masks"]

        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)

        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg.utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes
        )

        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        edge_masks_it = (
            iter(reversed(edge_masks)) if self.reverse_flow else iter(edge_masks)
        )

        # target-to-source message passing, one level of the dags at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = utils.make_adj(edge_index_masked, num_nodes)

            # nodes sending messages
            src_mask = pyg.utils.index_to_mask(edge_index_masked[self.j], num_nodes)

            # nodes receiving messages
            dst_mask = pyg.utils.index_to_mask(edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])

        return h

    def _forward_no_mp(self, x: Tensor) -> Tensor:
        """forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        """
        return self.mlp_prep(x)


class DagEncoder(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()
        input_dim = num_node_features + embed_dim
        self.mlp = utils.make_mlp(input_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_node: Tensor, dag_batch: pyg.data.Batch) -> Tensor:
        """returns a tensor of shape [num_dags, embed_dim]"""
        # include skip connection from raw input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.mlp(h_node), dag_batch.ptr)
        return h_dag


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim: int, mlp_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.mlp = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_dag: Tensor, obs_ptr: Tensor | None = None) -> Tensor:
        """returns a tensor of shape [num_observations, embed_dim]"""
        h_dag = self.mlp(h_dag)

        if obs_ptr is not None:
            # batch of observations
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob


class StagePolicyNetwork(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        emb_dims: dict[str, int],
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        input_dim = (
            num_node_features + emb_dims["node"] + emb_dims["dag"] + emb_dims["glob"]
        )

        self.mlp_score = utils.make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch, h_dict: dict[str, Tensor]) -> Tensor:
        """returns a tensor of shape [num_nodes,]"""

        stage_mask = dag_batch["stage_mask"]

        x = dag_batch.x[stage_mask]

        h_node = h_dict["node"][stage_mask]

        batch_masked = dag_batch.batch[stage_mask]
        h_dag_rpt = h_dict["dag"][batch_masked]

        if "num_stage_acts" in dag_batch:
            # batch of obsns
            num_stage_acts = dag_batch["num_stage_acts"]
        else:
            # single obs
            num_stage_acts = stage_mask.sum()

        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0
        )

        # residual connections to original features
        node_inputs = torch.cat([x, h_node, h_dag_rpt, h_glob_rpt], dim=1)

        node_scores = self.mlp_score(node_inputs).squeeze(-1)
        return node_scores


class ExecPolicyNetwork(nn.Module):
    def __init__(
        self,
        num_executors: int,
        num_dag_features: int,
        emb_dims: dict[str, int],
        mlp_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        input_dim = num_dag_features + emb_dims["dag"] + emb_dims["glob"] + 1

        self.mlp_score = utils.make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(
        self, dag_batch: pyg.data.Batch, h_dict: dict[str, Tensor], job_indices: Tensor
    ) -> Tensor:
        exec_mask = dag_batch["exec_mask"]

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        x_dag = x_dag[job_indices]

        h_dag = h_dict["dag"][job_indices]

        exec_mask = exec_mask[job_indices]

        if "num_exec_acts" in dag_batch:
            # batch of obsns
            num_exec_acts = dag_batch["num_exec_acts"][job_indices]
        else:
            # single obs
            num_exec_acts = exec_mask.sum()
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)
            exec_mask = exec_mask.unsqueeze(0)

        exec_actions = self._get_exec_actions(exec_mask)

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)

        x_h_dag_rpt = x_h_dag.repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

        dag_inputs = torch.cat([x_h_dag_rpt, h_glob_rpt, exec_actions], dim=1)

        dag_scores = self.mlp_score(dag_inputs).squeeze(-1)
        return dag_scores

    def _get_exec_actions(self, exec_mask: Tensor) -> Tensor:
        exec_actions = torch.arange(self.num_executors) / self.num_executors
        exec_actions = exec_actions.to(exec_mask.device)
        exec_actions = exec_actions.repeat(exec_mask.shape[0])
        exec_actions = exec_actions[exec_mask.view(-1)]
        exec_actions = exec_actions.unsqueeze(1)
        return exec_actions
