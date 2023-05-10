from typing import Tuple, Optional, Union, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.ops import MLP
from torch_scatter import segment_csr
import numpy as np
from gymnasium.core import ObsType, ActType

from .base import BaseScheduler
from spark_sched_sim import graph_utils



NUM_NODE_FEATURES = 3

NUM_DAG_FEATURES = 2

NUM_GLOBAL_FEATURES = 1

ACT_FN = lambda inplace, neg_slope=.2: nn.LeakyReLU(neg_slope, inplace)



class DecimaScheduler(BaseScheduler):

    def __init__(
        self,
        num_executors: int,
        training_mode: bool = True,
        state_dict_path: str = None,
        dim_embed: int = 8,
        optim_class: torch.optim.Optimizer = torch.optim.Adam,
        optim_lr: float = .001,
        max_grad_norm: Optional[float] = None
    ):
        name = 'Decima'
        if state_dict_path:
            name += f':{state_dict_path}'
        super().__init__(name)

        self.actor = ActorNetwork(num_executors, dim_embed)

        self.num_executors = num_executors

        self.optim_class = optim_class

        self.optim_lr = optim_lr

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.actor.load_state_dict(state_dict)

        self.actor.train(training_mode)

        self.training_mode = training_mode

        self.max_grad_norm = max_grad_norm



    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device



    def build(self, device: torch.device = None) -> None:
        if device is not None:
            self.actor.to(device)

        params = self.actor.parameters()
        self.optim = self.optim_class(params, lr=self.optim_lr)



    @torch.no_grad()
    def schedule(self, obs: ObsType, greedy=False) -> ActType:
        '''assumes that `DecimaObsWrapper` is providing observations of the environment 
        and `DecimaActWrapper` is receiving actions returned from here.
        '''
        dag_batch, stage_to_job_map = graph_utils.obs_to_pyg(obs)
        dag_batch.to(self.device, non_blocking=True)
        self.actor.embedding_forward(dag_batch)

        # first: select a stage
        stage_scores = self.actor.stage_forward().cpu()
        schedulable_stage_mask = torch.tensor(obs['schedulable_stage_mask'], dtype=bool)
        stage_idx, stage_lgprob = self._masked_sample(stage_scores, schedulable_stage_mask, greedy)
        job_idx = stage_to_job_map[stage_idx]

        # second: select a parallelism limit, conditioned on the selected stage's job
        prlsm_lim_scores = self.actor.prlsm_lim_forward(job_idx).cpu()
        valid_prlsm_lim_mask = torch.from_numpy(np.vstack(obs['valid_prlsm_lim_mask']))
        prlsm_lim, prlsm_lim_lgprob = self._masked_sample(prlsm_lim_scores, valid_prlsm_lim_mask[job_idx], greedy)

        action = {
            'stage_idx': stage_idx.item(),
            'job_idx': job_idx.item(),
            'prlsm_lim': prlsm_lim.item()
        }

        lgprob = stage_lgprob + prlsm_lim_lgprob

        return action, lgprob.item()



    def evaluate_actions(
        self, 
        obsns: graph_utils.ObsBatch,
        actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_idx, prlsm_lim_selections = [col.clone() for col in actions.T]

        num_nodes_per_obs = obsns.dag_batch['num_nodes_per_obs']
        obs_ptr = obsns.dag_batch['obs_ptr']
        job_idx += obs_ptr[:-1]

        obsns.dag_batch.to(self.device)
        self.actor.embedding_forward(obsns.dag_batch)
        stage_scores = self.actor.stage_forward()
        prlsm_lim_scores = self.actor.prlsm_lim_forward(job_idx)

        stage_scores = self._apply_mask(stage_scores.cpu(), obsns.schedulable_stage_masks)
        prlsm_lim_scores = self._apply_mask(prlsm_lim_scores.cpu(), obsns.valid_prlsm_lim_masks[job_idx])

        stage_lgprobs, stage_entropies = \
            self._evaluate_node_actions(
                stage_scores, 
                stage_selections, 
                num_nodes_per_obs
            )
        
        prlsm_lim_lgprobs, prlsm_lim_entropies = \
            self._evaluate(
                prlsm_lim_scores, 
                prlsm_lim_selections
            )

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + prlsm_lim_lgprobs

        action_entropies = stage_entropies + prlsm_lim_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return action_lgprobs, action_entropies



    def update_parameters(self, loss=None) -> None:
        if loss:
            # accumulate gradients
            loss.backward()

        if self.max_grad_norm:
            # clip grads
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    self.max_grad_norm,
                    error_if_nonfinite=True
                )
            except:
                print('infinite grad; skipping update.')
                self.optim.zero_grad()
                return

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()



    ## internal methods

    @classmethod
    def _apply_mask(cls, t: Tensor, msk: Tensor) -> Tensor:
        '''masks model outputs in-place'''
        min_real = torch.finfo(t.dtype).min
        return t.masked_fill(~msk, min_real)



    @classmethod
    def _masked_sample(cls, logits, mask, greedy):
        logits = cls._apply_mask(logits, mask)
        c = Categorical(logits=logits)
        if greedy:
            samp = logits.argmax()
        else:
            samp = c.sample()
        lgprob = c.log_prob(samp)
        return samp, lgprob
    

    @classmethod
    def _evaluate(cls, logits, selections):
        c = Categorical(logits=logits)
        return c.log_prob(selections), c.entropy()



    @classmethod
    def _evaluate_node_actions(
        cls,
        all_node_scores: Tensor,
        all_node_selections: Tensor,
        num_nodes_per_obs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        node_scores_split, node_selections_split, node_counts = \
            cls._split_node_experience(all_node_scores, all_node_selections, num_nodes_per_obs)

        # evaluate actions for each chunk, vectorized
        node_lgprobs, node_entropies = zip(*[
            cls._evaluate(node_scores.view(-1, node_count), node_selections)
            for node_scores, node_selections, node_count in zip(
                node_scores_split, node_selections_split, node_counts
            )
        ])

        return torch.cat(node_lgprobs), torch.cat(node_entropies)



    @classmethod
    def _split_node_experience(
        cls,
        all_node_scores: Tensor, 
        all_node_selections: Tensor, 
        num_nodes_per_obs: Tensor
    ) -> Tuple[Iterable[Tensor], Iterable[Tensor], Tensor]:
        '''splits the node score/selection batches into subbatches, where each 
        each sample within a subbatch has the same node count.
        '''
        batch_size = len(num_nodes_per_obs)

        # find indices where node count changes
        node_count_change_mask = num_nodes_per_obs[:-1] != num_nodes_per_obs[1:]

        ptr = 1 + node_count_change_mask.nonzero().squeeze()
        if ptr.shape == torch.Size():
            # ptr is zero-dimentional; not allowed in torch.cat
            ptr = ptr.unsqueeze(0)
        ptr = torch.cat([torch.tensor([0]), ptr, torch.tensor([batch_size])])

        # unique node count within each chunk
        node_counts = num_nodes_per_obs[ptr[:-1]]

        # number of samples in each subbatch
        chunk_sizes = ptr[1:] - ptr[:-1]

        # split node scores into subbatches
        node_scores_split = torch.split(all_node_scores, list(chunk_sizes * node_counts))

        # split node selections into subbatches
        node_selections_split = torch.split(all_node_selections, list(chunk_sizes))

        return node_scores_split, node_selections_split, node_counts




class ActorNetwork(nn.Module):
    
    def __init__(self, num_executors: int, dim_embed: int):
        super().__init__()
        self.encoder = EncoderNetwork(dim_embed)
        self.policy_network = PolicyNetwork(num_executors, dim_embed)

        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()

        self.state = {}



    def embedding_forward(self, dag_batch):
        assert self.state == {}
        dag_batch['num_nodes_per_dag'] = dag_batch.ptr[1:] - dag_batch.ptr[:-1]

        self.state['dag_batch'] = dag_batch
        (self.state['node_embeddings'], 
         self.state['dag_summaries'], 
         self.state['global_summaries']) = \
            self.encoder(dag_batch)


    
    def stage_forward(self):
        assert self.state != {}
        return self.policy_network.stage_forward(**self.state)
    


    def prlsm_lim_forward(self, job_idx):
        assert self.state != {}
        prlsm_lim_scores = self.policy_network.prlsm_lim_forward(job_idx, **self.state)

        self.state.clear()

        return prlsm_lim_scores
    



class EncoderNetwork(nn.Module):

    def __init__(self, dim_embed):
        super().__init__()

        # node embeddings
        self.mlp_node_prep = MLP(5, [16, 8, dim_embed], activation_layer=ACT_FN)
        self.mlp_node_msg = MLP(8, [16, 8, dim_embed], activation_layer=ACT_FN)
        self.mlp_node_update = MLP(8, [16, 8, dim_embed], activation_layer=ACT_FN)

        # dag summaries
        self.mlp_dag_msg = MLP(5 + dim_embed, [16, 8, dim_embed], activation_layer=ACT_FN)

        # global summaries
        self.mlp_glob_msg = MLP(dim_embed, [16, 8, dim_embed], activation_layer=ACT_FN)



    def forward(self, dag_batch):
        node_embeddings = self._embed_nodes(dag_batch)
        dag_summaries = self._summarize_dags(dag_batch, node_embeddings)
        global_summaries = self._summarize_global_states(dag_batch, dag_summaries)
        return node_embeddings, dag_summaries, global_summaries
    


    def _embed_nodes(self, dag_batch):
        # preprocess node features
        x = self.mlp_node_prep(dag_batch.x)

        edge_mask_batch = dag_batch['edge_mask_batch']
        
        message_passing_depth = edge_mask_batch.shape[0]
        if message_passing_depth == 0:
            return x
        
        num_nodes = dag_batch.x.shape[0]

        # target-to-source message passing, one level of the dag (batch) at a time
        for edge_mask in edge_mask_batch:
            # mask out the edges not involved in this message passing step,
            # and convert to sparse adjacency matrix format
            edge_index = dag_batch.edge_index[:, edge_mask]
            adj = graph_utils.make_adj(edge_index, num_nodes)

            # the bias terms from the update MLP will give non-zero values
            # to nodes that were not part of this message passing step, so
            # we need to explicitly mask those out. We only intend to pass
            # messages to the nodes found in `edge_index[0]`.
            node_mask = torch.zeros(num_nodes, dtype=bool)
            node_mask[edge_index[0]] = True
            node_mask = node_mask.unsqueeze(1).to(x.device)

            # message
            y = self.mlp_node_msg(x)

            # aggregate
            y = torch.sparse.mm(adj, y)

            # update
            x = x + node_mask * self.mlp_node_update(y)
        
        return x
    


    def _summarize_dags(self, dag_batch, node_embeddings):
        # add skip connection to original input
        x = torch.cat([dag_batch.x, node_embeddings], dim=1)

        # message
        x = self.mlp_dag_msg(x)

        # aggregate
        x = segment_csr(x, dag_batch.ptr)

        return x



    def _summarize_global_states(self, dag_batch, dag_summaries):
        # message
        x = self.mlp_glob_msg(dag_summaries)

        # aggregate
        if hasattr(dag_batch, 'obs_ptr'):
            # batch of observations
            x = segment_csr(x, dag_batch['obs_ptr'])
        else:
            # single observation
            x = x.sum(0).unsqueeze(0)

        return x
        
        
        

class PolicyNetwork(nn.Module):

    def __init__(self, num_executors, dim_embed):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_executors = num_executors
        self.mlp_node_score = MLP(5 + (3 * dim_embed), [32, 16, 8, 1], activation_layer=ACT_FN)
        self.mlp_dag_score = MLP(3 + (2 * dim_embed) + 1, [32, 16, 8, 1], activation_layer=ACT_FN)

    
    
    def stage_forward(
        self, 
        dag_batch,
        node_embeddings, 
        dag_summaries, 
        global_summaries
    ):
        num_nodes = dag_batch.x.shape[0]

        dag_summaries_repeat = \
            dag_summaries.repeat_interleave(
                dag_batch['num_nodes_per_dag'],
                output_size=num_nodes,
                dim=0
            )
        
        if hasattr(dag_batch, 'num_nodes_per_obs'):
            # batch of observations
            num_nodes_per_obs = dag_batch['num_nodes_per_obs']
        else:
            # single observation
            num_nodes_per_obs = dag_batch.x.shape[0]
        
        global_summaries_repeat = \
            global_summaries.repeat_interleave(
                num_nodes_per_obs,
                output_size=num_nodes,
                dim=0
            )

        # add skip connection to original node features
        node_inputs = torch.cat(
            [
                dag_batch.x, 
                node_embeddings, 
                dag_summaries_repeat, 
                global_summaries_repeat
            ], 
            dim=1
        )

        node_scores = self.mlp_node_score(node_inputs)

        return node_scores.squeeze(-1)
    
    
    
    def prlsm_lim_forward(
        self, 
        job_idx,
        dag_batch,
        dag_summaries, 
        global_summaries,
        **kwargs
    ):
        dag_idxs = dag_batch.ptr[:-1]
        dag_features = dag_batch.x[dag_idxs, 0:3]
        batch_size = job_idx.numel()

        executor_actions = torch.arange(self.num_executors) / self.num_executors
        executor_actions = executor_actions.repeat(batch_size).unsqueeze(1).to(dag_features.device)

        dag_features = dag_features[job_idx]
        dag_summaries = dag_summaries[job_idx]
        if batch_size == 1:
            dag_features = dag_features.unsqueeze(0)
            dag_summaries = dag_summaries.unsqueeze(0)

        dag_features_merged = torch.cat([dag_features, dag_summaries], dim=1)

        num_total_actions = executor_actions.shape[0]

        dag_features_merged_repeat = \
            dag_features_merged.repeat_interleave(
                self.num_executors,
                output_size=num_total_actions,
                dim=0
            )

        global_summaries_repeat = \
            global_summaries.repeat_interleave(
                self.num_executors, 
                output_size=num_total_actions,
                dim=0
            )
        
        dag_inputs = torch.cat(
            [
                dag_features_merged_repeat,
                global_summaries_repeat,
                executor_actions
            ], 
            dim=1
        )

        dag_scores = self.mlp_dag_score(dag_inputs)

        return dag_scores.squeeze(-1).view(batch_size, self.num_executors)

    