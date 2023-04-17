from typing import Tuple, Optional, Union, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.ops import MLP
from torch.optim.lr_scheduler import StepLR
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch_scatter import segment_csr
import numpy as np
from gymnasium.core import ObsType, ActType


from .base import BaseScheduler
from spark_sched_sim import graph_utils



NUM_NODE_FEATURES = 3

NUM_DAG_FEATURES = 2

NUM_GLOBAL_FEATURES = 1

ACT_FN = nn.LeakyReLU



class DecimaScheduler(BaseScheduler):

    def __init__(
        self,
        num_executors: int,
        training_mode: bool = True,
        state_dict_path: str = None,
        dim_embed: int = 8,
        optim_class: torch.optim.Optimizer = torch.optim.Adam,
        optim_lr: float = .001,
        max_grad_norm: float = .5
    ):
        super().__init__('Decima')

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
        self.lr_scheduler = StepLR(self.optim, 30, .5)



    @torch.no_grad()
    def schedule(self, obs: ObsType, greedy=False) -> ActType:
        '''assumes that `DecimaObsWrapper` is providing
        observations of the environment and `DecimaActWrapper` 
        is receiving actions returned from here.
        '''
        dag_batch = graph_utils.obs_to_pyg(obs['dag_batch'])
        batch = dag_batch.batch.clone() # save a CPU copy
        dag_batch = dag_batch.to(self.device, non_blocking=True)

        # no computational graphs needed during the episode
        outputs = self.actor(dag_batch)
        node_scores, dag_scores = [out.cpu() for out in outputs]

        schedulable_stage_mask = \
            torch.tensor(obs['schedulable_stage_mask'], dtype=bool)

        valid_prlsm_lim_mask = \
            torch.from_numpy(np.vstack(obs['valid_prlsm_lim_mask']))

        self._mask_outputs(
            (node_scores, dag_scores),
            (schedulable_stage_mask, valid_prlsm_lim_mask)
        )

        action, lgprob = self._sample_action(
            node_scores, 
            dag_scores, 
            batch, 
            greedy
        )

        return action, lgprob



    def evaluate_actions(
        self, 
        obsns: graph_utils.ObsBatch,
        actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        
        # save CPU copies of some attributes
        obs_ptr = obsns.dag_batch['obs_ptr']
        num_nodes_per_dag = obsns.dag_batch['num_nodes_per_dag']
        num_nodes_per_obs = obsns.dag_batch['num_nodes_per_obs']

        obsns.dag_batch.to(self.device)
        model_outputs = self.actor(obsns.dag_batch)

        # move model outputs to CPU
        node_scores_batch, dag_scores_batch = [out.cpu() for out in model_outputs]

        self._mask_outputs(
            (node_scores_batch, dag_scores_batch),
            (obsns.schedulable_stage_masks, obsns.valid_prlsm_lim_masks)
        )

        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        node_selections, dag_idxs, dag_selections = \
            [col.clone() for col in actions.T]

        (all_node_probs, 
         node_lgprobs, 
         node_entropies) = \
            self._evaluate_node_actions(
                node_scores_batch,
                node_selections,
                num_nodes_per_obs
            )

        dag_probs = \
            self._compute_dag_probs(
                all_node_probs, 
                num_nodes_per_dag
            )
        
        dag_lgprobs, dag_entropies = \
            self._evaluate_dag_actions(
                dag_scores_batch, 
                dag_idxs, 
                dag_selections,
                dag_probs,
                obs_ptr
            )

        # aggregate the evaluations for nodes and dags
        action_lgprobs = node_lgprobs + dag_lgprobs
        action_entropies = (node_entropies + dag_entropies) * self._entropy_scale(num_nodes_per_obs)

        return action_lgprobs, action_entropies



    def update_parameters(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        
        # compute gradients
        loss.backward()

        # clip grads
        try:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.max_grad_norm,
                error_if_nonfinite=True
            )
        except:
            print('infinite grad; skipping update.')
            return

        # update model parameters
        self.optim.step()



    ## internal methods

    @classmethod
    def _mask_outputs(
        cls,
        outputs: tuple[Tensor, Tensor],
        masks: tuple[Tensor, Tensor]
    ):
        '''masks model outputs in-place'''
        node_scores, dag_scores = outputs
        schedulable_stage_mask, valid_prlsm_lim_mask = masks

        # mask node scores
        node_scores[~schedulable_stage_mask] = float('-inf')

        # mask dag scores
        dag_scores.masked_fill_(~valid_prlsm_lim_mask, float('-inf'))



    @classmethod
    def _sample_action(cls, node_scores, dag_scores, batch, greedy):
        # select the next stage to schedule
        c_stage = Categorical(logits=node_scores)
        if greedy:
            stage_idx = torch.argmax(node_scores)
        else:
            stage_idx = c_stage.sample()
        lgprob_stage = c_stage.log_prob(stage_idx)

        # select the parallelism limit for the selected stage's job
        job_idx = batch[stage_idx]
        dag_scores = dag_scores[job_idx]
        c_pl = Categorical(logits=dag_scores)
        if greedy:
            prlsm_lim = torch.argmax(dag_scores, dim=-1)
        else:
            prlsm_lim = c_pl.sample()
        lgprob_pl = c_pl.log_prob(prlsm_lim)

        lgprob = lgprob_stage + lgprob_pl

        act = {
            'stage_idx': stage_idx.item(),
            'job_idx': job_idx.item(),
            'prlsm_lim': prlsm_lim.item()
        }
        
        return act, lgprob.item()



    @classmethod
    def _compute_dag_probs(cls, all_node_probs, num_nodes_per_dag):
        '''for each dag, compute the probability of it
        being selected by summing over the probabilities
        of each of its nodes being selected
        '''
        dag_indptr = num_nodes_per_dag.cumsum(0)
        dag_indptr = torch.cat([torch.tensor([0]), dag_indptr], 0)
        dag_probs = segment_csr(all_node_probs, dag_indptr)
        return dag_probs



    def _entropy_scale(self, num_nodes_per_obs):
        entropy_norm = torch.log(self.num_executors * num_nodes_per_obs)
        entropy_scale = 1 / torch.max(torch.tensor(1), entropy_norm)
        return entropy_scale



    @classmethod
    def _evaluate_node_actions(
        cls,
        node_scores_batch: Tensor,
        node_action_batch: Tensor,
        num_nodes_per_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        '''splits the node action batch into subbatches 
        (see subroutine below), then for each subbatch, evaluates
        actions using vectorized computations. Finally, merges the 
        evaluations from the subbatches together. This is faster than 
        either 
        - separately computing attributes for each sample in the
        batch, because vectorized computations are not utilized
        at all, or
        - stacking the whole batch together with padding and doing 
        one large vectorized computation, because the backward 
        pass becomes very expensive

        Args:
            node_scores_batch: flat batch of node scores from the actor
                network, with shape (total_num_nodes,)
            node_selection_batch: batch of node actions with shape
                (num_actions,)
            num_nodes_per_obs: stores the number of nodes in each
                observation with shape (num_actions,)
        Returns:
            tuple (all_node_probs, node_lgprobs, node_entropies), where
                all_node_probs is the probability of each node getting
                    selected, shape (total_num_nodes,)
                node_lgprobs is the log-probability of the selected nodes
                    actually getting selected, shape (num_actions,)
                node_entropies is the node entropy for each model
                    output, shape (num_actions,)
        '''

        def _eval_node_actions(node_scores, node_selection):
            c = Categorical(logits=node_scores)
            node_lgprob = c.log_prob(node_selection)
            node_entropy = c.entropy()
            return c.probs, node_lgprob, node_entropy

        (node_scores_subbatches, 
         node_selection_subbatches, 
         subbatch_node_counts) = \
            cls._split_node_experience(
                node_scores_batch, 
                node_action_batch, 
                num_nodes_per_obs
            )

        # evaluate actions for each subbatch, vectorized
        all_node_probs = []
        node_lgprobs = []
        node_entropies = []

        gen = zip(node_scores_subbatches,
                  node_selection_subbatches,
                  subbatch_node_counts)

        for (node_scores_subbatch, 
             node_selection_subbatch,
             node_count) in gen:

            node_scores_subbatch = \
                node_scores_subbatch.view(-1, node_count)

            (node_probs, 
             node_lgprob_subbatch, 
             node_entropy_subbatch) = \
                _eval_node_actions(node_scores_subbatch,
                                   node_selection_subbatch)

            all_node_probs += [torch.flatten(node_probs)]
            node_lgprobs += [node_lgprob_subbatch]
            node_entropies += [node_entropy_subbatch]

        ## collate the subbatch attributes

        # for each node ever seen, records the probability
        # that that node is selected out of all the
        # nodes within its observation
        all_node_probs = torch.cat(all_node_probs)

        # for each observation, records the log probability
        # of its node selection
        node_lgprobs = torch.cat(node_lgprobs)

        # for each observation, records its node entropy
        node_entropies = torch.cat(node_entropies)

        return all_node_probs, node_lgprobs, node_entropies



    @classmethod
    def _split_node_experience(
        cls,
        node_scores_batch: Tensor, 
        node_selection_batch: Tensor, 
        num_nodes_per_obs: Tensor
    ) -> Tuple[Iterable[Tensor], Iterable[Tensor], Tensor]:
        '''splits the node score/selection batches into
        subbatches, where each each sample within a subbatch
        has the same node count.
        '''
        batch_size = len(num_nodes_per_obs)

        # find indices where node count changes
        node_count_change_mask = \
            num_nodes_per_obs[:-1] != num_nodes_per_obs[1:]

        ptr = 1 + node_count_change_mask.nonzero().squeeze()
        if ptr.shape == torch.Size():
            # ptr is zero-dimentional; not allowed in torch.cat
            ptr = ptr.unsqueeze(0)
        ptr = torch.cat([torch.tensor([0]), 
                         ptr, 
                         torch.tensor([batch_size])])

        # unique node count within each subbatch
        subbatch_node_counts = num_nodes_per_obs[ptr[:-1]]

        # number of samples in each subbatch
        subbatch_sizes = ptr[1:] - ptr[:-1]

        # split node scores into subbatches
        node_scores_split = \
            torch.split(node_scores_batch, 
                        list(subbatch_sizes * subbatch_node_counts))

        # split node selections into subbatches
        node_selection_split = \
            torch.split(node_selection_batch, 
                        list(subbatch_sizes))

        return node_scores_split, \
               node_selection_split, \
               subbatch_node_counts



    @classmethod
    def _evaluate_dag_actions(
        cls,
        dag_scores_batch, 
        dag_idxs, 
        dag_selections,
        dag_probs,
        obs_ptr
    ):
        dag_idxs += obs_ptr[:-1]

        dag_lgprob_batch = \
            Categorical(logits=dag_scores_batch[dag_idxs]) \
                .log_prob(dag_selections)

        # can't have rows where all the entries are
        # -inf when computing entropy, so for all such 
        # rows, set the first entry to be 0. then the 
        # entropy for these rows becomes 0.
        inf_counts = torch.isinf(dag_scores_batch).sum(1)
        allinf_rows = (inf_counts == dag_scores_batch.shape[1])
        dag_scores_batch[allinf_rows, 0] = 0

        # compute expected entropy over dags for each obs.
        # each dag is weighted by the probability of it 
        # being selected. sum is segmented over observations.
        entropy_per_dag = Categorical(logits=dag_scores_batch).entropy()
        dag_entropy_batch = segment_csr(dag_probs * entropy_per_dag, obs_ptr)
        
        return dag_lgprob_batch, dag_entropy_batch




class ActorNetwork(nn.Module):
    
    def __init__(self, num_executors: int, dim_embed: int):
        super().__init__()
        self.encoder = EncoderNetwork(dim_embed)
        self.policy_network = PolicyNetwork(num_executors, dim_embed)
        

        
    def forward(self, dag_batch: Batch):
        '''
        Args:
            dag_batch (torch_geometric.data.Batch): PyG batch of job dags
            num_dags_per_obs (optional torch.Tensor): if dag_batch is a nested
                batch of dag_batches for many separate observations, then
                this argument specifies how many dags are in each observation.
                If it is not provided, then the dag_batch is assumed to not be
                nested.
        Returns:
            node scores and dag scores, and if dag_batch is a batch of batches, 
            then additionally returns the number of nodes in each observation and 
            an tensor containing the starting node index for each observation
        '''

        dag_batch['num_nodes_per_dag'] = dag_batch.ptr[1:] - dag_batch.ptr[:-1]

        if not hasattr(dag_batch, 'obs_ptr'):
            dag_batch['num_dags_per_obs'] = dag_batch.num_graphs
            dag_batch['num_nodes_per_obs'] = dag_batch.x.shape[0]

        node_embeddings, dag_summaries, global_summaries = self.encoder(dag_batch)

        node_scores, dag_scores = \
            self.policy_network(
                dag_batch,
                node_embeddings, 
                dag_summaries, 
                global_summaries
            )

        return node_scores, dag_scores
    



class EncoderNetwork(nn.Module):

    def __init__(self, dim_embed):
        super().__init__()

        # node embeddings
        self.mlp_node_prep = MLP(5, [16, 8], activation_layer=ACT_FN)
        self.mlp_node_msg = MLP(8, [16, 8], activation_layer=ACT_FN)
        self.mlp_node_update = MLP(8, [16, 8], activation_layer=ACT_FN)

        # dag summaries
        self.mlp_dag_msg = MLP(5 + dim_embed, [16, 8, dim_embed], activation_layer=ACT_FN)

        # global summaries
        self.mlp_glob_msg = MLP(dim_embed, [16, 8, dim_embed], activation_layer=ACT_FN)

        self.num_nodes = -1
        self.edge_index = None
        self.level_edge_index_list = None
        self.level_mask_batch = None



    def forward(self, dag_batch):
        node_embeddings = self._embed_nodes(dag_batch)
        dag_summaries = self._summarize_dags(dag_batch, node_embeddings)
        global_summaries = self._summarize_global_states(dag_batch, dag_summaries)
        return node_embeddings, dag_summaries, global_summaries
    


    def _embed_nodes(self, dag_batch):
        num_nodes = dag_batch.x.shape[0]

        # preprocess node features
        x = self.mlp_node_prep(dag_batch.x)

        if self.edge_index is None or self.num_nodes != num_nodes or \
            not torch.equal(dag_batch.edge_index, self.edge_index):
            self.num_nodes = num_nodes
            self.edge_index = dag_batch.edge_index.clone()
            self.level_edge_index_list, self.level_mask_batch = \
                graph_utils.construct_message_passing_levels(dag_batch)
        
        if self.level_edge_index_list is None:
            # no message passing to be done
            return x
        
        # target-to-source message passing, one level of the dag (batch) at a time
        for edge_index, mask in zip(self.level_edge_index_list, self.level_mask_batch):
            adj = graph_utils.make_adj(edge_index, num_nodes)

            # message
            y = self.mlp_node_msg(x)

            # aggregate
            y = torch.sparse.mm(adj, y)

            # update
            y = self.mlp_node_update(y)
            x = x + mask.unsqueeze(1) * y
        
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
            x = x.sum(dim=0).unsqueeze(0)

        return x
        
        
        

class PolicyNetwork(nn.Module):

    def __init__(self, num_executors, dim_embed):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_executors = num_executors
        self.mlp_node_score = MLP(5 + (3 * dim_embed), [16, 8, 1], activation_layer=ACT_FN)
        self.mlp_dag_score = MLP(3 + (2 * dim_embed) + 1, [16, 8, 1], activation_layer=ACT_FN)
        


    def forward(
        self,   
        dag_batch, 
        node_embeddings,
        dag_summaries, 
        global_summaries
    ):
        node_scores = self._compute_node_scores(
            dag_batch, 
            node_embeddings, 
            dag_summaries, 
            global_summaries
        )

        dag_scores = self._compute_dag_scores(
            dag_batch,
            dag_summaries, 
            global_summaries
        )

        return node_scores, dag_scores

    
    
    def _compute_node_scores(
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
        
        global_summaries_repeat = \
            global_summaries.repeat_interleave(
                dag_batch['num_nodes_per_obs'], 
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
    
    
    
    def _compute_dag_scores(
        self, 
        dag_batch,
        dag_summaries, 
        global_summaries
    ):
        dag_idxs = dag_batch.ptr[:-1]
        dag_features = dag_batch.x[dag_idxs, 0:3]

        executor_actions = torch.arange(self.num_executors, device=dag_features.device)
        executor_actions = executor_actions.repeat(dag_batch.num_graphs).unsqueeze(1)

        # add skip connection to original dag features
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
                dag_batch['num_dags_per_obs'] * self.num_executors, 
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

        return dag_scores.squeeze(-1).view(dag_batch.num_graphs, self.num_executors)

    