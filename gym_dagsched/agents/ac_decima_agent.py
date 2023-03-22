from typing import Tuple, Optional, Union, Iterable
from torch import Tensor
from itertools import chain

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data, Batch
from torch_scatter import segment_add_csr
from torch_sparse import matmul
from torchvision.ops import MLP
from torch.nn import Tanh
import numpy as np
from gymnasium.core import ObsType, ActType
from torch.optim.lr_scheduler import LambdaLR

from .base_agent import BaseAgent
from ..utils.graph import obs_to_pyg, ObsBatch



NUM_NODE_FEATURES = 2

NUM_DAG_FEATURES = 2

NUM_GLOBAL_FEATURES = 1



class DecimaAgent(BaseAgent):

    def __init__(
        self,
        num_workers: int,
        training_mode: bool = True,
        state_dict_path: str = None,
        dim_embed: int = 8,
        optim_class: torch.optim.Optimizer = torch.optim.Adam,
        optim_lr: float = .001,
        max_grad_norm: float = .5
    ):
        super().__init__('Decima')

        self.actor = ActorNetwork(num_workers, dim_embed)
        # self.critic = MLP(dim_embed, [64, 32, 8, 1])
        self.critic = CriticNetwork(dim_embed)

        self.num_workers = num_workers

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
            self.critic.to(device)

        self.ac_opt = self.optim_class([
            {'params': self.actor.parameters(), 'lr': self.optim_lr},
            {'params': self.critic.parameters(), 'lr': .005}
        ])

        lam_act = lambda i: 1
        lam_crit = lambda i: .995**i
        self.sched = LambdaLR(self.ac_opt, [lam_act, lam_crit])

        self.inner_opt = torch.optim.Adam(self.critic.parameters(), lr=.001)



    def predict_values(self, obsns):
        return self.critic(
            obsns.nested_dag_batch.to(self.device),
            obsns.num_dags_per_obs.to(self.device)
        ).cpu()

    # def predict_values(self, obsns):
    #     global_embeddings = self.actor.global_embed(
    #         obsns.nested_dag_batch.to(self.device),
    #         obsns.num_dags_per_obs.to(self.device)
    #     )
    #     return self.critic(global_embeddings).cpu()



    @torch.no_grad()
    def predict(self, obs: ObsType) -> ActType:
        '''assumes that `DecimaObsWrapper` is providing
        observations of the environment and `DecimaActWrapper` 
        is receiving actions returned from here.
        '''
        dag_batch = obs_to_pyg(obs['dag_batch'])
        batch = dag_batch.batch.clone() # save a CPU copy
        dag_batch = dag_batch.to(self.device, non_blocking=True)

        # no computational graphs needed during the episode
        actor_outputs = self.actor(dag_batch)
        node_scores, dag_scores = [out.cpu() for out in actor_outputs]

        schedulable_op_mask = \
            torch.tensor(obs['schedulable_op_mask'], dtype=bool)

        valid_prlsm_lim_mask = \
            torch.from_numpy(np.vstack(obs['valid_prlsm_lim_mask']))
        
        self._apply_mask(node_scores, schedulable_op_mask)
        self._apply_mask(dag_scores, valid_prlsm_lim_mask)

        action, lgprob = self._sample_action(node_scores, dag_scores, batch)

        return action, lgprob



    def evaluate_actions(
        self, 
        obsns: ObsBatch,
        actions: Tensor
    ) -> tuple[Tensor, Tensor]:

        actor_outputs = \
            self.actor(
                obsns.nested_dag_batch.to(self.device),
                obsns.num_dags_per_obs.to(self.device)
            )

        # move actor outputs to CPU
        (node_scores_batch, 
         dag_scores_batch,
         num_nodes_per_obs, 
         obs_indptr) = \
            [out.cpu() for out in actor_outputs]
        
        self._apply_mask(node_scores_batch, obsns.schedulable_op_masks)
        self._apply_mask(dag_scores_batch, obsns.valid_prlsm_lim_masks)

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
                obsns.num_nodes_per_dag
            )
        
        dag_lgprobs, dag_entropies = \
            self._evaluate_dag_actions(
                dag_scores_batch, 
                dag_idxs, 
                dag_selections,
                dag_probs,
                obs_indptr
            )

        # aggregate the evaluations for nodes and dags
        action_lgprobs = node_lgprobs + dag_lgprobs
        action_entropies = (node_entropies + dag_entropies) # * \
                           #self._get_entropy_scale(num_nodes_per_obs)

        return action_lgprobs, action_entropies



    def update_parameters(self, loss: torch.Tensor) -> None:
        self.ac_opt.zero_grad()
        
        # compute gradients
        loss.backward()

        # clip grads
        try:
            params = chain(self.actor.parameters(), self.critic.parameters())
            torch.nn.utils.clip_grad_norm_(
                params, 
                self.max_grad_norm,
                error_if_nonfinite=True
            )
        except:
            # infinite grad, skip update
            return

        # update model parameters
        self.ac_opt.step()



    ## internal methods

    @classmethod
    def _apply_mask(cls, t: Tensor, msk: Tensor) -> Tensor:
        '''masks model outputs in-place'''
        min_real = torch.finfo(t.dtype).min
        t.masked_fill_(~msk, min_real)



    @classmethod
    def _sample_action(cls, node_scores, dag_scores, batch):
        # select the next operation to schedule
        c_op = Categorical(logits=node_scores)
        op_idx = c_op.sample()
        lgprob_op = c_op.log_prob(op_idx)

        # select the parallelism limit for the selected op's job
        job_idx = batch[op_idx]
        dag_scores = dag_scores[job_idx]
        c_pl = Categorical(logits=dag_scores)
        prlsm_lim = c_pl.sample()
        lgprob_pl = c_pl.log_prob(prlsm_lim)

        lgprob = lgprob_op + lgprob_pl

        act = {
            'op_idx': op_idx.item(),
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
        dag_probs = segment_add_csr(all_node_probs, dag_indptr)
        return dag_probs



    def _get_entropy_scale(self, num_nodes_per_obs):
        entropy_norm = torch.log(self.num_workers * num_nodes_per_obs)
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
        obs_indptr
    ):
        dag_idxs += obs_indptr[:-1]

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
        dag_entropy_batch = \
            segment_add_csr(dag_probs * entropy_per_dag, obs_indptr)
        
        return dag_lgprob_batch, dag_entropy_batch



class ActorNetwork(nn.Module):
    
    def __init__(self, num_workers: int, dim_embed: int):
        super().__init__()
        self.encoder = GraphEncoderNetwork(dim_embed)
        self.policy_network = PolicyNetwork(num_workers, dim_embed)
        

        
    def forward(
        self, 
        dag_batch: Batch,
        num_dags_per_obs: Optional[Tensor] = None
    ) -> Union[tuple[Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
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

        is_data_batched = (num_dags_per_obs is not None)

        (obs_indptr, 
         num_nodes_per_dag, 
         num_nodes_per_obs, 
         num_dags_per_obs) = \
            self._bookkeep(dag_batch, num_dags_per_obs)

        (node_embeddings, 
         dag_embeddings, 
         global_embeddings) = \
            self.encoder(dag_batch, obs_indptr)

        node_scores, dag_scores = \
            self.policy_network(
                node_embeddings, 
                dag_embeddings, 
                global_embeddings,
                num_nodes_per_dag,
                num_nodes_per_obs,
                num_dags_per_obs
            )

        ret = (node_scores, dag_scores)
        if is_data_batched:
            ret += (num_nodes_per_obs, obs_indptr)
        return ret


    @torch.no_grad()
    def global_embed(self, dag_batch, num_dags_per_obs):
        obs_indptr, *_, num_dags_per_obs = self._bookkeep(dag_batch, num_dags_per_obs)
        *_, global_embeddings = self.encoder(dag_batch, obs_indptr)
        return global_embeddings


    def _bookkeep(self, dag_batch, num_dags_per_obs):
        num_nodes_per_dag = dag_batch.ptr[1:] - dag_batch.ptr[:-1]

        if num_dags_per_obs is None:
            num_dags_per_obs = dag_batch.num_graphs
            num_nodes_per_obs = dag_batch.x.shape[0]
            obs_indptr = None
        else:
            # data is batched
            batch_size = len(num_dags_per_obs)
            device = dag_batch.x.device
            obs_indptr = torch.zeros(batch_size+1, 
                                     device=device, 
                                     dtype=torch.long)
            torch.cumsum(num_dags_per_obs, 0, out=obs_indptr[1:])
            
            num_nodes_per_obs = \
                segment_add_csr(num_nodes_per_dag, obs_indptr)

        return obs_indptr, \
               num_nodes_per_dag, \
               num_nodes_per_obs, \
               num_dags_per_obs
    


class CriticNetwork(nn.Module):
    def __init__(self, dim_embed):
        super().__init__()
        self.encoder = GraphEncoderNetwork(dim_embed)
        self.value_network = MLP(
            dim_embed, 
            [64, 32, 8, 1] #,
            # activation_layer=torch.nn.Tanh,
            # inplace=None
        )

    def forward(self, dag_batch, num_dags_per_obs=None):
        obs_indptr = self._bookkeep(dag_batch, num_dags_per_obs)
        *_, global_embeddings = self.encoder(dag_batch, obs_indptr)
        return self.value_network(global_embeddings)
    
    def _bookkeep(self, dag_batch, num_dags_per_obs):
        if num_dags_per_obs is None:
            num_dags_per_obs = dag_batch.num_graphs
            obs_indptr = None
        else:
            # data is batched
            batch_size = len(num_dags_per_obs)
            device = dag_batch.x.device
            obs_indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
            torch.cumsum(num_dags_per_obs, 0, out=obs_indptr[1:])
        return obs_indptr



class GCNConv(MessagePassing):

    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')
        self.mlp_prep = MLP(in_ch, [32, 16, out_ch])
        self.mlp_proc = MLP(out_ch, [32, 16, out_ch])
        self.mlp_agg = MLP(out_ch, [32, 16, out_ch])
        

    def forward(self, x, edge_index):
        x_prep = self.mlp_prep(x)

        x_proc = self.mlp_proc(x_prep)
        x_agg = self.propagate(edge_index, x=x_proc)

        x_out = x_prep + x_agg
        return x_out


    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    
    def update(self, aggr_out):
        return self.mlp_agg(aggr_out)


    

class GraphEncoderNetwork(nn.Module):

    def __init__(self, dim_embed):
        super().__init__()

        self.graph_conv = GCNConv(NUM_NODE_FEATURES, dim_embed)

        self.mlp_dag = MLP(
            NUM_DAG_FEATURES + NUM_NODE_FEATURES + dim_embed,
            [32, 16, dim_embed]
        )

        self.mlp_global = MLP(
            NUM_GLOBAL_FEATURES + dim_embed, 
            [32, 16, dim_embed]
        )



    def forward(self, dag_batch, obs_indptr):
        # node-level embeddings
        node_features = dag_batch.x[:, 3:]
        node_embeddings = self.graph_conv(node_features, dag_batch.adj)

        # dag-level embeddings
        node_combined = torch.cat([node_features, node_embeddings], dim=1)
        node_combined_agg = \
            global_add_pool(
                node_combined, 
                dag_batch.batch, 
                size=dag_batch.num_graphs
            )
        dag_features = dag_batch.x[dag_batch.ptr[:-1], 1:3]
        x = torch.cat([dag_features, node_combined_agg], dim=1)
        dag_embeddings = self.mlp_dag(x)

        # global-level embeddings
        if obs_indptr is None:
            # data is not batched -> only one global embedding
            dag_embeddings_agg = dag_embeddings.sum(dim=0).unsqueeze(0)
            global_features = dag_batch.x[0, 0].reshape([1, 1])
        else:
            # data is batched -> one global embedding per observation
            dag_embeddings_agg = segment_add_csr(dag_embeddings, obs_indptr)
            global_features = dag_batch.x[obs_indptr[:-1], 0].unsqueeze(-1)
        x = torch.cat([global_features, dag_embeddings_agg], dim=1)
        global_embeddings = self.mlp_global(x)

        return node_embeddings, dag_embeddings, global_embeddings
        
        
        

class PolicyNetwork(nn.Module):

    def __init__(self, num_workers, dim_embed):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_workers = num_workers

        self.mlp_node_score = MLP(
            3 * dim_embed,
            [32, 16, 1]
        )

        self.mlp_dag_score = MLP(
            (2 * dim_embed) + 1,
            [32, 16, 1]
        )
        


    def forward(
        self,   
        node_embeddings,
        dag_embeddings, 
        global_embeddings,
        num_nodes_per_dag,
        num_nodes_per_obs,
        num_dags_per_obs
    ):
        node_scores = self._compute_node_scores(
            node_embeddings, 
            dag_embeddings, 
            global_embeddings, 
            num_nodes_per_dag, 
            num_nodes_per_obs
        )

        dag_scores = self._compute_dag_scores(
            dag_embeddings, 
            global_embeddings, 
            num_dags_per_obs
        )

        return node_scores, dag_scores

    
    
    def _compute_node_scores(
        self, 
        node_embeddings, 
        dag_embeddings, 
        global_embeddings,      
        num_nodes_per_dag, 
        num_nodes_per_obs
    ):
        num_nodes = node_embeddings.shape[0]

        dag_embeddings_rpt = \
            dag_embeddings.repeat_interleave(
                num_nodes_per_dag, 
                output_size=num_nodes,
                dim=0
            )
        
        global_embeddings_rpt = \
            global_embeddings.repeat_interleave(
                num_nodes_per_obs, 
                output_size=num_nodes,
                dim=0
            )

        node_inputs = \
            torch.cat(
                [
                    node_embeddings, 
                    dag_embeddings_rpt, 
                    global_embeddings_rpt
                ], 
                dim=1
            )

        node_scores = self.mlp_node_score(node_inputs).squeeze(-1)

        return node_scores
    
    
    
    def _compute_dag_scores(
        self, 
        dag_embeddings, 
        global_embeddings,
        num_dags_per_obs
    ):
        num_dags = dag_embeddings.shape[0]

        worker_actions = torch.arange(self.num_workers, device=dag_embeddings.device)
        worker_actions_rpt = worker_actions.repeat(num_dags).unsqueeze(1)

        num_total_actions = worker_actions_rpt.shape[0]

        dag_embeddings_rpt = \
            dag_embeddings.repeat_interleave(
                self.num_workers,
                output_size=num_total_actions,
                dim=0
            )

        global_embeddings_rpt = \
            global_embeddings.repeat_interleave(
                num_dags_per_obs * self.num_workers, 
                output_size=num_total_actions,
                dim=0
            )
        
        dag_inputs = \
            torch.cat(
                [
                    dag_embeddings_rpt,
                    global_embeddings_rpt,
                    worker_actions_rpt
                ], 
                dim=1
            )

        dag_scores = self.mlp_dag_score(dag_inputs) \
                         .squeeze(-1) \
                         .view(num_dags, self.num_workers)

        return dag_scores

    