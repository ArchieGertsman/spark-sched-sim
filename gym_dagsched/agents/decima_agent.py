from typing import Tuple, Optional, Union, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data, Batch
from torch_scatter import segment_add_csr
from torch_sparse import matmul
import numpy as np
from gymnasium.core import ObsType, ActType

from .base_agent import BaseAgent
from ..utils.graph import obs_to_pyg, ObsBatch



NUM_NODE_FEATURES = 5

NUM_DAG_FEATURES = 2



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

        self.actor_network = \
            ActorNetwork(num_workers, dim_embed)

        self.num_workers = num_workers

        self.optim_class = optim_class

        self.optim_lr = optim_lr

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.actor_network.load_state_dict(state_dict)

        self.actor_network.train(training_mode)

        self.training_mode = training_mode

        self.max_grad_norm = max_grad_norm



    @property
    def device(self) -> torch.device:
        return next(self.actor_network.parameters()).device



    def build(self, device: torch.device = None) -> None:
        if device is not None:
            self.actor_network.to(device)

        params = self.actor_network.parameters()
        self.optim = self.optim_class(params, lr=self.optim_lr)



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
        outputs = self.actor_network(dag_batch)
        node_scores, dag_scores = [out.cpu() for out in outputs]

        schedulable_op_mask = \
            torch.tensor(obs['schedulable_op_mask'], dtype=bool)

        valid_prlsm_lim_mask = \
            torch.from_numpy(np.vstack(obs['valid_prlsm_lim_mask']))

        self._mask_outputs(
            (node_scores, dag_scores),
            (schedulable_op_mask, valid_prlsm_lim_mask)
        )

        action = self._sample_action(node_scores, dag_scores, batch)

        return action



    def evaluate_actions(
        self, 
        obsns: ObsBatch,
        actions: Tensor
    ) -> tuple[Tensor, Tensor]:

        model_outputs = \
            self.actor_network(
                obsns.nested_dag_batch.to(self.device),
                obsns.num_dags_per_obs.to(self.device)
            )

        # move model outputs to CPU
        (node_scores_batch, 
         dag_scores_batch, 
         num_nodes_per_obs, 
         obs_indptr) = \
            [out.cpu() for out in model_outputs]

        self._mask_outputs(
            (node_scores_batch, dag_scores_batch),
            (obsns.schedulable_op_masks, obsns.valid_prlsm_lim_masks)
        )

        # split columns of `actions` into separate tensors
        node_selections, dag_idxs, dag_selections = \
            [act.long() for act in actions.T]

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
        action_entropies = (node_entropies + dag_entropies) * \
                           self._get_entropy_scale(num_nodes_per_obs)

        return action_lgprobs, action_entropies



    def update_parameters(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        
        # compute gradients
        loss.backward()

        # clip grads
        params = self.actor_network.parameters()
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

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
        schedulable_op_mask, valid_prlsm_lim_mask = masks

        # mask node scores
        node_scores[~schedulable_op_mask] = float('-inf')

        # mask dag scores
        dag_scores.masked_fill_(~valid_prlsm_lim_mask, float('-inf'))



    @classmethod
    def _sample_action(cls, node_scores, dag_scores, batch):
        # select the next operation to schedule
        op_idx = Categorical(logits=node_scores).sample()

        # select the parallelism limit for the selected op's job
        job_idx = batch[op_idx]
        dag_scores = dag_scores[job_idx]
        prlsm_lim = Categorical(logits=dag_scores).sample()

        return {
            'op_idx': op_idx.item(),
            'job_idx': job_idx.item(),
            'prlsm_lim': prlsm_lim.item()
        }



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

        self.encoder = \
            GraphEncoderNetwork(dim_embed)

        self.policy_network = \
            PolicyNetwork(num_workers, dim_embed)
        

        
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
                dag_batch,
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




def make_mlp(in_ch, out_ch, h1=16, h2=8):
    return nn.Sequential(
        nn.Linear(in_ch, h1),
        nn.ReLU(inplace=True),
        nn.Linear(h1, h2),
        nn.ReLU(inplace=True),
        nn.Linear(h2, out_ch)
    )
        



class GCNConv(MessagePassing):

    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')
        self.mlp_prep = make_mlp(in_ch, out_ch)
        self.mlp_proc = make_mlp(out_ch, out_ch)
        self.mlp_agg = make_mlp(out_ch, out_ch)
        


    def forward(self, x, edge_index):
        # lift input into a higher dimension
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

        self.mlp_node = \
            make_mlp(NUM_NODE_FEATURES + dim_embed, dim_embed)

        self.mlp_dag = make_mlp(dim_embed, dim_embed)



    def forward(self, dag_batch, obs_indptr):
        node_embeddings = \
            self._compute_node_embeddings(dag_batch)

        dag_embeddings = \
            self._compute_dag_embeddings(dag_batch, 
                                         node_embeddings)

        global_embeddings = \
            self._compute_global_embeddings(dag_embeddings, 
                                            obs_indptr)

        return node_embeddings, \
               dag_embeddings, \
               global_embeddings

    

    def _compute_node_embeddings(self, dag_batch):
        '''one embedding per node, per dag'''
        assert hasattr(dag_batch, 'adj')
        # achieve flow from target to source by *not* taking 
        # transpose of `adj`
        return self.graph_conv(dag_batch.x, dag_batch.adj)
    


    def _compute_dag_embeddings(self, 
                                dag_batch, 
                                node_embeddings):
        '''one embedding per dag'''

        # merge original node features with new node embeddings
        nodes_merged = \
            torch.cat([dag_batch.x, node_embeddings], dim=1)

        # pass combined node features through mlp
        nodes_merged = self.mlp_node(nodes_merged)

        # for each dag, add together its nodes
        # to obtain its dag embedding
        dag_embeddings = \
            global_add_pool(nodes_merged, 
                            dag_batch.batch,
                            size=dag_batch.num_graphs)

        return dag_embeddings



    def _compute_global_embeddings(self, 
                                   dag_embeddings, 
                                   obs_indptr):
        '''one embedding per observation'''

        # pass dag embeddings through mlp
        dag_embeddings = self.mlp_dag(dag_embeddings)

        # for each observation, add together its dags
        # to obtain its global embedding
        if obs_indptr is None:
            z = dag_embeddings.sum(dim=0).unsqueeze(0)
        else:
            z = segment_add_csr(dag_embeddings, obs_indptr)

        return z
        
        
        

class PolicyNetwork(nn.Module):

    def __init__(self, num_workers, dim_embed):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_workers = num_workers

        self.dim_node_merged = NUM_NODE_FEATURES + (3 * dim_embed)
        self.mlp_node_score = make_mlp(self.dim_node_merged, 1)

        self.dim_dag_merged = NUM_DAG_FEATURES + (2 * dim_embed) + 1
        self.mlp_dag_score = make_mlp(self.dim_dag_merged, 1)
        


    def forward(
        self,   
        dag_batch, 
        node_embeddings,
        dag_embeddings, 
        global_embeddings,
        num_nodes_per_dag,
        num_nodes_per_obs,
        num_dags_per_obs
    ):
        node_features = dag_batch.x

        node_scores = self._compute_node_scores(
            node_features, 
            node_embeddings, 
            dag_embeddings, 
            global_embeddings, 
            num_nodes_per_dag, 
            num_nodes_per_obs
        )

        dag_idxs = dag_batch.ptr[:-1]
        dag_features = \
            node_features[dag_idxs, :NUM_DAG_FEATURES]

        dag_scores = self._compute_dag_scores(
            dag_features, 
            dag_embeddings, 
            global_embeddings, 
            num_dags_per_obs,
            dag_batch.num_graphs
        )

        return node_scores, dag_scores

    
    
    def _compute_node_scores(
        self, 
        node_features, 
        node_embeddings, 
        dag_embeddings, 
        global_embeddings,      
        num_nodes_per_dag, 
        num_nodes_per_obs
    ):
        num_nodes = node_features.shape[0]

        dag_embeddings_repeat = \
            dag_embeddings.repeat_interleave(
                num_nodes_per_dag, 
                output_size=num_nodes,
                dim=0
            )
        
        global_embeddings_repeat = \
            global_embeddings.repeat_interleave(
                num_nodes_per_obs, 
                output_size=num_nodes,
                dim=0
            )

        node_inputs = \
            torch.cat([node_features, 
                       node_embeddings, 
                       dag_embeddings_repeat, 
                       global_embeddings_repeat], dim=1)

        node_scores = self.mlp_node_score(node_inputs) \
                          .squeeze(-1)

        return node_scores
    
    
    
    def _compute_dag_scores(
        self, 
        dag_features, 
        dag_embeddings, 
        global_embeddings,
        num_dags_per_obs,
        num_total_dags
    ):
        device = dag_features.device
        worker_actions = \
            torch.arange(self.num_workers, device=device)

        worker_actions = worker_actions.repeat(num_total_dags) \
                                       .unsqueeze(1)

        dag_features_merged = \
            torch.cat([dag_features, dag_embeddings], dim=1)

        num_total_actions = worker_actions.shape[0]

        dag_features_merged_repeat = \
            dag_features_merged.repeat_interleave(
                self.num_workers,
                output_size=num_total_actions,
                dim=0
            )

        global_embeddings_repeat = \
            global_embeddings.repeat_interleave(
                num_dags_per_obs * self.num_workers, 
                output_size=num_total_actions,
                dim=0
            )
        
        dag_inputs = torch.cat([dag_features_merged_repeat,
                                global_embeddings_repeat,
                                worker_actions], dim=1)

        dag_scores = self.mlp_dag_score(dag_inputs) \
                         .squeeze(-1) \
                         .view(num_total_dags, self.num_workers)

        return dag_scores

    