import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch_scatter import segment_add_csr
from torch_sparse import matmul
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .base_agent import BaseAgent
from ..utils.device import device
from ..utils.device import device as default_device
from ..utils.graph import make_adj




class DecimaAgent(BaseAgent):
    def __init__(self,
                 num_workers,
                 training_mode=True,
                 state_dict_path=None,
                 device=None,
                 num_node_features=5, 
                 num_dag_features=3, 
                 dim_embed=8,
                 optim_class=torch.optim.Adam,
                 optim_lr=.001):
        super().__init__('Decima')

        self.actor_network = \
            ActorNetwork(num_node_features, 
                         num_dag_features, 
                         num_workers,
                         dim_embed)

        self.num_workers = num_workers

        self.optim_class = optim_class

        self.optim_lr = optim_lr

        if device is None:
            device = default_device

        self.actor_network.to(device)

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, 
                                    map_location=device)
            self.actor_network.load_state_dict(state_dict)

        self.actor_network.train(training_mode)

        self.training_mode = training_mode

        self.cached_dag_batch = None



    def build(self, ddp=False, device=None):
        if ddp:
            assert device is not None
            self.actor_network.to(device)
            self.actor_network = DDP(self.actor_network, 
                                     device_ids=[device])

        params = self.actor_network.parameters()
        self.optim = self.optim_class(params, self.optim_lr)



    def predict(self, obs):
        '''assumes that `DecimaWrapper` is providing
        observations of the environment and receiving
        actions returned here.
        '''
        dag_batch = self._to_pyg(obs['dag_batch'])
        schedulable_ops_mask = torch.tensor(obs['schedulable_op_mask'], dtype=bool)
        valid_prlsm_lim_mask = torch.from_numpy(np.vstack(obs['valid_prlsm_lim_mask']))

        did_update_dag_batch = True

        if did_update_dag_batch:
            # the whole dag batch object was
            # updated because the number of
            # nodes changed, so send the new
            # object to the GPU
            dag_batch_clone = dag_batch.clone()
            dag_batch_clone.edge_index = None
            self.cached_dag_batch = dag_batch_clone \
                .to(device, non_blocking=True)
        else:
            # only send new node features to
            # the GPU; reuse cached dag batch
            # for everything else
            self.cached_dag_batch.x = dag_batch.x \
                .to(device, non_blocking=True)

        with torch.no_grad():
            # no computational graphs needed during 
            # the episode, only model outputs.
            node_scores, dag_scores = \
                self.actor_network(self.cached_dag_batch)

        node_scores, dag_scores = \
            self._mask_outputs(node_scores.cpu(), 
                               dag_scores.cpu(),
                               schedulable_ops_mask,
                               valid_prlsm_lim_mask)

        action = self._sample_action(node_scores, 
                                     dag_scores,
                                     dag_batch.batch)

        return action



    def evaluate_actions(self, obsns, actions):
        (nested_dag_batch,
         num_dags_per_obs,
         num_nodes_per_dag,
         valid_ops_masks,
         valid_prlsm_lim_masks) = obsns

        nested_dag_batch.to(default_device)

        # re-feed all the inputs from the entire
        # episode back through the model, this time
        # recording a computational graph
        model_outputs = \
            self.actor_network(nested_dag_batch,
                               num_dags_per_obs)

        # move model outputs to CPU
        (node_scores_batch, 
         dag_scores_batch, 
         num_nodes_per_obs, 
         obs_indptr) = \
            [t.cpu() for t in model_outputs]

        self._mask_outputs(node_scores_batch, 
                           dag_scores_batch,
                           valid_ops_masks,
                           valid_prlsm_lim_masks)

        (node_selections, 
         dag_idxs, 
         dag_selections) = \
            [torch.tensor(lst) for lst in zip(*(act.values() for act in actions))]

        (all_node_probs, 
         node_lgprobs, 
         node_entropies) = \
            self._evaluate_node_actions(node_scores_batch,
                                        node_selections,
                                        num_nodes_per_obs)

        dag_probs = self._compute_dag_probs(all_node_probs, 
                                            num_nodes_per_dag)
        
        dag_lgprobs, dag_entropies = \
            self._evaluate_dag_actions(dag_scores_batch, 
                                       dag_idxs + obs_indptr[:-1], 
                                       dag_selections,
                                       dag_probs,
                                       obs_indptr)

        action_lgprobs = node_lgprobs + dag_lgprobs

        action_entropies = \
            (node_entropies + dag_entropies) * \
            self._get_entropy_scale(num_nodes_per_obs)

        return action_lgprobs, action_entropies



    def update_parameters(self, loss, num_envs=None):
        # compute gradients
        self.optim.zero_grad()
        loss.backward()

        if num_envs is not None:
            # we want the sum of grads over all the
            # workers, but DDP gives average, so
            # scale the grads back
            for param in self.actor_network.parameters():
                param.grad.mul_(num_envs)

        # update model parameters
        self.optim.step()



    ## internal methods

    def _to_pyg(self, raw_dag_batch):
        ptr = np.array(raw_dag_batch['ptr'])
        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        num_active_nodes = raw_dag_batch['data'].nodes.shape[0]
        num_active_jobs = len(num_nodes_per_dag)

        # construct PyG `Batch` object
        x = raw_dag_batch['data'].nodes
        edge_index = torch.from_numpy(raw_dag_batch['data'].edge_links.T)
        batch = np.repeat(np.arange(num_active_jobs), num_nodes_per_dag)
        adj = make_adj(edge_index, num_active_nodes)
        dag_batch = Batch(x=torch.from_numpy(x), 
                          edge_index=edge_index, 
                          batch=torch.from_numpy(batch), 
                          ptr=torch.from_numpy(ptr),
                          adj=adj)
        dag_batch._num_graphs = num_active_jobs

        return dag_batch



    @classmethod
    def _mask_outputs(cls,
                      node_scores,
                      dag_scores,
                      valid_ops_mask,
                      valid_prlsm_lim_mask):
        # mask node scores
        node_scores[~valid_ops_mask] = float('-inf')

        # mask dag scores
        dag_scores.masked_fill_(~valid_prlsm_lim_mask, 
                                float('-inf'))

        return node_scores, dag_scores



    @classmethod
    def _sample_action(cls,
                       node_scores, 
                       dag_scores, 
                       batch):
        '''Returns a tuple `(env_action, raw_action)`, where 
        `env_action` is the action sent to `DecimaWrapper`, and 
        `raw_action` can be stored in experience by a training 
        algorithm.
        '''
        # select the next operation to schedule
        op_idx = Categorical(logits=node_scores).sample()

        # select the number of workers to schedule
        job_idx = batch[op_idx]
        prlsm_lim = Categorical(logits=dag_scores[job_idx]).sample()

        return {
            'op_idx': op_idx.item(),
            'job_idx': job_idx.item(),
            'prlsm_lim': prlsm_lim.item()
        }



    @classmethod
    def _translate_op(cls, op, job_ptr, active_jobs_ids):
        '''Returns:
        - `op`: translation of the policy sample so that 
        the environment can find the corresponding operation
        - `job_idx`: index of the job that the selected op 
        belongs to
        '''
        job_idx = (op >= job_ptr).sum() - 1

        job_id = active_jobs_ids[job_idx]
        active_op_idx = op - job_ptr[job_idx]
        
        op = (job_id, active_op_idx)

        return op, job_idx



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
    def _evaluate_node_actions(cls,
                               node_scores_batch,
                               node_selection_batch,
                               num_nodes_per_obs):
        '''splits the node score/selection batches into subbatches 
        (see subroutine below), then for each subbatch, comptues 
        attributes (action log-probability and entropy) using 
        vectorized computations. Finally, merges the attributes 
        from the subbatches together. This is faster than 
        either 
        - separately computing attributes for each sample in the
        batch, because vectorized computations are not utilized
        at all, or
        - stacking the whole batch together with padding and doing 
        one large vectorized computation, because the backward 
        pass becomes very expensive
        '''

        def _eval_node_actions(node_scores, node_selection):
            c = Categorical(logits=node_scores)
            node_lgprob = c.log_prob(node_selection)
            node_entropy = c.entropy()
            return c.probs, node_lgprob, node_entropy

        (node_scores_subbatches, 
        node_selection_subbatches, 
        subbatch_node_counts) = \
            cls._split_node_experience(node_scores_batch, 
                                       node_selection_batch, 
                                       num_nodes_per_obs)

        # for each subbatch, compute the node
        # action attributes, vectorized
        all_node_probs = []
        node_lgprobs = []
        node_entropies = []

        for (node_scores_subbatch, 
            node_selection_subbatch,
            node_count) in zip(node_scores_subbatches,
                            node_selection_subbatches,
                            subbatch_node_counts):
            node_scores_subbatch = \
                node_scores_subbatch.view(-1, node_count)

            node_probs, node_lgprob_subbatch, node_entropy_subbatch = \
                _eval_node_actions(node_scores_subbatch,
                                   node_selection_subbatch)

            all_node_probs += [torch.flatten(node_probs)]
            node_lgprobs += [node_lgprob_subbatch]
            node_entropies += [node_entropy_subbatch]

        ## concatenate the subbatch attributes together

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
    def _split_node_experience(cls,
                               node_scores_batch, 
                               node_selection_batch, 
                               num_nodes_per_obs):
        '''splits the node score/selection batches into
        subbatches, where each each sample within a subbatch
        has the same node count.
        '''
        batch_size = len(num_nodes_per_obs)

        # find indices where op count changes
        op_count_change_mask = \
            num_nodes_per_obs[:-1] != num_nodes_per_obs[1:]
        ptr = 1 + op_count_change_mask.nonzero().squeeze()
        if ptr.shape == torch.Size():
            # ptr is zero-dimentional; not allowed in torch.cat
            ptr = ptr.unsqueeze(0)
        ptr = torch.cat([torch.tensor([0]), 
                        ptr, 
                        torch.tensor([batch_size])])

        # unique op count within each subbatch
        subbatch_node_counts = num_nodes_per_obs[ptr[:-1]]

        # number of samples in each subbatch
        subbatch_sizes = ptr[1:] - ptr[:-1]

        # split node scores into subbatches
        node_scores_split = \
            torch.split(node_scores_batch, 
                        list(subbatch_node_counts * subbatch_sizes))

        # split node selections into subbatches
        node_selection_split = \
            torch.split(node_selection_batch, 
                        list(subbatch_sizes))

        return node_scores_split, \
               node_selection_split, \
               subbatch_node_counts



    @classmethod
    def _evaluate_dag_actions(cls,
                              dag_scores_batch, 
                              dag_idx_batch, 
                              dag_selection_batch,
                              dag_probs,
                              obs_indptr):
        dag_lgprob_batch = \
            Categorical(logits=dag_scores_batch[dag_idx_batch]) \
                .log_prob(dag_selection_batch)

        # can't have rows where all the entries are
        # -inf when computing entropy, so for all such 
        # rows, set the first entry to be 0. then the 
        # entropy for these rows becomes 0.
        inf_counts = torch.isinf(dag_scores_batch).sum(1)
        allinf_rows = (inf_counts == dag_scores_batch.shape[1])
        # dag_scores_batch[allinf_rows, 0] = 0
        dag_scores_batch[allinf_rows] = 0

        # compute expected entropy over dags for each obs.
        # each dag is weighted by the probability of it 
        # being selected. sum is segmented over observations.
        entropy_per_dag = \
            Categorical(logits=dag_scores_batch).entropy()
        dag_entropy_batch = \
            segment_add_csr(dag_probs * entropy_per_dag, 
                            obs_indptr)
        
        return dag_lgprob_batch, dag_entropy_batch




class ActorNetwork(nn.Module):
    def __init__(self, 
                 num_node_features, 
                 num_dag_features, 
                 num_workers,
                 dim_embed):
        super().__init__()

        self.encoder = \
            GraphEncoderNetwork(num_node_features, 
                                dim_embed)

        self.policy_network = \
            PolicyNetwork(num_node_features, 
                          num_dag_features,
                          num_workers,
                          dim_embed)
        

        
    def forward(self, 
                dag_batch, 
                num_dags_per_obs=None):
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
            self.policy_network(dag_batch,
                                node_embeddings, 
                                dag_embeddings, 
                                global_embeddings,
                                num_nodes_per_dag,
                                num_nodes_per_obs,
                                num_dags_per_obs)

        ret = (node_scores, dag_scores)
        if is_data_batched:
            ret += (num_nodes_per_obs, obs_indptr)
        return ret



    def _bookkeep(self, dag_batch, num_dags_per_obs):
        num_nodes_per_dag = \
            dag_batch.ptr[1:] - dag_batch.ptr[:-1]

        if num_dags_per_obs is None:
            num_dags_per_obs = dag_batch.num_graphs
            num_nodes_per_obs = dag_batch.x.shape[0]
            obs_indptr = None
        else:
            batch_size = len(num_dags_per_obs)
            obs_indptr = torch.zeros(batch_size+1, 
                                     device=device, 
                                     dtype=torch.long)
            torch.cumsum(num_dags_per_obs, 0, out=obs_indptr[1:])
            
            num_nodes_per_obs = \
                segment_add_csr(num_nodes_per_dag, 
                                obs_indptr)

        return obs_indptr, \
               num_nodes_per_dag, \
               num_nodes_per_obs, \
               num_dags_per_obs




def make_mlp(in_ch, out_ch, h1=16, h2=8):
    return nn.Sequential(nn.Linear(in_ch, h1),
                         nn.ReLU(inplace=True),
                         nn.Linear(h1, h2),
                         nn.ReLU(inplace=True),
                         nn.Linear(h2, out_ch))
        



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
    def __init__(self, num_node_features, dim_embed):
        super().__init__()
        self.graph_conv = GCNConv(num_node_features, 
                                  dim_embed)

        self.mlp_node = make_mlp(num_node_features + dim_embed, 
                                 dim_embed)

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
        # achieve flow from leaves to roots by *not* taking 
        # transpose of `adj`
        return self.graph_conv(dag_batch.x, dag_batch.adj)
    


    def _compute_dag_embeddings(self, 
                                dag_batch, 
                                node_embeddings):
        '''one embedding per dag'''

        # merge original node features with new node embeddings
        nodes_merged = \
            torch.cat([dag_batch.x, node_embeddings], 
                      dim=1)

        # pass combined node features through mlp
        nodes_merged = self.mlp_node(nodes_merged)

        # for each dag, add together its nodes
        # to obtain its dag embedding
        dag_embeddings = \
            gnn.global_add_pool(nodes_merged, 
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
    def __init__(self, 
                 num_node_features, 
                 num_dag_features, 
                 num_workers,
                 dim_embed):
        super().__init__()
        self.num_dag_features = num_dag_features
        self.dim_embed = dim_embed
        self.num_workers = num_workers

        self.dim_node_merged = num_node_features + (3 * dim_embed)
        self.mlp_node_score = make_mlp(self.dim_node_merged, 1)

        self.dim_dag_merged = num_dag_features + (2 * dim_embed) + 1
        self.mlp_dag_score = make_mlp(self.dim_dag_merged, 1)
        


    def forward(self,   
                dag_batch, 
                node_embeddings,
                dag_embeddings, 
                global_embeddings,
                num_nodes_per_dag,
                num_nodes_per_obs,
                num_dags_per_obs):
        node_features = dag_batch.x

        node_scores = \
            self._compute_node_scores(node_features, 
                                      node_embeddings, 
                                      dag_embeddings, 
                                      global_embeddings, 
                                      num_nodes_per_dag, 
                                      num_nodes_per_obs)

        dag_idxs = dag_batch.ptr[:-1]
        dag_features = \
            node_features[dag_idxs, :self.num_dag_features]

        dag_scores = \
            self._compute_dag_scores(dag_features, 
                                     dag_embeddings, 
                                     global_embeddings, 
                                     num_dags_per_obs,
                                     dag_batch.num_graphs)

        return node_scores, dag_scores

    
    
    def _compute_node_scores(self, 
                             node_features, 
                             node_embeddings, 
                             dag_embeddings, 
                             global_embeddings,      
                             num_nodes_per_dag, 
                             num_nodes_per_obs):
        num_nodes = node_features.shape[0]

        dag_embeddings_repeat = \
            dag_embeddings \
                .repeat_interleave(num_nodes_per_dag, 
                                   dim=0,
                                   output_size=num_nodes)
        
        global_embeddings_repeat = \
            global_embeddings \
                .repeat_interleave(num_nodes_per_obs, 
                                   dim=0,
                                   output_size=num_nodes)

        node_inputs = \
            torch.cat([node_features, 
                       node_embeddings, 
                       dag_embeddings_repeat, 
                       global_embeddings_repeat], 
                      dim=1)

        node_scores = \
            self.mlp_node_score(node_inputs).squeeze(-1)

        return node_scores
    
    
    
    def _compute_dag_scores(self, 
                            dag_features, 
                            dag_embeddings, 
                            global_embeddings,
                            num_dags_per_obs,
                            num_total_dags):
        worker_actions = torch.arange(self.num_workers, 
                                      device=device)
        worker_actions = \
            worker_actions.repeat(num_total_dags) \
                          .unsqueeze(1)

        dag_features_merged = \
            torch.cat([dag_features, dag_embeddings], 
                      dim=1)

        num_total_actions = worker_actions.shape[0]

        dag_features_merged_repeat = \
            dag_features_merged \
                .repeat_interleave(self.num_workers, 
                                   dim=0,
                                   output_size=num_total_actions)

        global_embeddings_repeat = \
            global_embeddings \
                .repeat_interleave(num_dags_per_obs * self.num_workers, 
                                   dim=0,
                                   output_size=num_total_actions)
        
        dag_inputs = \
            torch.cat([dag_features_merged_repeat,
                       global_embeddings_repeat,
                       worker_actions], 
                      dim=1)

        dag_scores = \
            self.mlp_dag_score(dag_inputs).squeeze(-1)

        dag_scores = dag_scores.view(num_total_dags, 
                                     self.num_workers)

        return dag_scores

    