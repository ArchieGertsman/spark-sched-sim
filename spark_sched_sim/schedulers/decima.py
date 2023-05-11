from typing import Tuple, Optional, Union, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.ops import MLP
from torch_scatter import segment_csr
from gymnasium.core import ObsType, ActType

from .scheduler import Scheduler
from spark_sched_sim import graph_utils



NUM_NODE_FEATURES = 5

NUM_DAG_FEATURES = 3

LEAKY_RELU = lambda inplace, neg_slope=.2: nn.LeakyReLU(neg_slope, inplace)



class DecimaScheduler(Scheduler):

    def __init__(
        self,
        num_executors,
        dim_embed,
        optim_class,
        optim_lr,
        act_fn=LEAKY_RELU,
        state_dict_path=None,
        max_grad_norm=None
    ):
        name = 'Decima'
        if state_dict_path:
            name += f':{state_dict_path}'
        super().__init__(name)

        self.actor = ActorNetwork(num_executors, dim_embed, act_fn)
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            self.actor.load_state_dict(state_dict)
        self.num_executors = num_executors
        self.optim_class = optim_class
        self.optim_lr = optim_lr
        self.max_grad_norm = max_grad_norm



    @property
    def device(self):
        return next(self.actor.parameters()).device



    def train(self):
        '''call only on an instance that is about to be trained'''
        self.actor.train()
        self.optim = self.optim_class(self.actor.parameters(), lr=self.optim_lr)



    @torch.no_grad()
    def schedule(self, obs: ObsType) -> ActType:
        '''assumes that `DecimaObsWrapper` is providing observations of the environment 
        and `DecimaActWrapper` is receiving actions returned from here.
        '''
        obs_tensor_dict = graph_utils.obs_to_torch(obs)

        dag_batch = obs_tensor_dict['dag_batch']
        dag_batch.to(self.device, non_blocking=True)

        # 1. compute embeddings
        embedding_dict = self.actor.encoder(dag_batch)

        # 2. select a stage
        stage_probs = self.actor.stage_policy_network(
            dag_batch, embedding_dict, obs_tensor_dict['stage_mask']
        )
        stage_idx, stage_lgprob = self._sample(stage_probs)
        job_idx = obs_tensor_dict['stage_to_job_map'][stage_idx]

        # 3. select the number of executors to add to that stage, conditioned on that stage's job
        exec_probs = self.actor.exec_policy_network(
            dag_batch, embedding_dict, job_idx, obs_tensor_dict['exec_mask']
        )
        num_exec, exec_lgprob = self._sample(exec_probs)

        action = {
            'stage_idx': stage_idx.item(),
            'job_idx': job_idx.item(),
            'num_exec': num_exec.item()
        }

        lgprob = stage_lgprob + exec_lgprob

        return action, lgprob.item()



    def evaluate_actions(self, obsns_dict: dict, actions: Tensor) -> tuple[Tensor, Tensor]:
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_indices, exec_selections = [col.clone() for col in actions.T]

        dag_batch = obsns_dict['dag_batch']
        num_nodes_per_obs = dag_batch['num_nodes_per_obs']
        obs_ptr = dag_batch['obs_ptr']
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model, this time with grads enabled
        dag_batch.to(self.device)
        embedding_dict = self.actor.encoder(dag_batch)
        stage_probs = self.actor.stage_policy_network(
            dag_batch, embedding_dict, obsns_dict['stage_mask']
        )
        exec_probs = self.actor.exec_policy_network(
            dag_batch, embedding_dict, job_indices, obsns_dict['exec_mask']
        )

        stage_lgprobs, stage_entropies = self._evaluate_node_actions(
            stage_probs, stage_selections, num_nodes_per_obs
        )
        prlsm_lim_lgprobs, prlsm_lim_entropies = self._evaluate(exec_probs, exec_selections)

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + prlsm_lim_lgprobs

        action_entropies = stage_entropies + prlsm_lim_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return action_lgprobs, action_entropies



    def update_parameters(self, loss=None):
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
                return

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()



    ## internal methods

    @classmethod
    def _sample(cls, probs):
        c = Categorical(probs=probs)
        samp = c.sample()
        lgprob = c.log_prob(samp)
        return samp, lgprob
    


    @classmethod
    def _evaluate(cls, probs, selections):
        c = Categorical(probs=probs)
        return c.log_prob(selections), c.entropy()



    @classmethod
    def _evaluate_node_actions(cls, node_probs, node_selections, num_nodes_per_obs):
        node_probs_split, node_selections_split, node_counts = \
            cls._split_node_experience(node_probs, node_selections, num_nodes_per_obs)

        # evaluate actions for each chunk, vectorized
        node_lgprobs, node_entropies = zip(*[
            cls._evaluate(node_probs.view(-1, node_count), node_selections)
            for node_probs, node_selections, node_count in zip(
                node_probs_split, node_selections_split, node_counts
            )
        ])

        return torch.cat(node_lgprobs), torch.cat(node_entropies)



    @classmethod
    def _split_node_experience(cls, node_probs, node_selections, num_nodes_per_obs):
        '''splits the batch of node distirbutions and selections into chunks, where 
        each sample within a chunk has the same node count. Allows for vectorized
        processing at the chunk level, instead of processing each sample individually.
        '''
        batch_size = len(num_nodes_per_obs)

        # find indices where node count changes
        node_count_change_mask = num_nodes_per_obs[:-1] != num_nodes_per_obs[1:]

        ptr = 1 + node_count_change_mask.nonzero().squeeze()
        if ptr.shape == torch.Size():
            # ptr is zero-dimentional; not allowed in torch.cat
            ptr = ptr.unsqueeze(0)
        ptr = torch.cat([torch.tensor([0]), ptr, torch.tensor([batch_size])])

        # unique node count of each chunk
        node_counts = num_nodes_per_obs[ptr[:-1]]

        # number of samples per chunk
        chunk_sizes = ptr[1:] - ptr[:-1]

        # chunk the node distributions
        node_probs_split = torch.split(node_probs, list(chunk_sizes * node_counts))

        # chunk the node selections
        node_selections_split = torch.split(node_selections, list(chunk_sizes))

        return node_probs_split, node_selections_split, node_counts




class ActorNetwork(nn.Module):
    def __init__(self, num_executors: int, dim_embed: int, act_fn):
        super().__init__()
        self.encoder = EncoderNetwork(dim_embed, act_fn)
        self.stage_policy_network = StagePolicyNetwork(dim_embed, act_fn)
        self.exec_policy_network = ExecPolicyNetwork(dim_embed, act_fn, num_executors)

        # initialize all biases to be zero
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()



class EncoderNetwork(nn.Module):
    def __init__(self, dim_embed, act_fn):
        super().__init__()
        self.node_encoder = NodeEncoder(dim_embed, act_fn)
        self.dag_encoder = DagEncoder(dim_embed, act_fn)
        self.global_encoder = GlobalEncoder(dim_embed, act_fn)


    def forward(self, dag_batch):
        node_embeddings = self.node_encoder(dag_batch)
        dag_summaries = self.dag_encoder(dag_batch, node_embeddings)
        global_summaries = self.global_encoder(dag_batch, dag_summaries)
        return {
            'node_embeddings': node_embeddings,
            'dag_summaries': dag_summaries,
            'global_summaries': global_summaries
        }



class NodeEncoder(nn.Module):
    def __init__(self, dim_embed, act_fn):
        super().__init__()
        self.mlp_node_prep = MLP(NUM_NODE_FEATURES, [16, 8, dim_embed], activation_layer=act_fn)
        self.mlp_node_msg = MLP(dim_embed, [16, 8, dim_embed], activation_layer=act_fn)
        self.mlp_node_update = MLP(dim_embed, [16, 8, dim_embed], activation_layer=act_fn)


    def forward(self, dag_batch):
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

            # the biases from `mlp_node_update` will give non-zero values
            # to nodes that are not part of this message passing step, so
            # those need to be masked out. Messages should only be passed 
            # to the nodes found in `edge_index[0]`.
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
    


class DagEncoder(nn.Module):
    def __init__(self, dim_embed, act_fn):
        super().__init__()
        self.mlp_dag_msg = MLP(
            NUM_NODE_FEATURES + dim_embed, 
            [16, 8, dim_embed], 
            activation_layer=act_fn
        )


    def forward(self, dag_batch, node_embeddings):
        # add skip connection to original input
        x = torch.cat([dag_batch.x, node_embeddings], dim=1)

        # message
        x = self.mlp_dag_msg(x)

        # aggregate
        x = segment_csr(x, dag_batch.ptr)

        return x
    


class GlobalEncoder(nn.Module):
    def __init__(self, dim_embed, act_fn):
        super().__init__()
        self.mlp_glob_msg = MLP(
            dim_embed, 
            [16, 8, dim_embed], 
            activation_layer=act_fn
        )


    def forward(self, dag_batch, dag_summaries):
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
    


def masked_softmax(logits, mask):
    min_real = torch.finfo(logits.dtype).min
    return torch.softmax(logits.masked_fill(~mask, min_real), -1)
        
        
        
class StagePolicyNetwork(nn.Module):
    def __init__(self, dim_embed, act_fn):
        super().__init__()
        self.dim_embed = dim_embed
        self.mlp_node_score = MLP(
            NUM_NODE_FEATURES + (3 * dim_embed), 
            [32, 16, 8, 1], 
            activation_layer=act_fn
        )

    
    def forward(self, dag_batch, embedding_dict, stage_mask):
        num_nodes = dag_batch.x.shape[0]

        dag_summaries_repeat = \
            embedding_dict['dag_summaries'].repeat_interleave(
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
            embedding_dict['global_summaries'].repeat_interleave(
                num_nodes_per_obs,
                output_size=num_nodes,
                dim=0
            )

        # add skip connection to original node features
        node_inputs = torch.cat(
            [
                dag_batch.x, 
                embedding_dict['node_embeddings'], 
                dag_summaries_repeat, 
                global_summaries_repeat
            ], 
            dim=1
        )

        node_scores = self.mlp_node_score(node_inputs).squeeze(-1).cpu()
        node_probs = masked_softmax(node_scores, stage_mask)
        return node_probs

    


class ExecPolicyNetwork(nn.Module):
    def __init__(self, dim_embed, act_fn, num_executors):
        super().__init__()
        self.dim_embed = dim_embed
        self.mlp_dag_score = MLP(
            NUM_DAG_FEATURES + (2 * dim_embed) + 1, 
            [32, 16, 8, 1], 
            activation_layer=act_fn
        )
        self.num_executors = num_executors
    
    
    def forward(self, dag_batch, embedding_dict, job_indices, exec_mask):
        dag_idxs = dag_batch.ptr[:-1]
        dag_features = dag_batch.x[dag_idxs, 0:3]
        batch_size = job_indices.numel()

        executor_actions = torch.arange(self.num_executors) / self.num_executors
        executor_actions = executor_actions \
            .repeat(batch_size).unsqueeze(1).to(dag_features.device)

        dag_features = dag_features[job_indices]
        dag_summaries = embedding_dict['dag_summaries'][job_indices]
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
            embedding_dict['global_summaries'].repeat_interleave(
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

        dag_scores = self.mlp_dag_score(dag_inputs) \
            .squeeze(-1).view(batch_size, self.num_executors).cpu()
        dag_probs = masked_softmax(dag_scores, exec_mask)
        return dag_probs