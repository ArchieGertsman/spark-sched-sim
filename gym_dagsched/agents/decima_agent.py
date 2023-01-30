import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as gnn
from torch_scatter import segment_add_csr
from torch_sparse import matmul

from gym_dagsched.utils.device import device
from .base_agent import BaseAgent




class DecimaAgent(BaseAgent):
    def __init__(self, 
                 num_workers,
                 mode='train',
                 state_dict_path=None,
                 device=None,
                 num_node_features=5, 
                 num_dag_features=3, 
                 dim_embed=8):
        super().__init__('Decima')

        self.actor_network = \
            ActorNetwork(num_node_features, 
                         num_dag_features, 
                         num_workers,
                         dim_embed)

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            self.actor_network.load_state_dict(state_dict)
        
        if device is not None:
            self.actor_network.to(device)

        if mode == 'train':
            self.actor_network.train()
        elif mode == 'eval':
            self.actor_network.eval()
        else:
            raise Exception('invalid mode')

        self.mode = mode

        self.cached_dag_batch = None



    def invoke(self, obs):
        '''assumes that `DecimaWrapper` is providing
        observations of the environment and receiving
        actions returned here.
        '''
        (did_update_dag_batch,
         dag_batch,
         valid_ops_mask,
         valid_prlsm_lim_mask,
         active_job_ids, 
         *_) = obs

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

        node_scores = node_scores.cpu()
        node_scores[~valid_ops_mask] = float('-inf')

        dag_scores = dag_scores.cpu()
        valid_prlsm_lim_mask = \
            torch.from_numpy(valid_prlsm_lim_mask)
        dag_scores.masked_fill_(~valid_prlsm_lim_mask, 
                                float('-inf'))

        env_action, raw_action = \
            self.sample_action(node_scores, 
                               dag_scores,
                               dag_batch.ptr.numpy(),
                               active_job_ids)

        if self.mode == 'train':
            return env_action, raw_action
        else:
            return env_action



    def sample_action(self,
                      node_scores, 
                      dag_scores, 
                      job_ptr,
                      active_job_ids):
        '''Returns a tuple `(env_action, raw_action)`, where 
        `env_action` is the action sent to `DecimaWrapper`, and 
        `raw_action` can be stored in experience by a training 
        algorithm.
        '''
        # select the next operation to schedule
        op_sample = \
            Categorical(logits=node_scores).sample()

        op_env, job_idx = \
            self.translate_op(op_sample.item(), 
                              job_ptr,
                              active_job_ids)

        # select the number of workers to schedule
        num_workers_sample = \
            Categorical(logits=dag_scores[job_idx]).sample()

        # action that's recorded in experience, 
        # later used during training
        raw_action = (op_sample, 
                      job_idx, 
                      num_workers_sample)

        # action that gets sent to the env
        num_workers_env = 1 + num_workers_sample.item()
        env_action = (op_env, num_workers_env)

        return env_action, raw_action



    @classmethod
    def translate_op(cls, op, job_ptr, active_jobs_ids):
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

    