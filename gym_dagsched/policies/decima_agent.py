from time import time

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.nn as gnn
from torch_scatter import segment_add_csr

from gym_dagsched.utils.device import device



# def make_mlp(in_ch, out_ch, h1=32, h2=16, h3=8):
def make_mlp(in_ch, out_ch, h1=16, h2=8):
    return nn.Sequential(
        nn.Linear(in_ch, h1),   nn.ReLU(inplace=True),
        nn.Linear(h1, h2),      nn.ReLU(inplace=True),
        nn.Linear(h2, out_ch)
    )
        



class GCNConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp1 = make_mlp(in_ch, 8)
        self.mlp2 = make_mlp(8, out_ch)
        

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.mlp1(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    
    def update(self, aggr_out):
        x = self.mlp2(aggr_out)
        return x


    

class GraphEncoderNetwork(nn.Module):
    def __init__(self, num_node_features, dim_embed):
        super().__init__()
        self.graph_conv = GCNConv(num_node_features, dim_embed)
        self.mlp_node = make_mlp(num_node_features + dim_embed, dim_embed)
        self.mlp_dag = make_mlp(dim_embed, dim_embed)



    def forward(self, dag_batch, env_indptr):
        node_embeddings = \
            self._compute_node_embeddings(dag_batch)

        dag_embeddings = \
            self._compute_dag_embeddings(dag_batch, node_embeddings)

        global_embeddings = \
            self._compute_global_embeddings(dag_embeddings, env_indptr)

        return node_embeddings, dag_embeddings, global_embeddings

    

    def _compute_node_embeddings(self, dag_batch):
        '''one embedding per node, per job, per env'''
        return self.graph_conv(dag_batch.x, dag_batch.edge_index)
    


    def _compute_dag_embeddings(self, dag_batch, node_embeddings):
        '''one embedding per job, per env'''

        # merge original node features with new node embeddings
        nodes_merged = \
            torch.cat([dag_batch.x, node_embeddings], dim=1)

        # pass combined node features through mlp
        nodes_merged = self.mlp_node(nodes_merged)

        # for each dag, add together its nodes
        # to obtain its dag embedding
        dag_embeddings = \
            gnn.global_add_pool(nodes_merged, dag_batch.batch)

        return dag_embeddings



    def _compute_global_embeddings(self, dag_embeddings, env_indptr):
        '''one embedding per env'''

        # pass dag embeddings through mlp
        dag_embeddings = self.mlp_dag(dag_embeddings)

        # for each env, add together its dags
        # to obtain its global embedding
        z = segment_add_csr(dag_embeddings, env_indptr)
        return z
        
        
        

class PolicyNetwork(nn.Module):
    def __init__(self, num_node_features, num_dag_features, dim_embed):
        super().__init__()
        self.num_dag_features = num_dag_features
        self.dim_embed = dim_embed

        self.dim_node_merged = num_node_features + (3 * dim_embed)
        self.mlp_node_score = make_mlp(self.dim_node_merged, 1)

        self.dim_dag_merged = num_dag_features + (2 * dim_embed) + 1
        self.mlp_dag_score = make_mlp(self.dim_dag_merged, 1)
        


    def forward(self,   
        dag_batch, 
        node_embeddings,
        dag_embeddings, 
        global_embeddings,
        num_ops_per_job,
        num_ops_per_env,
        num_jobs_per_env,
        n_workers
    ):
        node_features = dag_batch.x

        op_scores = self._compute_op_scores(
            node_features, 
            node_embeddings, 
            dag_embeddings, 
            global_embeddings, 
            num_ops_per_job, 
            num_ops_per_env)

        dag_idxs = dag_batch.ptr[:-1]
        dag_features = \
            node_features[dag_idxs, :self.num_dag_features]

        prlvl_scores= self._compute_prlvl_scores(
            dag_features, 
            dag_embeddings, 
            global_embeddings, 
            n_workers, 
            num_jobs_per_env)

        return op_scores, prlvl_scores

    
    
    def _compute_op_scores(self, 
        node_features, 
        node_embeddings, 
        dag_embeddings, 
        global_embeddings,      
        num_ops_per_job, 
        num_ops_per_env
    ):
        dag_embeddings_repeat = \
            dag_embeddings.repeat_interleave( 
                num_ops_per_job, 
                dim=0,
                output_size=node_features.shape[0])
        
        global_embeddings_repeat = \
            global_embeddings.repeat_interleave( 
                num_ops_per_env, 
                dim=0,
                output_size=node_features.shape[0])

        node_inputs = torch.cat(
            [
                node_features, 
                node_embeddings, 
                dag_embeddings_repeat, 
                global_embeddings_repeat
            ], 
            dim=1)

        node_scores = self.mlp_node_score(node_inputs).squeeze(-1)

        return node_scores
    
    
    
    def _compute_prlvl_scores(self, 
        dag_features, 
        dag_embeddings, 
        global_embeddings, 
        n_workers, 
        num_jobs_per_env
    ):
        num_total_jobs = num_jobs_per_env \
            if not torch.is_tensor(num_jobs_per_env) \
            else num_jobs_per_env.sum()

        worker_actions = torch.arange(n_workers, device=device)
        worker_actions = worker_actions.repeat(num_total_jobs).unsqueeze(1)

        dag_features_merged = torch.cat(
            [
                dag_features, 
                dag_embeddings
            ], 
            dim=1)

        dag_features_merged_repeat = \
            dag_features_merged.repeat_interleave(
                n_workers, 
                dim=0,
                output_size=worker_actions.shape[0])

        global_embeddings_repeat = \
            global_embeddings.repeat_interleave( 
                num_jobs_per_env * n_workers, 
                dim=0,
                output_size=worker_actions.shape[0])
        
        dag_inputs = torch.cat(
            [
                dag_features_merged_repeat,
                global_embeddings_repeat,
                worker_actions
            ], 
            dim=1)

        dag_inputs = dag_inputs.view(
            num_total_jobs, 
            n_workers, 
            dag_inputs.shape[1])

        dag_scores = self.mlp_dag_score(dag_inputs).squeeze(-1)

        return dag_scores

    


class ActorNetwork(nn.Module):
    def __init__(self, num_node_features, num_dag_features, dim_embed=4):
        super().__init__()

        self.encoder = GraphEncoderNetwork(num_node_features, dim_embed)

        self.policy_network = PolicyNetwork(
            num_node_features, 
            num_dag_features, 
            dim_embed
        )
        

        
    def forward(self, dag_batch, num_jobs_per_env, n_workers):
        env_indptr, num_ops_per_job, num_ops_per_env = \
            self._bookkeep(num_jobs_per_env, dag_batch)

        node_embeddings, dag_embeddings, global_embeddings = \
            self.encoder(dag_batch, env_indptr)

        op_scores, prlvl_scores= self.policy_network(
            dag_batch,
            node_embeddings, 
            dag_embeddings, 
            global_embeddings,
            num_ops_per_job,
            num_ops_per_env,
            num_jobs_per_env,
            n_workers
        )
        
        return op_scores, prlvl_scores, num_ops_per_env, env_indptr



    def _bookkeep(self, num_jobs_per_env, dag_batch):
        num_ops_per_job = dag_batch.ptr[1:] - dag_batch.ptr[:-1]

        if not torch.is_tensor(num_jobs_per_env):
            env_indptr = torch.tensor([0, num_jobs_per_env], device=device)
            num_ops_per_env = num_ops_per_job.sum()
        else:
            num_envs = len(num_jobs_per_env)
            env_indptr = torch.zeros(num_envs+1, device=device, dtype=torch.long)
            torch.cumsum(num_jobs_per_env, 0, out=env_indptr[1:])
            
            num_ops_per_env = segment_add_csr(num_ops_per_job, env_indptr)

        return env_indptr, num_ops_per_job, num_ops_per_env
    