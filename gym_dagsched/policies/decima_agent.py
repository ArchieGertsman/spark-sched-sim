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
        self.conv1 = GCNConv(num_node_features, dim_embed)
        self.mlp_node = make_mlp(num_node_features + dim_embed, dim_embed)
        self.mlp_dag = make_mlp(dim_embed, dim_embed)


    def forward(self, dag_batch, env_indptr):
        node_embeddings = self._compute_node_level_embeddings(dag_batch)
        dag_embeddings = self._compute_dag_level_embeddings(dag_batch, node_embeddings)
        global_embeddings = self._compute_global_embeddings(dag_embeddings, env_indptr)
        return node_embeddings, dag_embeddings, global_embeddings

    
    def _compute_node_level_embeddings(self, dag_batch):
        return self.conv1(dag_batch.x, dag_batch.edge_index)
    

    def _compute_dag_level_embeddings(self, dag_batch, node_embeddings):
        # merge original node features with new node embeddings
        nodes_merged = torch.cat([dag_batch.x, node_embeddings], dim=1)

        # pass combined node features through mlp
        nodes_merged = self.mlp_node(nodes_merged)

        # for each dag, add together its nodes
        # to obtain the dag embeding
        y = gnn.global_add_pool(nodes_merged, dag_batch.batch)
        return y
    

    def _compute_global_embeddings(self, dag_embeddings, env_indptr):
        # pass dag embeddings through mlp
        dag_embeddings = self.mlp_dag(dag_embeddings)

        # for each env, add together its dags
        # to obtain the global embedding
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

        # self._init_prlvl_scores()
        


    def forward(self,   
        node_features, 
        node_embeddings,
        dag_embeddings, 
        global_embeddings,
        num_ops_per_job,
        num_ops_per_env,
        num_jobs_per_env,
        n_workers,
        batch_ptr
    ):
        op_scores = self._compute_op_scores(
            node_features, 
            node_embeddings, 
            dag_embeddings, 
            global_embeddings, 
            num_ops_per_job, 
            num_ops_per_env)

        dag_features = node_features[batch_ptr[:-1], :self.num_dag_features]

        prlvl_scores= self._compute_prlvl_scores(
            dag_features, 
            dag_embeddings, 
            global_embeddings, 
            n_workers, 
            num_jobs_per_env)

        return op_scores, prlvl_scores


    # def _init_prlvl_scores(self):
    #     num_envs = 8
    #     max_env_jobs = 11
    #     self.n_workers = 50

    #     max_jobs = num_envs * max_env_jobs

    #     # self.worker_actions = torch.empty(
    #     #     self.n_workers * max_jobs, 
    #     #     device=device)

    #     self.dag_features = torch.empty(
    #         (max_jobs, self.num_dag_features + self.dim_embed),
    #         device=device)

    #     self.dag_features_repeat = torch.empty(
    #         (self.n_workers * max_jobs, self.dag_features.shape[1]),
    #         device=device)

    #     self.global_embeddings_repeat = torch.empty(
    #         (self.n_workers * max_jobs, self.dim_embed),
    #         device=device)

    #     self.dags_merged = torch.empty(
    #         (self.n_workers * max_jobs, self.dim_dag_merged),
    #         device=device)

    
    
    def _compute_op_scores(self, 
        node_features, 
        node_embeddings, 
        dag_embeddings, 
        global_embeddings,      
        num_ops_per_job, 
        num_ops_per_env
    ):
        dag_embeddings_repeat = \
            torch.repeat_interleave(
                dag_embeddings, 
                num_ops_per_job, 
                dim=0)
        
        global_embeddings_repeat = \
            torch.repeat_interleave(
                global_embeddings, 
                num_ops_per_env, 
                dim=0)

        nodes_merged = torch.cat([
            node_features, 
            node_embeddings, 
            dag_embeddings_repeat, 
            global_embeddings_repeat], dim=1)

        node_scores = self.mlp_node_score(nodes_merged).squeeze(-1)
        return node_scores


    # def _compute_prlvl_scores(self, 
    #     dag_features, 
    #     dag_embeddings, 
    #     global_embeddings, 
    #     n_workers,
    #     num_jobs_per_env
    # ):
    #     num_total_jobs = num_jobs_per_env.sum().item()

    #     worker_actions = torch.arange(self.n_workers, device=device)
    #     worker_actions = worker_actions.repeat(num_total_jobs).unsqueeze(1)

    #     torch.cat(
    #         [dag_features, dag_embeddings], 
    #         dim=1,
    #         out=self.dag_features)

    #     torch.repeat_interleave(
    #         self.dag_features[:num_total_jobs], 
    #         self.n_workers, 
    #         dim=0,
    #         out=self.dag_features_repeat)

    #     torch.repeat_interleave(
    #         global_embeddings, 
    #         num_jobs_per_env * self.n_workers, 
    #         dim=0,
    #         out=self.global_embeddings_repeat)

    #     num_inputs = num_total_jobs * self.n_workers
        
    #     torch.cat([
    #         self.dag_features_repeat[:num_inputs],
    #         self.global_embeddings_repeat[:num_inputs],
    #         worker_actions], 
    #         dim=1,
    #         out=self.dags_merged)

    #     dags_merged = self.dags_merged[:num_inputs]

    #     dags_merged = dags_merged.reshape(
    #         num_total_jobs, 
    #         self.n_workers, 
    #         self.dim_dag_merged)

    #     dag_scores = self.mlp_dag_score(dags_merged).squeeze(-1)

    #     return dag_scores
    
    
    def _compute_prlvl_scores(self, 
        dag_features, 
        dag_embeddings, 
        global_embeddings, 
        n_workers, 
        num_jobs_per_env
    ):
        num_total_jobs = num_jobs_per_env.sum().item()

        worker_actions = torch.arange(n_workers, device=device)
        worker_actions = worker_actions.repeat(num_total_jobs).unsqueeze(1)

        dag_features = torch.cat(
            [dag_features, dag_embeddings], 
            dim=1)

        dag_features_repeat = \
            torch.repeat_interleave(
                dag_features, 
                n_workers, 
                dim=0)

        global_embeddings_repeat = \
            torch.repeat_interleave(
                global_embeddings, 
                num_jobs_per_env * n_workers, 
                dim=0)
        
        dags_merged = torch.cat([
            dag_features_repeat,
            global_embeddings_repeat,
            worker_actions], dim=1)

        dags_merged = dags_merged.reshape(
            num_total_jobs, 
            n_workers, 
            dags_merged.shape[1])

        dag_scores = self.mlp_dag_score(dags_merged).squeeze(-1)

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
            dag_batch.x,
            node_embeddings, 
            dag_embeddings, 
            global_embeddings,
            num_ops_per_job,
            num_ops_per_env,
            num_jobs_per_env,
            n_workers,
            dag_batch.ptr
        )
        
        return op_scores, prlvl_scores, num_ops_per_env, env_indptr


    def _bookkeep(self, num_jobs_per_env, dag_batch):
        num_envs = len(num_jobs_per_env)
        env_indptr = torch.zeros(num_envs+1, device=device, dtype=torch.long)
        torch.cumsum(num_jobs_per_env, 0, out=env_indptr[1:])
        
        num_ops_per_job = dag_batch.ptr[1:] - dag_batch.ptr[:-1]
        num_ops_per_env = segment_add_csr(num_ops_per_job, env_indptr)

        return env_indptr, num_ops_per_job, num_ops_per_env
    