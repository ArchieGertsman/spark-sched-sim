import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.nn as gnn

from gym_dagsched.utils.device import device


def make_mlp(in_ch, out_ch, h1=32, h2=16):
    return nn.Sequential(
            nn.Linear(in_ch, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_ch)
        )
        


class GCNConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp1 = make_mlp(in_ch, 8)
        self.mlp2 = make_mlp(8, out_ch)
        

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.mlp1(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    
    def update(self, aggr_out):
        return self.mlp2(aggr_out)
    
    

class GraphEncoderNetwork(nn.Module):
    def __init__(self, in_ch, dim_embed):
        super().__init__()
        self.conv1 = GCNConv(in_ch, dim_embed)
        self.mlp_dag = make_mlp(dim_embed, dim_embed)
        self.mlp_global = make_mlp(dim_embed, dim_embed)
        self.total_time = 0.


    def forward(self, dag_batch):
        x = self._compute_node_level_embeddings(dag_batch)
        y = self._compute_dag_level_embeddings(x, dag_batch.batch)
        z = self._compute_global_embedding(y)
        return x, y, z

    
    def _compute_node_level_embeddings(self, dag_batch):
        return self.conv1(dag_batch.x, dag_batch.edge_index)
    

    def _compute_dag_level_embeddings(self, x, batch_tensor):
        y = gnn.global_add_pool(x, batch_tensor)
        return self.mlp_dag(y)
    

    def _compute_global_embedding(self, y):
        z = torch.sum(y, dim=0)
        return self.mlp_global(z)
        
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, dim_embed, num_workers):
        super().__init__()
        self.mlp_op_score = make_mlp(3*dim_embed, 1)
        self.mlp_prlvl_score = make_mlp(2*dim_embed+1, 1)
        self.num_workers = num_workers
        self.total_time = [0,0]
        
        
    def forward(
        self, 
        num_ops_per_dag, 
        num_dags, 
        x, y, z, 
        op_msk, 
        prlvl_msk
    ):
        # 6.130%
        ops = self._compute_ops(num_ops_per_dag, x, y, z, op_msk)

        # 6.953%
        prlvl = self._compute_prlvl(num_dags, y, z, prlvl_msk)

        return ops, prlvl
    
    
    def _compute_ops(self, num_ops_per_dag, x, y, z, op_msk):
        y_ops = torch.repeat_interleave(y, num_ops_per_dag, dim=0)
        
        num_total_ops = num_ops_per_dag.sum(dim=0)
        z_ops = z.repeat(num_total_ops, 1)
        
        ops = torch.cat([x,y_ops,z_ops], dim=1)
        ops = self.mlp_op_score(ops).squeeze(-1)
        ops -= (1-op_msk)*1000
        ops = torch.softmax(ops, dim=0)

        return ops
    
    
    def _compute_prlvl(self, num_dags, y, z, prlvl_msk):
        limits = torch.arange(1, self.num_workers+1, device=device)
        limits = limits.repeat(num_dags).unsqueeze(1)
        y_prlvl = torch.repeat_interleave(y, self.num_workers, dim=0)
        z_prlvl = z.repeat(num_dags * self.num_workers, 1)
        
        prlvl = torch.cat([limits, y_prlvl, z_prlvl], dim=1)
        prlvl = prlvl.reshape(num_dags, self.num_workers, prlvl.shape[1])
        prlvl = self.mlp_prlvl_score(prlvl).squeeze(-1)
        prlvl -= (1-prlvl_msk)*1000
        prlvl = torch.softmax(prlvl, dim=1)
        return prlvl

    
    
class ActorNetwork(nn.Module):
    def __init__(self, in_ch, dim_embed, num_workers):
        super().__init__()
        self.encoder = GraphEncoderNetwork(in_ch, dim_embed)
        self.policy_network = PolicyNetwork(dim_embed, num_workers)
        
        
    def forward(self, dag_batch, op_msk, prlvl_msk):
        x, y, z = self.encoder(dag_batch)

        # t0 = time.time()
        num_dags = dag_batch.num_graphs
        ops, prlvl = self.policy_network(
            dag_batch.num_ops_per_dag, 
            num_dags, 
            x, y, z, 
            op_msk, 
            prlvl_msk)

        # t1 = time.time()
        # print(t1-t0)
        
        return ops, prlvl
