from time import time

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as gnn
from torch_scatter import segment_add_csr
from torch_sparse import SparseTensor
from torch_sparse import matmul

from gym_dagsched.utils.device import device



def make_mlp(in_ch, out_ch, h1=16, h2=8):
    return nn.Sequential(
        nn.Linear(in_ch, h1),   nn.ReLU(inplace=True),
        nn.Linear(h1, h2),      nn.ReLU(inplace=True),
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

        x_agg = self.propagate(edge_index, x=x_prep)

        x_out = x_prep + x_agg

        return x_out


    def message_and_aggregate(self, adj_t, x):
        x = self.mlp_proc(x)
        return matmul(adj_t, x, reduce=self.aggr)

    
    def update(self, aggr_out):
        return self.mlp_agg(aggr_out)


    

class GraphEncoderNetwork(nn.Module):
    def __init__(self, num_node_features, dim_embed):
        super().__init__()

        self.graph_conv = GCNConv(num_node_features, dim_embed)

        self.mlp_node = make_mlp(num_node_features + dim_embed, dim_embed)
        self.mlp_dag = make_mlp(dim_embed, dim_embed)



    def forward(self, dag_batch):
        node_embeddings = \
            self._compute_node_embeddings(dag_batch)

        dag_embeddings = \
            self._compute_dag_embeddings(dag_batch, node_embeddings)

        global_embeddings = \
            self._compute_global_embeddings(dag_embeddings)

        return node_embeddings, dag_embeddings, global_embeddings

    

    def _compute_node_embeddings(self, dag_batch):
        '''one embedding per node, per job, per env'''

        # achieve flow from leaves to roots
        # by *not* taking transpose of `adj`
        node_embeddings = self.graph_conv(
            dag_batch.x, dag_batch.adj)

        return node_embeddings
    


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
            gnn.global_add_pool(
                nodes_merged, 
                dag_batch.batch,
                size=dag_batch.num_graphs)

        return dag_embeddings



    def _compute_global_embeddings(self, dag_embeddings):
        '''one embedding per env'''

        # pass dag embeddings through mlp
        dag_embeddings = self.mlp_dag(dag_embeddings)

        # add together all the dag embeddings
        # to get the global embedding
        global_embedding = dag_embeddings.sum(dim=0)
        return global_embedding
        
        
        

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
        n_workers
    ):
        node_features = dag_batch.x

        op_scores = self._compute_op_scores(
            node_features, 
            node_embeddings, 
            dag_embeddings, 
            global_embeddings, 
            num_ops_per_job)

        dag_idxs = dag_batch.ptr[:-1]
        dag_features = \
            node_features[dag_idxs, :self.num_dag_features]

        num_jobs = dag_batch.num_graphs

        prlvl_scores= self._compute_prlvl_scores(
            dag_features, 
            dag_embeddings, 
            global_embeddings, 
            n_workers,
            num_jobs)

        return op_scores, prlvl_scores

    
    
    def _compute_op_scores(self, 
        node_features, 
        node_embeddings, 
        dag_embeddings, 
        global_embedding,      
        num_ops_per_job
    ):
        total_num_ops = node_features.shape[0]

        dag_embeddings_repeat = \
            dag_embeddings.repeat_interleave( 
                num_ops_per_job, 
                dim=0,
                output_size=node_features.shape[0])
        
        global_embedding_repeat = \
            global_embedding.unsqueeze(0) \
            .repeat_interleave( 
                total_num_ops, 
                dim=0,
                output_size=node_features.shape[0])

        node_inputs = torch.cat(
            [
                node_features, 
                node_embeddings, 
                dag_embeddings_repeat, 
                global_embedding_repeat
            ], 
            dim=1)

        node_scores = \
            self.mlp_node_score(node_inputs).squeeze(-1)

        return node_scores
    
    
    
    def _compute_prlvl_scores(self, 
        dag_features, 
        dag_embeddings, 
        global_embedding, 
        n_workers,
        num_jobs
    ):
        worker_actions = torch.arange(n_workers, device=device)
        worker_actions = \
            worker_actions.repeat(num_jobs).unsqueeze(1)

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

        global_embedding_repeat = \
            global_embedding.unsqueeze(0) \
            .repeat_interleave( 
                num_jobs * n_workers, 
                dim=0,
                output_size=worker_actions.shape[0])
        
        dag_inputs = torch.cat(
            [
                dag_features_merged_repeat,
                global_embedding_repeat,
                worker_actions
            ], 
            dim=1)

        dag_inputs = dag_inputs.view(
            num_jobs, 
            n_workers, 
            dag_inputs.shape[1])

        dag_scores = self.mlp_dag_score(dag_inputs).squeeze(-1)

        return dag_scores

    


class ActorNetwork(nn.Module):
    def __init__(self, num_node_features, num_dag_features, dim_embed=8):
        super().__init__()

        self.encoder = GraphEncoderNetwork(num_node_features, dim_embed)

        self.policy_network = PolicyNetwork(
            num_node_features, 
            num_dag_features, 
            dim_embed
        )

        # used for caching the dag structure.
        # while the node features get 
        # updated on every forward pass, the 
        # active nodes in the system don't
        # always change.
        self.dag_batch = None
        

        
    def forward(self, 
        node_features, 
        new_dag_batch,
        n_workers
    ):
        self._update_dag_batch(node_features, new_dag_batch)

        node_embeddings, dag_embeddings, global_embeddings = \
            self.encoder(self.dag_batch)

        num_ops_per_job = self.dag_batch.ptr[1:] - self.dag_batch.ptr[:-1]

        op_scores, prlvl_scores = self.policy_network(
            self.dag_batch,
            node_embeddings, 
            dag_embeddings, 
            global_embeddings,
            num_ops_per_job,
            n_workers
        )
        
        return op_scores, prlvl_scores
    


    def _update_dag_batch(self, node_features, new_dag_batch):
        '''if the number of ops in the system has
        changed since last time, then the environment 
        should have sent a new dag batch, which will 
        replace the locally cached version. Otherwise, 
        the environment should have sent `None`, and 
        the locally cached dag batch will be reused 
        in this forward pass. 
        
        Node features are updated regardless, since 
        they are fresh each time.
        '''
        if new_dag_batch is not None:
            self.dag_batch = new_dag_batch

            num_nodes = node_features.shape[0]

            self.dag_batch.adj = SparseTensor(
                row=new_dag_batch.edge_index[0], 
                col=new_dag_batch.edge_index[1],
                sparse_sizes=(num_nodes, num_nodes),
                trust_data=True,
                is_sorted=True)
        else:
            assert self.dag_batch is not None

        self.dag_batch.x = node_features