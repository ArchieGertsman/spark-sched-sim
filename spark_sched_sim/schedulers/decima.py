import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse

from .neural import *
from ..wrappers import DAGNNObsWrapper
from spark_sched_sim import graph_utils


LEAKY_RELU = lambda: nn.LeakyReLU(negative_slope=.2, inplace=True)


class DecimaScheduler(NeuralScheduler):
    '''Original Decima architecture, which uses asynchronous message passing
    as in DAGNN.
    Paper: https://dl.acm.org/doi/abs/10.1145/3341302.3342080
    '''

    def __init__(
        self,
        num_executors,
        num_node_features=5,
        num_dag_features=3,
        dim_embed=16,
        state_dict_path=None,
        optim_class=None,
        optim_lr=None,
        max_grad_norm=None
    ):
        name = 'Decima'
        if state_dict_path:
            name += f':{state_dict_path}'

        actor = ActorNetwork(
            num_executors, num_node_features, num_dag_features, dim_embed)
        
        obs_wrapper_cls = DAGNNObsWrapper

        super().__init__(
            name,
            actor,
            obs_wrapper_cls,
            num_executors,
            state_dict_path,
            optim_class,
            optim_lr,
            max_grad_norm
        )



class ActorNetwork(nn.Module):
    def __init__(
        self, 
        num_executors, 
        num_node_features, 
        num_dag_features, 
        dim_emb
    ):
        super().__init__()
        self.encoder = EncoderNetwork(num_node_features, dim_emb)

        emb_dims = {
            'node': dim_emb,
            'dag': dim_emb,
            'glob': dim_emb
        }
        # hid_dims = [32, 16, 8, 1]
        hid_dims = [64, 64, 1]
        act_fn = LEAKY_RELU

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, hid_dims, act_fn)

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, hid_dims, act_fn)
        
        self._reset_biases()
        

    def _reset_biases(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()



class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, dim_embed):
        super().__init__()
        self.node_encoder = NodeEncoder(num_node_features, dim_embed)
        self.dag_encoder = DagEncoder(num_node_features, dim_embed)
        self.global_encoder = GlobalEncoder(dim_embed)

    def forward(self, dag_batch):
        '''
            Returns:
                a dict of representations at three different levels:
                node, dag, and global.
        '''
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        try:
            # batch of obsns
            obs_ptr = dag_batch['obs_ptr']
            h_glob = self.global_encoder(h_dag, obs_ptr)
        except:
            # single obs
            h_glob = self.global_encoder(h_dag)

        h_dict = {
            'node': h_node, # shape: (num_nodes, dim_embed * num_layers)
            'dag': h_dag,   # shape: (num_graphs, dim_embed)
            'glob': h_glob  # shape: (num_obsns, dim_embed)
        }

        return h_dict



class NodeEncoder(nn.Module):
    def __init__(self, num_node_features, dim_embed, reverse_flow=True):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        # hid_dims = [16, 8, dim_embed]
        hid_dims = [32, 16, dim_embed]
        act_fn = LEAKY_RELU

        self.mlp_prep = make_mlp(num_node_features, hid_dims, act_fn)
        self.mlp_msg = make_mlp(dim_embed, hid_dims, act_fn)
        self.mlp_update = make_mlp(dim_embed, hid_dims, act_fn)


    def forward(self, dag_batch):
        edge_masks = dag_batch['edge_masks']
        
        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)
        
        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes)
        
        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        edge_masks_it = iter(reversed(edge_masks)) \
            if self.reverse_flow else iter(edge_masks)

        # target-to-source message passing, one level of the dag (batch) at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = graph_utils.make_adj(edge_index_masked, num_nodes)

            # nodes sending messages
            src_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.j], num_nodes)

            # nodes receiving messages
            dst_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])

        return h
    

    def _forward_no_mp(self, x):
        '''forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        '''
        return self.mlp_prep(x)
    


class DagEncoder(nn.Module):
    def __init__(self, num_node_features, dim_embed):
        super().__init__()
        self.mlp = make_mlp(
            num_node_features + dim_embed,
            # hid_dims=[16, 8, dim_embed], 
            hid_dims=[32, 16, dim_embed], 
            act_fn=LEAKY_RELU)

    def forward(self, h_node, dag_batch):
        # include original input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.mlp(h_node), dag_batch.ptr)
        return h_dag
    


class GlobalEncoder(nn.Module):
    def __init__(self, dim_embed):
        super().__init__()
        self.mlp = make_mlp(
            dim_embed,
            # hid_dims=[16, 8, dim_embed], 
            hid_dims=[32, 16, dim_embed], 
            act_fn=LEAKY_RELU)

    def forward(self, h_dag, obs_ptr=None):
        h_dag = self.mlp(h_dag)

        if obs_ptr is not None:
            # batch of observations
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob