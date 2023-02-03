from typing import NamedTuple

import torch
from torch_geometric.data import Batch
import numpy as np

from ..utils.pyg import add_adj



class Observation(NamedTuple):
    dag_batch: object
    valid_ops_mask: object
    valid_prlsm_lim_mask: object
    num_jobs: int
    num_source_workers: int
    wall_time: float
    action: object
    reward: float



def collect_rollout(env, agent, seed, options):  
    obs, info = env.reset(seed=seed, options=options)

    done = False

    # save rollout from each step
    # of the episode, to later be used
    # in leaning
    rollout_buffer = []
    
    while not done:
        # unpack the current observation
        obs = preprocess_obs(obs, agent.num_workers)

        (dag_batch,
         schedulable_ops_mask,
         valid_prlsm_lim_mask) = obs

        raw_action = agent(obs)

        op_idx, _, prlsm_lim = raw_action
        env_action = {
            'op_idx': op_idx.item(),
            'prlsm_lim': 1 + prlsm_lim.item()
        }

        obs, reward, terminated, truncated, info = \
            env.step(env_action)

        rollout_buffer += \
            [Observation(dag_batch.clone(),
                         schedulable_ops_mask,
                         valid_prlsm_lim_mask,
                         dag_batch.num_graphs,
                         obs['num_workers_to_schedule'],
                         info['wall_time'],
                         raw_action,
                         reward)]

        done = (terminated or truncated)

    return rollout_buffer



def preprocess_obs(obs, num_workers):
    num_active_jobs = len(obs['worker_counts'])
    num_active_nodes = obs['graph'].nodes.shape[0]
    num_nodes_per_dag = np.bincount(obs['batch'])
    ptr = np.zeros(num_active_jobs + 1, dtype=int)
    np.cumsum(num_nodes_per_dag, out=ptr[1:])

    x = np.zeros((num_active_nodes, 5), dtype=np.float32)
    x[:, :2] = obs['graph'].nodes
    x[:, 2] = obs['num_workers_to_schedule']
    x[:, 3] = np.repeat(obs['worker_counts'], num_nodes_per_dag)
    if obs['source_job_idx'] < num_active_jobs:
        i = obs['source_job_idx']
        x[ptr[i]:ptr[i+1], 4] = 1

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(obs['graph'].edge_links.T)
    dag_batch = Batch(x=x, edge_index=edge_index)
    dag_batch.batch = torch.tensor(obs['batch'])
    dag_batch._num_graphs = num_active_jobs
    dag_batch.ptr = torch.from_numpy(ptr)
    add_adj(dag_batch)

    schedulable_ops_mask = np.array(obs['schedulable_op_mask'], dtype=bool)

    valid_prlsm_lim_mask = np.zeros((num_active_jobs, num_workers), dtype=bool)

    for i, worker_count in enumerate(obs['worker_counts']):
        min_prlsm_lim = worker_count + 1
        if i == obs['source_job_idx']:
            min_prlsm_lim -= obs['num_workers_to_schedule']

        assert 0 < min_prlsm_lim
        assert     min_prlsm_lim <= num_workers + 1

        valid_prlsm_lim_mask[i, (min_prlsm_lim-1):] = 1

    return dag_batch, \
           schedulable_ops_mask, \
           valid_prlsm_lim_mask



def construct_nested_dag_batch(dag_batch_list, 
                               num_dags_per_obs):
    '''extracts the inputs to the model from each
    step of the episode, then stacks them into a
    large batch.
    '''
    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    num_nodes_per_dag = nested_dag_batch.batch.bincount()
    ptr = num_nodes_per_dag.cumsum(dim=0)
    nested_dag_batch.ptr = \
        torch.cat([torch.tensor([0]), ptr], dim=0)
    nested_dag_batch._num_graphs = sum(num_dags_per_obs)
    add_adj(nested_dag_batch)

    return nested_dag_batch, num_nodes_per_dag