
import numpy as np
from gymnasium import (
    ObservationWrapper,
    ActionWrapper
)
from gymnasium.spaces import (
    Discrete,
    MultiBinary,
    Box,
    Dict,
    Sequence,
    Graph,
    GraphInstance
)


class DecimaObsWrapper(ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Dict({
            'dag_batch': Dict({
                'data': Graph(node_space=Box(0, np.inf, (5,)),
                              edge_space=Discrete(1)),
                'ptr': Sequence(Discrete(1))
            }),
            'schedulable_op_mask': Sequence(Discrete(2)),
            'valid_prlsm_lim_mask': Sequence(MultiBinary(env.num_workers))
        })


    
    def observation(self, obs):
        ptr = np.array(obs['dag_batch']['ptr'])
        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        num_active_nodes = obs['dag_batch']['data'].nodes.shape[0]
        worker_counts = obs['worker_counts']
        num_active_jobs = len(worker_counts)

        # build node features
        nodes = np.zeros((num_active_nodes, 5), dtype=np.float32)
        nodes[:, :2] = obs['dag_batch']['data'].nodes
        nodes[:, 2] = obs['num_workers_to_schedule']
        nodes[:, 3] = np.repeat(worker_counts, num_nodes_per_dag)
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i]:ptr[i+1], 4] = 1

        # update op action space to reflect the current number of active ops
        self.observation_space['dag_batch']['ptr'].feature_space.n = num_active_nodes+1
        # self.action_space['op_idx'].n = num_active_nodes

        obs = {
            'dag_batch': {
                'data': GraphInstance(nodes=nodes, 
                                      edges=obs['dag_batch']['data'].edges, 
                                      edge_links=obs['dag_batch']['data'].edge_links),
                'ptr': obs['dag_batch']['ptr']
            },
            'schedulable_op_mask': obs['schedulable_op_mask'],
            'valid_prlsm_lim_mask': obs['valid_prlsm_lim_mask']
        }

        return obs



class DecimaActWrapper(ActionWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.action_space = Dict({
            'op_idx': Discrete(1),
            'job_idx': Discrete(env.num_total_jobs),
            'prlsm_lim': Discrete(env.num_workers)
        })



    def action(self, act):
        return {
            'op_idx': act['op_idx'],
            'prlsm_lim': 1 + act['prlsm_lim']
        }