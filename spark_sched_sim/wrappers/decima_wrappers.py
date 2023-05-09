
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

from ..graph_utils import construct_message_passing_masks


class DecimaObsWrapper(ObservationWrapper):
    '''transforms observations in preparation for the Decima model. Is also responsible
    for constructing message passing paths in the form of edge masks.
    '''
    
    def __init__(self, env):
        super().__init__(env)

        self.num_executors = env.num_executors

        self.observation_space = Dict({
            'dag_batch': Dict({
                'data': Graph(node_space=Box(-np.inf, np.inf, (5,)),
                              edge_space=Discrete(1)),
                'ptr': Sequence(Discrete(1))
            }),
            'stage_mask': Sequence(Discrete(2)),
            'exec_mask': MultiBinary(self.num_executors),
            'edge_mask_batch': MultiBinary((1,1))
        })

        # cache message passing data, because it doesn't always change between observations
        self.msg_passing_cache = {
            'num_nodes': -1,
            'edge_links': None,
            'edge_mask_batch': None
        }


    
    def observation(self, obs):
        self._check_msg_passing_cache(obs)

        data = obs['dag_batch']['data']

        graph_instance = \
            GraphInstance(
                nodes=self._build_node_features(obs), 
                edges=data.edges, 
                edge_links=data.edge_links
            )
        stage_mask = data.nodes[:, 2].astype(bool)
        exec_mask = np.zeros(self.num_executors, dtype=bool)
        exec_mask[:obs['num_executors_to_schedule']] = True

        obs = {
            'dag_batch': {
                'data': graph_instance,
                'ptr': obs['dag_batch']['ptr']
            },
            'stage_mask': stage_mask,
            'exec_mask': exec_mask,
            'edge_mask_batch': self.msg_passing_cache['edge_mask_batch']
        }

        # update dimensions of spaces
        self.observation_space['dag_batch']['ptr'].feature_space.n = data.nodes.shape[0]+1
        self.observation_space['edge_mask_batch'].n = obs['edge_mask_batch'].shape

        return obs
    


    def _build_node_features(self, obs):
        data = obs['dag_batch']['data']
        num_nodes = data.nodes.shape[0]
        
        nodes = np.zeros((num_nodes, 5), dtype=np.float32)

        num_executors_to_schedule = obs['num_executors_to_schedule']
        nodes[:, 0] = num_executors_to_schedule / self.num_executors

        ptr = np.array(obs['dag_batch']['ptr'])
        executor_counts = obs['executor_counts']
        num_active_jobs = len(executor_counts)
        nodes[:, 1] = -1
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i] : ptr[i+1], 1] = 1

        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        nodes[:, 2] = np.repeat(executor_counts, num_nodes_per_dag) / self.num_executors

        num_remaining_tasks = data.nodes[:, 0]
        nodes[:, 3] = num_remaining_tasks / 200

        most_recent_duration = data.nodes[:, 1]
        nodes[:, 4] = num_remaining_tasks * most_recent_duration * 1e-5

        return nodes
    


    def _check_msg_passing_cache(self, obs):
        data = obs['dag_batch']['data']
        num_nodes = data.nodes.shape[0]

        if self.msg_passing_cache['edge_links'] is None or \
           num_nodes != self.msg_passing_cache['num_nodes'] or \
           not np.array_equal(data.edge_links, self.msg_passing_cache['edge_links']):
            # dag batch has changed, so synchronize the cache
            self.msg_passing_cache = {
                'num_nodes': num_nodes,
                'edge_links': data.edge_links,
                'edge_mask_batch': construct_message_passing_masks(data.edge_links, num_nodes)
            }




class DecimaActWrapper(ActionWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.action_space = Dict({
            'stage_idx': Discrete(1),
            'job_idx': Discrete(1),
            'num_exec': Discrete(env.num_executors)
        })


    def action(self, act):
        return {
            'stage_idx': act['stage_idx'],
            'num_exec': 1 + act['num_exec']
        }