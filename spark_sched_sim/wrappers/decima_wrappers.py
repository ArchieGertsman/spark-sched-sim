
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
    
    def __init__(self, env):
        super().__init__(env)

        self.num_executors = env.num_executors

        self.observation_space = Dict({
            'dag_batch': Dict({
                'data': Graph(node_space=Box(-np.inf, np.inf, (5,)),
                              edge_space=Discrete(1)),
                'ptr': Sequence(Discrete(1))
            }),
            'schedulable_stage_mask': Sequence(Discrete(2)),
            'valid_prlsm_lim_mask': Sequence(MultiBinary(self.num_executors)),
            'edge_mask_batch': MultiBinary((1,1))
        })

        # cache message passing data, because it doesn't always change between observations
        self.num_nodes = -1
        self.edge_links = None
        self.edge_mask_batch = None


    
    def observation(self, obs):
        ptr = np.array(obs['dag_batch']['ptr'])
        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        num_nodes = obs['dag_batch']['data'].nodes.shape[0]
        executor_counts = obs['executor_counts']
        num_active_jobs = len(executor_counts)

        # build node features
        nodes = np.zeros((num_nodes, 5), dtype=np.float32)

        nodes[:, 0] = obs['num_executors_to_schedule'] / self.num_executors

        nodes[:, 1] = -1
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i]:ptr[i+1], 1] = 1

        nodes[:, 2] = np.repeat(executor_counts, num_nodes_per_dag) / self.num_executors

        num_remaining_tasks = obs['dag_batch']['data'].nodes[:, 0]
        nodes[:, 3] = num_remaining_tasks / 200

        most_recent_duration = obs['dag_batch']['data'].nodes[:, 1]
        nodes[:, 4] = num_remaining_tasks * most_recent_duration * 1e-5

        edge_links = obs['dag_batch']['data'].edge_links

        if self.edge_links is None or num_nodes != self.num_nodes or \
            not np.array_equal(edge_links, self.edge_links):
            # dag batch has changed, so update message passing data
            self.num_nodes = num_nodes
            self.edge_links = edge_links
            self.edge_mask_batch = construct_message_passing_masks(edge_links, num_nodes)

        self.observation_space['dag_batch']['ptr'].feature_space.n = num_nodes+1
        self.observation_space['edge_mask_batch'].n = self.edge_mask_batch.shape

        graph_instance = \
            GraphInstance(
                nodes=nodes, 
                edges=obs['dag_batch']['data'].edges, 
                edge_links=obs['dag_batch']['data'].edge_links
            )

        obs = {
            'dag_batch': {
                'data': graph_instance,
                'ptr': obs['dag_batch']['ptr']
            },
            'schedulable_stage_mask': obs['schedulable_stage_mask'],
            'valid_prlsm_lim_mask': obs['valid_prlsm_lim_mask'],
            'edge_mask_batch': self.edge_mask_batch
        }

        return obs



class DecimaActWrapper(ActionWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.action_space = Dict({
            'stage_idx': Discrete(1),
            'job_idx': Discrete(1),
            'prlsm_lim': Discrete(env.num_executors)
        })



    def action(self, act):
        return {
            'stage_idx': act['stage_idx'],
            'prlsm_lim': 1 + act['prlsm_lim']
        }