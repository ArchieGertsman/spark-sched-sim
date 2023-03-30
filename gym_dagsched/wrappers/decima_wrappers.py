
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

        self.num_workers = env.num_workers

        self.observation_space = Dict({
            'dag_batch': Dict({
                'data': Graph(node_space=Box(-np.inf, np.inf, (6,)),
                              edge_space=Discrete(1)),
                'ptr': Sequence(Discrete(1))
            }),
            'schedulable_op_mask': Sequence(Discrete(2)),
            'valid_prlsm_lim_mask': Sequence(MultiBinary(self.num_workers))
        })


    
    def observation(self, obs):
        ptr = np.array(obs['dag_batch']['ptr'])
        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        num_active_nodes = obs['dag_batch']['data'].nodes.shape[0]
        worker_counts = obs['worker_counts']
        num_active_jobs = len(worker_counts)

        # build node features
        nodes = np.zeros((num_active_nodes, 6), dtype=np.float32)

        nodes[:, 0] = obs['num_workers_to_schedule'] / self.num_workers

        nodes[:, 1] = -2
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i]:ptr[i+1], 1] = 2

        nodes[:, 2] = np.repeat(worker_counts, num_nodes_per_dag) / self.num_workers

        num_remaining_tasks = obs['dag_batch']['data'].nodes[:, 0]
        most_recent_duration = obs['dag_batch']['data'].nodes[:, 1]
        schedulable = obs['dag_batch']['data'].nodes[:, 2]
        nodes[:, 3] = num_remaining_tasks / 200
        nodes[:, 4] = most_recent_duration * num_remaining_tasks * 1e-5
        nodes[:, 5] = 2*schedulable - 1

        # update op action space to reflect the current number of active ops
        self.observation_space['dag_batch']['ptr'].feature_space.n = num_active_nodes+1

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