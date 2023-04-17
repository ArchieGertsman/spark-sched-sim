
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

        self.num_executors = env.num_executors

        self.observation_space = Dict({
            'dag_batch': Dict({
                'data': Graph(node_space=Box(-np.inf, np.inf, (5,)),
                              edge_space=Discrete(1)),
                'ptr': Sequence(Discrete(1))
            }),
            'schedulable_stage_mask': Sequence(Discrete(2)),
            'valid_prlsm_lim_mask': Sequence(MultiBinary(self.num_executors))
        })


    
    def observation(self, obs):
        ptr = np.array(obs['dag_batch']['ptr'])
        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        num_active_nodes = obs['dag_batch']['data'].nodes.shape[0]
        executor_counts = obs['executor_counts']
        num_active_jobs = len(executor_counts)

        # build node features
        nodes = np.zeros((num_active_nodes, 5), dtype=np.float32)

        nodes[:, 0] = obs['num_executors_to_schedule'] / self.num_executors

        nodes[:, 1] = -1
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i]:ptr[i+1], 1] = 1

        nodes[:, 2] = np.repeat(executor_counts, num_nodes_per_dag) / self.num_executors

        num_remaining_tasks = obs['dag_batch']['data'].nodes[:, 0]
        most_recent_duration = obs['dag_batch']['data'].nodes[:, 1]
        nodes[:, 3] = num_remaining_tasks / 200
        nodes[:, 4] = num_remaining_tasks * most_recent_duration * 1e-5

        # update stage action space to reflect the current number of active stages
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
            'schedulable_stage_mask': obs['schedulable_stage_mask'],
            'valid_prlsm_lim_mask': obs['valid_prlsm_lim_mask']
        }

        return obs



class DecimaActWrapper(ActionWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        self.action_space = Dict({
            'stage_idx': Discrete(1),
            'job_idx': Discrete(env.num_total_jobs),
            'prlsm_lim': Discrete(env.num_executors)
        })



    def action(self, act):
        return {
            'stage_idx': act['stage_idx'],
            'prlsm_lim': 1 + act['prlsm_lim']
        }