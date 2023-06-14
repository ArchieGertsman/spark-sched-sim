import numpy as np
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium.spaces import *

from .. import graph_utils as utils


NUM_NODE_FEATURES = 5



class NeuralActWrapper(ActionWrapper):
    '''converts a neural scheduler's actions to the environment's format'''

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
    


class NeuralObsWrapper(ObservationWrapper):
    '''transforms environment observations into a format that's more suitable 
    for neural schedulers.
    '''
    def __init__(self, env):
        super().__init__(env)

        self.num_executors = env.num_executors

        self.observation_space = Dict({
            'dag_batch': Graph(
                node_space=Box(-np.inf, np.inf, (NUM_NODE_FEATURES,)), 
                edge_space=Discrete(1)),
            'dag_ptr': Sequence(Discrete(1)),
            'stage_mask': Sequence(Discrete(2)),
            'exec_mask': MultiBinary(self.num_executors)
        })


    def observation(self, obs):
        dag_batch = obs['dag_batch']

        graph_instance = GraphInstance(
            nodes=self._build_node_features(obs), 
            edges=dag_batch.edges, 
            edge_links=dag_batch.edge_links
        )
        stage_mask = dag_batch.nodes[:, 2].astype(bool)
        exec_mask = np.zeros(self.num_executors, dtype=bool)
        exec_mask[:obs['num_executors_to_schedule']] = True

        obs = {
            'dag_batch': graph_instance,
            'dag_ptr': obs['dag_ptr'],
            'stage_mask': stage_mask,
            'exec_mask': exec_mask
        }

        self.observation_space['dag_ptr'].feature_space.n = \
            dag_batch.nodes.shape[0] + 1

        return obs
    

    def _build_node_features(self, obs):
        dag_batch = obs['dag_batch']
        num_nodes = dag_batch.nodes.shape[0]
        
        nodes = np.zeros((num_nodes, NUM_NODE_FEATURES), dtype=np.float32)

        num_executors_to_schedule = obs['num_executors_to_schedule']
        nodes[:, 0] = num_executors_to_schedule / self.num_executors

        ptr = np.array(obs['dag_ptr'])
        executor_counts = obs['executor_counts']
        num_active_jobs = len(executor_counts)
        nodes[:, 1] = -1
        if obs['source_job_idx'] < num_active_jobs:
            i = obs['source_job_idx']
            nodes[ptr[i] : ptr[i+1], 1] = 1

        num_nodes_per_dag = ptr[1:] - ptr[:-1]
        nodes[:, 2] = np.repeat(
            executor_counts, num_nodes_per_dag) / self.num_executors

        num_remaining_tasks = dag_batch.nodes[:, 0]
        nodes[:, 3] = num_remaining_tasks / 200

        most_recent_duration = dag_batch.nodes[:, 1]
        nodes[:, 4] = num_remaining_tasks * most_recent_duration * 1e-5

        # is_schedulable = dag_batch.nodes[:, 2]
        # nodes[:, 5] = 2 * is_schedulable - 1

        return nodes



class DAGNNObsWrapper(NeuralObsWrapper):
    '''Observation wrapper for DAGNN-based schedulers. 
    Builds edge masks for each topological generation of the dag
    for asynchronous message passing.
    '''
    def __init__(self, env):
        super().__init__(env)

        self.observation_space['edge_masks'] = MultiBinary((1,1))

        # cache edge masks, because dag batch doesn't always change 
        # between observations
        self._cache = {
            'num_nodes': -1,
            'edge_links': None,
            'edge_masks': None
        }


    def observation(self, obs):
        obs = super().observation(obs)
        self._check_cache(obs)
        obs['edge_masks'] = self._cache['edge_masks']
        self.observation_space['edge_masks'].n = obs['edge_masks'].shape
        return obs
    

    def _check_cache(self, obs):
        dag_batch = obs['dag_batch']
        num_nodes = dag_batch.nodes.shape[0]

        if self._cache['edge_links'] is None \
            or num_nodes != self._cache['num_nodes'] \
            or not np.array_equal(
                dag_batch.edge_links, self._cache['edge_links']):
            # dag batch has changed, so synchronize the cache
            self._cache = {
                'num_nodes': num_nodes,
                'edge_links': dag_batch.edge_links,
                'edge_masks': utils.make_dag_layer_edge_masks(
                    edge_links=dag_batch.edge_links, num_nodes=num_nodes)
            }
    


class TransformerObsWrapper(NeuralObsWrapper):
    '''Observation wrapper for transformer-based schedulers.
    Computes transitive closure of edges for DAGRA (reachability-based 
    attention), and depth of each node for DAGPE (positional encoding).
    '''
    def __init__(self, env):
        super().__init__(env)

        max_depth = 100
        self.observation_space['node_depth'] = Sequence(Discrete(max_depth))

        self._cache = {
            'num_nodes': -1,
            'edge_links': None,
            'edge_links_tc': None,
            'node_depth': None
        }


    def observation(self, obs):
        obs = super().observation(obs)

        self._check_cache(obs)
        edge_links_tc = self._cache['edge_links_tc']
        node_depth = self._cache['node_depth']

        dag_batch = obs['dag_batch']
        obs['dag_batch'] = GraphInstance(
            dag_batch.nodes, dag_batch.edges, edge_links_tc)
        obs['node_depth'] = node_depth

        return obs
    

    def _check_cache(self, obs):
        dag_batch = obs['dag_batch']
        num_nodes = dag_batch.nodes.shape[0]

        if self._cache['edge_links'] is None \
            or num_nodes != self._cache['num_nodes'] \
            or not np.array_equal(
                dag_batch.edge_links, self._cache['edge_links']):
            # dag batch has changed, so synchronize the cache

            if dag_batch.edge_links.shape[0] > 0:
                G = utils.np_to_nx(dag_batch.edge_links, num_nodes)
                edge_links_tc = utils.transitive_closure(G=G) #, bidirectional=False)
                depth = utils.node_depth(G=G)
            else:
                edge_links_tc = dag_batch.edge_links
                depth = np.zeros(num_nodes)
            
            self._cache = {
                'num_nodes': num_nodes,
                'edge_links': dag_batch.edge_links,
                'edge_links_tc': edge_links_tc,
                'node_depth': depth
            }