from typing import Any
from numpy import ndarray
from gymnasium import Wrapper, ActionWrapper, ObservationWrapper
import numpy as np
import gymnasium.spaces as sp

from . import utils

NUM_NODE_FEATURES = 5


class DecimaEnvWrapper(Wrapper):
    def __init__(self, env):
        env = DecimaActWrapper(env)
        env = DecimaObsWrapper(env)
        super().__init__(env)


class DecimaActWrapper(ActionWrapper):
    """converts Decima's actions to the environment's format"""

    def __init__(self, env) -> None:
        super().__init__(env)

        self.action_space = sp.Dict(
            {
                "stage_idx": sp.Discrete(1),
                "job_idx": sp.Discrete(1),
                "num_exec": sp.Discrete(env.unwrapped.num_executors),
            }
        )

    def action(self, act: dict[str, Any]) -> dict[str, Any]:
        return {"stage_idx": act["stage_idx"], "num_exec": 1 + act["num_exec"]}


class DecimaObsWrapper(ObservationWrapper):
    """transforms environment observations into a format that's more suitable for Decima"""

    def __init__(
        self, env, num_tasks_scale: int = 200, work_scale: float = 1e5
    ) -> None:
        super().__init__(env)

        self.num_tasks_scale = num_tasks_scale
        self.work_scale = work_scale
        self.num_executors = env.unwrapped.num_executors

        # cache edge masks, because dag batch doesn't always change between observations
        self._cache: dict[str, Any] = {
            "num_nodes": -1,
            "edge_links": None,
            "edge_masks": None,
        }

        self.observation_space = sp.Dict(
            {
                "dag_batch": sp.Graph(
                    node_space=sp.Box(-np.inf, np.inf, (NUM_NODE_FEATURES,)),
                    edge_space=sp.Discrete(1),
                ),
                "dag_ptr": sp.Sequence(sp.Discrete(1)),
                "stage_mask": sp.Sequence(sp.Discrete(2)),
                "exec_mask": sp.Sequence(sp.MultiBinary(self.num_executors)),
                "edge_masks": sp.MultiBinary((1, 1)),
            }
        )

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        dag_batch = obs["dag_batch"]

        exec_supplies = np.array(obs["exec_supplies"])
        num_committable_execs = obs["num_committable_execs"]
        gap = np.maximum(self.num_executors - exec_supplies, 0)

        # cap on number of execs that can be committed to each job
        commit_caps = np.minimum(gap, num_committable_execs)

        j_src = obs["source_job_idx"]
        num_jobs = exec_supplies.size
        if j_src < num_jobs:
            commit_caps[j_src] = num_committable_execs

        graph_instance = sp.GraphInstance(
            nodes=self._build_node_features(obs, commit_caps),
            edges=dag_batch.edges,
            edge_links=dag_batch.edge_links,
        )

        stage_mask = dag_batch.nodes[:, 2].astype(bool)

        exec_mask = np.zeros((num_jobs, self.num_executors), dtype=bool)
        for j, cap in enumerate(commit_caps):
            exec_mask[j, :cap] = True

        self._validate_cache(obs)

        obs = {
            "dag_batch": graph_instance,
            "dag_ptr": obs["dag_ptr"],
            "stage_mask": stage_mask,
            "exec_mask": exec_mask,
            "edge_masks": self._cache["edge_masks"],
        }

        self.observation_space["dag_ptr"].feature_space.n = dag_batch.nodes.shape[0] + 1
        self.observation_space["edge_masks"].n = obs["edge_masks"].shape
        return obs

    def _build_node_features(
        self, obs: dict[str, Any], commit_caps: ndarray
    ) -> ndarray:
        dag_batch = obs["dag_batch"]
        num_nodes = dag_batch.nodes.shape[0]
        ptr = np.array(obs["dag_ptr"])
        node_counts = ptr[1:] - ptr[:-1]
        exec_supplies = obs["exec_supplies"]
        num_active_jobs = len(exec_supplies)
        source_job_idx = obs["source_job_idx"]

        nodes = np.zeros((num_nodes, NUM_NODE_FEATURES), dtype=np.float32)

        # how many exec can be added to each node
        nodes[:, 0] = np.repeat(commit_caps, node_counts) / self.num_executors

        # whether or not a node belongs to the source job
        nodes[:, 1] = -1
        if source_job_idx < num_active_jobs:
            i = source_job_idx
            nodes[ptr[i] : ptr[i + 1], 1] = 1

        # current supply of executors for each node's job
        nodes[:, 2] = np.repeat(exec_supplies, node_counts) / self.num_executors

        # number of remaining tasks in each node
        num_remaining_tasks = dag_batch.nodes[:, 0]
        nodes[:, 3] = num_remaining_tasks / self.num_tasks_scale

        # approximate remaining work in each node
        most_recent_duration = dag_batch.nodes[:, 1]
        nodes[:, 4] = num_remaining_tasks * most_recent_duration / self.work_scale

        return nodes

    def _validate_cache(self, obs: dict[str, Any]) -> None:
        dag_batch = obs["dag_batch"]
        num_nodes = dag_batch.nodes.shape[0]

        if (
            self._cache["edge_links"] is None
            or num_nodes != self._cache["num_nodes"]
            or not np.array_equal(dag_batch.edge_links, self._cache["edge_links"])
        ):
            # dag batch has changed, so synchronize the cache
            self._cache = {
                "num_nodes": num_nodes,
                "edge_links": dag_batch.edge_links,
                "edge_masks": utils.make_dag_layer_edge_masks(
                    (dag_batch.edge_links, num_nodes)
                ),
            }
