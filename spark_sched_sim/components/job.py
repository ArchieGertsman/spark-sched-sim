from collections.abc import Generator
import numpy as np
import networkx as nx

from .stage import Stage
from .executor import Executor


class Job:
    """An object representing a job in the system, containing a set of stages with dependencies stored in a dag."""

    def __init__(
        self, id_: int, stages: list[Stage], dag: nx.DiGraph, t_arrival: float
    ) -> None:
        # unique identifier of this job
        self.id_ = id_

        # list of objects of all the stages that belong to this job
        self.stages = stages

        # all incomplete stages
        self.active_stages = stages.copy()

        # incomplete stages whose parents have completed
        self.frontier_stages: set[Stage] = set()

        # networkx dag storing the stage dependencies
        self.dag = dag

        # time that this job arrived into the system
        self.t_arrival = t_arrival

        # time that this job completed, i.e. when the last
        # stage completed
        self.t_completed = np.inf

        # set of executors that are local to this job
        self.local_executors: set[int] = set()

        # count of stages who have no remaining tasks
        self.saturated_stage_count = 0

        self._init_frontier()

    @property
    def pool_key(self) -> tuple[int, None]:
        return (self.id_, None)

    @property
    def completed(self) -> bool:
        return not self.num_active_stages

    @property
    def saturated(self) -> bool:
        return self.saturated_stage_count == len(self.stages)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def num_active_stages(self) -> int:
        return len(self.active_stages)

    def record_stage_completion(self, stage: Stage) -> bool:
        """increments the count of completed stages"""
        self.active_stages.remove(stage)
        self.frontier_stages.remove(stage)

        new_stages = self._find_new_frontier_stages(stage)
        self.frontier_stages |= new_stages

        return bool(new_stages)

    def get_children_stages(self, stage: Stage) -> Generator[Stage, None, None]:
        return (self.stages[stage_id] for stage_id in self.dag.successors(stage.id_))

    def get_parent_stages(self, stage: Stage) -> Generator[Stage, None, None]:
        return (self.stages[stage_id] for stage_id in self.dag.predecessors(stage.id_))

    def attach_executor(self, executor: Executor) -> None:
        assert executor.task is None
        self.local_executors.add(executor.id_)
        executor.job_id = self.id_

    def detach_executor(self, executor: Executor) -> None:
        self.local_executors.remove(executor.id_)
        executor.job_id = None
        executor.task = None

    # internal methods

    def _init_frontier(self) -> None:
        """returns a set containing all the stages which are
        source nodes in the dag, i.e. which have no dependencies
        """
        assert not self.frontier_stages
        self.frontier_stages |= self._get_source_stages()

    def _check_dependencies(self, stage_id: int) -> bool:
        """searches to see if all the dependencies of stage with id `stage_id` are satisfied."""
        for dep_id in self.dag.predecessors(stage_id):
            if not self.stages[dep_id].completed:
                return False

        return True

    def _get_source_stages(self) -> set[Stage]:
        return set(
            self.stages[node] for node, in_deg in self.dag.in_degree() if in_deg == 0
        )

    def _find_new_frontier_stages(self, stage: Stage) -> set[Stage]:
        """if ` stage` is completed, returns all of its successors whose other dependencies are also
        completed, if any exist.
        """
        if not stage.completed:
            return set()

        new_stages = set()
        # search through stage's children
        for suc_stage_id in self.dag.successors(stage.id_):
            # if all dependencies are satisfied, then add this child to the frontier
            new_stage = self.stages[suc_stage_id]
            if not new_stage.completed and self._check_dependencies(suc_stage_id):
                new_stages.add(new_stage)

        return new_stages
