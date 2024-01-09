from typing import cast


PoolKey = tuple[int | None, int | None]
OptPoolKey = PoolKey | None
JobPoolKey = tuple[int, None]
StagePoolKey = tuple[int, int]
CommonPoolKey = tuple[None, None]

COMMON_POOL_KEY = (None, None)


class ExecutorTracker:
    """Maintains all executor assignments. These include:
    - current location of each executor, called a 'pool'
    - commitments from one executor pool to another, and
    - the executors moving between pools.

    The following executor pools exist:
    - Placeholder pool (key `None`): not an actual pool; it never contains any executors.
      the executor source is set to null pool when no pool is ready to schedule its executors.
    - General pool (key `(None, None)`): the pool where executors reside when they are not at
      any job. All executors start out in the common pool.
    - Job pool (key `(job_id, None)`): pool of idle executors at a job
    - Operation pool (key `(job_id, stage_id)`): pool of executors at a stage, including idle
      and busy
    """

    def __init__(self, num_executors: int) -> None:
        self.num_executors = num_executors

    def reset(self) -> None:
        # executor id -> key of pool where the executor currently resides
        self._executor_locations: dict[int, OptPoolKey] = {
            executor_id: COMMON_POOL_KEY for executor_id in range(self.num_executors)
        }

        # pool key -> set of id's of executors who reside at this pool
        self._pools: dict[OptPoolKey, set[int]] = {
            None: set(),
            COMMON_POOL_KEY: set(range(self.num_executors)),
        }

        # pool key A ->
        #   (pool key B ->
        #       number of commitments from
        #       pool A to pool B)
        self._commitments: dict[OptPoolKey, dict[OptPoolKey, int]] = {
            None: {},
            COMMON_POOL_KEY: {},
        }

        # pool key -> total number of outgoing commitments from this pool
        self._num_commitments_from: dict[OptPoolKey, int] = {
            None: 0,
            COMMON_POOL_KEY: 0,
        }

        # stage pool key -> total number of commitments to stage
        self._num_commitments_to_stage: dict[PoolKey, int] = {COMMON_POOL_KEY: 0}

        # stage pool key -> number of executors moving to stage
        self._num_moving_to_stage: dict[StagePoolKey, int] = {}

        # job id -> size of job's pool plus the total number of external commitments
        # and executors moving to any of its stages
        self._total_executor_count: dict[int | None, int] = {None: 0}

        # initialize executor source
        self._curr_source: OptPoolKey = COMMON_POOL_KEY

    def add_job_pool(self, pool_key: JobPoolKey) -> None:
        if pool_key in self._pools:
            raise ValueError("job pool already exists")

        job_id, _ = pool_key
        self._pools[pool_key] = set()
        self._commitments[pool_key] = {}
        self._num_commitments_from[pool_key] = 0
        self._total_executor_count[job_id] = 0

    def add_stage_pool(self, pool_key: StagePoolKey) -> None:
        if pool_key in self._pools:
            raise ValueError("stage pool already exists")

        job_id, _ = pool_key
        if (job_id, None) not in self._pools:
            raise ValueError(f"job with id {job_id} does not exist")

        self._pools[pool_key] = set()
        self._commitments[pool_key] = {}
        self._num_commitments_from[pool_key] = 0
        self._num_commitments_to_stage[pool_key] = 0
        self._num_moving_to_stage[pool_key] = 0

    def get_source(self) -> OptPoolKey:
        return self._curr_source

    def source_job_id(self) -> int | None:
        if not self._curr_source or self._curr_source is COMMON_POOL_KEY:
            return None
        else:
            return self._curr_source[0]

    def num_committable_execs(self) -> int:
        num_uncommitted = (
            len(self._pools[self._curr_source])
            - self._num_commitments_from[self._curr_source]
        )
        assert num_uncommitted >= 0, "[num_committable_execs]"
        return num_uncommitted

    def common_pool_has_executors(self) -> bool:
        return bool(self._pools[COMMON_POOL_KEY])

    def num_executors_moving_to_stage(self, stage_pool_key: StagePoolKey) -> int:
        return self._num_moving_to_stage[stage_pool_key]

    def num_commitments_to_stage(self, stage_pool_key: StagePoolKey) -> int:
        return self._num_commitments_to_stage[stage_pool_key]

    def exec_supply(self, job_id: int) -> int:
        return self._total_executor_count[job_id]

    def update_executor_source(self, pool_key: PoolKey) -> None:
        self._curr_source = pool_key

    def clear_executor_source(self) -> None:
        self._curr_source = None

    def get_source_commitments(self) -> dict[OptPoolKey, int]:
        return self._commitments[self._curr_source].copy()

    def get_pool(self, pool_key: OptPoolKey) -> set[int]:
        return self._pools[pool_key].copy()

    def pool_size(self, pool_key: OptPoolKey) -> int:
        return len(self._pools[pool_key])

    def get_source_pool(self) -> set[int]:
        return self.get_pool(self._curr_source)

    def executor_location(self, executor_id: int) -> OptPoolKey:
        return self._executor_locations[executor_id]

    def add_commitment(self, num_executors: int, dst_pool_key: PoolKey) -> None:
        assert self._curr_source, "[add_commitment]"
        src_job_id = self._curr_source[0]
        dst_job_id = dst_pool_key[0]

        self._increment_commitments(dst_pool_key, n=num_executors)

        if dst_job_id != src_job_id:
            self._total_executor_count[dst_job_id] += num_executors

    def remove_commitment(self, executor_id: int, dst_pool_key: PoolKey) -> PoolKey:
        src_pool_key = self._executor_locations[executor_id]
        assert src_pool_key, "[remove_commitment]"

        if dst_pool_key not in self._commitments[src_pool_key]:
            raise ValueError(f"no commitments from {src_pool_key} to {dst_pool_key}")

        src_job_id = src_pool_key[0]
        dst_job_id = dst_pool_key[0]

        # update commitment from source to dest stage
        self._decrement_commitments(src_pool_key, dst_pool_key)

        if dst_job_id != src_job_id:
            self._total_executor_count[dst_job_id] -= 1
            assert self._total_executor_count[dst_job_id] >= 0

        return src_pool_key

    def peek_commitment(self, pool_key: OptPoolKey) -> OptPoolKey:
        try:
            return next(iter(self._commitments[pool_key]))
        except (KeyError, StopIteration):
            # no outgoing commitments from this pool
            return None

    def record_executor_arrival(self, stage_pool_key: StagePoolKey) -> None:
        self._num_moving_to_stage[stage_pool_key] -= 1
        assert self._num_moving_to_stage[stage_pool_key] >= 0

    def move_executor_to_pool(
        self, executor_id: int, new_pool_key: OptPoolKey, send: bool = False
    ) -> None:
        if send and (
            not new_pool_key or new_pool_key[0] is None or new_pool_key[1] is None
        ):
            raise ValueError("can only send executors to stages")

        old_pool_key = self._executor_locations[executor_id]

        if old_pool_key is not None:
            # remove executor from old pool
            self._pools[old_pool_key].remove(executor_id)
            self._executor_locations[executor_id] = None

        if not send:
            # directly move executor into new pool
            self._executor_locations[executor_id] = new_pool_key
            self._pools[new_pool_key].add(executor_id)
            return

        # send the executor to the stage

        new_pool_key = cast(StagePoolKey, new_pool_key)

        self._num_moving_to_stage[new_pool_key] += 1

        old_job_id = old_pool_key[0] if old_pool_key is not None else None
        new_job_id = new_pool_key[0]
        assert old_job_id != new_job_id

        self._total_executor_count[new_job_id] += 1
        if old_job_id is not None:
            self._total_executor_count[old_job_id] -= 1
            assert self._total_executor_count[old_job_id] >= 0

    # internal methods

    def _increment_commitments(self, dst_pool_key: PoolKey, n: int) -> None:
        try:
            self._commitments[self._curr_source][dst_pool_key] += n
        except KeyError:
            # key not in dict yet
            self._commitments[self._curr_source][dst_pool_key] = n

        self._num_commitments_from[self._curr_source] += n
        self._num_commitments_to_stage[dst_pool_key] += n

        supply = len(self._pools[self._curr_source])
        demand = self._num_commitments_from[self._curr_source]
        assert supply >= demand

    def _decrement_commitments(
        self, src_pool_key: PoolKey, dst_pool_key: PoolKey
    ) -> None:
        self._commitments[src_pool_key][dst_pool_key] -= 1
        self._num_commitments_from[src_pool_key] -= 1
        self._num_commitments_to_stage[dst_pool_key] -= 1

        assert self._num_commitments_from[src_pool_key] >= 0
        assert self._num_commitments_to_stage[dst_pool_key] >= 0

        if self._commitments[src_pool_key][dst_pool_key] == 0:
            self._commitments[src_pool_key].pop(dst_pool_key)
