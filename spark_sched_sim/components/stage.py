from .task import Task


class Stage:
    def __init__(
        self, id: int, job_id: int, num_tasks: int, rough_task_duration: float
    ) -> None:
        self.id_ = id
        self.job_id = job_id
        self.most_recent_duration = rough_task_duration
        self.num_tasks = num_tasks
        self.remaining_tasks = [
            Task(id_=i, stage_id=self.id_, job_id=self.job_id) for i in range(num_tasks)
        ]
        self.num_remaining_tasks = num_tasks
        self.num_executing_tasks = 0
        self.num_completed_tasks = 0
        self.is_schedulable = False

    def __hash__(self) -> int:
        return hash(self.pool_key)

    def __eq__(self, other) -> bool:
        if type(other) is type(self):
            return self.pool_key == other.pool_key
        else:
            return False

    @property
    def pool_key(self) -> tuple[int, int]:
        return (self.job_id, self.id_)

    @property
    def job_pool_key(self) -> tuple[int, None]:
        return (self.job_id, None)

    @property
    def completed(self) -> bool:
        return self.num_completed_tasks == self.num_tasks

    @property
    def num_saturated_tasks(self) -> int:
        return self.num_executing_tasks + self.num_completed_tasks

    @property
    def next_task_id(self) -> int:
        return self.num_saturated_tasks

    @property
    def approx_remaining_work(self) -> float:
        return self.most_recent_duration * self.num_remaining_tasks

    def launch_next_task(self) -> Task:
        assert self.num_saturated_tasks < self.num_tasks
        task = self.remaining_tasks.pop()
        self.num_remaining_tasks -= 1
        self.num_executing_tasks += 1
        return task

    def record_task_completion(self) -> None:
        self.num_executing_tasks -= 1
        self.num_completed_tasks += 1
