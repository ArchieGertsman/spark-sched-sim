from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any
import shutil
import os
import os.path as osp
import sys
from copy import deepcopy
import json
import pathlib

import numpy as np
import torch
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from schedulers import make_scheduler, TrainableScheduler
from .rollout_worker import RolloutWorkerSync, RolloutWorkerAsync, RolloutBuffer
from .utils import Baseline, ReturnsCalculator


CfgType = dict[str, Any]


class Trainer(ABC):
    """Base training algorithm class. Each algorithm must implement the
    abstract method `train_on_rollouts`
    """

    def __init__(
        self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType
    ) -> None:
        self.seed = train_cfg["seed"]
        torch.manual_seed(self.seed)

        self.scheduler_cls = agent_cfg["agent_cls"]

        self.device = torch.device(
            train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # number of training iterations
        self.num_iterations: int = train_cfg["num_iterations"]

        # number of unique job sequences per iteration
        self.num_sequences: int = train_cfg["num_sequences"]

        # number of rollouts per job sequence
        self.num_rollouts: int = int(train_cfg["num_rollouts"])

        self.artifacts_dir: str = train_cfg["artifacts_dir"]
        pathlib.Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

        self.stdout_dir = osp.join(self.artifacts_dir, "stdout")
        self.tb_dir = osp.join(self.artifacts_dir, "tb")
        self.checkpointing_dir = osp.join(self.artifacts_dir, "checkpoints")
        self.use_tensorboard: bool = train_cfg["use_tensorboard"]
        self.checkpointing_freq: int = train_cfg["checkpointing_freq"]
        self.env_cfg = env_cfg

        self.baseline = Baseline(self.num_sequences, self.num_rollouts)

        self.rollout_duration: float | None = train_cfg.get("rollout_duration")

        assert ("reward_buff_cap" in train_cfg) ^ (
            "beta_discount" in train_cfg
        ), "must provide exactly one of `reward_buff_cap` and `beta_discount` in config"

        if "reward_buff_cap" in train_cfg:
            self.return_calc = ReturnsCalculator(buff_cap=train_cfg["reward_buff_cap"])
        else:
            beta: float = train_cfg["beta_discount"]
            env_cfg |= {"beta": beta}
            self.return_calc = ReturnsCalculator(beta=beta)

        self.scheduler_cfg = (
            agent_cfg
            | {"num_executors": env_cfg["num_executors"]}
            | {k: train_cfg[k] for k in ["opt_cls", "opt_kwargs", "max_grad_norm"]}
        )
        scheduler = make_scheduler(self.scheduler_cfg)
        assert isinstance(scheduler, TrainableScheduler), "scheduler must be trainable."
        self.scheduler: TrainableScheduler = scheduler

    def train(self) -> None:
        """trains the model on different job arrival sequences.
        For each job sequence:
        - multiple rollouts are collected in parallel, asynchronously
        - the rollouts are gathered at the center, where model parameters are
            updated, and
        - new model parameters are scattered to the rollout workers
        """
        self._setup()

        # every n'th iteration, save the best model from the past n iterations,
        # where `n = self.model_save_freq`
        best_state = None

        exception: Exception | None = None

        print("Beginning training.\n", flush=True)

        for i in range(self.num_iterations):
            state_dict = deepcopy(self.scheduler.state_dict())

            # # move params to GPU for learning
            self.scheduler.to(self.device, non_blocking=True)

            # scatter
            for conn in self.conns:
                conn.send({"state_dict": state_dict})

            # gather
            results = []
            for i, conn in enumerate(self.conns):
                res = conn.recv()
                if isinstance(res, Exception):
                    print(f"An exception occured in process {i}", flush=True)
                    exception = res
                    break
                results += [res]

            if exception:
                break

            rollout_buffers, rollout_stats_list = zip(
                *[(res["rollout_buffer"], res["stats"]) for res in results if res]
            )

            # update parameters
            learning_stats = self.train_on_rollouts(rollout_buffers)

            # return params to CPU before scattering updated state dict to the rollout workers
            self.scheduler.to("cpu", non_blocking=True)

            avg_num_jobs = self.return_calc.avg_num_jobs or np.mean(
                [stats["avg_num_jobs"] for stats in rollout_stats_list]
            )

            # check if model is the current best
            if not best_state or avg_num_jobs < best_state["avg_num_jobs"]:
                best_state = self._capture_state(
                    i, avg_num_jobs, state_dict, rollout_stats_list
                )

            if (i + 1) % self.checkpointing_freq == 0:
                self._checkpoint(i, best_state)
                best_state = None

            if self.use_tensorboard:
                ep_lens = [len(buff) for buff in rollout_buffers if buff]
                self._write_stats(i, learning_stats, rollout_stats_list, ep_lens)

            print(
                f"Iteration {i+1} complete. Avg. # jobs: " f"{avg_num_jobs:.3f}",
                flush=True,
            )

        self._cleanup()

        if exception:
            raise exception

    @abstractmethod
    def train_on_rollouts(
        self, rollout_buffers: Iterable[RolloutBuffer]
    ) -> dict[str, Any]:
        pass

    # internal methods

    def _preprocess_rollouts(
        self, rollout_buffers: Iterable[RolloutBuffer]
    ) -> dict[str, tuple]:
        (
            obsns_list,
            actions_list,
            wall_times_list,
            rewards_list,
            lgprobs_list,
            resets_list,
        ) = zip(
            *(
                (
                    buff.obsns,
                    buff.actions,
                    buff.wall_times,
                    buff.rewards,
                    buff.lgprobs,
                    buff.resets,
                )
                for buff in rollout_buffers
                if buff is not None
            )
        )

        returns_list = self.return_calc(
            rewards_list,
            wall_times_list,
            resets_list,
        )

        wall_times_list = tuple([wall_times[:-1] for wall_times in wall_times_list])
        baselines_list = self.baseline(wall_times_list, returns_list)

        return {
            "obsns_list": obsns_list,
            "actions_list": actions_list,
            "returns_list": returns_list,
            "baselines_list": baselines_list,
            "lgprobs_list": lgprobs_list,
        }

    def _setup(self) -> None:
        # logging
        shutil.rmtree(self.stdout_dir, ignore_errors=True)
        os.mkdir(self.stdout_dir)
        sys.stdout = open(osp.join(self.stdout_dir, "main.out"), "a")

        if self.use_tensorboard:
            self.summary_writer = SummaryWriter(self.tb_dir)

        # model checkpoints
        shutil.rmtree(self.checkpointing_dir, ignore_errors=True)
        os.mkdir(self.checkpointing_dir)

        # torch
        torch.multiprocessing.set_start_method("spawn")
        # print('cuda available:', torch.cuda.is_available())
        # torch.autograd.set_detect_anomaly(True)

        self.scheduler.train()

        self._start_rollout_workers()

    def _cleanup(self) -> None:
        self._terminate_rollout_workers()

        if self.use_tensorboard:
            self.summary_writer.close()

        print("\nTraining complete.", flush=True)

    def _capture_state(
        self, i: int, avg_num_jobs: float, state_dict: dict, stats_list: Iterable[dict]
    ) -> dict[str, Any]:
        return {
            "iteration": i,
            "avg_num_jobs": np.round(avg_num_jobs, 3),
            "state_dict": state_dict,
            "completed_job_count": int(
                np.mean([stats["num_completed_jobs"] for stats in stats_list])
            ),
        }

    def _checkpoint(self, i: int, best_state: dict) -> None:
        dir = osp.join(self.checkpointing_dir, f"{i+1}")
        os.mkdir(dir)
        best_sd = best_state.pop("state_dict")
        torch.save(best_sd, osp.join(dir, "model.pt"))
        with open(osp.join(dir, "state.json"), "w") as fp:
            json.dump(best_state, fp)

    def _start_rollout_workers(self) -> None:
        self.procs = []
        self.conns = []

        base_seeds = self.seed + np.arange(self.num_sequences)
        base_seeds = np.repeat(base_seeds, self.num_rollouts)
        seed_step = self.num_sequences
        lock = mp.Lock()
        for rank, base_seed in enumerate(base_seeds):
            conn_main, conn_sub = mp.Pipe()
            self.conns += [conn_main]

            proc = mp.Process(
                target=RolloutWorkerAsync(self.rollout_duration)
                if self.rollout_duration
                else RolloutWorkerSync(),
                args=(
                    rank,
                    conn_sub,
                    self.env_cfg,
                    self.scheduler_cfg,
                    self.stdout_dir,
                    int(base_seed),
                    seed_step,
                    lock,
                ),
            )

            self.procs += [proc]
            proc.start()

        for proc in self.procs:
            proc.join(5)

    def _terminate_rollout_workers(self) -> None:
        for conn in self.conns:
            conn.send(None)

        for proc in self.procs:
            proc.join()

    def _write_stats(
        self,
        epoch: int,
        learning_stats: dict,
        stats_list: Iterable[dict],
        ep_lens: list[int],
    ) -> None:
        episode_stats = learning_stats | {
            "avg num concurrent jobs": np.mean(
                [stats["avg_num_jobs"] for stats in stats_list]
            ),
            "avg job duration": np.mean(
                [stats["avg_job_duration"] for stats in stats_list]
            ),
            "completed jobs count": np.mean(
                [stats["num_completed_jobs"] for stats in stats_list]
            ),
            "job arrival count": np.mean(
                [stats["num_job_arrivals"] for stats in stats_list]
            ),
            "episode length": np.mean(ep_lens),
        }

        for name, stat in episode_stats.items():
            self.summary_writer.add_scalar(name, stat, epoch)
