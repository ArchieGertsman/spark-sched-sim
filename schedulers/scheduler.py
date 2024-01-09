from abc import ABC, abstractmethod
from collections.abc import Iterable
from gymnasium import Wrapper
from torch import Tensor

import torch
import torch.nn as nn


class Scheduler(ABC):
    """Interface for all schedulers"""

    name: str
    env_wrapper_cls: type[Wrapper] | None

    @abstractmethod
    def schedule(self, obs: dict) -> tuple[dict, dict]:
        pass


class TrainableScheduler(Scheduler, nn.Module):
    """Interface for all trainable schedulers"""

    optim: torch.optim.Optimizer | None
    max_grad_norm: float | None

    @abstractmethod
    def evaluate_actions(
        self, obsns: Iterable[dict], actions: Iterable[tuple]
    ) -> dict[str, Tensor]:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def update_parameters(self, loss: Tensor | None = None) -> None:
        assert self.optim

        if loss:
            # accumulate gradients
            loss.backward()

        if self.max_grad_norm:
            # clip grads
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()
