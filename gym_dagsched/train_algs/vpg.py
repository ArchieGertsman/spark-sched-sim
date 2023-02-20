from typing import Tuple, Optional
from torch import Tensor

import numpy as np
import torch
import torch.profiler

from .base_alg import BaseAlg
from ..utils.graph import ObsBatch




class VPG(BaseAlg):
    '''Vanilla Policy Gradient'''

    def __init__(
        self,
        env_kwargs: dict,
        num_iterations: int = 500,
        num_epochs: int = 4,
        batch_size: int = 512,
        num_envs: int = 4,
        seed: int = 42,
        log_dir: str = 'log',
        summary_writer_dir: Optional[str] = None,
        model_save_dir: str = 'models',
        model_save_freq: int = 20,
        optim_class: torch.optim.Optimizer = torch.optim.Adam,
        optim_lr: float = 3e-4,
        max_grad_norm: float = .5,
        gamma: float = .99,
        max_time_mean_init: float = np.inf,
        max_time_mean_growth: float = 0.,
        max_time_mean_clip_range: float = 0.,
        entropy_weight_init: float = 1.,
        entropy_weight_decay: float = 1e-3,
        entropy_weight_min: float = 1e-4
    ):  
        super().__init__(
            env_kwargs,
            num_iterations,
            num_epochs,
            batch_size,
            num_envs,
            seed,
            log_dir,
            summary_writer_dir,
            model_save_dir,
            model_save_freq,
            optim_class,
            optim_lr,
            max_grad_norm,
            gamma,
            max_time_mean_init,
            max_time_mean_growth,
            max_time_mean_clip_range,
            entropy_weight_init,
            entropy_weight_decay,
            entropy_weight_min
        )



    def _compute_loss(
        self,
        obsns: ObsBatch, 
        actions: Tensor, 
        advantages: Tensor,
        old_lgprobs: Tensor
    ) -> Tuple[Tensor, float, float]:

        action_lgprobs, action_entropies = self.agent.evaluate_actions(obsns, actions)

        policy_loss = -(advantages * action_lgprobs).mean()
        entropy_loss = -action_entropies.mean()
        total_loss = policy_loss + self.entropy_weight * entropy_loss

        return total_loss, policy_loss.item(), entropy_loss.item()