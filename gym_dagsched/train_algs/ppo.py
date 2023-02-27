from typing import Optional
from torch import Tensor

import numpy as np
import torch
import torch.profiler
import torch.nn.functional as F

from .base_alg import BaseAlg
from ..utils.graph import ObsBatch



class PPO(BaseAlg):
    '''Proximal Policy Optimization'''

    def __init__(
        self,
        env_kwargs: dict,
        num_iterations: int = 500,
        num_epochs: int = 4,
        batch_size: Optional[int] = 512,
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
        entropy_weight_min: float = 1e-4,
        clip_range: float = .2,
        target_kl: Optional[float] = None,
        value_weight: float = .5
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
            entropy_weight_min,
            target_kl
        )

        self.clip_range = clip_range
        self.value_weight = value_weight



    def _compute_loss(
        self,
        obsns: ObsBatch,
        actions: Tensor,
        value_targets: Tensor,
        advantages: Tensor,
        old_lgprobs: Tensor
    ) -> tuple[Tensor, float, float]:

        values, lgprobs, entropies = self.agent.evaluate_actions(obsns, actions)

        # normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(lgprobs - old_lgprobs)
        policy_loss1 = advantages * ratio
        policy_loss2 = advantages * \
            torch.clamp(
                ratio, 
                1 - self.clip_range, 
                1 + self.clip_range
            )

        policy_loss = -torch.min(policy_loss1, policy_loss2).mean() * 1e3
        entropy_loss = -entropies.mean()
        value_loss = F.mse_loss(value_targets, values.flatten())

        total_loss = policy_loss + \
                     self.entropy_weight * entropy_loss + \
                     self.value_weight * value_loss

        with torch.no_grad():
            log_ratio = lgprobs - old_lgprobs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

        return total_loss, policy_loss.item(), entropy_loss.item(), value_loss.item(), approx_kl_div