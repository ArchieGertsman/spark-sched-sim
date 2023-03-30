from typing import Optional, Iterable
from torch import Tensor

import numpy as np
import torch
import torch.profiler

from .base_alg import BaseAlg
from .rollouts import RolloutBuffer
from ..utils.graph import collate_obsns
from ..utils.baselines import compute_baselines




class VPG(BaseAlg):
    '''Vanilla Policy Gradient'''

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
        max_time_mean_ceil: float = 0.,
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
            max_time_mean_ceil,
            entropy_weight_init,
            entropy_weight_decay,
            entropy_weight_min
        )
    


    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        
        (obsns_list, 
         actions_list, 
         wall_times_list, 
         rewards_list) = \
            zip(*((buff.obsns, 
                   buff.actions, 
                   buff.wall_times, 
                   buff.rewards)
                  for buff in rollout_buffers)) 

        returns_list = self.return_calc(rewards_list, wall_times_list)
        baselines_list = compute_baselines(wall_times_list, returns_list)

        self.agent.optim.zero_grad()

        num_samples = 0
        policy_loss_tot = 0.
        entropy_loss_tot = 0.

        gen = zip(obsns_list, actions_list, returns_list, baselines_list)
        for obsns, actions, returns, baselines in gen:
            num_samples += len(obsns)
            obsns = collate_obsns(obsns)
            actions = torch.tensor([list(act.values()) for act in actions])
            adv = torch.from_numpy(returns - baselines).float()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            lgprobs, entropies, logit_loss = self.agent.evaluate_actions(obsns, actions)

            policy_loss = -(lgprobs * adv).sum()
            policy_loss_tot += policy_loss.item()

            entropy_loss = -entropies.sum()
            entropy_loss_tot += entropy_loss.item()

            loss = policy_loss + self.entropy_weight * entropy_loss + .001 * logit_loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), 
            .5,
            error_if_nonfinite=True
        )

        self.agent.optim.step()

        for buff, rewards, times in zip(rollout_buffers, rewards_list, wall_times_list):
            buff.sum_rewards = np.sum(rewards) / (times[-1] / (1e3 * 60))

        return policy_loss_tot / num_samples, \
               entropy_loss_tot / num_samples, \
               0, \
               0