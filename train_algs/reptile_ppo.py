from typing import Optional, Iterable
from itertools import chain
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_alg import BaseAlg
from .rollouts import RolloutBuffer, RolloutDataset, ValueDataset
from spark_sched_sim.graph_utils import ObsBatch, collate_obsns



class ReptilePPO(BaseAlg):
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
        max_time_mean_ceil: float = np.inf,
        entropy_weight_init: float = 1.,
        entropy_weight_decay: float = 1e-3,
        entropy_weight_min: float = 1e-4,
        clip_range: float = .2,
        target_kl: Optional[float] = None
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

        self.target_kl = target_kl
        self.clip_range = clip_range



    def _compute_loss(
        self,
        obsns: ObsBatch,
        actions: Tensor,
        advantages: Tensor,
        old_lgprobs: Tensor,
    ) -> tuple[Tensor, float, float, float]:

        lgprobs, entropies = self.agent.evaluate_actions(obsns, actions)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(lgprobs - old_lgprobs)
        policy_loss1 = advantages * ratio
        policy_loss2 = advantages * \
            torch.clamp(
                ratio, 
                1 - self.clip_range, 
                1 + self.clip_range
            )

        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + self.entropy_weight * entropy_loss

        with torch.no_grad():
            log_ratio = lgprobs - old_lgprobs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

        return total_loss, policy_loss.item(), entropy_loss.item(), approx_kl_div
    


    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        
        critic_clone = deepcopy(self.agent.critic)
        
        critic_clone.load_state_dict({name:
            torch.mean(torch.stack([rb.sd[name] for rb in rollout_buffers]), axis=0)
            for name in critic_clone.state_dict()
        })

        self.agent.ac_opt.zero_grad()
        for p_old, p_new in zip(self.agent.critic.parameters(), critic_clone.parameters()):
            p_old.grad = p_old - p_new
        self.agent.ac_opt.step()
        
        # old_sd = self.agent.critic.state_dict()
        # self.agent.critic.load_state_dict({name:
        #     old_sd[name] + .01 * (torch.mean(torch.stack([rb.sd[name] for rb in rollout_buffers])) - old_sd[name])
        #     for name in old_sd
        # })

        policy_dataloader = self._make_dataloader(rollout_buffers)

        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []
        continue_training = True
        for _ in range(2):
            if not continue_training:
                break

            for obsns, actions, advantages, old_lgprobs in policy_dataloader:
                total_loss, policy_loss, entropy_loss, approx_kl_div = \
                    self._compute_loss(
                        obsns, 
                        actions, 
                        advantages,
                        old_lgprobs
                    )

                policy_losses += [policy_loss]
                entropy_losses += [entropy_loss]
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    print(f"Early stopping due to reaching max kl: {approx_kl_div:.3f}")
                    continue_training = False
                    break

                self.agent.update_parameters(total_loss)


        # self.agent.sched.step()


        return np.mean(policy_losses), \
               np.mean(entropy_losses), \
               np.mean([rb.value_loss for rb in rollout_buffers]), \
               np.mean(approx_kl_divs)




    def _make_dataloader(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> DataLoader:
        '''creates a dataset out of the new rollouts, and returns a 
        dataloader that loads minibatches from that dataset
        '''

        # separate the rollout data into lists
        (obsns_list, 
         actions_list, 
         wall_times_list, 
         rewards_list, 
         lgprobs_list,
         returns_list,
         values_list) = \
            zip(*((buff.obsns, 
                   buff.actions, 
                   buff.wall_times, 
                   buff.rewards, 
                   buff.lgprobs,
                   buff.returns,
                   buff.values)
                  for buff in rollout_buffers)) 
        
        # self.return_calc(rewards_list, wall_times_list)

        returns = torch.hstack(returns_list)
        values = torch.hstack(values_list)
        advantages = returns - values

        obsns = {i: obs for i, obs in enumerate(chain(*obsns_list))}
        actions = torch.tensor([list(act.values()) for act in chain(*actions_list)])
        old_lgprobs = torch.from_numpy(np.hstack(lgprobs_list))
        policy_dataloader = \
            DataLoader(
                dataset=RolloutDataset(
                    obsns, 
                    actions, 
                    advantages, 
                    old_lgprobs
                ),
                batch_size=(old_lgprobs.numel() // self.batch_size + 1),
                shuffle=True,
                collate_fn=RolloutDataset.collate,
                generator=self.dataloader_gen
            )
        
        return policy_dataloader