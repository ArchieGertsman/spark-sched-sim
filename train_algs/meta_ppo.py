from typing import Optional, Iterable
from itertools import chain

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchopt

from .base_alg import BaseAlg
from .rollouts import RolloutBuffer, RolloutDataset, ValueDataset
from spark_sched_sim.graph_utils import ObsBatch, collate_obsns
from .utils.baselines import compute_baselines



class MetaPPO(BaseAlg):
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
    ) -> tuple[Tensor, float, float]:

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
    


    def adapt_and_predict(
        self, 
        critic_sd,
        inner_opt_sd,
        obsns_adapt_list, 
        val_targs_adapt_list,
        obsns_pred_list,
        val_targs_pred_list
    ):
        obsns_adapt = {i: obs for i, obs in enumerate(chain(*obsns_adapt_list))}
        val_targs_adapt = torch.from_numpy(np.hstack(val_targs_adapt_list)).float()
        critic_dl = DataLoader(
            ValueDataset(obsns_adapt, val_targs_adapt),
            batch_size=(val_targs_adapt.numel() // self.num_envs + 1),
            collate_fn=ValueDataset.collate,
            shuffle=True,
            generator=self.dataloader_gen
        )

        # adapt
        print('start inner')
        for _ in range(2):
            for obsns, val_targs in critic_dl:
                vals = self.agent.predict_values(obsns)
                print('asdf', vals[-10:])
                inner_val_loss = F.mse_loss(vals.flatten(), val_targs)
                self.agent.inner_opt.step(inner_val_loss)

        # predict
        vals_pred = self.agent.predict_values(
            collate_obsns(chain(*obsns_pred_list))
        ).flatten()
        val_targs_pred = torch.from_numpy(np.hstack(val_targs_pred_list)).float()
        val_loss = F.mse_loss(vals_pred, val_targs_pred)
        print('bruh')
        print(vals_pred[-20:])
        print(val_targs_pred[-20:])
        print(val_loss)

        torchopt.recover_state_dict(self.agent.critic, critic_sd)
        torchopt.recover_state_dict(self.agent.inner_opt, inner_opt_sd)

        return vals_pred.detach(), val_loss
    


    def _learn_from_rollouts(
        self,
        rollout_buffers: Iterable[RolloutBuffer]
    ) -> tuple[float, float]:
        
        (obsns_list, wall_times_list, rewards_list) = \
            zip(*((buff.obsns, buff.wall_times, buff.rewards)
                  for buff in rollout_buffers)) 

        diff_rewards_list = self.return_calc.compute_diff_rewards(wall_times_list, rewards_list)

        # differential returns for each rollout
        diff_returns_list = []
        for diff_rewards, obsns in zip(rewards_list, obsns_list): # diff_rewards_list:
            diff_returns = np.zeros_like(diff_rewards)
            # with torch.no_grad():
            #     obs = collate_obsns([obsns[-1]])
            #     val_bootstrap = self.agent.predict_values(obs).item()
            diff_returns[-1] = diff_rewards[-1] # + .99 * val_bootstrap
            for i in reversed(range(len(diff_rewards)-1)):
                diff_returns[i] = diff_rewards[i] + .99 * diff_returns[i+1]
            diff_returns_list += [diff_returns]

        value_losses = []

        
        
        # for _ in range(3):
        critic_sd = torchopt.extract_state_dict(self.agent.critic) #, by='copy')
        inner_opt_sd = torchopt.extract_state_dict(self.agent.inner_opt) #, by='copy')

        # adapt critic on first half of rollouts, then predict values 
        # for second half of rollouts
        values_2nd, val_loss2 = self.adapt_and_predict(
            critic_sd,
            inner_opt_sd,
            obsns_list[:self.num_envs//2],
            diff_returns_list[:self.num_envs//2],
            obsns_list[self.num_envs//2:],
            diff_returns_list[self.num_envs//2:]
        )
        
        # adapt critic on second half of rollouts, then predict values 
        # for first half of rollouts
        values_1st, val_loss1 = self.adapt_and_predict(
            critic_sd,
            inner_opt_sd,
            obsns_list[self.num_envs//2:],
            diff_returns_list[self.num_envs//2:],
            obsns_list[:self.num_envs//2],
            diff_returns_list[:self.num_envs//2]
        )

        value_loss = (val_loss1 + val_loss2) / 2
        value_losses += [value_loss.item()]
        self.agent.update_parameters(value_loss)



        adapted_values = torch.hstack([values_1st, values_2nd])
        diff_returns = torch.from_numpy(np.hstack(diff_returns_list)).float()

        # monte carlo advantage estimation using adapted values
        advantages = diff_returns - adapted_values

        policy_dataloader = self._make_dataloader(rollout_buffers, advantages)

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

        return np.mean(policy_losses), \
               np.mean(entropy_losses), \
               np.mean(value_losses), \
               np.mean(approx_kl_divs)




    def _make_dataloader(
        self,
        rollout_buffers: Iterable[RolloutBuffer],
        advantages
    ) -> DataLoader:
        '''creates a dataset out of the new rollouts, and returns a 
        dataloader that loads minibatches from that dataset
        '''

        # separate the rollout data into lists
        (obsns_list, actions_list, lgprobs_list) = \
            zip(*((buff.obsns, buff.actions, buff.lgprobs)
                  for buff in rollout_buffers)) 

        # flatten observations into a dict for fast access time
        obsns = {i: obs for i, obs in enumerate(chain(*obsns_list))}
        actions = torch.tensor([list(act.values()) for act in chain(*actions_list)])
        old_lgprobs = torch.from_numpy(np.hstack(lgprobs_list))
        
        rollout_dataset = \
            RolloutDataset(
                obsns, 
                actions, 
                advantages, 
                old_lgprobs
            )

        policy_dataloader = \
            DataLoader(
                dataset=rollout_dataset,
                batch_size=(old_lgprobs.numel() // self.batch_size + 1),
                shuffle=True,
                collate_fn=RolloutDataset.collate,
                generator=self.dataloader_gen
            )

        return policy_dataloader

