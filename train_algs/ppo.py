from typing import Optional, Iterable
from itertools import chain

import numpy as np
import torch
from torch import Tensor
import torch.profiler
from torch.utils.data import DataLoader

from .base_alg import BaseAlg
from .rollouts import RolloutBuffer, RolloutDataset
from spark_sched_sim.graph_utils import ObsBatch, collate_obsns
from .utils.baselines import compute_baselines



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
        max_time_mean_ceil: float = np.inf,
        entropy_weight_init: float = 1.,
        entropy_weight_decay: float = 1e-3,
        entropy_weight_min: float = 1e-4,
        clip_range: float = .2,
        target_kl: float = .01
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
        
        (obsns_list, 
         actions_list, 
         old_lgprobs_list,
         wall_times_list,
         rewards_list) = \
            zip(*((buff.obsns, 
                   buff.actions, 
                   buff.lgprobs,
                   buff.wall_times, 
                   buff.rewards)
                  for buff in rollout_buffers)) 

        returns_list = self.return_calc(rewards_list, wall_times_list)
        baselines_list = compute_baselines(wall_times_list, returns_list)

        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []

        continue_training = True

        for _ in range(self.num_epochs):
            if not continue_training:
                break

            gen = zip(obsns_list, actions_list, old_lgprobs_list, returns_list, baselines_list)
            for obsns, actions, old_lgprobs, returns, baselines in gen:
                obsns = collate_obsns(obsns)
                actions = torch.tensor([list(act.values()) for act in actions])
                old_lgprobs = torch.tensor(old_lgprobs)
                adv = torch.from_numpy(returns - baselines).float()

                total_loss, policy_loss, entropy_loss, approx_kl_div = \
                    self._compute_loss(
                        obsns, 
                        actions, 
                        adv,
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
               0, \
               np.mean(approx_kl_divs)



    


    # def _learn_from_rollouts(
    #     self,
    #     rollout_buffers: Iterable[RolloutBuffer]
    # ) -> tuple[float, float]:

    #     policy_dataloader = self._make_dataloader(rollout_buffers)

    #     policy_losses = []
    #     entropy_losses = []
    #     approx_kl_divs = []

    #     continue_training = True

    #     # update policy for one epoch
    #     for _ in range(self.num_epochs):
    #         if not continue_training:
    #             break

    #         for obsns, actions, advantages, old_lgprobs in policy_dataloader:
    #             total_loss, policy_loss, entropy_loss, approx_kl_div = \
    #                 self._compute_loss(
    #                     obsns, 
    #                     actions, 
    #                     advantages,
    #                     old_lgprobs
    #                 )

    #             policy_losses += [policy_loss]
    #             entropy_losses += [entropy_loss]
    #             approx_kl_divs.append(approx_kl_div)

    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 print(f"Early stopping due to reaching max kl: {approx_kl_div:.3f}")
    #                 continue_training = False
    #                 break

    #             self.agent.update_parameters(total_loss)

    #     return np.mean(policy_losses), \
    #            np.mean(entropy_losses), \
    #            0, \
    #            np.mean(approx_kl_divs)




    # def _make_dataloader(
    #     self,
    #     rollout_buffers: Iterable[RolloutBuffer]
    # ) -> DataLoader:
    #     '''creates a dataset out of the new rollouts, and returns a 
    #     dataloader that loads minibatches from that dataset
    #     '''

    #     # separate the rollout data into lists
    #     (obsns_list, 
    #      actions_list, 
    #      wall_times_list, 
    #      rewards_list, 
    #      lgprobs_list) = \
    #         zip(*((buff.obsns, 
    #                buff.actions, 
    #                buff.wall_times, 
    #                buff.rewards,
    #                buff.lgprobs)
    #               for buff in rollout_buffers)) 
        
    #     N = self.num_envs # // 2

    #     # flatten observations and actions into a dict for fast access time
    #     obsns = {i: obs for i, obs in enumerate(chain(*obsns_list[:N]))}
    #     actions = torch.tensor([list(act.values()) for act in chain(*actions_list[:N])])
    #     old_lgprobs = torch.from_numpy(np.hstack(lgprobs_list[:N]))

    #     # deltas_list = []
    #     # num_jobs_list = []
    #     # for times, job_minutes in zip(wall_times_list, rewards_list):
    #     #     times = np.concatenate([np.array([0.]), times])
    #     #     deltas = (times[1:] - times[:-1]) / (1e3 * 60)
    #     #     deltas_list += [deltas]
    #     #     # num_jobs_list += [job_minutes / deltas]
    #     #     num_jobs = np.zeros_like(job_minutes)
    #     #     for t, (jm, d) in enumerate(zip(job_minutes, deltas)):
    #     #         if d != 0:
    #     #             num_jobs[t] = jm / d
    #     #         else:
    #     #             num_jobs[t] = num_jobs[t-1]
    #     #     num_jobs_list += [num_jobs]


    #     # num_jobs_baselines_list = compute_baselines(wall_times_list, num_jobs_list)

    #     # returns_list = []
    #     # baselines_list = []
    #     # for deltas, job_minutes, num_jobs_baseline in zip(deltas_list, rewards_list, num_jobs_baselines_list):
    #     #     returns = np.zeros_like(job_minutes)
    #     #     baselines = np.zeros_like(job_minutes)
    #     #     r = 0
    #     #     b = 0
    #     #     for t in reversed(range(len(deltas))):
    #     #         r += job_minutes[t]
    #     #         b += deltas[t] * num_jobs_baseline[t]
    #     #         returns[t] = r
    #     #         baselines[t] = b
    #     #     returns_list += [returns]
    #     #     baselines_list += [baselines]

    #     diff_returns_list = self.return_calc(rewards_list, wall_times_list)
    #     # baselines_list = compute_baselines(wall_times_list, diff_returns_list)

    #     baselines_list = []
    #     for i in range(self.num_envs // 4):
    #         baselines_list += compute_baselines(
    #             wall_times_list[4*i : 4*(i+1)], 
    #             diff_returns_list[4*i : 4*(i+1)]
    #         )

        
    #     # t_samp_list = wall_times_list[:N]
    #     # r_samp_list = diff_returns_list[:N]
    #     # t_greedy_list = wall_times_list[N:]
    #     # r_greedy_list = diff_returns_list[N:]

    #     # adv_list = []
    #     # for t_samp, r_samp, t_greedy, r_greedy in zip(
    #     #     t_samp_list, 
    #     #     r_samp_list,
    #     #     t_greedy_list,
    #     #     r_greedy_list
    #     # ):
    #     #     baseline = np.interp(t_samp, t_greedy, r_greedy)
    #     #     adv_list += [r_samp - baseline]

    #     # advantages = torch.from_numpy(np.hstack(adv_list)).float()

    #     advantages = torch.from_numpy(
    #         np.hstack(diff_returns_list) - np.hstack(baselines_list)
    #     ).float()
        

    #     for buff, rewards, times in zip(rollout_buffers, rewards_list, wall_times_list):
    #         buff.sum_rewards = 0 #np.mean(rewards[rewards != 0])


    #     # adv_list = []
    #     # for rewards, times, values in zip(rewards_list, wall_times_list, values_list):
    #     #     times = np.concatenate([np.array([0.]), times])
    #     #     deltas = (times[1:] - times[:-1]) / (1e3 * 60)

    #     #     dv = values[1:] - values[:-1]
    #     #     dv = np.concatenate([dv, np.array([0])])

    #     #     td_err = rewards - deltas * self.return_calc.avg_num_jobs + dv

    #     #     adv = np.zeros_like(td_err)
    #     #     a = 0
    #     #     for t in reversed(range(len(td_err))):
    #     #         a = td_err[t] + self.gae_lam * a
    #     #         adv[t] = a
    #     #     adv_list += [adv]
    #     # advantages = torch.from_numpy(np.hstack(adv_list)).float()

    #     # for buff, adv, values in zip(rollout_buffers, adv_list, values_list):
    #     #     buff.returns = adv + values
        
    #     rollout_dataset = \
    #         RolloutDataset(
    #             obsns, 
    #             actions, 
    #             advantages, 
    #             old_lgprobs
    #         )

    #     num_samples = old_lgprobs.numel()
    #     batch_size = num_samples // self.batch_size + 1

    #     policy_dataloader = \
    #         DataLoader(
    #             dataset=rollout_dataset,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             collate_fn=RolloutDataset.collate,
    #             generator=self.dataloader_gen
    #         )
        
    #     return policy_dataloader