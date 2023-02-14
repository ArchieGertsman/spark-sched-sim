from typing import Optional
from itertools import chain
from gymnasium.core import ObsType, ActType
from torch import Tensor

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.profiler

from .base_alg import BaseAlg
from ..utils.rollout_buffer import RolloutBuffer
from ..utils.graph import collate_obsns, ObsBatch




class PPO(BaseAlg):

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
        old_lgprobs: Tensor,
        clip_range: float = .2
    ) -> tuple[Tensor, float, float]:
        '''clipped loss unique to PPO'''
        lgprobs, entropies = \
            self.agent.evaluate_actions(obsns, actions)

        ratio = torch.exp(lgprobs - old_lgprobs)
        loss1 = advantages * ratio
        loss2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(loss1, loss2).mean()

        entropy_loss = entropies.sum()
        total_loss = policy_loss + self.entropy_weight * entropy_loss

        return total_loss, policy_loss.item(), entropy_loss.item()



    def _learn_from_rollouts(
        self,
        rollout_buffers: list[RolloutBuffer]
    ) -> tuple[float, float]:

        # separate the rollout data into lists
        obsns_list, actions_list, wall_times_list, rewards_list = \
            zip(*((buff.obsns, buff.actions, buff.wall_times, buff.rewards)
                  for buff in rollout_buffers))      

        advantages_list = self._compute_advantages(rewards_list, wall_times_list)

        # make a new dataset out of the new rollouts, and make a 
        # dataloader that loads minibatches from that dataset
        dataloader = \
            self._make_dataloader(obsns_list, actions_list, advantages_list)

        action_losses = []
        entropies = []

        # run multiple learning epochs with minibatching
        for _ in range(self.num_epochs):
            for obsns, actions, advantages, old_lgprobs in dataloader:
                total_loss, action_loss, entropy_loss = \
                    self._compute_loss(
                        obsns, 
                        actions, 
                        advantages,
                        old_lgprobs
                    )

                action_losses += [action_loss]
                entropies += [entropy_loss / advantages.numel()]

                self.agent.update_parameters(total_loss)

        return np.sum(action_losses), np.mean(entropies)



    def _make_dataloader(
        self,
        obsns_list: list[list[ObsType]],
        actions_list: list[list[ActType]],
        advantages_list: list[np.ndarray]
    ) -> DataLoader:

        old_lgprobs = \
            self._compute_old_lgprobs(obsns_list, actions_list)
        
        rollout_dataset = \
            PPORolloutDataset(
                obsns_list, 
                actions_list, 
                advantages_list, 
                old_lgprobs
            )

        dataloader = \
            DataLoader(
                dataset=rollout_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=PPORolloutDataset.collate
            )

        return dataloader



    def _compute_old_lgprobs(
        self, 
        obsns_list: list[list[ObsType]], 
        actions_list: list[list[ActType]]
    ) -> Tensor:
        obsns = collate_obsns(chain(*obsns_list))
        actions = Tensor([[int(a) for a in act.values()] 
                          for act in chain(*actions_list)])
        with torch.no_grad():
            old_lgprobs, _ = \
                self.agent.evaluate_actions(obsns, actions)

        return old_lgprobs




class PPORolloutDataset(Dataset):
    def __init__(
        self, 
        obsns_list: list[list[ObsType]],
        actions_list: list[list[ActType]],
        advantages_list: list[np.ndarray],
        old_lgprobs: Tensor 
    ):
        '''
        Args:
            obsns_list: list of observation lists for each rollout
            actions_list: list of action lists for each rollout
            advantages_list: list of advantages for each rollout
            old_lgprobs: already flattened tensor of all old log-
                probabilities for all the rollouts
        '''
        super().__init__()
        self.obsns = {i: obs for i, obs in enumerate(chain(*obsns_list))}
        self.actions = {i: act for i, act in enumerate(chain(*actions_list))}
        self.advantages = np.hstack(advantages_list)
        self.old_lgprobs = old_lgprobs



    def __getitem__(
        self, 
        idx: int
    ) -> tuple[ObsType, ActType, float, float]:
        return self.obsns[idx], \
               self.actions[idx], \
               self.advantages[idx], \
               self.old_lgprobs[idx]



    def __len__(self):
        return len(self.advantages)



    @classmethod
    def collate(
        cls, 
        batch: list[tuple[ObsType, ActType, np.ndarray, Tensor]]
    ) -> tuple[ObsBatch, Tensor, Tensor, Tensor]:
        '''
        Args:
            batch: list of (obs, action, advantage, old_lgprob) 
                tuples to be collated into a minibatch
        Returns:
            tuple of collated observations, actions, advantages, 
                and old log-probabilities
        '''
        obsns, actions, advantages, old_lgprobs = zip(*batch)
        obsns = collate_obsns(obsns)
        actions = Tensor([[int(a) for a in act.values()] for act in actions])
        advantages = torch.from_numpy(np.hstack(advantages))
        old_lgprobs = torch.stack(old_lgprobs)
        return obsns, actions, advantages, old_lgprobs