from typing import List, Tuple, Optional
from itertools import chain
from gymnasium.core import ObsType, ActType
from torch import Tensor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.profiler

from .base_alg import BaseAlg
from ..utils.baselines import compute_baselines
from ..utils.rollout_buffer import RolloutBuffer
from ..utils.graph import collate_obsns, ObsBatch




class Reinforce(BaseAlg):

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
        advantages: Tensor
    ) -> Tuple[Tensor, float, float]:

        action_lgprobs, action_entropies = \
            self.agent.evaluate_actions(obsns, actions)

        action_loss = -(advantages * action_lgprobs).sum()
        entropy_loss = action_entropies.sum()
        total_loss = action_loss + self.entropy_weight * entropy_loss

        return total_loss, action_loss.item(), entropy_loss.item()



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
            for obsns, actions, advantages in dataloader:
                total_loss, action_loss, entropy_loss = \
                    self._compute_loss(
                        obsns, 
                        actions, 
                        advantages
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

        rollout_dataset = \
            ReinforceRolloutDataset(
                obsns_list, 
                actions_list, 
                advantages_list
            )

        dataloader = \
            DataLoader(
                dataset=rollout_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=ReinforceRolloutDataset.collate
            )

        return dataloader



class ReinforceRolloutDataset(Dataset):
    def __init__(
        self, 
        obsns_list: list[list[ObsType]],
        actions_list: list[list[ActType]],
        advantages_list: list[np.ndarray]
    ):
        '''
        Args:
            obsns_list: list of observation lists for each rollout
            actions_list: list of action lists for each rollout
            advantages_list: list of advantages for each rollout
        '''
        super().__init__()
        self.obsns = {i: obs for i, obs in enumerate(chain(*obsns_list))}
        self.actions = {i: act for i, act in enumerate(chain(*actions_list))}
        self.advantages = np.hstack(advantages_list)



    def __getitem__(
        self, 
        idx: int
    ) -> tuple[ObsType, ActType, float, float]:
        return self.obsns[idx], \
               self.actions[idx], \
               self.advantages[idx]



    def __len__(self):
        return len(self.advantages)



    @classmethod
    def collate(
        cls, 
        batch: list[tuple[ObsType, ActType, np.ndarray]]
    ) -> tuple[ObsBatch, Tensor, Tensor]:
        '''
        Args:
            batch: list of (obs, action, advantage) tuples 
                to be collated into a minibatch
        Returns:
            tuple of collated observations, actions, and advantages
        '''
        obsns, actions, advantages = zip(*batch)
        obsns = collate_obsns(obsns)
        actions = Tensor([[int(a) for a in act.values()] for act in actions])
        advantages = torch.from_numpy(np.hstack(advantages))
        return obsns, actions, advantages