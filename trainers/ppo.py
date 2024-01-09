from collections.abc import Iterable
from itertools import chain
from typing import SupportsFloat
from torch import Tensor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .trainer import Trainer


EPS = 1e-8


class RolloutDataset(Dataset):
    def __init__(self, obsns, acts, advgs, lgprobs):
        self.obsns = obsns
        self.acts = acts
        self.advgs = advgs
        self.lgprobs = lgprobs

    def __len__(self):
        return len(self.obsns)

    def __getitem__(self, idx):
        return self.obsns[idx], self.acts[idx], self.advgs[idx], self.lgprobs[idx]


# def collate_fn(batch):
#     obsns, acts, advgs, lgprobs = zip(*batch)
#     obsns = collate_obsns(obsns)
#     acts = torch.stack(acts)
#     advgs = torch.stack(advgs)
#     lgprobs = torch.stack(lgprobs)
#     return obsns, acts, advgs, lgprobs


class PPO(Trainer):
    """Proximal Policy Optimization"""

    def __init__(self, agent_cfg, env_cfg, train_cfg):
        super().__init__(agent_cfg, env_cfg, train_cfg)

        self.entropy_coeff = train_cfg.get("entropy_coeff", 0.0)
        self.clip_range = train_cfg.get("clip_range", 0.2)
        self.target_kl = train_cfg.get("target_kl", 0.01)
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.num_batches = train_cfg.get("num_batches", 3)

    def train_on_rollouts(self, rollout_buffers):
        data = self._preprocess_rollouts(rollout_buffers)

        returns = np.array(list(chain(*data["returns_list"])))
        baselines = np.concatenate(data["baselines_list"])

        dataset = RolloutDataset(
            obsns=list(chain(*data["obsns_list"])),
            acts=list(chain(*data["actions_list"])),
            advgs=returns - baselines,
            lgprobs=list(chain(*data["lgprobs_list"])),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset) // self.num_batches + 1,
            shuffle=True,
            collate_fn=lambda batch: zip(*batch),
        )

        return self._train(dataloader)

    def _train(self, dataloader):
        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []
        continue_training = True

        for _ in range(self.num_epochs):
            if not continue_training:
                break

            for obsns, actions, advgs, old_lgprobs in dataloader:
                loss, info = self._compute_loss(obsns, actions, advgs, old_lgprobs)

                kl = info["approx_kl_div"]

                policy_losses += [info["policy_loss"]]
                entropy_losses += [info["entropy_loss"]]
                approx_kl_divs.append(kl)

                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    print(f"Early stopping due to reaching max kl: " f"{kl:.3f}")
                    continue_training = False
                    break

                self.scheduler.update_parameters(loss)

        return {
            "policy loss": np.abs(np.mean(policy_losses)),
            "entropy": np.abs(np.mean(entropy_losses)),
            "approx kl div": np.abs(np.mean(approx_kl_divs)),
        }

    def _compute_loss(
        self,
        obsns: Iterable[dict],
        acts: Iterable[tuple],
        advantages: Iterable[SupportsFloat],
        old_lgprobs: Iterable[SupportsFloat],
    ) -> tuple[Tensor, dict[str, SupportsFloat]]:
        """CLIP loss"""
        eval_res = self.scheduler.evaluate_actions(obsns, acts)

        advgs = torch.tensor(advantages).float()
        advgs = (advgs - advgs.mean()) / (advgs.std() + EPS)

        log_ratio = eval_res["lgprobs"] - torch.tensor(old_lgprobs)
        ratio = log_ratio.exp()

        policy_loss1 = advgs * ratio
        policy_loss2 = advgs * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        entropy_loss = -eval_res["entropies"].mean()

        loss = policy_loss + self.entropy_coeff * entropy_loss

        with torch.no_grad():
            approx_kl_div = ((ratio - 1) - log_ratio).mean().item()

        return loss, {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl_div": approx_kl_div,
        }
