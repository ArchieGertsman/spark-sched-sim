from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .trainer import Trainer
from .utils import compute_baselines
from spark_sched_sim import graph_utils


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
        return self.obsns[idx], self.acts[idx], self.advgs[idx], \
               self.lgprobs[idx]
    

def collate_fn(batch):
    obsns, acts, advgs, lgprobs = zip(*batch)
    obsns = graph_utils.collate_obsns(obsns)
    acts = torch.stack(acts)
    advgs = torch.stack(advgs)
    lgprobs = torch.stack(lgprobs)
    return obsns, acts, advgs, lgprobs



class PPO(Trainer):
    '''Proximal Policy Optimization'''

    def __init__(
        self,
        scheduler_cls,
        device,
        log_options,
        checkpoint_options,
        env_kwargs,
        model_kwargs,
        seed=42,
        async_rollouts=False,
        entropy_coeff=1e-4,
        clip_range=.2,
        target_kl=.01,
        num_epochs=3,
        num_batches=10
    ):  
        super().__init__(
            scheduler_cls,
            device,
            log_options,
            checkpoint_options,
            env_kwargs,
            model_kwargs,
            seed,
            async_rollouts
        )

        self.entropy_coeff = entropy_coeff
        self.clip_range = clip_range
        self.target_kl = target_kl
        self.num_epochs = num_epochs
        self.num_batches = num_batches
    

    def learn_from_rollouts(self, rollout_buffers):
        (obsns_list, actions_list, wall_times_list, 
         rewards_list, lgprobs_list, resets_list) = zip(*(
            (
                buff.obsns, buff.actions, buff.wall_times, 
                buff.rewards, buff.lgprobs, buff.resets
            )
            for buff in rollout_buffers 
            if buff is not None
        )) 

        returns_list = self.return_calc(
            rewards_list, wall_times_list, resets_list)

        wall_times_list = [wall_times[:-1] for wall_times in wall_times_list]
        baselines_list = compute_baselines(wall_times_list, returns_list)

        returns = np.array(list(chain(*returns_list)))
        baselines = np.concatenate(baselines_list)

        dataset = RolloutDataset(
            obsns = list(chain(*obsns_list)),
            acts = torch.tensor(list(chain(*actions_list))),
            advgs = torch.from_numpy(returns - baselines).float(),
            lgprobs = torch.tensor(list(chain(*lgprobs_list))))

        dataloader = DataLoader(
            dataset,
            batch_size = len(dataset) // self.num_batches + 1,
            shuffle = True,
            collate_fn = collate_fn)

        return self._learn(dataloader)
    

    def _learn(self, dataloader):
        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []
        continue_training = True

        for _ in range(self.num_epochs):
            if not continue_training:
                break

            for obsns, actions, advgs, old_lgprobs in dataloader:
                total_loss, policy_loss, entropy_loss, approx_kl_div = \
                    self._compute_loss(obsns, actions, advgs, old_lgprobs)

                policy_losses += [policy_loss]
                entropy_losses += [entropy_loss]
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None \
                    and approx_kl_div > 1.5 * self.target_kl:
                    print(f'Early stopping due to reaching max kl: '
                            f'{approx_kl_div:.3f}')
                    continue_training = False
                    break

                self.agent.update_parameters(total_loss)
        
        return {
            'policy loss': np.abs(np.mean(policy_losses)),
            'entropy': np.abs(np.mean(entropy_losses)),
            'approx kl div': np.abs(np.mean(approx_kl_divs))
        }
    

    def _compute_loss(self, obsns, acts, advgs, old_lgprobs):
        '''CLIP loss'''
        lgprobs, entropies = self.agent.evaluate_actions(obsns, acts)

        advgs = (advgs - advgs.mean()) / (advgs.std() + EPS)
        log_ratio = lgprobs - old_lgprobs
        ratio = log_ratio.exp()

        policy_loss1 = advgs * ratio
        policy_loss2 = advgs * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        entropy_loss = -entropies.mean()

        loss = policy_loss + self.entropy_coeff * entropy_loss

        with torch.no_grad():
            approx_kl_div = ((ratio - 1) - log_ratio).mean().item()

        return loss, policy_loss.item(), entropy_loss.item(), approx_kl_div