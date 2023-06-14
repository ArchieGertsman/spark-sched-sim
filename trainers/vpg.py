from typing import Optional, Iterable
from torch import Tensor

import numpy as np
import torch
import torch.profiler

from .trainer import Trainer
from .rollout_worker import RolloutBuffer
from .utils import compute_baselines
from spark_sched_sim.graph_utils import collate_obsns




class VPG(Trainer):
    '''Vanilla Policy Gradient'''

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
        entropy_coeff=1e-4
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

        policy_losses = []
        entropy_losses = []

        gen = zip(obsns_list, actions_list, returns_list, baselines_list, lgprobs_list)
        for obsns, actions, returns, baselines, old_lgprobs in gen:
            obsns = collate_obsns(obsns)
            actions = torch.tensor(actions)
            lgprobs, entropies = self.agent.evaluate_actions(obsns, actions)

            # re-computed log-probs don't exactly match the original ones,
            # but it doesn't seem to affect training
            # with torch.no_grad():
            #     diff = (lgprobs - torch.tensor(old_lgprobs)).abs()
            #     assert lgprobs.allclose(torch.tensor(old_lgprobs))

            adv = torch.from_numpy(returns - baselines).float()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(lgprobs * adv).mean()
            policy_losses += [policy_loss.item()]

            entropy_loss = -entropies.mean()
            entropy_losses += [entropy_loss.item()]

            loss = policy_loss + self.entropy_coeff * entropy_loss
            loss.backward()

        self.agent.update_parameters()
        
        return {
            'policy loss': np.mean(policy_losses),
            'entropy': np.mean(entropy_losses)
        }