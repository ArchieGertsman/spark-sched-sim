import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.utils import clamp_probs
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_scatter import segment_csr
from gymnasium.core import ObsType, ActType
import torch_geometric.utils as pyg_utils

from .scheduler import Scheduler
from spark_sched_sim import graph_utils



class NeuralScheduler(Scheduler):
    '''Base class for all neural schedulers'''

    def __init__(
        self,
        name,
        actor,
        obs_wrapper_cls,
        num_executors,
        state_dict_path,
        optim_class,
        optim_lr,
        max_grad_norm
    ):
        super().__init__(name)

        self.actor = actor
        self.obs_wrapper_cls = obs_wrapper_cls
        self.num_executors = num_executors
        self.optim_class = optim_class
        self.optim_lr = optim_lr
        self.max_grad_norm = max_grad_norm
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            self.actor.load_state_dict(state_dict)
        self.lr_scheduler = None


    @property
    def device(self):
        return next(self.actor.parameters()).device


    def train(self):
        '''call only on an instance that is about to be trained'''
        assert self.optim_class and self.optim_lr, 'optimizer options were not provided'
        self.actor.train()
        self.optim = self.optim_class(self.actor.parameters(), lr=self.optim_lr, weight_decay=1e-6)
        # self.lr_scheduler = StepLR(self.optim, 30, gamma=.75)
        # self.lr_scheduler = CosineAnnealingLR(self.optim, 30)


    @torch.no_grad()
    def schedule(self, obs: ObsType) -> ActType:
        dag_batch = graph_utils.obs_to_pyg(obs)
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch['stage_mask']

        dag_batch.to(self.device, non_blocking=True)

        # 1. compute node, dag, and global representations
        h_dict = self.actor.encoder(dag_batch)

        # 2. select a schedulable stage
        stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
        stage_idx, stage_lgprob = self._sample(stage_scores)

        # retrieve index of selected stage's job
        stage_idx_glob = pyg_utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob]

        # 3. select the number of executors to add to that stage, conditioned 
        # on that stage's job
        exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_idx)
        num_exec, exec_lgprob = self._sample(exec_scores)

        action = {
            'stage_idx': stage_idx.item(),
            'job_idx': job_idx.item(),
            'num_exec': num_exec.item()
        }

        lgprob = stage_lgprob + exec_lgprob

        return action, lgprob.item()
    

    @classmethod
    def _sample(cls, logits):
        pi = Categorical(logits=logits)
        samp = pi.sample()
        lgprob = pi.log_prob(samp)
        return samp, lgprob


    def evaluate_actions(self, dag_batch, actions):
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_indices, exec_selections = \
            [col.clone() for col in actions.T]

        num_stage_acts = dag_batch['num_stage_acts']
        num_exec_acts = dag_batch['num_exec_acts']
        num_nodes_per_obs = dag_batch['num_nodes_per_obs']
        obs_ptr = dag_batch['obs_ptr']
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model, this time with grads enabled
        dag_batch.to(self.device)
        h_dict = self.actor.encoder(dag_batch)
        stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
        exec_scores = self.actor.exec_policy_network(
            dag_batch, h_dict, job_indices)

        stage_lgprobs, stage_entropies = self._evaluate(
            stage_scores.cpu(), num_stage_acts, stage_selections)
        
        exec_lgprobs, exec_entropies = self._evaluate(
            exec_scores.cpu(), num_exec_acts, exec_selections)

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + exec_lgprobs

        action_entropies = stage_entropies + exec_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return action_lgprobs, action_entropies
    

    @classmethod
    def _evaluate(cls, scores, counts, selections):
        ptr = counts.cumsum(0)
        ptr = torch.cat([torch.tensor([0]), ptr], 0)
        selections += ptr[:-1]
        probs = pyg_utils.softmax(scores, ptr=ptr)
        probs = clamp_probs(probs)
        log_probs = probs.log()
        selection_log_probs = log_probs[selections]
        entropies = -segment_csr(log_probs * probs, ptr)
        return selection_log_probs, entropies


    def update_parameters(self, loss=None):
        if loss:
            # accumulate gradients
            loss.backward()

        if self.max_grad_norm:
            # clip grads
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    self.max_grad_norm,
                    error_if_nonfinite=True
                )
            except:
                print('infinite grad; skipping update.')
                return

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()



def make_mlp(input_dim, hid_dims, act_fn):
    mlp = nn.Sequential()
    prev_dim = input_dim
    for i, dim in enumerate(hid_dims):
        mlp.append(nn.Linear(prev_dim, dim))
        if i == len(hid_dims) - 1:
            break
        mlp.append(act_fn())
        prev_dim = dim
    return mlp



class StagePolicyNetwork(nn.Module):
    def __init__(self, num_node_features, emb_dims, hid_dims, act_fn):
        super().__init__()
        input_dim = num_node_features + \
            emb_dims['node'] + emb_dims['dag'] + emb_dims['glob']
        self.mlp_score = make_mlp(input_dim, hid_dims, act_fn)


    def forward(self, dag_batch, h_dict):
        stage_mask = dag_batch['stage_mask']

        x = dag_batch.x[stage_mask]

        h_node = h_dict['node'][stage_mask]

        batch_masked = dag_batch.batch[stage_mask]
        h_dag_rpt = h_dict['dag'][batch_masked]

        try:
            num_stage_acts = dag_batch['num_stage_acts'] # batch of obsns
        except:
            num_stage_acts = stage_mask.sum() # single obs

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0)

        # residual connections to original features
        node_inputs = torch.cat(
            [
                x, 
                h_node, 
                h_dag_rpt, 
                h_glob_rpt
            ], 
            dim=1
        )

        node_scores = self.mlp_score(node_inputs).squeeze(-1)
        return node_scores
    


class ExecPolicyNetwork(nn.Module):
    def __init__(self, num_executors, num_dag_features, emb_dims, hid_dims, act_fn):
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        input_dim = num_dag_features + emb_dims['dag'] + emb_dims['glob'] + 1
        self.mlp_score = make_mlp(input_dim, hid_dims, act_fn)

    
    def forward(self, dag_batch, h_dict, job_indices):
        exec_mask = dag_batch['exec_mask']

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, :self.num_dag_features]
        x_dag = x_dag[job_indices]

        h_dag = h_dict['dag'][job_indices]

        try:
            # batch of obsns
            num_exec_acts = dag_batch['num_exec_acts']
        except:
            # single obs
            num_exec_acts = exec_mask.sum()
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)
            exec_mask = exec_mask.unsqueeze(0)

        exec_actions = self._get_exec_actions(exec_mask)

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)

        x_h_dag_rpt = x_h_dag.repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], 
            dim=0)

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], 
            dim=0)
        
        dag_inputs = torch.cat(
            [
                x_h_dag_rpt,
                h_glob_rpt,
                exec_actions
            ], 
            dim=1
        )

        dag_scores = self.mlp_score(dag_inputs).squeeze(-1)
        return dag_scores
    
    
    def _get_exec_actions(self, exec_mask):
        exec_actions = torch.arange(self.num_executors) / self.num_executors
        exec_actions = exec_actions.to(exec_mask.device)
        exec_actions = exec_actions.repeat(exec_mask.shape[0])
        exec_actions = exec_actions[exec_mask.view(-1)]
        exec_actions = exec_actions.unsqueeze(1)
        return exec_actions