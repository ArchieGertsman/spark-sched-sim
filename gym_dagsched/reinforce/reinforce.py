import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')
from time import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
from torch_scatter import segment_add_csr
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..envs.vec_dagsched_env import VecDagSchedEnv
from ..utils.metrics import avg_job_duration
from ..utils.device import device



def learn_from_trajectories(
    optim,
    entropy_weight,
    action_lgps_batch, 
    returns_batch, 
    entropies_batch
):
    '''given a list of trajectories from multiple MDP episodes
    that were repeated on a fixed job arrival sequence, update the model 
    parameters using the REINFORCE algorithm as in the Decima paper.
    '''

    # TODO: try normalizing advantages by std
    baselines = returns_batch.mean(axis=0)
    advantages_batch = returns_batch - baselines

    action_lgprobs = action_lgps_batch.flatten()
    advantages = advantages_batch.flatten()

    policy_loss  = -action_lgprobs @ advantages
    entropy_loss = entropy_weight * entropies_batch.sum()

    ep_len = baselines.numel()
    # num_envs = baselines.shape[0]

    optim.zero_grad()
    loss = (policy_loss + entropy_loss) / ep_len
    loss.backward()
    optim.step()

    return loss.item()



def invoke_agent(agent, obs_batch, num_jobs_per_env, num_ops_per_job, n_workers):
    dag_batch, op_msk_batch, prlvl_msk_batch = obs_batch 

    ts = np.zeros(4)

    t = time()
    num_jobs_per_env = num_jobs_per_env.to(device=device)
    dag_batch = dag_batch.to(device=device)
    ts[0] = time() - t

    t = time()
    op_scores_batch, prlvl_scores_batch, num_ops_per_env, env_indptr = \
        agent(
            dag_batch,
            num_jobs_per_env,
            n_workers
        )
    ts[1] = time() - t

    t = time()
    op_scores_batch, prlvl_scores_batch, num_ops_per_env, env_indptr = \
        op_scores_batch.cpu(), prlvl_scores_batch.cpu(), num_ops_per_env.cpu(), env_indptr.cpu()
    ts[2] = time() - t

    t = time()

    op_msk_batch = op_msk_batch[num_ops_per_job[:,None] > torch.arange(op_msk_batch.shape[1])]
    op_scores_batch[(~op_msk_batch).nonzero()] = torch.finfo(torch.float).min
    op_scores_list = torch.split(op_scores_batch, num_ops_per_env.tolist())
    op_scores_batch = pad_sequence(op_scores_list, padding_value=torch.finfo(torch.float).min).t()

    idx = (~prlvl_msk_batch).nonzero()
    prlvl_scores_batch[idx[:,0], idx[:,1]] = torch.finfo(torch.float).min
    ts[3] = time() - t

    return op_scores_batch, prlvl_scores_batch, env_indptr, ts



def sample_action_batch(vec_env, op_scores_batch, prlvl_scores_batch, env_indptr):
    # batched operation selection
    op_idx_batch, op_idx_lgp_batch, op_entropy_batch = \
        sample_op_idx_batch(op_scores_batch)

    job_idx_batch, op_id_batch = vec_env.translate_op_selections(op_idx_batch)

    # batched parallelism level selection
    prlvl_batch, prlvl_lgp_batch, prlvl_entropy_batch = \
        sample_prlvl_batch(prlvl_scores_batch, job_idx_batch, env_indptr)

    action_batch = (op_id_batch, prlvl_batch)
    action_lgp_batch = op_idx_lgp_batch + prlvl_lgp_batch
    entropy_batch = op_entropy_batch + prlvl_entropy_batch

    return action_batch, action_lgp_batch, entropy_batch



def sample_op_idx_batch(op_scores_batch):
    '''sample index of next operation for each env.
    Returns the operation selections, the log-probability 
    of each selection, and the entropy of policy distribution
    '''
    c = Categorical(logits=op_scores_batch)
    op_idx_batch = c.sample()
    lgp_batch = c.log_prob(op_idx_batch)
    entropy_batch = c.entropy()
    return op_idx_batch, lgp_batch, entropy_batch



def sample_prlvl_batch(prlvl_scores_batch, job_idx_batch, env_indptr):
    '''sample parallelism level for the job of each selected 
    operation for each env.
    Returns the parallelism level selections, the log-probability
    of each selection, and the 
    '''
    prlvl_scores_batch_selected = prlvl_scores_batch[job_idx_batch]
    c = Categorical(logits=prlvl_scores_batch_selected)
    prlvl_batch = c.sample()
    lgp_batch = c.log_prob(prlvl_batch)
    entropy_batch = compute_prlvl_entropy_batch(prlvl_scores_batch, env_indptr)
    return prlvl_batch, lgp_batch, entropy_batch



def compute_prlvl_entropy_batch(prlvl_scores_batch, env_indptr):
    entropy = Categorical(logits=prlvl_scores_batch).entropy()
    entropy_batch = segment_add_csr(entropy, env_indptr)
    return entropy_batch



def compute_returns_batch(rewards_batch, discount):
    rewards_batch = np.array(rewards_batch)
    r = rewards_batch[...,::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return torch.from_numpy(y[...,::-1].copy()).float()



def train(
    agent,
    optim,
    n_sequences,
    n_ep_per_seq,
    discount,
    entropy_weight_init,
    entropy_weight_decay,
    entropy_weight_min,
    n_job_arrivals, 
    n_init_jobs, 
    mjit,
    n_workers,
    initial_mean_ep_len,
    ep_len_growth,
    min_ep_len,
    writer
):
    '''train the model on multiple different job arrival sequences'''
    
    print('cuda available:', torch.cuda.is_available())

    vec_env = VecDagSchedEnv(n=n_ep_per_seq)

    agent.to(device)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    # ep_lens = np.zeros(n_sequences)
    # ep_durations = np.zeros(n_sequences)

    vec_env.run()

    df = pd.DataFrame(
        index=np.arange(100,1000+100,100), 
        columns=['total', 't_env', 't_reset', 't_step_total'])

    for epoch in range(n_sequences):
        t_start = time()
        t_policy = 0
        t_sample = 0
        t_env = 0

        # ep_len = np.random.geometric(1/mean_ep_len)
        # ep_len = max(ep_len, min_ep_len)
        ep_len = mean_ep_len

        print(f'beginning training on sequence {epoch+1} with ep_len={ep_len}')

        t = time()
        obs_batch = vec_env.reset(n_job_arrivals, n_init_jobs, mjit, n_workers)
        t_env += time() - t

        prev_episode_stats = vec_env.get_prev_episode_stats()
        if prev_episode_stats is not None:
            write_tensorboard(
                writer,
                epoch,
                loss,
                ep_len,
                prev_episode_stats
            )

        done_batch = torch.zeros(n_ep_per_seq, dtype=torch.bool)

        action_lgps_batch = torch.zeros((vec_env.n, ep_len))
        rewards_batch = torch.zeros((vec_env.n, ep_len))
        entropies_batch = torch.zeros((vec_env.n, ep_len))
        
        participating_idxs = torch.arange(vec_env.n)
        
        ts_total = np.zeros(4)

        i = 0
        while i < ep_len and not done_batch.all():

            t = time()
            op_scores_batch, prlvl_scores_batch, env_indptr, ts = \
                invoke_agent(
                    agent, 
                    obs_batch, 
                    vec_env.num_jobs_per_env,
                    vec_env.num_ops_per_job,
                    n_workers
                )
            t_policy += time() - t
            ts_total += ts

            t = time()
            action_batch, action_lgp_batch, entropy_batch = \
                sample_action_batch(
                    vec_env, 
                    op_scores_batch, 
                    prlvl_scores_batch,
                    env_indptr
                )
            t_sample += time() - t

            t = time()
            obs_batch, reward_batch, done_batch = \
                vec_env.step(action_batch)
            t_env += time() - t

            rewards_batch[:, i] = reward_batch
            action_lgps_batch[participating_idxs, i] = action_lgp_batch
            entropies_batch[participating_idxs, i] = entropy_batch
            
            participating_idxs = (~done_batch).nonzero().flatten()
            
            i += 1

        # print('avg wall time:', np.mean([env.wall_time for env in vec_env.envs]))


        returns_batch = compute_returns_batch(rewards_batch.detach(), discount)

        t = time()
        loss = learn_from_trajectories(
            optim, 
            entropy_weight,
            action_lgps_batch, 
            returns_batch, 
            entropies_batch)
        t_learn = time() - t


        t_total = time() - t_start

        print(f'{t_total:.2f}')
        print(f'{t_policy:.2f}, {t_sample:.2f}, {t_env:.2f}, {t_learn:.2f}')
        print(f'{vec_env.t_reset:.2f}, {vec_env.t_step:.2f}, {vec_env.t_parse:.2f}, {vec_env.t_subbatch:.2f}')
        a = [f'{t:.3f}' for t in ts_total]
        print('ts:', a)

        df.loc[ep_len] = (t_total, t_env, vec_env.t_reset, vec_env.t_step)


        # avg_job_durations = np.array([avg_job_duration(env) for env in vec_env.envs])
        # n_completed_jobs_list = [env.n_completed_jobs for env in vec_env.envs]


        mean_ep_len += ep_len_growth

        entropy_weight = max(
            entropy_weight - entropy_weight_decay, 
            entropy_weight_min)

        # t_total = time() - t_start
        # print(t_total)

        # ep_lens[epoch] = ep_len
        # ep_durations[epoch] = t_total


    # np.save('bruh.npy', np.stack([ep_lens, ep_durations]))

    vec_env.terminate()

    writer.close()

    df.to_csv('timing.csv')




def write_tensorboard(
    writer, 
    epoch, 
    loss, 
    ep_len, 
    prev_episode_stats
):
    avg_job_duration_mean, n_completed_jobs_mean = prev_episode_stats
    print(n_completed_jobs_mean)
    writer.add_scalar('episode length', ep_len, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('avg job duration', avg_job_duration_mean, epoch)
    writer.add_scalar('n completed jobs', n_completed_jobs_mean, epoch)
