import sys
sys.path.append('./gym_dagsched/data_generation/tpch/')
from time import time
from copy import deepcopy
import os

import torch
from torch.distributions import Categorical
import numpy as np
from scipy.signal import lfilter
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
from torch.nn.parallel import DistributedDataParallel as DDP

from ..envs.dagsched_env_async_wrapper import DagSchedEnvAsyncWrapper
from ..utils.metrics import avg_job_duration
from ..utils.device import device
from ..utils.profiler import Profiler





def train(
    model,
    optim_type,
    optim_lr,
    n_sequences,
    num_envs,
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
    '''trains the model on multiple different job arrival sequences, where
    `n_ep_per_seq` episodes are repeated on each sequence in parallel
    using multiprocessing'''

    # set up torch.distributed for IPC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    datagen_state = np.random.RandomState(69)

    model.to(device)

    procs, conns = \
        launch_subprocesses(
            num_envs, 
            model, 
            datagen_state, 
            optim_type, 
            optim_lr,
            discount)

    mean_ep_len = initial_mean_ep_len
    entropy_weight = entropy_weight_init

    for epoch in range(n_sequences):
        # sample the length of the current episode
        # ep_len = np.random.geometric(1/mean_ep_len)
        # ep_len = max(ep_len, min_ep_len)
        # ep_len = min(ep_len, 4500)
        ep_len = mean_ep_len

        print(f'beginning training on sequence {epoch+1} with ep_len = {ep_len}', flush=True)

        # send episode data to each of the subprocesses, 
        # which starts the episodes
        for conn in conns:
            conn.send((
                n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers, 
                ep_len, 
                entropy_weight))

        # wait for model update
        losses, avg_job_durations, n_completed_jobs_list = \
            list(zip(*[conn.recv() for conn in conns]))

        if writer:
            write_tensorboard(
                writer, 
                epoch, 
                ep_len,
                np.mean(losses),
                np.mean(avg_job_durations),
                np.mean(n_completed_jobs_list)
            )

        # increase the mean episode length
        mean_ep_len += ep_len_growth

        # decrease the entropy weight
        entropy_weight = max(
            entropy_weight - entropy_weight_decay, 
            entropy_weight_min)

    terminate_subprocesses(conns, procs)



def launch_subprocesses(
    num_envs, 
    model, 
    datagen_state, 
    optim_type,
    optim_lr,
    discount
):
    procs = []
    conns = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        proc = Process(
            target=run_episodes, 
            args=(
                rank, 
                num_envs, 
                model,
                datagen_state, 
                discount, 
                optim_type,
                optim_lr,
                conn_sub))

        proc.start()

    return procs, conns



def terminate_subprocesses(conns, procs):
    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]




def configure_subproc(rank, num_envs):
    sys.stdout = open(f'log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    dist.init_process_group('gloo', rank=rank, world_size=num_envs)

    torch.cuda.set_per_process_memory_fraction(1/num_envs, device=device)

    # IMPORTANT! ensures that the different child processes
    # don't all generate the same random numbers. Otherwise,
    # each process would produce an identical episode.
    torch.manual_seed(rank)
    np.random.seed(rank)


def run_episodes(
    rank, 
    num_envs, 
    model,
    datagen_state, 
    discount, 
    optim_type,
    optim_lr,
    conn
):
    '''subprocess function which runs episodes and trains the model 
    by communicating with the parent process'''
    configure_subproc(rank, num_envs)

    prof = Profiler()

    # this subprocess's copy of the environment
    env = DagSchedEnvAsyncWrapper(rank, datagen_state)

    # set up local model and optimizer
    # local_model = deepcopy(model)
    local_model = DDP(model, device_ids=[device])
    optim = optim_type(local_model.parameters(), lr=optim_lr)

    # streams = []
    # for _ in range(num_envs):
    #     streams += [torch.cuda.Stream(device=device)]
    # s = streams[rank]
    

    while data := conn.recv():
        # receive episode data from parent process
        (   
            n_job_arrivals, 
            n_init_jobs, 
            mjit, 
            n_workers,
            ep_len, 
            entropy_weight
        ) = data

        prof.enable()

        # with torch.cuda.stream(s):
        action_lgprobs, rewards, entropies = \
            run_episode(
                rank,
                env,
                n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers,
                ep_len,
                local_model
            )

        torch.cuda.synchronize()

        returns = compute_returns(rewards.detach(), discount)

        # compute baselines
        baselines = returns.clone()
        dist.all_reduce(baselines)
        baselines /= num_envs
            
        # compute advantages
        advantages = returns - baselines

        # compute loss
        policy_loss = -action_lgprobs @ advantages
        entropy_loss = entropy_weight * entropies.sum()
        loss = (policy_loss + entropy_loss) / ep_len

        # update model
        optim.zero_grad()
        # with torch.cuda.stream(s):
        loss.backward()
        # for param in local_model.parameters():
        #     dist.all_reduce(param.grad)
        optim.step()
        

        prof.disable()

        send_ep_stats(conn, loss, env)




def send_ep_stats(conn, loss, env):
    '''sends statistics back to main process for tensorboard 
    logging
    '''
    conn.send((
        loss.item(),
        avg_job_duration(env),
        env.n_completed_jobs))



def write_tensorboard(
    writer, 
    epoch, 
    ep_len, 
    loss,
    avg_job_duration,
    n_completed_jobs
):
    writer.add_scalar('episode length', ep_len, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('avg job duration', avg_job_duration, epoch)
    writer.add_scalar('n completed jobs', n_completed_jobs, epoch)





def run_episode(
    rank,
    env,
    n_job_arrivals,
    n_init_jobs,
    mjit,
    n_workers,
    ep_len,
    agent
):
    obs = env.reset(n_job_arrivals, n_init_jobs, mjit, n_workers)
    done = False

    action_lgprobs = torch.zeros(ep_len)
    rewards = torch.zeros(ep_len)
    entropies = torch.zeros(ep_len)

    


    # if rank == 0:
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=979, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('log/perf'),
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True)

    # prof.start()

    for i in range(ep_len):
        if done:
            break

        op_scores, prlvl_scores = \
            invoke_agent(
                agent, 
                obs,
                n_workers
            )

        action, action_lgp, entropy = \
            sample_action(
                op_scores, 
                prlvl_scores,
                env.active_job_ids, 
                env.op_counts[env.active_job_ids]
            )

        obs, reward, done = env.step(action)

        rewards[i] = reward
        action_lgprobs[i] = action_lgp
        entropies[i] = entropy

    #     # if rank == 0:
    #     prof.step()


    # # if rank == 0:
    # prof.stop()

    return action_lgprobs, rewards, entropies    







def invoke_agent(agent, obs, n_workers):
    dag_batch, op_msk, prlvl_msk = obs

    # num_jobs = torch.tensor([dag_batch.num_graphs], dtype=int, device=device)
    num_jobs = dag_batch.num_graphs
    dag_batch = dag_batch.to(device=device, non_blocking=True)

    op_scores, prlvl_scores, _, _ = \
        agent(
            dag_batch,
            num_jobs,
            n_workers
        )

    op_scores, prlvl_scores = op_scores.cpu(), prlvl_scores.cpu()

    op_scores[(~op_msk).nonzero()] = torch.finfo(torch.float).min

    idx = (~prlvl_msk).nonzero()
    prlvl_scores[idx[:,0], idx[:,1]] = torch.finfo(torch.float).min

    return op_scores, prlvl_scores







def sample_action(op_scores, prlvl_scores, active_job_ids, op_counts):
    # operation selection
    op_idx, op_idx_lgp, op_entropy = sample_op_idx(op_scores)
    job_idx, op_id = translate_op_selection(op_idx, active_job_ids, op_counts)

    # parallelism level selection
    prlvl, prlvl_lgp, prlvl_entropy = sample_prlvl(prlvl_scores, job_idx)

    action = (op_id, prlvl)
    action_lgp = op_idx_lgp + prlvl_lgp
    entropy = op_entropy + prlvl_entropy

    return action, action_lgp, entropy



def translate_op_selection(op_idx, active_job_ids, op_counts):
    cum = torch.cumsum(op_counts, 0)
    job_idx = (op_idx >= cum).sum()

    job_id = active_job_ids[job_idx]
    op_id = op_idx - (cum[job_idx-1].item() if job_idx > 0 else 0)
    op_id = (job_id, op_id)

    return job_idx, op_id




def sample_op_idx(op_scores):
    '''sample index of next operation for each env.
    Returns the operation selections, the log-probability 
    of each selection, and the entropy of policy distribution
    '''
    c = Categorical(logits=op_scores)
    op_idx = c.sample()
    lgp = c.log_prob(op_idx)
    entropy = c.entropy()
    return op_idx.item(), lgp, entropy



def sample_prlvl(prlvl_scores, job_idx):
    '''sample parallelism level for the job of each selected 
    operation for each env.
    Returns the parallelism level selections, the log-probability
    of each selection, and the 
    '''
    prlvl_scores_selected = prlvl_scores[job_idx]
    c = Categorical(logits=prlvl_scores_selected)
    prlvl = c.sample()
    lgp = c.log_prob(prlvl)
    entropy = Categorical(logits=prlvl_scores).entropy().sum()
    return (1 + prlvl).item(), lgp, entropy



def compute_returns(rewards, discount):
    rewards = np.array(rewards)
    r = rewards[...,::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return torch.from_numpy(y[...,::-1].copy()).float()




