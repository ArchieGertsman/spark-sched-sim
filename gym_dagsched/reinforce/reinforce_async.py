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

        returns_sh = torch.zeros(5000)
        wall_times_sh = torch.zeros(5000)

        proc = Process(
            target=run_episodes, 
            args=(
                rank, 
                num_envs, 
                model,
                datagen_state, 
                discount,
                returns_sh,
                wall_times_sh,
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
    returns_sh,
    wall_times_sh,
    optim_type,
    optim_lr,
    conn
):
    '''subprocess function which runs episodes and trains the model 
    by communicating with the parent process'''
    configure_subproc(rank, num_envs)



    






    prof = Profiler()
    # prof = None

    # this subprocess's copy of the environment
    env = DagSchedEnvAsyncWrapper(rank, datagen_state)

    # set up local model and optimizer
    # local_model = deepcopy(model)
    local_model = DDP(model.to(device), device_ids=[device])
    optim = optim_type(local_model.parameters(), lr=optim_lr)

    # streams = []
    # for _ in range(num_envs):
    #     streams += [torch.cuda.Stream(device=device)]
    # s = streams[rank]
    

    i = 0
    while data := conn.recv():
        i += 1

        # receive episode data from parent process
        (   
            n_job_arrivals, 
            n_init_jobs, 
            mjit, 
            n_workers,
            ep_len, 
            entropy_weight
        ) = data


        if prof:
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

        returns = compute_returns(rewards, discount)
        # returns_sh[:ep_len] = returns

        # # notify main proc that returns are ready
        # conn.send(None)

        # # wait for main proc to compute baselines
        # baselines = conn.recv()

        # compute baselines
        baselines = returns.clone()
        dist.all_reduce(baselines)
        baselines /= num_envs
            
        # compute advantages
        advantages = returns - baselines

        # compute loss
        policy_loss = -action_lgprobs @ advantages
        print('POLICY LOSS:', policy_loss)
        entropy_loss = entropy_weight * entropies.sum()
        print('ENTROPY LOSS:', entropy_loss)

        loss = (policy_loss + entropy_loss) / 1e4

        # update model
        optim.zero_grad()
        # with torch.cuda.stream(s):

        loss.backward()


        # for param in local_model.parameters():
        #     dist.all_reduce(param.grad)

        for param in local_model.parameters():
            param.grad.mul_(num_envs)

        optim.step()

        if rank == 0 and i % 10 == 0:
            torch.save(model.state_dict(), 'model.pt')
        

        if prof:
            prof.disable()

        send_ep_stats(conn, loss, env)




def send_ep_stats(conn, loss, env):
    '''sends statistics back to main process for tensorboard 
    logging
    '''
    conn.send((
        loss.item(),
        avg_job_duration(env) * 1e-3,
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
    model
):
    obs = env.reset(n_job_arrivals, n_init_jobs, mjit, n_workers)
    done = False

    action_lgprobs = torch.zeros(ep_len)
    rewards = torch.zeros(ep_len)
    entropies = torch.zeros(ep_len)



    # if rank == 0:
    #     prof = torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=50, warmup=1, active=20),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('log/perf'),
    #         record_shapes=True,
    #         with_stack=True,
    #         profile_memory=True)

    #     prof.start()

    job_ptr = None
    

    for i in range(ep_len):
        if done:
            break

        (node_features,
        new_dag_batch, 
        valid_ops_mask, 
        active_job_ids, 
        n_source_workers) = obs

        if new_dag_batch is not None:
            job_ptr = new_dag_batch.ptr.numpy()
        else:
            assert job_ptr is not None

        op_scores, prlvl_scores = \
            invoke_agent(
                model,
                node_features,
                new_dag_batch,
                valid_ops_mask,
                n_source_workers
            )

        action, action_lgp, entropy = \
            sample_action(
                op_scores, 
                prlvl_scores,
                job_ptr,
                active_job_ids,
            )

        # entropy_scale = 1 / (np.log(n_source_workers * num_ops))
        # entropy = entropy_scale * entropy

        obs, reward, done = env.step(action)

        rewards[i] = reward
        action_lgprobs[i] = action_lgp
        entropies[i] = entropy

    #     if rank == 0:
    #         prof.step()

    # if rank == 0:
    #     prof.stop()

    return action_lgprobs, rewards, entropies    







def invoke_agent(
    agent, 
    node_features,
    new_dag_batch,
    valid_ops_mask,
    n_workers
):
    if new_dag_batch is not None:
        new_dag_batch = new_dag_batch.clone() \
            .to(device, non_blocking=True)

    node_features = node_features \
        .to(device, non_blocking=True)

    op_scores, prlvl_scores = \
        agent(
            node_features,
            new_dag_batch,
            n_workers
        )

    op_scores, prlvl_scores = op_scores.cpu(), prlvl_scores.cpu()

    op_scores[(~valid_ops_mask).nonzero()] = torch.finfo(torch.float).min

    # rows, cols = (~valid_prlvl_msk).nonzero()
    # prlvl_scores[rows, cols] = torch.finfo(torch.float).min

    return op_scores, prlvl_scores







def sample_action(
    op_scores, 
    prlvl_scores, 
    job_ptr,
    active_job_ids,
):
    # operation selection
    op, op_lgp, op_entropy = \
        sample_op(op_scores)

    job_idx, op = \
        translate_op(
            op, 
            job_ptr,
            active_job_ids)

    # parallelism level selection
    prlvl, prlvl_lgp, prlvl_entropy = \
        sample_prlvl(prlvl_scores, job_idx)

    action = (op, prlvl)
    action_lgp = op_lgp + prlvl_lgp
    entropy = op_entropy + prlvl_entropy

    return action, action_lgp, entropy



def translate_op(op, job_ptr, active_jobs_ids):
    job_idx = (op >= job_ptr).sum() - 1

    job_id = active_jobs_ids[job_idx]
    active_op_idx = op - job_ptr[job_idx]
    
    op = (job_id, active_op_idx)

    return job_idx, op




def sample_op(op_scores):
    '''sample index of next operation for each env.
    Returns the operation selections, the log-probability 
    of each selection, and the entropy of policy distribution
    '''
    c = Categorical(logits=op_scores)
    op = c.sample()
    lgp = c.log_prob(op)
    entropy = c.entropy()
    return op.item(), lgp, entropy



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




