import sys

from attr import dataclass


sys.path.append('./gym_dagsched/data_generation/tpch/')
import os

import torch
from torch.distributions import Categorical
import numpy as np
from scipy.signal import lfilter
import torch.profiler
import torch.distributed as dist
from torch.multiprocessing import Pipe, Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch_geometric.data import Batch

from ..envs.dagsched_env_async_wrapper import DagSchedEnvAsyncWrapper
from ..utils.metrics import avg_job_duration
from ..utils.device import device
from ..utils.profiler import Profiler
from ..utils.baselines import compute_piecewise_linear_fit_baselines
from ..utils.pyg import add_adj





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
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    datagen_state = np.random.RandomState(69)

    (procs, 
    conns, 
    returns_sh_list, 
    wall_times_sh_list,
    baselines_sh_list) = \
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
        max_ep_len = mean_ep_len

        print(f'beginning training on sequence {epoch+1} with ep_len = {max_ep_len}', flush=True)

        # send episode data to each of the subprocesses, 
        # which starts the episodes
        for conn in conns:
            conn.send((
                n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers, 
                max_ep_len, 
                entropy_weight))

        # wait for returns and wall times
        ep_lens = [conn.recv() for conn in conns]


        baselines_list = \
            compute_piecewise_linear_fit_baselines(
                returns_sh_list, 
                wall_times_sh_list,
                ep_lens)


        gen = zip(
            baselines_sh_list, 
            baselines_list, 
            ep_lens, 
            conns)

        # send baselines
        for baselines_sh, baselines, ep_len, conn in gen:
            baselines_sh[:ep_len] = torch.from_numpy(baselines)
            conn.send(None)


        # wait for model update
        losses, avg_job_durations, n_completed_jobs_list, returns_list = \
            list(zip(*[conn.recv() for conn in conns]))


        if writer:
            write_tensorboard(
                writer, 
                epoch, 
                ep_len,
                np.mean(losses),
                np.mean(avg_job_durations),
                np.mean(n_completed_jobs_list),
                np.mean(returns_list)
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

    returns_sh_list = []
    wall_times_sh_list = []
    baselines_sh_list = []

    for rank in range(num_envs):
        conn_main, conn_sub = Pipe()
        conns += [conn_main]

        returns_sh = torch.zeros(5000)
        returns_sh_list += [returns_sh]

        wall_times_sh = torch.zeros(5000)
        wall_times_sh_list += [wall_times_sh]

        baselines_sh = torch.zeros(5000)
        baselines_sh_list += [baselines_sh]

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
                baselines_sh,
                optim_type,
                optim_lr,
                conn_sub))

        proc.start()

    return procs, \
        conns, \
        returns_sh_list, \
        wall_times_sh_list, \
        baselines_sh_list



def terminate_subprocesses(conns, procs):
    [conn.send(None) for conn in conns]
    [proc.join() for proc in procs]




def setup(rank, num_envs):
    sys.stdout = open(f'log/proc/{rank}.out', 'a')

    torch.set_num_threads(1)

    dist.init_process_group('gloo', rank=rank, world_size=num_envs)

    torch.cuda.set_per_process_memory_fraction(1/num_envs, device=device)

    # IMPORTANT! ensures that the different child processes
    # don't all generate the same random numbers. Otherwise,
    # each process would produce an identical episode.
    torch.manual_seed(rank)
    np.random.seed(rank)


def cleanup():
    dist.destroy_process_group()







def run_episodes(
    rank, 
    num_envs, 
    model,
    datagen_state, 
    discount, 
    returns_sh,
    wall_times_sh,
    baselines_sh,
    optim_type,
    optim_lr,
    conn
):
    '''subprocess function which runs episodes and trains the model 
    by communicating with the parent process'''

    setup(rank, num_envs)

    env = DagSchedEnvAsyncWrapper(rank, datagen_state)

    model = DDP(
        model.to(device), 
        device_ids=[device],
        gradient_as_bucket_view=True)

    optim = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim_type,
        lr=optim_lr
    )
    
    i = 0
    while data := conn.recv():
        i += 1

        # receive episode data from parent process
        (n_job_arrivals, 
         n_init_jobs, 
         mjit, 
         n_workers,
         max_ep_len, 
         entropy_weight) = data


        prof = Profiler()
        # prof = None

        if prof:
            prof.enable()


        experience = \
            run_episode(
                rank,
                env,
                n_job_arrivals, 
                n_init_jobs, 
                mjit, 
                n_workers,
                max_ep_len,
                model
            )

        wall_times, rewards = \
            list(zip(*[
                (exp.wall_time, exp.reward) 
                for exp in experience]))

        ep_len = len(experience)

        returns = compute_returns(rewards, discount)

        # write returns and wall times to shared tensors
        returns_sh[:ep_len] = returns
        wall_times_sh[:ep_len] = torch.tensor(wall_times)

        # notify main proc that returns and wall times are ready
        conn.send(ep_len)

        # wait for main proc to compute baselines
        conn.recv()
        baselines = baselines_sh[:ep_len]
            
        # compute advantages
        advantages = returns - baselines


        loss = learn_from_experience(
            model,
            optim,
            experience,
            advantages,
            entropy_weight,
            num_envs
        )


        # if rank == 0 and i % 10 == 0:
        #     torch.save(model.state_dict(), 'model.pt')
        
        if prof:
            prof.disable()

        # send stats
        conn.send((
            loss,
            avg_job_duration(env) * 1e-3,
            env.n_completed_jobs,
            returns[0]))

    cleanup()





def learn_from_experience(
    model,
    optim,
    experience,
    advantages,
    entropy_weight,
    num_envs
):
    model.train()

    batched_model_inputs = \
        extract_batched_model_inputs(experience)

    node_scores_batch, dag_scores_batch = \
        model(*batched_model_inputs)

    action_lgprobs, action_entropies = \
        action_attributes(
            node_scores_batch, 
            dag_scores_batch, 
            experience
        )

    loss = compute_loss(
        action_lgprobs, 
        action_entropies, 
        advantages, 
        entropy_weight
    )
    
    optim.zero_grad()

    loss.backward()

    torch.cuda.synchronize()

    # DDP averages grads over the workers, 
    # while we just want them to be summed,
    # so scale them back.
    scale_grads(model, num_envs)

    optim.step()

    return loss.item()



def scale_grads(model, num_envs):
    for param in model.parameters():
        # TODO: figure out why some grads are None...
        if param.grad is not None:
            param.grad.mul_(num_envs)



def extract_batched_model_inputs(experience):
    (dag_batch_list,
     num_jobs_per_obs, 
     num_source_workers_list) = \
        list(zip(*[
            (exp.dag_batch,
             exp.num_jobs,
             exp.num_source_workers)
            for exp in experience
        ]))

    nested_dag_batch = Batch.from_data_list(dag_batch_list)
    ptr = nested_dag_batch.batch.bincount().cumsum(dim=0)
    nested_dag_batch.ptr = torch.cat([torch.tensor([0]), ptr], dim=0)
    add_adj(nested_dag_batch)
    nested_dag_batch.to(device)

    n_workers_batch = torch.tensor(num_source_workers_list)

    num_jobs_per_obs = torch.tensor(num_jobs_per_obs)

    return nested_dag_batch, n_workers_batch, num_jobs_per_obs




def action_attributes(
    node_scores_batch, 
    dag_scores_batch, 
    experience,
):
    action_lgprobs = []
    action_entropies = []

    gen = zip(
        node_scores_batch,
        dag_scores_batch,
        experience
    )

    for node_scores, dag_scores, exp in gen:
        dag_scores = dag_scores.view(exp.num_jobs, exp.num_source_workers)

        node_selection, dag_idx, dag_selection = exp.action

        node_scores = node_scores.cpu()
        invalid_op_indices = (~exp.valid_ops_mask).nonzero()
        node_scores[invalid_op_indices] = torch.finfo(torch.float).min
        node_lgprob, node_entropy = \
            node_action_attributes(node_scores, node_selection)

        dag_scores = dag_scores.cpu()
        dag_lgprob, dag_entropy = \
            dag_action_attributes(dag_scores, dag_idx, dag_selection)

        num_nodes = node_scores.numel()
        ent_scale = 1 / (exp.num_source_workers * num_nodes)

        action_lgp = node_lgprob + dag_lgprob
        action_ent = ent_scale * (node_entropy + dag_entropy)

        action_lgprobs += [action_lgp]
        action_entropies += [action_ent]

    action_lgprobs = torch.stack(action_lgprobs)
    action_entropies = torch.stack(action_entropies)

    return action_lgprobs, action_entropies

    


def compute_loss(
    action_lgprobs, 
    action_entropies, 
    advantages, 
    entropy_weight
):
    policy_loss = -action_lgprobs @ advantages
    entropy_loss = entropy_weight * action_entropies.sum()

    loss = (policy_loss + entropy_loss) / 1e4

    return loss


def node_action_attributes(node_scores, node_selection):
    c_node = Categorical(logits=node_scores)
    node_lgprob = c_node.log_prob(node_selection)
    node_entropy = c_node.entropy()
    return node_lgprob, node_entropy


def dag_action_attributes(dag_scores, dag_idx, dag_selection):
    dag_lgprob = \
        Categorical(logits=dag_scores[dag_idx]) \
            .log_prob(dag_selection)

    dag_entropy = \
        Categorical(logits=dag_scores) \
            .entropy().sum()
    
    return dag_lgprob, dag_entropy




def write_tensorboard(
    writer, 
    epoch, 
    ep_len, 
    loss,
    avg_job_duration,
    n_completed_jobs,
    returns
):
    writer.add_scalar('episode length', ep_len, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('avg job duration', avg_job_duration, epoch)
    writer.add_scalar('n completed jobs', n_completed_jobs, epoch)
    writer.add_scalar('returns', returns, epoch)





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
    model.eval()

    obs = env.reset(n_job_arrivals, n_init_jobs, mjit, n_workers)
    done = False

    job_ptr = None

    dag_batch_device = None


    experience = []
    

    for i in range(ep_len):
        if done:
            break


        (did_update_dag_batch,
        dag_batch,
        valid_ops_mask, 
        active_job_ids, 
        num_source_workers,
        wall_time) = obs


        if did_update_dag_batch:
            job_ptr = dag_batch.ptr.numpy()
            dag_batch_device = dag_batch.clone() \
                .to(device, non_blocking=True)
        else:
            dag_batch_device.x = dag_batch.x \
                .to(device, non_blocking=True)


        node_scores, dag_scores = \
            invoke_agent(
                model,
                dag_batch_device,
                valid_ops_mask,
                num_source_workers
            )

        raw_action, parsed_action = \
            sample_action(
                node_scores, 
                dag_scores,
                job_ptr,
                active_job_ids,
            )

        # entropy_scale = 1 / (num_source_workers * num_ops)
        # entropy = entropy_scale * entropy


        obs, reward, done = env.step(parsed_action)

        num_jobs = len(active_job_ids)

        experience += [Experience(
            dag_batch.clone(),
            valid_ops_mask,
            num_jobs,
            num_source_workers,
            wall_time,
            raw_action,
            reward
        )]


    return experience




@dataclass
class Experience:
    dag_batch: object
    valid_ops_mask: object
    num_jobs: int
    num_source_workers: int
    wall_time: float
    action: object
    reward: float




def invoke_agent(
    model,
    dag_batch_device,
    valid_ops_mask,
    worker_count
):
    node_scores, dag_scores = \
        model(
            dag_batch_device,
            worker_count
        )

    node_scores = node_scores.cpu()
    node_scores[(~valid_ops_mask).nonzero()] = torch.finfo(torch.float).min

    dag_scores = dag_scores.cpu()
    dag_scores = dag_scores.view(
        dag_batch_device.num_graphs, 
        worker_count)

    return node_scores, dag_scores







def sample_action(
    node_scores, 
    dag_scores, 
    job_ptr,
    active_job_ids,
):
    # select the next operation to schedule
    op_sample = Categorical(logits=node_scores).sample()

    job_idx, op = \
        translate_op(
            op_sample.item(), 
            job_ptr,
            active_job_ids)

    # select the number of workers to schedule
    num_workers_sample = \
        Categorical(logits=dag_scores[job_idx]).sample()

    # action recorded in experience,
    # used for training later
    raw_action = (op_sample, job_idx, num_workers_sample)

    # action sent to env
    parsed_action = (op, 1+num_workers_sample.item())

    return raw_action, parsed_action



def translate_op(op, job_ptr, active_jobs_ids):
    job_idx = (op >= job_ptr).sum() - 1

    job_id = active_jobs_ids[job_idx]
    active_op_idx = op - job_ptr[job_idx]
    
    op = (job_id, active_op_idx)

    return job_idx, op



def compute_returns(rewards_list, discount):
    rewards = np.array(rewards_list)

    r = rewards[...,::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    y = y[...,::-1].copy()

    return torch.from_numpy(y).float()




