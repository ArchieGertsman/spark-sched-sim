import gymnasium as gym
import numpy as np
import torch

from .schedulers import RandomScheduler, DecimaScheduler
from .wrappers import DecimaObsWrapper, DecimaActWrapper
from .graph_utils import collate_dag_batches


EPS = 1e-3

NUM_EXEC = 50

ENV_KWARGS = {
    'num_executors': NUM_EXEC,
    'num_init_jobs': 1,
    'num_job_arrivals': 50,
    'job_arrival_rate': 1/25000,
    'moving_delay': 2000.
}

ENV_ID = 'spark_sched_sim:SparkSchedSimEnv-v0'


def check_obs(env, obs):
    # first compare data of schedulable stages
    msk = obs['schedulable_stage_mask']
    assert np.sum(msk) == len(env.schedulable_stages)
    data = obs['dag_batch']['data']
    valid_nodes = data.nodes[np.nonzero(msk)[0]]
    assert valid_nodes.shape[0] == len(env.schedulable_stages)
    for stage, x in zip(env.schedulable_stages, valid_nodes):
        assert stage.num_remaining_tasks == int(x[0])
        assert np.allclose(x[1], stage.most_recent_duration)

    # now compare all active stages
    i = 0
    for job_id in env.active_job_ids:
        job = env.jobs[job_id]
        active_stages = job.active_stages
        for stage, x in zip(active_stages, data.nodes[i : i + len(active_stages)]):
            assert stage.num_remaining_tasks == int(x[0])
            assert np.allclose(x[1], stage.most_recent_duration)
        i += len(active_stages)



def check_wrapped_obs(env, obs):
    data = obs['dag_batch']['data']
    src_job_id = env.exec_state.source_job_id()
    i = 0
    for job_id in env.active_job_ids:
        active_stages = env.jobs[job_id].active_stages
        for stage, x in zip(active_stages, data.nodes[i : i + len(active_stages)]):
            assert np.allclose(x[0], env.exec_state.num_executors_to_schedule() / env.num_executors)
            assert np.allclose(x[1], 2 * int(job_id == src_job_id) - 1)
            assert np.allclose(x[2], env.exec_state.total_executor_count(job_id) / env.num_executors)
            assert np.allclose(x[3], stage.num_remaining_tasks / 200)
            assert np.allclose(x[4], stage.num_remaining_tasks * stage.most_recent_duration * 1e-5)
        i += len(active_stages)



# def test_obsns():
#     env = gym.make(ENV_ID, **ENV_KWARGS)
#     sched = RandomScheduler()

#     obs, _ = env.reset(seed=0)
#     check_obs(env, obs)

#     done = False
#     while not done:
#         act = sched(obs)
#         obs, _, trunc, term, _ = env.step(act)
#         check_obs(env, obs)
#         done = trunc or term



# def test_wrapper():
#     base_env = gym.make(ENV_ID, **ENV_KWARGS)
#     env = DecimaObsWrapper(DecimaActWrapper(base_env))
#     sched = DecimaScheduler(NUM_EXEC, training_mode=False)

#     obs, _ = env.reset(seed=0)
#     check_wrapped_obs(env, obs)

#     done = False
#     while not done:
#         act, *_ = sched(obs)
#         obs, _, trunc, term, _ = env.step(act)
#         check_wrapped_obs(env, obs)
#         done = trunc or term



def test_collate():
    base_env = gym.make(ENV_ID, **ENV_KWARGS)
    env = DecimaObsWrapper(DecimaActWrapper(base_env))
    sched = DecimaScheduler(NUM_EXEC, training_mode=False)

    obsns = []

    obs, _ = env.reset(seed=0)
    obsns += [obs]

    done = False
    while not done:
        act, *_ = sched(obs)
        obs, _, trunc, term, _ = env.step(act)
        obsns += [obs]
        done = trunc or term

    nested_dag_batch, num_dags_per_obs, num_nodes_per_dag = \
        collate_dag_batches([obs['dag_batch'] for obs in obsns])
    
    assert num_dags_per_obs.numel() == len(obsns)
    
    dag_counter = 0
    node_counter = 0
    for obs, n in zip(obsns, num_dags_per_obs):
        num_nodes = num_nodes_per_dag[dag_counter : dag_counter + n].sum().item()

        # test that node data matches
        x1 = obs['dag_batch']['data'].nodes
        x2 = nested_dag_batch.x[node_counter : node_counter + num_nodes].numpy()
        assert num_nodes == x1.shape[0]
        assert np.allclose(x1, x2)

        # test that edges match
        edges1 = obs['dag_batch']['data'].edge_links
        edges2 = nested_dag_batch.subgraph(torch.arange(node_counter, node_counter + num_nodes)).edge_index.t().numpy()
        assert (edges1 == edges2).all()

        dag_counter += n
        node_counter += num_nodes


