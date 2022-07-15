import sys

from gym_dagsched.data_generation.tpch_datagen import TPCHDataGen
sys.path.append('./gym_dagsched/data_generation/tpch/')
from multiprocessing import Process, SimpleQueue

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.multiprocessing import set_start_method

from gym_dagsched.envs.dagsched_env import DagSchedEnv
from gym_dagsched.policies.heuristics import fcfs, max_children, srt, lrt
from gym_dagsched.policies.decima_agent import ActorNetwork
from gym_dagsched.utils.metrics import avg_job_duration, makespan
from gym_dagsched.data_generation.random_datagen import RandomDataGen
from gym_dagsched.reinforce.reinforce_base import sample_action
from gym_dagsched.utils.device import device



def decima(env, policy):
    obs = env._observe()
    if obs is None or env.n_active_jobs == 0:
        next_op, prlvl = None, 0
    else:
        dag_batch, op_msk, prlvl_msk = obs
        num_ops_per_dag = dag_batch.num_ops_per_dag
        ops_probs, prlvl_probs = policy(
            dag_batch.to(device=device), 
            num_ops_per_dag.to(device=device),
            op_msk, 
            prlvl_msk)
        next_op, prlvl, _, _ = \
            sample_action(env, ops_probs, prlvl_probs)
    return next_op, prlvl



def launch_subprocesses(heurs, models):
    procs = []
    in_qs = []
    out_qs = []
    for heur, model in zip(heurs, models):
        in_q = SimpleQueue()
        in_qs += [in_q]
        out_q = SimpleQueue()
        out_qs += [out_q]
        proc = Process(
            target=episode_runner, 
            args=(in_q, out_q, heur, model))
        proc.start()
    return procs, in_qs, out_qs



def terminate_subprocesses(in_qs, procs):
    for in_q in in_qs:
        in_q.put(None)

    for proc in procs:
        proc.join()



def episode_runner(in_q, out_q, heur, model):
    env = DagSchedEnv()

    while episode_data := in_q.get():
        initial_timeline, workers = episode_data

        env.reset(initial_timeline, workers)
        done = False
        while not done:
            next_op, n_workers = heur(env) if model is None else heur(env, model)
            _, _, done = env.step(next_op, n_workers)

        out_q.put(avg_job_duration(env))



if __name__ == '__main__':
    set_start_method('spawn')

    n_workers = 10

    policy0 = ActorNetwork(5, 8, n_workers)
    policy0.load_state_dict(torch.load('policy0.pt'))
    policy0.eval()
    policy0.to(device)

    policy1 = ActorNetwork(5, 8, n_workers)
    policy1.load_state_dict(torch.load('policy1.pt'))
    policy1.eval()
    policy1.to(device)

    datagen = RandomDataGen(
        max_ops=8,
        max_tasks=4,
        mean_task_duration=2000.,
        n_worker_types=1)
    # datagen = TPCHDataGen()

    n_seq = 10

    heurs = [srt, decima, decima]
    models = [None, policy0, policy1]
    heur_names = ['srt', 'decima0', 'decima1']

    ajds = np.zeros((n_seq, len(heurs)))

    procs, in_qs, out_qs = \
        launch_subprocesses(heurs, models)

    for i in range(n_seq):
        initial_timeline = datagen.initial_timeline(
            n_job_arrivals=100, n_init_jobs=0, mjit=2000.)

        workers = datagen.workers(n_workers=n_workers)

        for in_q in in_qs:
            in_q.put((initial_timeline, workers))

        ajd_list = [out_q.get() for out_q in out_qs]
        ajds[i] = np.array(ajd_list)
        print(ajds[i])

    seq_nums = np.arange(n_seq)

    for i,name in enumerate(heur_names):
        plt.plot(seq_nums, ajds[:,i], label=name)
    plt.legend()
    plt.title('AJD for Various Job Sequences')
    plt.xlabel('Sequence Number')
    plt.ylabel('Average Job Duration')
    plt.savefig('bruhfig.png')

    terminate_subprocesses(in_qs, procs)