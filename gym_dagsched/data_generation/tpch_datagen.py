
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

from .datagen import DataGen
from ..entities.job import Job
from ..entities.operation import Operation


class TPCHDataGen(DataGen):

    def __init__(self, np_random):
        super().__init__(np_random)


    def _job(self, id, t_arrival):
        tpch_sizes = ['2g','5g','10g','20g','50g','80g','100g']
        query_size = tpch_sizes[self.np_random.randint(len(tpch_sizes))]
        
        tpch_num = 22
        query_idx = str(self.np_random.randint(tpch_num) + 1)

        query_path = f'./gym_dagsched/data_generation/tpch/{query_size}/'
        
        adj_matrix = np.load(
            query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
        task_durations = np.load(
            query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True).item()
        
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        assert adj_matrix.shape[0] == len(task_durations)

        n_ops = adj_matrix.shape[0]
        ops = []
        for op_id in range(n_ops):
            task_duration = task_durations[op_id]
            e = next(iter(task_duration['first_wave']))

            num_tasks = len(task_duration['first_wave'][e]) + \
                        len(task_duration['rest_wave'][e])

            # remove fresh duration from first wave duration
            # drag nearest neighbor first wave duration to empty spots
            self._pre_process_task_duration(task_duration)
            rough_duration = np.mean(
                [i for l in task_duration['first_wave'].values() for i in l] + \
                [i for l in task_duration['rest_wave'].values() for i in l] + \
                [i for l in task_duration['fresh_durations'].values() for i in l])

            # generate a node
            # op = Operation(op_id, id, num_tasks, np.array([rough_duration]))
            op = Operation(op_id, id, num_tasks, task_duration, rough_duration)
            ops += [op]

        # generate DAG
        dag = nx.convert_matrix.from_numpy_matrix(
            adj_matrix, create_using=nx.DiGraph)
        for _,_,d in dag.edges(data=True):
            d.clear()
        job = Job(id_=id, ops=ops, dag=dag, t_arrival=t_arrival)
        job.local_workers = set()
        
        return job



    def _pre_process_task_duration(self, task_duration):
        # remove fresh durations from first wave
        clean_first_wave = {}
        for e in task_duration['first_wave']:
            clean_first_wave[e] = []
            fresh_durations = SetWithCount()
            # O(1) access
            for d in task_duration['fresh_durations'][e]:
                fresh_durations.add(d)
            for d in task_duration['first_wave'][e]:
                if d not in fresh_durations:
                    clean_first_wave[e].append(d)
                else:
                    # prevent duplicated fresh duration blocking first wave
                    fresh_durations.remove(d)

        # fill in nearest neighour first wave
        last_first_wave = []
        for e in sorted(clean_first_wave.keys()):
            if len(clean_first_wave[e]) == 0:
                clean_first_wave[e] = last_first_wave
            last_first_wave = clean_first_wave[e]

        # swap the first wave with fresh durations removed
        task_duration['first_wave'] = clean_first_wave








class SetWithCount(object):
    """
    allow duplication in set
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]