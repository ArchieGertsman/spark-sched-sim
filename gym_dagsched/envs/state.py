from enum import Enum, auto
from dataclasses import dataclass

import networkx as nx


class OpState(Enum):
    ''' for each operation, there are three nodes
    present in the graph
    '''
    COMMITTED = auto()

    MOVING = auto()

    PRESENT = auto()



class Attr(Enum):
    '''Node and edge attribute keys'''

    # edge attr: number of commitments from src to dst
    NUM_COMMIT = auto()

    # node attr: number of incoming commitments to node
    # NOTE: same as in-degree weighted by `NUM_COMMIT`,
    #       but stored separately for immediate lookup
    NUM_IN_COMMIT = auto()

    # node attr: number of outgoing commitments from node
    # NOTE: same as out-degree weighted by `NUM_COMMIT`,
    #       but stored separately for immediate lookup
    NUM_OUT_COMMIT = auto()



'''classes for each type of node in the graph'''

@dataclass(frozen=True, eq=True)
class WorkerNode:
    worker_id: int


@dataclass(frozen=True, eq=True)
class NullNode:
    pass


@dataclass(frozen=True, eq=True)
class JobNode:
    job_id: int


@dataclass(frozen=True, eq=True)
class OpNode:
    job_id: int
    op_id: int
    state: OpState



def _zero_if_exception(func):
    '''decorator which tries to run
    the function, but returns zero
    in the case of an exception
    '''
    def aux(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return 0
    return aux



class State:
    def reset(self, num_workers):
        self.G = nx.DiGraph()

        self.add_worker_nodes(num_workers)

        self.G.add_nodes_from([(NullNode(), 
                                {Attr.NUM_IN_COMMIT: 0,
                                 Attr.NUM_OUT_COMMIT: 0})])

        # add all the workers to the graph, and connect
        # them to the 'null' node initially
        edge_gen = ((WorkerNode(worker_id), NullNode()) 
                    for worker_id in range(num_workers))
        self.G.add_edges_from(edge_gen)

        # set inital worker source to be the 'null' node
        # since it initially contains all the workers
        self._curr_source = NullNode()



    def add_worker_nodes(self, num_workers):
        node_gen = (WorkerNode(worker_id)
                    for worker_id in range(num_workers))
        self.G.add_nodes_from(node_gen)



    def add_job(self, job_id):
        # using `add_nodes_from` instead of `add_node` 
        # in order to set enum-keyed attributes, because 
        # `add_node` only allows string keys
        node = (JobNode(job_id), 
                {Attr.NUM_IN_COMMIT: 0,
                 Attr.NUM_OUT_COMMIT: 0})
        self.G.add_nodes_from([node])



    def add_op(self, job_id, op_id):
        node_gen = ((OpNode(job_id, op_id, state), 
                     {Attr.NUM_IN_COMMIT: 0,
                      Attr.NUM_OUT_COMMIT: 0})
                    for state in OpState)
        self.G.add_nodes_from(node_gen)



    def remove_op(self, job_id, op_id):
        for state in OpState:
            self._remove_node(OpNode(job_id, 
                                     op_id, 
                                     state)) 
            


    def remove_job(self, job_id):
        self._remove_node(JobNode(job_id))



    def all_source_workers_committed(self):
        return self.num_uncommitted_source_workers() == 0



    def num_uncommitted_source_workers(self):
        return self._num_uncommitted_workers(self._curr_source)



    def null_pool_has_workers(self):
        return self.G.in_degree[NullNode()] > 0



    def source_job(self):
        if isinstance(self._curr_source, NullNode):
            return None
        else:
            return self._curr_source.job_id



    def get_source(self):
        if isinstance(self._curr_source, NullNode):
            return None, None
        elif isinstance(self._curr_source, JobNode):
            return self._curr_source.job_id, None
        else:
            return self._curr_source.job_id, \
                self._curr_source.op_id



    def is_worker_moving(self, worker_id):
        node = self._get_worker_location(worker_id)
        return isinstance(node, OpNode) and \
               node.state == OpState.MOVING



    def num_workers_moving_to_op(self, job_id, op_id):
        op_node = OpNode(job_id, op_id, OpState.MOVING)
        return self._num_workers_moving_to_op(op_node)



    def num_commitments_to_op(self, job_id, op_id):
        op_node = OpNode(job_id, op_id, OpState.COMMITTED)
        return self._num_commitments_to_op(op_node)



    def num_workers_at_source(self):
        return self._num_workers_at(self._curr_source)



    def num_workers_at(self, job_id=None, op_id=None):
        node = self._parse_source(job_id, op_id)
        return self._num_workers_at(node)



    def num_commitments_from(self, job_id=None, op_id=None):
        node = self._parse_source(job_id, op_id)
        return self._num_commitments_from(node)



    def update_worker_source(self, 
                             job_id=None, 
                             op_id=None):
        self._curr_source = \
            State._parse_source(job_id, op_id)



    def get_source_commitments(self):
        ''' returns a generator over all the operations that the
        current source has commitments to; generates tuples
            (job_id, op_id, NUM_COMMIT), 
        where
        - (job_id, op_id) identify an operation which the source
          is committed to
        - NUM_COMMIT is the number of commitments from
          the source to that operation
        '''
        succ = self.G.successors(self._curr_source)
        return [(op_node_dst.job_id, 
                 op_node_dst.op_id, 
                 self._num_commitments(self._curr_source, 
                                       op_node_dst))
                for op_node_dst in succ]



    def move_worker_to_job_pool(self, worker_id):
        op_node_old = self.remove_worker_from_pool(worker_id)
        print('moving worker to job pool:', worker_id, op_node_old)
        assert isinstance(op_node_old, OpNode)
        job_node = JobNode(op_node_old.job_id)
        self.G.add_edge(WorkerNode(worker_id), job_node)



    def move_worker_to_null_pool(self, worker_id):
        print('moving worker to null pool:', worker_id)
        _ = self.remove_worker_from_pool(worker_id)
        self.G.add_edge(WorkerNode(worker_id), NullNode())



    def get_source_workers(self):
        pred = self.G.predecessors(self._curr_source)
        return (worker_node.worker_id 
                for worker_node in pred)



    def add_commitment(self, 
                       num_workers, 
                       job_id_dst, 
                       op_id_dst, 
                       job_id_src=None, 
                       op_id_src=None):
        assert job_id_dst is not None and op_id_dst is not None

        source = State._parse_source(job_id_src, 
                                     op_id_src, 
                                     default=self._curr_source)

        op_node_dst = OpNode(job_id_dst, 
                             op_id_dst, 
                             OpState.COMMITTED)

        if self._num_commitments(source, op_node_dst) == 0:
            self._add_commitment_edge(source, op_node_dst)

        self._increment_commitments(source, 
                                    op_node_dst, 
                                    n=num_workers)



    def fulfill_commitment(self, 
                           worker_id, 
                           job_id_dst, 
                           op_id_dst, 
                           move):
        self.remove_commitment(worker_id, 
                               job_id_dst, 
                               op_id_dst)

        self.remove_worker_from_pool(worker_id)

        # add new connection from worker to dest operation
        self.assign_worker(worker_id, 
                           job_id_dst, 
                           op_id_dst, 
                           move)



    def remove_commitment(self, worker_id, job_id_dst, op_id_dst):
        assert not self.is_worker_moving(worker_id)

        worker_node = WorkerNode(worker_id)
        op_node_dst = OpNode(job_id_dst, 
                             op_id_dst, 
                             OpState.COMMITTED)

        assert nx.has_path(self.G, 
                           worker_node, 
                           op_node_dst)

        node_src = self._get_worker_location(worker_id)

        # update commitment from source to dest op
        self._decrement_commitments(node_src, op_node_dst)
        if self._num_commitments(node_src, op_node_dst) == 0:
            # all commitments are fulfilled from source to dest op
            # so remove their connection
            self.G.remove_edge(node_src, op_node_dst)

        return node_src



    def peek_commitment(self, job_id=None, op_id=None):
        try:
            node_src = State._parse_source(job_id, op_id)
            op_node_dst = next(self.G.successors(node_src))
            return op_node_dst.job_id, op_node_dst.op_id
        except:
            return None



    def assign_worker(self, 
                      worker_id, 
                      job_id_dst, 
                      op_id_dst, 
                      move):
        assert self.G.out_degree[WorkerNode(worker_id)] == 0
        state = OpState.MOVING if move else OpState.PRESENT
        op_node_new = OpNode(job_id_dst, op_id_dst, state)
        self.G.add_edge(WorkerNode(worker_id), op_node_new)



    def remove_worker_from_pool(self, worker_id):
        node_old = self._get_worker_location(worker_id)
        self.G.remove_edge(WorkerNode(worker_id), node_old)
        if not isinstance(node_old, OpNode) or \
           node_old.state == OpState.PRESENT:
            assert self._num_uncommitted_workers(node_old) >= 0
        return node_old



    def mark_worker_present(self, worker_id):
        op_node = self.remove_worker_from_pool(worker_id)

        assert isinstance(op_node, OpNode) and \
               op_node.state == OpState.MOVING

        self.assign_worker(worker_id, 
                           op_node.job_id, 
                           op_node.op_id, 
                           move=False)



    # internal functions

    @classmethod
    def _parse_source(cls, 
                      job_id=None, 
                      op_id=None, 
                      default=None):
        if op_id is not None:
            return OpNode(job_id, op_id, OpState.PRESENT)
        elif job_id is not None:
            return JobNode(job_id)
        else:
            return default if default else NullNode()



    def _remove_node(self, node):
        assert self.G.in_degree[node] == 0
        assert self.G.out_degree[node] == 0
        self.G.remove_node(node)



    def _add_commitment_edge(self, source, op_node_dst):
        edge = (source, op_node_dst, {Attr.NUM_COMMIT: 0})
        self.G.add_edges_from([edge])



    def _decrement_commitments(self, 
                               node_src, 
                               op_node_dst, 
                               n=1):
        assert isinstance(op_node_dst, OpNode)
        assert self._num_commitments(node_src, op_node_dst) > 0
        assert op_node_dst in self.G[node_src]

        self.G[node_src][op_node_dst][Attr.NUM_COMMIT] -= n
        self.G.nodes[node_src][Attr.NUM_OUT_COMMIT] -= n
        self.G.nodes[op_node_dst][Attr.NUM_IN_COMMIT] -= n



    def _increment_commitments(self, 
                               node_src, 
                               op_node_dst, 
                               n=1):
        assert isinstance(op_node_dst, OpNode)

        demand = self._num_commitments_from(node_src)
        supply = self._num_workers_at(node_src)
        assert demand + n <= supply

        assert op_node_dst in self.G[node_src]

        self.G[node_src][op_node_dst][Attr.NUM_COMMIT] += n
        self.G.nodes[node_src][Attr.NUM_OUT_COMMIT] += n
        self.G.nodes[op_node_dst][Attr.NUM_IN_COMMIT] += n



    @_zero_if_exception
    def _num_commitments_from(self, source):
        return self.G.nodes[source][Attr.NUM_OUT_COMMIT]



    @_zero_if_exception
    def _num_commitments_to_op(self, op_node):
        assert isinstance(op_node, OpNode) and \
               op_node.state == OpState.COMMITTED
        return self.G.nodes[op_node][Attr.NUM_IN_COMMIT]


    
    @_zero_if_exception
    def _num_workers_at(self, source):
        if isinstance(source, OpNode):
            assert source.state == OpState.PRESENT
        return self.G.in_degree[source]



    @_zero_if_exception
    def _num_workers_moving_to_op(self, op_node):
        assert isinstance(op_node, OpNode) and \
               op_node.state == OpState.MOVING
        return self.G.in_degree[op_node]



    @_zero_if_exception
    def _num_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        return self.G[node_src][op_node_dst][Attr.NUM_COMMIT]



    def _num_uncommitted_workers(self, source):
        num_uncommitted = \
            self._num_workers_at(source) - \
            self._num_commitments_from(source)
        assert num_uncommitted >= 0
        return num_uncommitted



    def _get_worker_location(self, worker_id):
        worker_node = WorkerNode(worker_id)
        dest_nodes = set(self.G[worker_node])

        # worker should be connected to exactly one node
        assert len(dest_nodes) == 1

        node = dest_nodes.pop()
        return node