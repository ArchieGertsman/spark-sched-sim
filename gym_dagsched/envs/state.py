from enum import Enum, auto
from dataclasses import dataclass

import networkx as nx


class OpState(Enum):
    ''' for each operation, there are two nodes
    present in the graph:
    - the 'source' node
        - incomming edges: from workers that
            are currently at or moving to this 
            operation
        - outgoing edges: to 'destination' 
            operations that this operation
            has made commitments to
    - the 'destination' node
        - incoming edges: from 'source' worker pools
            that have commitments to this operation
        - outgoing edges: none
    '''

    # source
    SRC = auto()

    # destination
    DST = auto()



class Attr(Enum):
    '''Node and edge attribute keys'''

    # edge attr: number of commitments from src to dst
    N_COMMITMENTS = auto()

    # node attr: is operation completed?
    COMPLETED = auto()

    # node attr: is the worker en route to some operation?
    MOVING = auto()



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



class State:


    @property
    def all_source_workers_committed(self):
        return self.num_uncommitted_source_workers == 0

    @property
    def num_uncommitted_source_workers(self):
        return self._n_workers_at(self._curr_source) - self._n_commitments_from(self._curr_source)

    @property
    def null_pool_has_workers(self):
        return self.G.in_degree(NullNode()) > 0

    @property
    def source_job(self):
        if isinstance(self._curr_source, NullNode):
            return None
        else:
            return self._curr_source.job_id



    ## reset: call in environment reset method

    def reset(self, n_workers):
        self.G = nx.DiGraph()

        self.add_worker_nodes(n_workers)

        # add all the workers to the graph, and connect
        # them to the 'null' node initially
        self.G.add_edges_from((
            (WorkerNode(worker_id), NullNode()) 
            for worker_id in range(n_workers)
        ))

        # set inital worker source to be the 'null' node
        # since it initially contains all the workers
        self._curr_source = NullNode()



    ## add / remove nodes

    def add_worker_nodes(self, n_workers):
        self.G.add_nodes_from((
            (WorkerNode(worker_id), {Attr.MOVING: False})
            for worker_id in range(n_workers)
        ))



    def add_job(self, job_id):
        # using `add_nodes_from` instead of `add_node` in order to set
        # COMPLETED attribute, which is an enum type, while `add_node` 
        # only allows string keys
        self.G.add_nodes_from([(JobNode(job_id), {Attr.COMPLETED: False})])



    def add_op(self, job_id, op_id):
        self.G.add_nodes_from((
            (OpNode(job_id, op_id, state), {Attr.COMPLETED: False})
            for state in OpState
        ))



    def remove_op(self, job_id, op_id):
        for state in OpState:
            self._remove_node(OpNode(job_id, op_id, state)) 
            


    def remove_job(self, job_id):
        self._remove_node(JobNode(job_id))



    def _remove_node(self, node):
        assert self.G.in_degree(node) == 0
        assert self.G.out_degree(node) == 0
        self.G.remove_node(node)



    def _add_commitment_edge(self, source, op_node_dst):
        self.G.add_edges_from([(source, op_node_dst, {Attr.N_COMMITMENTS: 0})])


    
    # set / get node and edge attributes

    def set_worker_moving(self, worker_id, flag):
        worker_node = WorkerNode(worker_id)
        self.G.nodes[worker_node][Attr.MOVING] = flag



    def is_worker_moving(self, worker_id):
        worker_node = WorkerNode(worker_id)
        return self.G.nodes[worker_node][Attr.MOVING]



    def mark_op_completed(self, job_id, op_id):
        op_node = OpNode(job_id, op_id, OpState.SRC)
        self.G.nodes[op_node][Attr.COMPLETED] = True



    def is_op_completed(self, op_node):
        assert isinstance(op_node, OpNode)
        return self.G.nodes[op_node][Attr.COMPLETED]


    def mark_job_completed(self, job_id):
        job_node = JobNode(job_id)
        self.G.nodes[job_node][Attr.COMPLETED] = True



    def is_job_completed(self, job_node):
        assert isinstance(job_node, JobNode)
        return self.G.nodes[job_node][Attr.COMPLETED]



    def n_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        if op_node_dst in self.G[node_src]:
            return self.G[node_src][op_node_dst][Attr.N_COMMITMENTS]
        else:
            return 0



    def n_workers_at(self, job_id=None, op_id=None):
        node = self._parse_source(job_id, op_id)
        if node not in self.G:
            return 0
        return self._n_workers_at(node)



    def n_workers_at_source(self):
        return self.G.in_degree(self._curr_source)



    def n_commitments_from(self, job_id=None, op_id=None):
        node = self._parse_source(job_id, op_id)
        if node not in self.G:
            return 0
        return self._n_commitments_from(node)



    def n_commitments_to(self, job_id=None, op_id=None):
        node = self._parse_source(job_id, op_id)
        if node not in self.G:
            return 0
        return self._n_commitments_to(node)



    def _decrement_commitments(self, node_src, op_node_dst, n=1):
        assert isinstance(op_node_dst, OpNode)
        assert self.n_commitments(node_src, op_node_dst) > 0
        assert op_node_dst in self.G[node_src]
        self.G[node_src][op_node_dst][Attr.N_COMMITMENTS] -= n



    def _increment_commitments(self, node_src, op_node_dst, n=1):
        assert isinstance(op_node_dst, OpNode)
        assert self._n_commitments_from(node_src) + n <= self._n_workers_at(node_src)
        assert op_node_dst in self.G[node_src]
        self.G[node_src][op_node_dst][Attr.N_COMMITMENTS] += n



    def _n_commitments_from(self, source):
        return self.G.out_degree(source, weight=Attr.N_COMMITMENTS)



    def _n_commitments_to(self, source):
        return self.G.in_degree(source, weight=Attr.N_COMMITMENTS)



    def _n_workers_at(self, source):
        return self.G.in_degree(source)
        



    ## helper methods

    def update_worker_source(self, job_id=None, op_id=None):
        self._curr_source = State._parse_source(job_id, op_id)



    def get_source_commitments(self):
        ''' returns a generator over all the operations that the
        current source has commitments to; generates tuples
            (job_id, op_id, n_commitments), 
        where
        - (job_id, op_id) identify an operation which the source
            is committed to
        - n_commitments is the number of commitments from
            the source to that operation
        '''
        return (
            (
                op_node_dst.job_id, 
                op_node_dst.op_id, 
                self.n_commitments(self._curr_source, op_node_dst)
            ) 
            for op_node_dst in self.G.successors(self._curr_source)
        )



    def move_worker_to_job_pool(self, worker_id):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        job_node = JobNode(op_node_old.job_id)
        self.G.add_edge(WorkerNode(worker_id), job_node)

        # the worker's previous operation may need to be 
        # removed from the graph
        self._clean_up_nodes(op_node_old)



    def move_worker_to_null_pool(self, worker_id):
        _ = self._remove_worker_from_pool(worker_id)
        self.G.add_edge(WorkerNode(worker_id), NullNode())



    def move_all_source_workers(self, is_job_saturated):
        move_func = self.move_worker_to_job_pool if not is_job_saturated \
                else self.move_worker_to_null_pool

        # need to create a frozen list because source 
        # workers change as we iterate
        source_workers = list(self.get_source_workers())

        for worker_id in source_workers:
            move_func(worker_id)



    def get_source_workers(self):
        return (
            worker_node.worker_id 
            for worker_node in self.G.predecessors(self._curr_source)
        )



    def add_commitment(self, 
        n_workers, 
        job_id_dst, 
        op_id_dst, 
        job_id_src=None, 
        op_id_src=None
    ):
        assert job_id_dst is not None and op_id_dst is not None

        source = State._parse_source(job_id_src, op_id_src, default=self._curr_source)

        op_node_dst = OpNode(job_id_dst, op_id_dst, OpState.DST)
        if self.n_commitments(source, op_node_dst) == 0:
            self._add_commitment_edge(source, op_node_dst)

        self._increment_commitments(source, op_node_dst, n=n_workers)



    def fulfill_commitment(self, worker_id, job_id_dst, op_id_dst):
        worker_node = WorkerNode(worker_id)
        op_node_dst = OpNode(job_id_dst, op_id_dst, OpState.DST)
        assert nx.has_path(self.G, worker_node, op_node_dst)

        # remove connection from worker to source
        node_src = self._remove_worker_from_pool(worker_id)

        # update commitment from source to dest op
        self._decrement_commitments(node_src, op_node_dst)
        if self.n_commitments(node_src, op_node_dst) == 0:
            # all commitments are fulfilled from source to dest op
            # so remove their connection
            self.G.remove_edge(node_src, op_node_dst)

        # add new connection from worker to dest operation
        is_worker_present = self._assign_worker(worker_id, job_id_dst, op_id_dst, node_src)

        # nodes may be need to be removed from the graph
        self._clean_up_nodes(node_src)

        return is_worker_present



    def reroute_worker(self, worker_id, job_id_new, op_id_new):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        assert self.is_worker_moving(worker_id)

        is_worker_present = \
            self._assign_worker(worker_id, job_id_new, op_id_new, op_node_old)
        return is_worker_present



    def peek_commitment(self, job_id=None, op_id=None):
        try:
            node_src = State._parse_source(job_id, op_id)
            op_node_dst = next(self.G.successors(node_src))
            return op_node_dst.job_id, op_node_dst.op_id
        except:
            return None



    def _assign_worker(self, worker_id, job_id_dst, op_id_dst, node_old):
        if isinstance(node_old, JobNode) or isinstance(node_old, OpNode):
            job_id_old = node_old.job_id
        else:
            job_id_old = None
        is_worker_present = (job_id_old == job_id_dst)
        self.set_worker_moving(worker_id, not is_worker_present)
        op_node_new = OpNode(job_id_dst, op_id_dst, OpState.SRC)
        self.G.add_edge(WorkerNode(worker_id), op_node_new)
        return is_worker_present



    def _get_worker_location(self, worker_id):
        worker_node = WorkerNode(worker_id)
        dest_nodes = set(self.G[worker_node])

        # worker should be connected to exactly one node
        assert len(dest_nodes) == 1

        node = dest_nodes.pop()
        return node



    def _remove_worker_from_pool(self, worker_id):
        node_old = self._get_worker_location(worker_id)
        self.G.remove_edge(WorkerNode(worker_id), node_old)
        return node_old



    @classmethod
    def _parse_source(cls, job_id=None, op_id=None, default=None):
        if op_id is not None:
            return OpNode(job_id, op_id, OpState.SRC)
        elif job_id is not None:
            return JobNode(job_id)
        else:
            return default if default else NullNode()



    def _clean_up_nodes(self, node):
        ''' if the node is an operation which has
        - completed,
        - released all its workers, and
        - fulfilled all its commitments
        then remove it from the graph. If it's
        the last operation within its job, then
        its job's node should be removed too.
        '''
        if isinstance(node, OpNode) and \
            self.is_op_completed(node) and \
            self.G.in_degree(node) == 0 and \
            self.G.out_degree(node) == 0:

            self.remove_op(node.job_id, node.op_id)

            job_node = JobNode(node.job_id)
            if self.is_job_completed(job_node):
                self.remove_job(node.job_id)