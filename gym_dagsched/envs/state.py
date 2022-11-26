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
        - incoming edges: from 'source' operations
            that have commitments to this operation
        - outgoing edges: none
    '''

    # source
    SRC: auto()

    # destination
    DST: auto()



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
        return self.n_source_commitments() == self.n_source_workers()

    @property
    def null_pool_has_workers(self):
        return self.G.in_degree(NullNode()) > 0



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
        self.curr_source = NullNode()



    ## add / remove nodes

    def add_worker_nodes(self, n_workers):
        self.G.add_nodes_from((
            (worker_id, {Attr.MOVING: False})
            for worker_id in range(n_workers)
        ))



    def add_job(self, job_id):
        self.G.add_node(JobNode(job_id))
        self.mark_job_completed(job_id, False)



    def add_op(self, job_id, op_id):
        self.G.add_nodes_from((
            OpNode(job_id, op_id, state)
            for state in OpState
        ))
        self.mark_op_completed(job_id, op_id, False)



    def remove_op(self, job_id, op_id):
        for state in OpState:
            self._remove_node(OpNode(job_id, op_id, state)) 
            


    def remove_job(self, job_id):
        self._remove_node(JobNode(job_id))



    def _remove_node(self, node):
        assert self.G.in_degree(node) == 0
        assert self.G.out_degree(node) == 0
        self.G.remove_node(node)


    
    # set / get node and edge attributes

    def set_worker_moving(self, worker_id, flag):
        worker_node = WorkerNode(worker_id)
        self.G.nodes[worker_node][Attr.MOVING] = flag



    def is_worker_moving(self, worker_id):
        worker_node = WorkerNode(worker_id)
        return self.G.nodes[worker_node][Attr.MOVING]



    def mark_op_completed(self, job_id, op_id, flag=True):
        op_node = OpNode(job_id, op_id, OpState.SRC)
        self.G.nodes[op_node][Attr.COMPLETED] = flag



    def is_op_completed(self, op_node):
        assert isinstance(op_node, OpNode)
        return self.G.nodes[op_node][Attr.COMPLETED]


    def mark_job_completed(self, job_id, flag=True):
        job_node = JobNode(job_id)
        self.G.nodes[job_node][Attr.COMPLETED] = flag



    def is_job_completed(self, job_node):
        assert isinstance(job_node, JobNode)
        return self.G.nodes[job_node][Attr.COMPLETED]



    def n_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        return self.G[node_src][op_node_dst][Attr.N_COMMITMENTS]



    def decrement_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        assert self.n_commitments(node_src, op_node_dst) > 0
        self.G[node_src][op_node_dst][Attr.N_COMMITMENTS] -= 1



    ## helper methods

    def _parse_source(self, job_id, op_id, default=None):
        if op_id is not None:
            return OpNode(job_id, op_id, OpState.SRC)
        elif job_id is not None:
            return JobNode(job_id)
        else:
            return default if default else NullNode()



    def update_worker_source(self, job_id=None, op_id=None):
        self.curr_source = self._parse_source(job_id, op_id)



    def n_source_commitments(self, source=None):
        if source is None:
            source = self.curr_source
        return self.G.out_degree(source, weight=Attr.N_COMMITMENTS)



    def n_commitments_to_op(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        return self.G[node_src][op_node_dst][Attr.N_COMMITMENTS]



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
                self.n_commitments_to_op(self.curr_source, op_node_dst)
            ) 
            for op_node_dst in self.G.successors(self.curr_source)
        )



    def get_worker_location(self, worker_id):
        worker_node = WorkerNode(worker_id)
        dest_nodes = set(self.G[worker_node])
        # worker should be connected to exactly one node
        assert len(dest_nodes) == 1

        node = dest_nodes.pop()
        return node



    def _remove_worker_from_pool(self, worker_id):
        node_old = self.get_worker_location(worker_id)
        self.G.remove_edge(worker_id, node_old)
        return node_old



    def move_worker_to_job_pool(self, worker_id):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        job_node = JobNode(op_node_old.job_id)
        self.G.add_edge(worker_id, job_node)

        # the worker's previous operation may need to be 
        # removed from the graph
        self.clean_up_nodes(op_node_old)



    def move_worker_to_null_pool(self, worker_id):
        _ = self._remove_worker_from_pool(worker_id)
        self.G.add_edge(WorkerNode(worker_id), NullNode())



    def get_source_workers(self):
        return (
            worker_node.worker_id 
            for worker_node in self.G.predecessors(self.curr_source)
        )



    def n_source_workers(self, source=None):
        if source is None:
            source = self.curr_source
        return self.G.in_degree(self.curr_source)



    def add_commitment(self, 
        n_workers, 
        job_id_dst, 
        op_id_dst, 
        job_id_src=None, 
        op_id_src=None
    ):
        assert job_id_dst is not None and op_id_dst is not None

        source = self._parse_source(job_id_src, op_id_src, default=self.curr_source)
        assert self.n_source_commitments(source) + n_workers <= self.n_source_workers(source)

        op_node_dst = OpNode(job_id_dst, op_id_dst, OpState.DST)
        if op_node_dst in self.G[source]:
            self.G[source][op_node_dst][Attr.N_COMMITMENTS] += n_workers
        else:
            self.G.add_edge(source, op_node_dst, n_workers=n_workers)



    def clean_up_nodes(self, node):
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



    def fulfill_commitment(self, worker_id, job_id_dst, op_id_dst):
        worker_node = WorkerNode(worker_id)
        op_node_dst = OpNode(job_id_dst, op_id_dst, OpState.DST)
        assert nx.has_path(self.G, worker_node, op_node_dst)

        # remove connection from worker to source
        node_src = self._remove_worker_from_pool(worker_id)

        # update commitment from source to dest op
        self.decrement_commitments(node_src, op_node_dst)
        if self.n_commitments(node_src, op_node_dst) == 0:
            # all commitments are fulfilled from source to dest op
            # so remove their connection
            self.G.remove_edge(node_src, op_node_dst)

        # add new connection from worker to dest operation
        worker_is_present = self.assign_worker(worker_id, job_id_dst, op_id_dst)

        # nodes may be need to be removed from the graph
        self.clean_up_nodes(node_src)

        return worker_is_present



    def reroute_worker(self, worker_id, job_id_new, op_id_new):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        assert self.is_worker_moving(worker_id)

        is_worker_present = self.assign_worker(worker_id, job_id_new, op_id_new)
        return is_worker_present



    def assign_worker(self, worker_id, job_id_dst, op_id_dst, job_id_old=None):
        is_worker_present = (job_id_old == job_id_dst)
        self.set_worker_moving(worker_id, not is_worker_present)
        op_node_new = OpNode(job_id_dst, op_id_dst, OpState.SRC)
        self.G.add_edge(WorkerNode(worker_id), op_node_new)
        return is_worker_present



    def peek_commitment(self, job_id, op_id):
        try:
            op_node_src = OpNode(job_id, op_id, OpState.SRC)
            op_node_dst = next(self.G.successors(op_node_src))
            return op_node_dst.job_id, op_node_dst.op_id
        except:
            return None



    def get_worker_op(self, worker_id):
        node = self.get_worker_location(worker_id)
        if isinstance(node, OpNode):
            return node.job_id, node.op_id
        else:
            return None