from enum import Enum, auto
from dataclasses import dataclass

import networkx as nx

class WorkerState(Enum):
    COMMITTED = auto()
    MOVING = auto()
    PRESENT = auto()

    @classmethod
    def valid_update(cls, old_state, new_state):
        # worker remains at same operation but with an updated state
        if old_state is cls.COMMITTED:
            return new_state in [cls.MOVING, cls.PRESENT]
        elif old_state is cls.MOVING:
            return new_state is cls.PRESENT
        return False


class Attr(Enum):
    '''Node and edge attribtues'''
    N_COMMITMENTS = auto()
    COMPLETED = auto()


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
    state: WorkerState


class State:

    def reset(self, n_workers):
        self.G = nx.DiGraph()

        # add all the workers to the graph, and connect
        # them to the 'null' node initially
        self.G.add_edges_from(((WorkerNode(worker_id), NullNode()) for worker_id in n_workers))

        # set inital worker source to be the 'null' node
        # since it initially contains all the workers
        self.curr_source = NullNode()


    def _parse_source(self, job_id, op_id, default=None):
        if op_id is not None:
            return OpNode(job_id, op_id, WorkerState.PRESENT)
        elif job_id is not None:
            return JobNode(job_id)
        else:
            return NullNode() if not default else default


    def update_worker_source(self, job_id=None, op_id=None):
        self.curr_source = self._parse_source(job_id, op_id)


    def add_job(self, job_id):
        self.G.add_node(JobNode(job_id))
        self.mark_job_completed(job_id, False)


    def add_ops(self, job_id, op_ids):
        self.G.add_nodes_from((
            OpNode(job_id, op_id, state)
            for state in WorkerState
            for op_id in op_ids
        ))
        [self.mark_op_completed(job_id, op_id, False) for op_id in op_ids]


    def add_op(self, job_id, op_id):
        self.G.add_nodes_from((
            OpNode(job_id, op_id, state)
            for state in WorkerState
        ))
        self.mark_op_completed(job_id, op_id, False)


    def remove_op(self, job_id, op_id):
        for state in WorkerState:
            self._remove_node(OpNode(job_id, op_id, state)) 
            


    def remove_job(self, job_id):
        self._remove_node(JobNode(job_id))


    def _remove_node(self, node):
        assert self.G.in_degree(node) == 0
        assert self.G.out_degree(node) == 0
        self.G.remove_node(node)


    def n_source_commitments(self, source=None):
        if source is None:
            source = self.curr_source
        return self.G.out_degree(source, weight=Attr.N_COMMITMENTS)


    def get_source_commitments(self):
        return ((dest.job_id, dest.op_id, self.G[self.curr_source][dest][Attr.N_COMMITMENTS]) 
            for dest in self.G.successors(self.curr_source))


    def get_worker_location(self, worker_id):
        worker_node = WorkerNode(worker_id)
        dest_nodes = set(self.G[worker_node])
        assert len(dest_nodes) == 1 # worker should be connected to exactly one node
        node = dest_nodes.pop()
        return node


    def _remove_worker_from_pool(self, worker_id):
        node_old = self.get_worker_location(worker_id)
        # assert isinstance(op_node_old, OpNode)
        self.G.remove_edge(worker_id, node_old)
        return node_old


    def update_worker_state(self, worker_id, new_state):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        assert WorkerState.valid_update(op_node_old.state, new_state)
        op_node_new = OpNode(op_node_old.job_id, op_node_old.op_id, new_state)
        self.G.add_edge(worker_id, op_node_new)


    def move_worker_to_job_pool(self, worker_id):
        # TODO: may need to remove op from graph
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        job_node = JobNode(op_node_old.job_id)
        self.G.add_edge(worker_id, job_node)

        self.remove_node_if_needed(op_node_old)


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


    def add_commitment(self, n_workers, job_id_dst, op_id_dst, job_id_src=None, op_id_src=None):
        assert job_id_dst is not None and op_id_dst is not None

        source = self._parse_source(job_id_src, op_id_src, default=self.curr_source)
        assert self.n_source_commitments(source) + n_workers <= self.n_source_workers(source)

        op_node_dst = OpNode(job_id_dst, op_id_dst, WorkerState.COMMITTED)
        if op_node_dst in self.G[source]:
            self.G[source][op_node_dst][Attr.N_COMMITMENTS] += n_workers
        else:
            self.G.add_edge(source, op_node_dst, n_workers=n_workers)


    def mark_op_completed(self, job_id, op_id, flag=True):
        op_node = OpNode(job_id, op_id, WorkerState.PRESENT)
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


    def is_completed(self, node):
        assert isinstance(node, JobNode) or isinstance(node, OpNode)
        return self.G.nodes[node][Attr.COMPLETED]


    def n_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        return self.G[node_src][op_node_dst][Attr.N_COMMITMENTS]

    def decrement_commitments(self, node_src, op_node_dst):
        assert isinstance(op_node_dst, OpNode)
        assert self.n_commitments(node_src, op_node_dst) > 0
        self.G[node_src][op_node_dst][Attr.N_COMMITMENTS] -= 1



    def remove_node_if_needed(self, node):
        if isinstance(node, OpNode) and self.is_op_completed(node):
            # op has released all of its workers
            # so remove it from the graph
            self.remove_op(node.job_id, node.op_id)
        elif isinstance(node, JobNode) and self.is_job_completed(node):
            self.remove_job(node.job_id)



    def fulfill_commitment(self, worker_id, job_id_dst, op_id_dst):
        worker_node = WorkerNode(worker_id)
        op_node_dst = OpNode(job_id_dst, op_id_dst, WorkerState.COMMITTED)
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
        worker_state = WorkerState.PRESENT if node_src.job_id == job_id_dst \
                  else WorkerState.MOVING
        op_node_dst = OpNode(job_id_dst, op_id_dst, worker_state)
        self.G.add_edge(worker_node, op_node_dst)

        # if source is op or job and has completed and has released 
        # all its workers, then remove it from the graph
        self.remove_node_if_needed(node_src)


    @property
    def all_source_workers_committed(self):
        return self.n_source_commitments() == self.n_source_workers()


    @property
    def null_pool_has_workers(self):
        return self.G.in_degree(NullNode()) > 0


    def reroute_worker(self, worker_id, job_id_new, op_id_new):
        op_node_old = self._remove_worker_from_pool(worker_id)
        assert isinstance(op_node_old, OpNode)
        assert op_node_old.state is WorkerState.MOVING
        worker_node = WorkerNode(worker_id)
        op_node_new = OpNode(job_id_new, op_id_new, WorkerState.MOVING)
        self.G.add_edge(worker_node, op_node_new)


    def peek_commitment(self, job_id, op_id):
        try:
            op_node_src = OpNode(job_id, op_id, WorkerState.PRESENT)
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