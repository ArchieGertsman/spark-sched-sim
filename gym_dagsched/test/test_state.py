from ..envs.state import *


def test_reset():
    n_workers = 5
    state = State()

    state.reset(n_workers)

    assert len(state.G) == 6
    for worker_id in range(n_workers):
        worker_node = WorkerNode(worker_id)
        assert not state.G.nodes[worker_node][Attr.MOVING]
        assert NullNode() in state.G[worker_node]



def test_add_job():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_job(0)

    assert len(state.G) == (n_workers + 1 + 1)
    job_node = JobNode(0)
    assert job_node in state.G
    assert not state.G.nodes[job_node][Attr.COMPLETED]



def test_add_op():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)
    assert len(state.G) == (n_workers + 1 + 2)

    op_node_src = OpNode(0, 0, OpState.SRC)
    assert op_node_src in state.G
    assert not state.G.nodes[op_node_src][Attr.COMPLETED]

    op_node_dst = OpNode(0, 0, OpState.DST)
    assert op_node_dst in state.G
    assert not state.G.nodes[op_node_dst][Attr.COMPLETED]



def test_remove_job():
    n_workers = 5
    state = State()
    state.reset(n_workers)
    
    state.add_job(0)
    state.remove_job(0)

    assert len(state.G) == (n_workers + 1)
    assert JobNode(0) not in state.G



def test_remove_op():
    n_workers = 5
    state = State()
    state.reset(n_workers)
    
    state.add_op(0, 0)
    state.remove_op(0, 0)

    assert len(state.G) == (n_workers + 1)
    assert OpNode(0, 0, OpState.SRC) not in state.G
    assert OpNode(0, 0, OpState.DST) not in state.G



def test_parse_source():
    assert State.parse_source() == NullNode()
    assert State.parse_source(None, None, JobNode(0)) == JobNode(0)
    assert State.parse_source(0) == JobNode(0)
    assert State.parse_source(0, 0) == OpNode(0, 0, OpState.SRC)



def test_add_commitment():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)

    state.add_commitment(1, 0, 0)

    op_node_dst = OpNode(0, 0, OpState.DST)
    assert op_node_dst in state.G[NullNode()]
    assert state.n_commitments(NullNode(), op_node_dst) == 1

    state.add_commitment(1, 0, 0)
    assert state.n_commitments(NullNode(), op_node_dst) == 2



def test_get_worker_location():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    assert state.get_worker_location(0) == NullNode()



def test_remove_worker_from_pool():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    node_old = state.remove_worker_from_pool(0)
    assert node_old == NullNode()
    assert state.n_source_workers(NullNode()) == (n_workers - 1)



def test_assign_worker_1():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)

    state.remove_worker_from_pool(0)
    state.assign_worker(0, 0, 0, NullNode())

    op_node = OpNode(0, 0, OpState.SRC)

    # make sure the worker is at the new operation
    assert state.get_worker_location(0) == op_node

    # make sure that operation just has this one 
    # worker assigned to it
    assert state.n_source_workers(op_node) == 1

    # worker should be moving to the job since it
    # started out at the null pool
    assert state.is_worker_moving(0)



def test_fulfill_commitment_1():
    '''commitment from null pool to an operation'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)

    # add two commitments from null pool to op
    state.add_commitment(2, 0, 0)

    # fulfill one of them
    state.fulfill_commitment(0, 0, 0)

    # should have only 1 left over
    assert state.n_commitments(NullNode(), OpNode(0, 0, OpState.DST)) == 1

    op_node = OpNode(0, 0, OpState.SRC)

    # make sure the worker is at the new operation
    assert state.get_worker_location(0) == op_node

    # make sure that operation just has this one 
    # worker assigned to it
    assert state.n_source_workers(op_node) == 1
    assert state.n_source_workers(NullNode()) == (n_workers - 1)

    # worker should be moving to the job since it
    # started out at the null pool
    assert state.is_worker_moving(0)



def test_fulfill_commitment_2():
    '''commitment between operations within same job'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_job(0)
    state.add_op(0, 0)
    state.add_op(0, 1)

    state.remove_worker_from_pool(0)
    state.assign_worker(0, 0, 0, NullNode())

    # add one commitment from op 0 to op 1
    state.add_commitment(1, 0, 1, 0, 0)

    # fulfill one of them
    state.fulfill_commitment(0, 0, 1)

    op_node_src = OpNode(0, 0, OpState.SRC)
    op_node_dst = OpNode(0, 1, OpState.DST)

    # no more commitments from op 0 to op 1
    assert state.n_commitments(op_node_src, op_node_dst) == 0

    op_node_dst = OpNode(0, 1, OpState.SRC)

    # make sure the worker is at the new operation
    assert state.get_worker_location(0) == op_node_dst

    # make sure that operation just has this one 
    # worker assigned to it
    assert state.n_source_workers(op_node_dst) == 1
    assert state.n_source_workers(op_node_src) == 0

    # worker should not be moving since the two
    # operations are within the same job
    assert not state.is_worker_moving(0)



def test_move_worker_to_job_pool():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # add a job with one operation
    state.add_job(0)
    state.add_op(0, 0)

    # assign worker 0 to the op
    state.add_commitment(1, 0, 0)
    state.fulfill_commitment(0, 0, 0)

    # move worker 0 from op to job pool
    state.move_worker_to_job_pool(0)

    assert state.get_worker_location(0) == JobNode(0)
    assert state.n_source_workers(JobNode(0)) == 1
    assert state.n_source_workers(OpNode(0, 0, OpState.SRC)) == 0



def test_move_worker_to_null_pool():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # add an operation
    state.add_op(0, 0)

    # assign worker 0 to the op
    state.add_commitment(1, 0, 0)
    state.fulfill_commitment(0, 0, 0)

    # number of workers at null pool should have
    # decreased by 1
    assert state.n_source_workers(NullNode()) == (n_workers - 1)

    # move worker 0 from op to null pool
    state.move_worker_to_null_pool(0)

    assert state.get_worker_location(0) == NullNode()
    assert state.n_source_workers(NullNode()) == n_workers
    assert state.n_source_workers(OpNode(0, 0, OpState.SRC)) == 0



def test_clean_up_nodes_1():
    '''both op and job are completed'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # add a job with one operation
    state.add_job(0)
    state.add_op(0, 0)

    state.mark_job_completed(0)
    state.mark_op_completed(0, 0)

    state.clean_up_nodes(OpNode(0, 0, OpState.SRC))

    assert OpNode(0, 0, OpState.SRC) not in state.G
    assert OpNode(0, 0, OpState.DST) not in state.G
    assert JobNode(0) not in state.G



def test_clean_up_nodes_2():
    '''only op is completed'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # add a job with one operation
    state.add_job(0)
    state.add_op(0, 0)

    # state.mark_job_completed(0)
    state.mark_op_completed(0, 0)

    state.clean_up_nodes(OpNode(0, 0, OpState.SRC))

    assert OpNode(0, 0, OpState.SRC) not in state.G
    assert OpNode(0, 0, OpState.DST) not in state.G
    assert JobNode(0) in state.G



def test_clean_up_nodes_3():
    '''neither job nor op are completed'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # add a job with one operation
    state.add_job(0)
    state.add_op(0, 0)

    # state.mark_job_completed(0)
    # state.mark_op_completed(0, 0)

    state.clean_up_nodes(OpNode(0, 0, OpState.SRC))

    # test for no-op
    assert OpNode(0, 0, OpState.SRC) in state.G
    assert OpNode(0, 0, OpState.DST) in state.G
    assert JobNode(0) in state.G



def test_reroute_worker():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)
    state.add_op(0, 1)

    # send worker from null pool to op 0
    state.add_commitment(1, 0, 0)
    state.fulfill_commitment(0, 0, 0)

    # reroute worker from op 0 to op 1
    state.reroute_worker(0, 0, 1)

    assert state.get_worker_location(0) == OpNode(0, 1, OpState.SRC)
    assert state.n_source_workers(OpNode(0, 1, OpState.SRC)) == 1
    assert state.n_source_workers(OpNode(0, 0, OpState.SRC)) == 0
    assert state.n_source_workers(NullNode()) == (n_workers - 1)



def test_peek_commitment_1():
    '''there is one commitment from null pool to op'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)
    state.add_commitment(1, 0, 0)

    # peek commitment from null pool
    commitment = state.peek_commitment()

    assert commitment is not None

    job_id, op_id = commitment
    assert job_id == 0 and op_id == 0



def test_peek_commitment_2():
    '''no commitments'''
    n_workers = 5
    state = State()
    state.reset(n_workers)

    # peek commitment from null pool
    commitment = state.peek_commitment()

    assert commitment is None



def test_get_source_workers():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    worker_ids = list(state.get_source_workers())
    assert len(worker_ids) == n_workers



def test_get_source_commitments():
    n_workers = 5
    state = State()
    state.reset(n_workers)

    state.add_op(0, 0)
    state.add_commitment(1, 0, 0)

    state.add_op(0, 1)
    state.add_commitment(1, 0, 1)

    commitments = list(state.get_source_commitments())

    assert len(commitments) == 2
    assert commitments[0] == (0, 0, 1)
    assert commitments[1] == (0, 1, 1)
