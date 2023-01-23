


_GENERAL_POOL_KEY = (None, None)


class State:
    def reset(self, num_workers):
        # worker id -> pool key
        self._worker_locations = \
            {worker_id: _GENERAL_POOL_KEY
             for worker_id in range(num_workers)}

        # pool key -> set of worker id's
        self._pools = \
            {None: set(),
             _GENERAL_POOL_KEY: set(range(num_workers))}

        # pool key A -> 
        #   (pool key B -> 
        #       number of commitments from 
        #       pool A to pool B)
        self._commitments = \
            {None: {},
             _GENERAL_POOL_KEY: {}}

        # pool key -> total number of outgoing commitments
        self._num_commitments_from = \
            {None: 0,
             _GENERAL_POOL_KEY: 0}

        ## operations only
        # op pool key -> total number of commitments to op
        self._num_commitments_to = {}

        # op pool key -> number of workers moving to op
        self._num_moving_to = {}

        # initial worker source
        self._curr_source = _GENERAL_POOL_KEY



    def add_job(self, job_id):
        job_key = (job_id, None)
        self._pools[job_key] = set()
        self._commitments[job_key] = {}
        self._num_commitments_from[job_key] = 0



    def add_op(self, job_id, op_id):
        op_key = (job_id, op_id)
        self._pools[op_key] = set()
        self._commitments[op_key] = {}
        self._num_commitments_from[op_key] = 0
        self._num_commitments_to[op_key] = 0
        self._num_moving_to[op_key] = 0



    def all_source_workers_committed(self):
        return self.num_uncommitted_source_workers() == 0



    def num_uncommitted_source_workers(self):
        num_uncommitted = \
            len(self._pools[self._curr_source]) - \
            self._num_commitments_from[self._curr_source]
        assert num_uncommitted >= 0
        return num_uncommitted



    def general_pool_has_workers(self):
        return len(self._pools[_GENERAL_POOL_KEY]) > 0



    def source_job(self):
        if self._curr_source in [None, _GENERAL_POOL_KEY]:
            return None
        else:
            return self._curr_source[0]



    def get_source(self):
        return self._curr_source



    def num_workers_moving_to_op(self, job_id, op_id):
        return self._num_moving_to[(job_id, op_id)]



    def num_commitments_to_op(self, job_id, op_id):
        return self._num_commitments_to[(job_id, op_id)]



    def num_workers_at_source(self):
        return len(self._pools[self._curr_source])



    def num_workers_at(self, job_id=None, op_id=None):
        return len(self._pools[(job_id, op_id)])



    def num_commitments_from(self, job_id=None, op_id=None):
        return self._num_commitments_from[(job_id, op_id)]



    def update_worker_source(self, job_id=None, op_id=None):
        self._curr_source = (job_id, op_id)



    def clear_worker_source(self):
        self._curr_source = None



    def get_source_commitments(self):
        return self._commitments[self._curr_source].copy()



    def get_source_workers(self):
        return self._pools[self._curr_source]



    def add_commitment(self, num_workers, dst_pool_key):
        self._increment_commitments(dst_pool_key, n=num_workers)



    def remove_commitment(self, worker_id, dst_pool_key):
        src_pool_key = self._worker_locations[worker_id]

        # update commitment from source to dest op
        self._decrement_commitments(src_pool_key, dst_pool_key)

        return src_pool_key



    def peek_commitment(self, pool_key):
        try:
            return next(iter(self._commitments[pool_key]))
        except:
            # no outgoing commitments from this pool
            return None



    def mark_worker_present(self, worker_id, dst_pool_key):
        self._num_moving_to[dst_pool_key] -= 1
        assert self._num_moving_to[dst_pool_key] >= 0

        self.move_worker_to_pool(worker_id, dst_pool_key)



    def move_worker_to_pool(self, 
                            worker_id, 
                            new_pool_key, 
                            send=False):
        old_pool_key = self._worker_locations[worker_id]
        if old_pool_key is not None:
            self._pools[old_pool_key].remove(worker_id)
            self._worker_locations[worker_id] = None

        if send:
            self._num_moving_to[new_pool_key] += 1
        else:
            self._worker_locations[worker_id] = new_pool_key
            self._pools[new_pool_key].add(worker_id)


    
    # internal methods

    def _increment_commitments(self, dst_pool_key, n=1):
        try:
            self._commitments[self._curr_source][dst_pool_key] += n
        except:
            # key not in dict yet
            self._commitments[self._curr_source][dst_pool_key] = n

        self._num_commitments_from[self._curr_source] += n
        self._num_commitments_to[dst_pool_key] += n

        supply = len(self._pools[self._curr_source])
        demand = self._num_commitments_from[self._curr_source]
        assert supply >= demand


    
    def _decrement_commitments(self, src_pool_key, dst_pool_key):
        self._commitments[src_pool_key][dst_pool_key] -= 1
        self._num_commitments_from[src_pool_key] -= 1
        self._num_commitments_to[dst_pool_key] -= 1

        assert self._num_commitments_from[src_pool_key] >= 0
        assert self._num_commitments_to[dst_pool_key] >= 0

        if self._commitments[src_pool_key][dst_pool_key] == 0:
            self._commitments[src_pool_key].pop(dst_pool_key)