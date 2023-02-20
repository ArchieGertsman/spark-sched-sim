import cProfile, pstats, io
from pstats import SortKey

class Profiler:
    '''context manager which profiles a block of code,
    then prints out the function calls sorted by cumulative
    execution time
    '''
    def __init__(self):
        self.pr = cProfile.Profile()


    def __enter__(self):
        self.pr.enable()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()

        # log stats
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s) \
                   .sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue(), flush=True)