import cProfile, pstats, io
from pstats import SortKey

class Profiler:
    def __init__(self):
        self.pr = cProfile.Profile()


    def enable(self):
        self.pr.enable()
        return self


    def disable(self):
        self.pr.disable()

        # log stats
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue(), flush=True)