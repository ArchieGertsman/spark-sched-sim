import numpy as np

def invalid_time():
        '''invalid time is defined to be infinity.'''
        return to_wall_time(np.inf)


def to_wall_time(t):
        """converts a float to a singleton array to
        comply with the Box space"""
        assert(t >= 0)
        return np.array([t], dtype=np.float32)


def triangle(n):
    return n*(n-1)//2


def mask_to_indices(mask):
    return np.argwhere(mask==1).flatten()