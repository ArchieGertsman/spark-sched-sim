from typing import List, Tuple
import numpy as np


def compute_baselines(
    ts_list: List[np.ndarray], 
    ys_list: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''linearly interpolated baselines

    args:
    - `ts_list`: list of wall time arrays (len: num envs)
    - 'ys_list': list of value (e.g. return) arrays (len: num envs)
    
    returns: list of basline arrays and list of std arrays (both len: num envs)
    '''
    ts_unique = np.unique(np.hstack(ts_list))

    # shape: (num envs, len(ts_unique))
    # y_hats[i, t] is the linear interpolation of 
    # (ts_list[i], ys_list[i]) at time t
    y_hats = np.vstack([np.interp(ts_unique, ts, ys) 
                        for ts, ys in zip(ts_list, ys_list)])

    # find baseline and std at each unique time point
    baselines = {}
    for t, y_hat in zip(ts_unique, y_hats.T):
        baselines[t] = y_hat.mean()

    baselines_list = [np.array([baselines[t] for t in ts]) for ts in ts_list]

    return baselines_list