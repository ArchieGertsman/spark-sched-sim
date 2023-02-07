from bisect import bisect_left
import numpy as np



def compute_baselines(ts_list, ys_list):
    '''piecewise linear fit baseline

    args:
    - `ts_list`: list of wall time arrays (len: num envs)
    - 'ys_list': list of value (e.g. return) arrays (len: num envs)
    
    returns: list of basline arrays and list of std arrays
    (both len: num envs)
    '''
    ts_unique = np.unique(np.hstack(ts_list))

    # find baseline and std at each unique time point
    baselines = {}
    stds = {}
    for t in ts_unique:
        values = [_pw_linear_fit(t, ts, ys) 
                  for ts, ys in zip(ts_list, ys_list)]
        baselines[t] = np.mean(values)
        stds[t] = np.std(values)

    baselines_list = \
        [np.array([baselines[t] for t in ts])
         for ts in ts_list]

    stds_list = \
        [np.array([stds[t] for t in ts])
         for ts in ts_list]

    return baselines_list, stds_list



def _pw_linear_fit(t, ts, ys):
    '''approximates `y(t)` according to a piecewise 
    linear fit on (ts, ys)
    '''
    idx = bisect_left(ts, t)
    if idx == len(ys):
        idx = -1

    intercept = ys[idx]

    y_hat = intercept

    if idx in [0, -1] or ts[idx] == t:
        return y_hat

    dy = ys[idx] - ys[idx-1]
    dt = ts[idx] - ts[idx-1]
    slope = dy / dt

    y_hat += slope * (t - ts[idx])

    return y_hat