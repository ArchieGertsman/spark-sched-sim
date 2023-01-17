from bisect import bisect_left
import numpy as np



def compute_baselines(ts_list, ys_list):
    '''piecewise linear fit baseline'''
    ts_unique = np.unique(np.hstack(ts_list))

    # find baseline value at each unique time point
    baseline_values = \
        {t: np.mean([pw_linear_fit(t, ts, ys)
                     for ts, ys in zip(ts_list, ys_list)])
         for t in ts_unique}

    # output baselines for each env
    baselines_list = \
        [np.array([baseline_values[t] for t in ts])
         for ts in ts_list]

    return baselines_list



def pw_linear_fit(t, ts, ys):
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