import bisect
import numpy as np




def compute_baselines(all_cum_rewards, all_wall_time):
    # do a piece-wise linear fit baseline
    # all_cum_rewards: list of lists of cumulative rewards
    # all_wall_time:   list of lists of physical time
    assert len(all_cum_rewards) == len(all_wall_time)

    # all unique wall time
    unique_wall_time = np.unique(np.hstack(all_wall_time))

    # for find baseline value for all unique time points
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_wall_time))

    # output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)

    return baselines