import matplotlib.pyplot as plt
import numpy as np

def plot_cell_55():
    random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity, similar_prev_day_activity_tuning
    random_cn_percent_pass = 0.2

    n_pairs = len(day_2_cn_idxs)

    fig4, ax4s = plt.subplots(2, 6, figsize=(24, 8))
    fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))
    fig6, ax6s = plt.subplots(2, 6, figsize=(24, 8))
    fig7, ax7s = plt.subplots(2, 6, figsize=(24, 8))

    n_random_neurons = 1000
    cn_idx_idx = 0

    for ts_metric, axs in zip(
        ('tuning', 'trial_resp', 'pre', 'post'),
        (ax4s, ax5s, ax6s, ax7s)
    ):   
        xs = []
        ys = []

        for pair_idx, ax in zip(range(n_pairs), axs.flatten()):
            cn_idx = int(day_2_cn_idxs[pair_idx])
            ts_slopes = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']
            dist_to_cn = day_2_dist_to_cn[pair_idx]

            ax.scatter(dist_to_cn, ts_slopes, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
            ax.scatter(dist_to_cn[cn_idx], ts_slopes[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
            slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, ts_slopes, ax=ax, color=c_vals[PAIR_COLORS[pair_idx]])

    #         ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, ts_slopes[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
            ax.axhline(0.0, color='lightgrey', zorder=-5)
            ax.legend()

            if pair_idx == 0 or pair_idx == 6:
                ax.set_ylabel('{} slope'.format(ts_metric))
            if pair_idx in (6, 7, 8, 9, 10):
                ax.set_xlabel('Distance to CN (um)')

    #         axs.flatten()[-1].scatter(ts_slopes[cn_idx], slope, marker='.', color=c_vals[PAIR_COLORS[pair_idx]])
            axs.flatten()[-1].errorbar(ts_slopes[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
            xs.append(ts_slopes[cn_idx])
            ys.append(slope)

        axs.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
        axs.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
        axs.flatten()[-1].set_xlabel('CN change')
        axs.flatten()[-1].set_ylabel('Distance vs. change slope')
        add_regression_line(xs, ys, ax=axs.flatten()[-1], color='k')

    fig4.suptitle('Distance to CN vs. (within-session) tuning slopes')
    fig5.suptitle('Distance to CN vs. (within-session) trial response slopes')
    fig6.suptitle('Distance to CN vs. (within-session) pre trial response slopes')
    fig7.suptitle('Distance to CN vs. (within-session) post trial response slopes')

if __name__ == '__main__':
    plot_cell_55()
