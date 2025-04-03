import matplotlib.pyplot as plt
import numpy as np

def plot_cell_57():
    random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity, similar_prev_day_activity_tuning
    random_cn_percent_pass = 0.2

    n_pairs = len(day_2_cn_idxs)
    ts_metric = 'post'

    fig1, ax1s = plt.subplots(2, 6, figsize=(24, 8))
    fig2, ax2s = plt.subplots(2, 6, figsize=(24, 8))

    n_random_neurons = 1000
    cn_idx_idx = 0

    percentiles = np.zeros((n_random_neurons, n_pairs,))

    xs = []
    ys = []

    for pair_idx, ax, axp in zip(range(n_pairs), ax1s.flatten(), ax2s.flatten()):
        cn_idx = int(day_2_cn_idxs[pair_idx])

    #     neuron_metric = np.nansum(delta_correlations[pair_idx], axis=-1)
        neuron_metric = np.nansum(np.abs(delta_correlations[pair_idx]), axis=-1) # Looks quite significant
    #     neuron_metric = nan_matmul(posts_2[pair_idx], delta_correlations[pair_idx]) # Not really significant

    #     top_changes = posts_2[pair_idx] > np.percentile(posts_2[pair_idx], 66)
    #     top_changes = posts_2[pair_idx] < np.percentile(posts_2[pair_idx], 33)
    #     neuron_metric = np.nansum(delta_correlations[pair_idx][:, top_changes], axis=-1)
        sort_idxs = np.argsort(neuron_metric)

        n_neurons = neuron_metric.shape[0]
        neuron_idxs = np.arange(n_neurons)
        cn_idx_sort_loc = np.where(sort_idxs == cn_idx)[0][0]

        ax.scatter(neuron_idxs, neuron_metric[sort_idxs], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
        ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, neuron_metric[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        ax.axhline(0.0, color='lightgrey', zorder=-5)

        # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
        neuron_percentiles = np.zeros((n_neurons,))
        for neuron_idx in range(n_neurons):
            neuron_percentiles[neuron_idx] = np.where(sort_idxs == neuron_idx)[0][0] / (n_neurons - 1)

        percentiles[0, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th

        candidate_random_cns = get_candidate_random_cns(
            cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass,
            prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
        )

        for neuron_idx in range(n_random_neurons - 1): 
            percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[np.random.choice(candidate_random_cns)]

        dist_to_cn = day_2_dist_to_cn[pair_idx]

        axp.scatter(dist_to_cn, neuron_metric, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
        axp.scatter(dist_to_cn[cn_idx], neuron_metric[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
        slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, neuron_metric, ax=axp, color=c_vals[PAIR_COLORS[pair_idx]])
        axp.legend()

        if pair_idx == 0 or pair_idx == 6:
            axp.set_ylabel('Neuron metric')
        if pair_idx in (6, 7, 8, 9, 10):
            axp.set_xlabel('Distance to CN (um)')

    #         axs.flatten()[-1].scatter(ts_slopes[cn_idx], slope, marker='.', color=c_vals[PAIR_COLORS[pair_idx]])
        ax2s.flatten()[-1].errorbar(neuron_metric[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        xs.append(neuron_metric[cn_idx])
        ys.append(slope)

    ax2s.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    ax2s.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    ax2s.flatten()[-1].set_xlabel('CN neuron metric')
    ax2s.flatten()[-1].set_ylabel('Distance vs. neuron metric slope')
    add_regression_line(xs, ys, ax=ax2s.flatten()[-1], color='k')

    # Mean across pairs
    _, bins, _ = ax1s.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
    sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
    ax1s.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                              label='CN Mean, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    # Median across pairs
    _, bins, _ = ax1s.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
    sort_idxs = np.argsort(np.median(percentiles, axis=-1))
    ax1s.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                               label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    ax1s.flatten()[-1].legend()

if __name__ == '__main__':
    plot_cell_57()
