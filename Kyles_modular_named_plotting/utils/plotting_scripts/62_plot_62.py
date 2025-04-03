import matplotlib.pyplot as plt
import numpy as np

def plot_cell_62():
    random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity
    random_cn_percent_pass = 0.2

    n_pairs = len(day_2_cn_idxs)
    ts_metric = 'post'

    fig1, ax1s = plt.subplots(2, 6, figsize=(24, 8))
    fig2, ax2s = plt.subplots(2, 6, figsize=(24, 8))

    n_random_neurons = 1000

    percentiles = np.zeros((n_random_neurons, n_pairs,))
    cn_idx_idx = 0

    xs = []
    ys = []

    MIN_DIST = 1000
    MIN_GROUPS_INDIRECT = 8

    for pair_idx, ax, axp in zip(range(n_pairs), ax1s.flatten(), ax2s.flatten()):
    #     cn_idx = int(day_1_cn_idxs[pair_idx])
        cn_idx = int(day_2_cn_idxs[pair_idx])

    #     neuron_metric = np.nanmean(day_1_indirects[pair_idx], axis=-1) # Negatively skewed, but not as much as Day 2
    #     neuron_metric = np.nanmean(day_2_indirects[pair_idx], axis=-1) # Quite negatively skewed
    #     neuron_metric = np.nanmean(np.abs(day_2_indirects[pair_idx]), axis=-1) # Kind of positively skewed
    #     neuron_metric = np.nanmean(delta_indirects[pair_idx], axis=-1) # Negatively skewed, barely significant
    #     neuron_metric = np.nanmean(np.abs(delta_indirects[pair_idx]), axis=-1) # Positively skewed, barely significant
    #     neuron_metric = np.nanmean(delta_indirects_pred[pair_idx], axis=-1) # Tiny bit positively skewed
    #     neuron_metric = np.nanmean(np.abs(delta_indirects_pred[pair_idx]), axis=-1) # Quite positively skewed

    #     r_squared_weight = (
    #         day_1_rsquared_indirects[pair_idx] / np.nansum(day_1_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
    #         np.nansum(day_1_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
    #     )
    #     r_squared_weight = (
    #         day_2_rsquared_indirects[pair_idx] / np.nansum(day_2_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
    #         np.nansum(day_2_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
    #     )
    #     r_squared_weight = (
    #         min_rsquared_indirects[pair_idx] / np.nansum(min_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
    #         np.nansum(min_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
    #     )
    #     neuron_metric = np.nanmean(r_squared_weight * day_1_indirects_pred[pair_idx], axis=-1)
    #     neuron_metric = np.nanmean(r_squared_weight * day_2_indirects_pred[pair_idx], axis=-1)
    #     neuron_metric = np.nanmean(r_squared_weight * delta_indirects_pred[pair_idx], axis=-1)
    #     neuron_metric = np.nanmean(r_squared_weight *  np.abs(delta_indirects_pred[pair_idx]), axis=-1) # Quite positively skewed

        # Enforces minimum indirect counts
        print('CN indirect count: {}'.format(indirect_counts[pair_idx][cn_idx]))
        neuron_metric = np.where(indirect_counts[pair_idx] < MIN_GROUPS_INDIRECT, np.nan, neuron_metric)

        assert ~np.isnan(neuron_metric[cn_idx]) # Neuron metric cannot be nan for the CN

        sort_idxs = np.argsort(neuron_metric) # Puts nans at the end

        n_neurons = neuron_metric.shape[0]
        n_nonnan_neurons =  np.sum(~np.isnan(neuron_metric))
        print('Pair {}, percent non-nan: {:.2f}'.format(pair_idx, n_nonnan_neurons/n_neurons))

        neuron_idxs = np.arange(n_neurons)
        nonnan_neuron_idxs = np.arange(n_nonnan_neurons)
        cn_idx_sort_loc = np.where(sort_idxs == cn_idx)[0][0]

        if pair_idx < 11: # Can only plot 11 spots
            ax.scatter(nonnan_neuron_idxs, neuron_metric[sort_idxs][:n_nonnan_neurons], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
            ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, neuron_metric[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
            ax.axhline(0.0, color='lightgrey', zorder=-5)

        # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
        # If the neuron_metric contains nans, doesn't count said neurons in the percentile computation
        neuron_percentiles = np.zeros((n_neurons,))
        for neuron_idx in range(n_neurons):
            if np.isnan(neuron_metric[neuron_idx]):
                neuron_percentiles[neuron_idx] = np.nan
            else:
                neuron_percentiles[neuron_idx] = np.where(sort_idxs == neuron_idx)[0][0] / (n_nonnan_neurons - 1)

        percentiles[cn_idx_idx, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th

        candidate_random_cns = get_candidate_random_cns(
            cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass, 
            prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
        )

        for neuron_idx in range(n_random_neurons - 1): 
            random_neuron_idx = np.random.choice(candidate_random_cns)
            while np.isnan(neuron_percentiles[random_neuron_idx]): # Redraw if nan
                random_neuron_idx = np.random.choice(candidate_random_cns)
            percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[random_neuron_idx]

        dist_to_cn = day_2_dist_to_cn[pair_idx]
        neuron_metric = np.where(dist_to_cn < MIN_DIST, neuron_metric, np.nan)

        if pair_idx < 11: # Can only plot 11 spots
            axp.scatter(dist_to_cn, neuron_metric, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
            axp.scatter(dist_to_cn[cn_idx], neuron_metric[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
            slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, neuron_metric, ax=axp, color=c_vals[PAIR_COLORS[pair_idx]])
            axp.legend()

        if pair_idx == 0 or pair_idx == 6:
            axp.set_ylabel('Neuron metric')
        if pair_idx in (6, 7, 8, 9, 10):
            axp.set_xlabel('Distance to CN (um)')

        ax2s.flatten()[-1].errorbar(neuron_metric[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        xs.append(neuron_metric[cn_idx])
        ys.append(slope)

    ax2s.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    ax2s.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    ax2s.flatten()[-1].set_xlabel('CN neuron metric')
    ax2s.flatten()[-1].set_ylabel('Distance vs. neuron metric slope')
    add_regression_line(xs, ys, ax=ax2s.flatten()[-1], color='k')
    ax2s.flatten()[-1].legend()

    # Mean across pairs
    _, bins, _ = ax1s.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
    sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
    ax1s.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                              label='CN Mean, p={:.3f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    # Median across pairs
    _, bins, _ = ax1s.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
    sort_idxs = np.argsort(np.median(percentiles, axis=-1))
    ax1s.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                               label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    ax1s.flatten()[-1].legend()

if __name__ == '__main__':
    plot_cell_62()
