import matplotlib.pyplot as plt
import numpy as np

def plot_cell_53():
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
        percentiles = np.zeros((n_random_neurons, n_pairs,))

        for pair_idx, ax in zip(range(n_pairs), axs.flatten()):
            cn_idx = int(day_2_cn_idxs[pair_idx])
            ts_slopes = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']
            change_sort_idxs = np.argsort(ts_slopes)

            n_neurons = ts_slopes.shape[0]
            neuron_idxs = np.arange(n_neurons)
            cn_idx_sort_loc = np.where(change_sort_idxs == cn_idx)[0][0]

            ax.scatter(neuron_idxs, ts_slopes[change_sort_idxs], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
            ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, ts_slopes[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
            ax.axhline(0.0, color='lightgrey', zorder=-5)

            # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
            neuron_percentiles = np.zeros((n_neurons,))
            for neuron_idx in range(n_neurons):
                neuron_percentiles[neuron_idx] = np.where(change_sort_idxs == neuron_idx)[0][0] / (n_neurons - 1)

            percentiles[cn_idx_idx, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th

            candidate_random_cns = get_candidate_random_cns(
                cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass,
                prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
            )

            for neuron_idx in range(n_random_neurons - 1): 
                percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[np.random.choice(candidate_random_cns)]

        # Mean across pairs
        _, bins, _ = axs.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
        sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
        axs.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                                  label='CN Mean, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

        # Median across pairs
        _, bins, _ = axs.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
        sort_idxs = np.argsort(np.median(percentiles, axis=-1))
        axs.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                                   label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

        axs.flatten()[-1].legend()

    fig4.suptitle('Tuning Slopes')
    fig5.suptitle('Trial Response Slopes')
    fig6.suptitle('Pre Trial Response Slopes')
    fig7.suptitle('Post Trial Response Slopes')

if __name__ == '__main__':
    plot_cell_53()
