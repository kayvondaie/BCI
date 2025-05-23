import matplotlib.pyplot as plt
import numpy as np

def plot_cell_22():
    n_bins = 20
    perc_bins = np.linspace(0, 1, n_bins)

    fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))

    for pair_idx, ax5 in zip(range(n_pairs), ax5s.flatten()):

        percent_nans_1 = np.concatenate(percent_dir_nans_1[pair_idx], axis=0)
        percent_nans_2 = np.concatenate(percent_dir_nans_2[pair_idx], axis=0)
        percent_nans = np.concatenate((percent_nans_1, percent_nans_1), axis=0)

        ax5.hist(percent_nans, bins=perc_bins, color=c_vals[PAIR_COLORS[pair_idx]],)

    fig6, ax6s = plt.subplots(2, 6, figsize=(24, 8))

    for pair_idx, ax6 in zip(range(n_pairs), ax6s.flatten()):

        percent_nans_1 = np.concatenate(percent_event_nans_1[pair_idx], axis=0)
        percent_nans_2 = np.concatenate(percent_event_nans_2[pair_idx], axis=0)
        percent_nans = np.concatenate((percent_nans_1, percent_nans_1), axis=0)

        ax6.hist(percent_nans, bins=perc_bins, color=c_vals[PAIR_COLORS[pair_idx]],)

    fig5.suptitle('Percent of direct neurons in an events that are nans (n_trials_back={})'.format(
         ps_stats_params['resp_ps_n_trials_back_mask']
    ), fontsize=20)

    fig6.suptitle('Percent a events that a direct neuron is nans (n_trials_back={})'.format(
         ps_stats_params['resp_ps_n_trials_back_mask']
    ), fontsize=20)

if __name__ == '__main__':
    plot_cell_22()
