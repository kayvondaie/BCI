import matplotlib.pyplot as plt
import numpy as np

def plot_cell_66():
    session_idxs = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19,)
    ps_stats_params = {
        'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    }

    P_VALUE_THRESH = 0.05 # Threshold for significance

    d_ps_all = []
    resp_ps_all = []
    ps_resp_pvals_all = []

    for session_idx in session_idxs:

        print(f'Session idx: {session_idx}')
        ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
        ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)

        n_ps_times = ps_fs.shape[0]
        n_neurons = ps_fs.shape[1]
        n_groups = int(np.max(ps_events_group_idxs))

        d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
        d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

        resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
            ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=True
        )

        d_ps_all.append(d_ps)
        resp_ps_all.append(resp_ps)
        ps_resp_pvals_all.append(resp_ps_extras['resp_ps_pvalues'])

    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4,))

    #### Bin plot #####

    d_bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
    n_bins = len(d_bins) - 1

    n_neurons_bins = np.zeros((n_bins,))
    n_sig_exc_bins = np.zeros((n_bins,))
    n_sig_inh_bins = np.zeros((n_bins,))

    for session_idx_idx in range(len(d_ps_all)):

        d_ps = d_ps_all[session_idx_idx]
        resp_ps = resp_ps_all[session_idx_idx]
        ps_resp_pvals = ps_resp_pvals_all[session_idx_idx]

        n_neurons = d_ps.shape[0]
        for neuron_idx in range(n_neurons):
            bin_idxs = np.digitize(d_ps[neuron_idx], d_bins) - 1

            for bin_idx in range(n_bins):
                n_neurons_bins[bin_idx] += np.where(bin_idxs == bin_idx)[0].shape[0]

            sig_idxs = np.where(ps_resp_pvals[neuron_idx] < P_VALUE_THRESH)[0]

            sig_ds = d_ps[neuron_idx, sig_idxs]
            sig_resp_ps = resp_ps[neuron_idx, sig_idxs]

            bin_idxs = np.digitize(sig_ds, d_bins) - 1

            for bin_idx in range(n_bins):
                n_sig_exc_bins[bin_idx] += np.where(
                    np.logical_and(bin_idxs == bin_idx, sig_resp_ps > 0.)
                )[0].shape[0]
                n_sig_inh_bins[bin_idx] += np.where(
                    np.logical_and(bin_idxs == bin_idx, sig_resp_ps < 0.)
                )[0].shape[0]

    bin_widths = np.array(d_bins[1:]) - np.array(d_bins[:-1])
    bin_locs = d_bins[:-1] + bin_widths / 2
    _ = ax4.bar(bin_locs, n_sig_exc_bins / n_neurons_bins, width=bin_widths, color=c_vals[0])
    _ = ax4.bar(bin_locs, n_sig_inh_bins / n_neurons_bins, width=bin_widths, color=c_vals[1],
                bottom = n_sig_exc_bins / n_neurons_bins)

    ax4.set_xlabel('Distance from neuron to PS group')
    ax4.set_ylabel('Percent significant (p < {})'.format(P_VALUE_THRESH))

    ax4.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=3)
    ax4.axvline(D_DIRECT, color='lightgrey', linestyle='dashed', zorder=3)
    ax4.axvline(D_FAR, color='lightgrey', linestyle='dashed', zorder=3)
    ax4.axhline(0.0, color='k', linestyle='dashed', zorder=3)
    ax4.axhline(P_VALUE_THRESH / 2, color='k', linestyle='dashed', zorder=3)
    ax4.axhline(P_VALUE_THRESH, color='k', linestyle='dashed', zorder=3)

    # ax4.plot(n_neurons_bins)
    # ax4.scatter(d_ps[neuron_idx, sig_idxs], resp_ps[neuron_idx, sig_idxs], marker='.', color=c_vals[2])

if __name__ == '__main__':
    plot_cell_66()
