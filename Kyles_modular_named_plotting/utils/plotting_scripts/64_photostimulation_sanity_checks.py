import matplotlib.pyplot as plt
import numpy as np

def plot_cell_64():
    session_idx = 11
    ps_stats_params = {
        'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    }
    exemplar_neuron_idx = 250

    P_VALUE_THRESH = 0.05 # Threshold for significance

    normalize_by_pre = True # Normalize by pre response in the plots

    ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
    ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
    print('CN idx:', data_dict['data']['conditioned_neuron'][session_idx] - 1)

    n_ps_times = ps_fs.shape[0]
    n_neurons = ps_fs.shape[1]
    n_groups = int(np.max(ps_events_group_idxs))

    d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

    resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
        ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=True
    )

    raw_resp_ps_by_group = resp_ps_extras['raw_resp_ps_by_group']

    raw_resp_ps_mean = resp_ps_extras['raw_resp_ps_mean']

    resp_ps_sem = resp_ps_extras['resp_ps_sem']
    resp_ps_pvalues = resp_ps_extras['resp_ps_pvalues']

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4,))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4,))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6,))
    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4,))

    if normalize_by_pre: # Normalize group mean responses by mean pre-PS response for each neuron separately

        for group_idx in range(n_groups): # Do this first or its already been noramlized
            raw_resp_ps_by_group[group_idx] = (
                raw_resp_ps_by_group[group_idx] / 
                np.mean(raw_resp_ps_mean[:, group_idx, IDXS_PRE_PS], axis=-1)[np.newaxis, :, np.newaxis]
            )

        raw_resp_ps_mean = raw_resp_ps_mean / np.mean(raw_resp_ps_mean[:, :, IDXS_PRE_PS], axis=-1, keepdims=True)

    direct_resps_ps = np.zeros((n_neurons, n_ps_times,))  
    indirect_resps_ps = np.zeros((n_neurons, n_ps_times,))  

    for neuron_idx in range(n_neurons):

        # # Plot raw photostim responses to first few photostimulation events
        # ax1.plot(ps_fs[:, neuron_idx, 1], color=c_vals[0])
        # ax1.plot(ps_fs[:, neuron_idx, 2], color=c_vals[1])
        # ax1.plot(ps_fs[:, neuron_idx, 3], color=c_vals[2])

        direct_idxs = np.where(d_ps[neuron_idx, :] < D_DIRECT)[0]
        indirect_idxs = np.where(d_ps[neuron_idx, :] > D_DIRECT)[0]
        #     indirect_idxs = np.where(d_ps[neuron_idx, :] > D_NEAR and d_ps[neuron_idx, :] < D_FAR)[0]

        direct_resps_ps[neuron_idx] = np.nanmean(raw_resp_ps_mean[neuron_idx, direct_idxs, :], axis=0) # Average over direct groups
        indirect_resps_ps[neuron_idx] = np.nanmean(raw_resp_ps_mean[neuron_idx, indirect_idxs, :], axis=0) # Average over indirect groups

        ax1.plot(direct_resps_ps[neuron_idx], color=c_vals_l[0], alpha=0.3, zorder=-5)
        ax1.plot(indirect_resps_ps[neuron_idx], color=c_vals_l[1], alpha=0.3, zorder=-5)

        # # Plot group averaged photostim responses
        # ax1.plot(raw_resp_ps_mean[neuron_idx, 3, :], color=c_vals[0], marker='.')
        # ax1.plot(raw_resp_ps_mean[neuron_idx, 4, :], color=c_vals[1], marker='.')
        # ax1.plot(raw_resp_ps_mean[neuron_idx, 5, :], color=c_vals[2], marker='.')

        if exemplar_neuron_idx == neuron_idx:

            ### Plot of indirect responses for a given neuron ###

            indirect_mean_resps_ps = raw_resp_ps_mean[neuron_idx, indirect_idxs, :] # (group_idxs, ps_times)

            resps_ps = (np.nanmean(indirect_mean_resps_ps[:, IDXS_POST_PS], axis=-1) -  # Post minus pre
                        np.nanmean(indirect_mean_resps_ps[:, IDXS_PRE_PS], axis=-1))

            max_indirect_resp_idxs = np.argsort(resps_ps)[::-1] # Max to min

            my_cmap = plt.cm.get_cmap('bwr')
            vmax = np.max(np.abs(resps_ps))
            vmin = -1 * vmax

            for max_indirect_resp_idx in max_indirect_resp_idxs[::-1]:
                cmap_color = my_cmap((resps_ps[max_indirect_resp_idx] - vmin) / (vmax - vmin))
                ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idx], color=cmap_color)

    #         ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idxs[0]], color=c_vals_l[1]) # Max
    #         ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idxs[-1]], color=c_vals_l[1]) # Min

            if normalize_by_pre:
                ax2.set_ylabel(f'Normalized Fl. - Neuron {neuron_idx}')
            else:
                ax2.set_ylabel(f'Raw Fl - Neuron {neuron_idx}')

            ### Plot of PS responses as a function of distance for a given neuron ###

    #         # Each element is (ps_times, n_neurons, n_events)
    #         raw_resps_ps_neuron = [raw_resp_ps_by_group[group_idx][:, neuron_idx, :] for group_idx in range(n_groups)] 

    #         resps_ps_all = [[] for _ in range(n_groups)]
    #         for group_idx in range(n_groups):
    #             for event_idx in range(raw_resps_ps_neuron[group_idx].shape[-1]):
    #                 resps_ps_all[group_idx].append(
    #                     np.nanmean(raw_resps_ps_neuron[group_idx][IDXS_POST_PS, event_idx], axis=0) - 
    #                     np.nanmean(raw_resps_ps_neuron[group_idx][IDXS_PRE_PS, event_idx], axis=0)
    #                 )

    #         resps_ps_means = np.zeros((n_groups,))
    #         resps_ps_mses = np.zeros((n_groups,))
    #         resps_ps_pvalues = np.zeros((n_groups,))

    #         for group_idx in range(n_groups):
    #             resps_ps_means[group_idx] = np.nanmean(resps_ps_all[group_idx])
    #             resps_ps_mses[group_idx] = np.nanstd(resps_ps_all[group_idx]) / np.sqrt(np.sum(~np.isnan(resps_ps_all[group_idx])))

    #             non_nans = [resps_ps_all[group_idx][idx] for idx in np.where(~np.isnan(resps_ps_all[group_idx]))[0]]
    #             _, pvalue = ttest_1samp(non_nans, 0)
    #             resps_ps_pvalues[group_idx] = pvalue

    #         print(resp_ps_sem[neuron_idx, :3])
    #         print(resps_ps_mses[:3])

    #         print(resp_ps_pvalues[neuron_idx, :3])
    #         print(resps_ps_pvalues[:3])

    #         ax3.errorbar(d_ps[neuron_idx, :], resps_ps_means, resps_ps_mses, fmt='.', color=c_vals_l[2], zorder=-1)
            ax3.errorbar(d_ps[neuron_idx, :], resp_ps[neuron_idx], resp_ps_sem[neuron_idx], fmt='.', color=c_vals_l[2], zorder=-1)
            sig_idxs = np.where(resp_ps_pvalues[neuron_idx] < P_VALUE_THRESH)[0]
            ax3.errorbar(d_ps[neuron_idx, sig_idxs], resp_ps[neuron_idx, sig_idxs], resp_ps_sem[neuron_idx, sig_idxs], fmt='.', color=c_vals[2], zorder=-1)

            ax3.set_xlabel(f'Distance - Neuron {neuron_idx}')
            ax3.set_ylabel(f'PS Resp. - Neuron {neuron_idx}')

            ax3.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
            ax3.axvline(D_DIRECT, color='lightgrey', linestyle='dashed', zorder=-5)
            ax3.axvline(D_FAR, color='lightgrey', linestyle='dashed', zorder=-5)
            ax3.axhline(0.0, color='k', linestyle='dashed', zorder=-3)

    # # Plot mean across all groups and neurons
    # ax1.plot(np.mean(raw_resp_ps_mean, axis=(0, 1)), color='k', marker='.')

    ax1.plot(np.nanmean(direct_resps_ps, axis=0), color=c_vals[0], marker='.', label='direct', zorder=5)
    ax1.plot(np.nanmean(indirect_resps_ps, axis=0), color=c_vals[1], marker='.', label='indirect', zorder=5)

    for ax in (ax1, ax2):
        # Draw dividing lines between sessions
        ax.axvline(np.max(IDXS_PRE_PS) + 0.5, color='lightgrey', linestyle='dashed', zorder=-5)
        ax.axvline(np.min(IDXS_POST_PS) - 0.5, color='lightgrey', linestyle='dashed', zorder=-5)
        ax.axvline(np.max(IDXS_POST_PS) + 0.5, color='lightgrey', linestyle='dashed', zorder=-5)

        ax.axhline(1.0, color='k', linestyle='dashed', zorder=-3)

        ax.set_xlabel('FStim time point')    
        ax.set_xlim((0, np.max(IDXS_POST_PS)+1))

    if normalize_by_pre:
        ax1.set_ylabel('Normalized Fl. (by pre-PS resp.)')
    else:
        ax1.set_ylabel('Raw Fl')

    ax1.set_ylim((0.75, 2.25))
    ax2.set_ylim((0.8, 1.75))
    # ax2.set_ylim((50, 150))

    #### Bin plot #####

    d_bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
    n_bins = len(d_bins) - 1

    n_neurons_bins = np.zeros((n_bins,))
    n_sig_exc_bins = np.zeros((n_bins,))
    n_sig_inh_bins = np.zeros((n_bins,))

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
    plot_cell_64()
