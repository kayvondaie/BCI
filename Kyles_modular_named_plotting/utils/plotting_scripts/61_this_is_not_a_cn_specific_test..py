import matplotlib.pyplot as plt
import numpy as np

def plot_cell_61():
    pair_idx = 8

    n_bins = 5

    group_sort = 'delta_tuning' 
    neuron_sort = None # delta_cc, delta_tuning

    n_groups = day_2_indirects[pair_idx].shape[-1]
    group_bins = np.round(np.linspace(0, n_groups, n_bins+1)).astype(np.int32)

    n_neurons = day_2_indirects[pair_idx].shape[0]
    neuron_bins = np.round(np.linspace(0, n_neurons, n_bins+1)).astype(np.int32)

    print('group_bins:', group_bins)
    print('neuron_bins:', neuron_bins)

    # direct_resp = direct_masks[pair_idx] * 1/2 * (
    #     day_1_resp_pss[pair_idx] + day_2_resp_pss[pair_idx]
    # ) 

    # direct_resp = direct_masks[pair_idx] * 1/2 * (
    #     day_1_resp_ps_preds[pair_idx] + day_2_resp_ps_preds[pair_idx]
    # ) 

    direct_resp = direct_masks[pair_idx] / np.nansum(direct_masks[pair_idx], axis=0) # Just include all direct, normalizing for count

    # indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_pss[pair_idx] - day_1_resp_pss[pair_idx])
    # indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_ps_preds[pair_idx] - day_1_resp_ps_preds[pair_idx])
    indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_ps_preds[pair_idx])

    if group_sort == 'delta_tuning':
        # ts_metric = 'tuning'
        ts_metric = 'post'
        # ts_metric = 'trial_resp'
        # ts_metric = 'pre'
        ts_metric_change = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']

        group_metric = nan_matmul(ts_metric_change, direct_resp)
    else:
        raise NotImplementedError('group_sort {} not recognized!')

    neuron_metric = None
    if neuron_sort is not None:
        if neuron_sort == 'delta_cc':
            neuron_metric = np.nanmean(indirect_delta_resp, axis=-1)
        else:
            raise NotImplementedError('neuron_sort {} not recognized!')

    # group_day_2_post[pair_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    group_sort_idxs = np.argsort(group_metric)
    group_bin_idxs = []

    for bin_idx in range(n_bins):
        group_bin_idxs.append(group_sort_idxs[group_bins[bin_idx]:group_bins[bin_idx+1]])

    neuron_sort_idxs = np.argsort(neuron_metric)
    neuron_bin_idxs = []

    for bin_idx in range(n_bins):
        neuron_bin_idxs.append(neuron_sort_idxs[neuron_bins[bin_idx]:neuron_bins[bin_idx+1]])

    neuron_metric_bin_means = np.zeros((n_bins, n_bins)) # group, neuron_idx

    running_group_count = 0

    bin_x = []
    bin_y_1 = []
    bin_y_se_1 = []
    bin_y_2 = []
    bin_y_se_2 = []

    for group_bin_idx in range(n_bins):

        n_bin_groups = indirect_delta_resp[:, group_bin_idxs[group_bin_idx]].shape[-1]

        bin_x.append(running_group_count + n_bin_groups/2)

        # average over groups first
        group_indirect_delta_resp = np.nanmean(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]], axis=0)
        group_indirect_delta_resp_std = (np.nanstd(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]], axis=0) /
                                         np.sqrt(np.nansum(~np.isnan(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)))

        ax1.errorbar(np.arange(n_bin_groups) + running_group_count, group_indirect_delta_resp, 
                     yerr=group_indirect_delta_resp_std, linestyle='None',
                     marker='.', color=c_vals_l[group_bin_idx])

        bin_y_1.append(np.nanmean(group_indirect_delta_resp)) # Mean across groups in bin
        bin_y_se_1.append(np.nanstd(group_indirect_delta_resp) / np.sqrt(n_bin_groups))

        group_indirect_abs_delta_resp = np.nanmean(np.abs(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)
        group_indirect_abs_delta_resp_std = (np.nanstd(np.abs(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0) /
                                             np.sqrt(np.nansum(~np.isnan(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)))

        ax2.errorbar(np.arange(n_bin_groups) + running_group_count, group_indirect_abs_delta_resp, 
                     yerr=group_indirect_abs_delta_resp_std, linestyle='None',
                     marker='.', color=c_vals_l[group_bin_idx])

        bin_y_2.append(np.nanmean(group_indirect_abs_delta_resp)) # Mean across groups in bin
        bin_y_se_2.append(np.nanstd(group_indirect_abs_delta_resp) / np.sqrt(n_bin_groups))

        running_group_count += n_bin_groups

    ax1.errorbar(bin_x, bin_y_1, yerr=bin_y_se_1, color='k', zorder=5)
    ax2.errorbar(bin_x, bin_y_2, yerr=bin_y_se_2, color='k', zorder=5)

    for ax in (ax1, ax2):
        ax.axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)

if __name__ == '__main__':
    plot_cell_61()
