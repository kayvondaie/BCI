import matplotlib.pyplot as plt
import numpy as np

def plot_cell_72():
    # Reset globals
    D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

    session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

    n_sessions = len(session_idxs)

    PS_RESPONSE_THRESH = 0.1

    exemplar_session_idx = 1

    # direct_response_mode: group_membership, ps_resp, pr_resp_thresh
    # d_ps_mode: default, small_indir, small_indir_dir
    # correlation_mode: all, pre+post, pre, post 
    fit_types = ( # (name, (direct_response_mode, d_ps_mode, correlation_mode))
        ('corr - group_membership', ('group_membership', 'default', 'all',)),
        ('corr - response_weight',  ('ps_resp', 'default', 'all',)),
        ('corr - threshold',        ('pr_resp_thresh', 'default', 'all',)),
        ('d_direct = 20',           ('ps_resp', 'small_dir', 'all',)),
        ('d_direct, d_near = 20',   ('ps_resp', 'small_dir_near', 'all',)),
        ('d_direct, d_near = 20, d_far=250',   ('ps_resp', 'small_dir_near_large_far', 'all',)),
        ('d_direct, d_near = 20, d_far=50',   ('ps_resp', 'small_dir_near_far', 'all',)),
        ('corr - pre + post',   ('ps_resp', 'small_dir_near_far', 'pre+post',)),
        ('corr - pre only',   ('ps_resp', 'small_dir_near_far', 'pre',)),
        ('corr - post only',   ('ps_resp', 'small_dir_near_far', 'post',)),
        ('corr - pre+post (default ds)',   ('ps_resp', 'default', 'pre+post',)),
    )

    n_fit_types = len(fit_types)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    # fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    direct_responses = [[] for _ in range(n_sessions)]
    slopes = np.zeros((n_fit_types, n_sessions,))
    log10_rsquares = np.zeros((n_fit_types, n_sessions,))
    log10_pvalues = np.zeros((n_fit_types, n_sessions,))

    for session_idx_idx, session_idx in enumerate(session_idxs):

        data_to_extract = ('d_ps', 'resp_ps', 'trial_start_fs', 'd_masks',)

        for fit_type_idx, fit_type in enumerate(fit_types):
            fit_type_name, fit_type_params = fit_type
            direct_response_mode, d_ps_mode, correlation_mode = fit_type_params

            if d_ps_mode == 'default':
                D_DIRECT, D_NEAR, D_FAR = 30, 30, 100
            elif d_ps_mode == 'small_dir':
                D_DIRECT, D_NEAR, D_FAR = 20, 30, 100
            elif d_ps_mode == 'small_dir_near':
                D_DIRECT, D_NEAR, D_FAR = 20, 20, 100
            elif d_ps_mode == 'small_dir_near_large_far':
                D_DIRECT, D_NEAR, D_FAR = 20, 20, 250
            elif d_ps_mode == 'small_dir_near_far':
                D_DIRECT, D_NEAR, D_FAR = 20, 20, 50
            else:
                raise ValueError()

            data = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)

            n_neurons = data['resp_ps'].shape[0]
            n_groups = data['resp_ps'].shape[1]

            dir_mask_weighted = data['dir_mask']
            indir_mask_weighted = data['indir_mask']

            flat_direct_filter = dir_mask_weighted.flatten() > 0
            flat_indirect_filter = indir_mask_weighted.flatten() > 0

    #     sum_dir_resp = np.empty((n_groups,))
    #     sum_dir_resp[:] = np.nan

    #     mean_dir_resp = sum_dir_resp.copy()
    #     sum_indir_resp = sum_dir_resp.copy()
    #     mean_indir_resp = sum_dir_resp.copy()

    #     for group_idx in range(n_groups):
    #         group_mask_dir = dir_mask_weighted[:, group_idx] > 0
    #         if group_mask_dir.shape[0] > 0: # No neuron catch
    #             sum_dir_resp[group_idx] = np.sum(data['resp_ps'][group_mask_dir, group_idx])
    #             mean_dir_resp[group_idx] = np.mean(data['resp_ps'][group_mask_dir, group_idx])

    #             direct_responses[session_idx_idx].append(data['resp_ps'][group_mask_dir, group_idx])

    #         group_mask_indir = indir_mask_weighted[:, group_idx] > 0
    #         if group_mask_indir.shape[0] > 0: # No neuron catch
    #             sum_indir_resp[group_idx] = np.sum(data['resp_ps'][group_mask_indir, group_idx])
    #             mean_indir_resp[group_idx] = np.mean(data['resp_ps'][group_mask_indir, group_idx])

    #     direct_responses[session_idx_idx] = np.concatenate(direct_responses[session_idx_idx], axis=0)

            if correlation_mode == 'all':
                pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
                # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
                pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)
            elif correlation_mode == 'pre+post':
                pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'])
            elif correlation_mode == 'pre':
                pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'], ts_trial=(-2, 0))
            elif correlation_mode == 'post':
                pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'], ts_trial=(0, 10))
            else:
                raise ValueError()

            if direct_response_mode == 'group_membership': # No response weighting
                group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted)
            elif direct_response_mode == 'ps_resp': # PS response weighting (might introduce spurious correlations)
                group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted * data['resp_ps']) 
            elif direct_response_mode == 'pr_resp_thresh': # Threshold response weighting
                threshold_mask = data['resp_ps'] > PS_RESPONSE_THRESH
                group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted * threshold_mask) 
            else:
                raise ValueError()

            # Control for what counts as photostimulation response
            resp_ps_plot = data['resp_ps'] # Default is just the photostimulation response

            ax_fit = None 
            if session_idx == exemplar_session_idx:
                ax1.scatter(group_cors.flatten()[flat_indirect_filter], 
                            resp_ps_plot.flatten()[flat_indirect_filter], 
                            color=c_vals_l[fit_type_idx % 9], alpha=0.05, marker='.')

                ax_fit = ax1

            slope, intercept, rvalue, pvalue, se = add_regression_line(
                group_cors.flatten()[flat_indirect_filter], 
                resp_ps_plot.flatten()[flat_indirect_filter],       
                ax=ax_fit,
            )

            print(' Session idx {} r^2: {:.1e}'.format(session_idx, rvalue**2))
            slopes[fit_type_idx, pair_idx] = slope
            log10_rsquares[fit_type_idx, session_idx_idx] = np.log10(rvalue**2)
            log10_pvalues[fit_type_idx, session_idx_idx] = np.log10(pvalue)

        # When done scanning over the various fit types
        if session_idx == exemplar_session_idx:
            ax1.set_xlabel('Correlation to group')
            ax1.set_ylabel('PS response')

            ax1.legend()

    # print('Mean rsquared log10: {:.1f} ( = {:.2e})'.format(
    #     np.mean(log10_rsquares), 10**np.mean(log10_rsquares)
    # ))    

    # ax2.violinplot(direct_responses)
    # ax2.boxplot(direct_responses, notch=True, sym='.')

    # ax2.set_xlabel('Session idx')
    # ax2.set_xticks((np.arange(1, n_sessions+1)))
    # ax2.set_xticklabels(session_idxs)

    # ax2.set_ylabel('Direct PS Response')

    # ax2.axhline(0.0, color='grey', zorder=-5)
    # ax2.axhline(PS_RESPONSE_THRESH, color=c_vals[1], zorder=-5)

    # Reset globals               
    D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))
    fit_type_names = []
    color_idxs = (0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,)

    assert len(color_idxs) == n_fit_types

    for fit_type_idx, fit_type in enumerate(fit_types):

        fit_type_names.append(fit_type[0])

        ax3.scatter(fit_type_idx * np.ones((n_sessions,)), log10_rsquares[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
        ax3.scatter(fit_type_idx, np.mean(log10_rsquares[fit_type_idx]), color='k')

        ax4.scatter(fit_type_idx * np.ones((n_sessions,)), log10_pvalues[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
        ax4.scatter(fit_type_idx, np.mean(log10_pvalues[fit_type_idx]), color='k')

    ax3.set_ylabel('log10($r^2$) $\Delta$group correlation - $\Delta$PS response')
    ax4.set_ylabel('log10(p) $\Delta$group correlation - $\Delta$PS response')

    ax3.axhline(-1, color='lightgrey', zorder=-5)
    ax3.axhline(-2, color='lightgrey', zorder=-5)
    ax3.axhline(-3, color='lightgrey', zorder=-5)

    ax4.axhline(np.log10(0.1), color='lightgrey', zorder=-5)
    ax4.axhline(np.log10(0.05), color='lightgrey', zorder=-5)
    ax4.axhline(np.log10(0.01), color='lightgrey', zorder=-5)

    for ax in (ax3, ax4):
        ax.set_xticks(np.arange(n_fit_types))

    ax3.set_xticklabels([])
    ax4.set_xticklabels(fit_type_names, rotation=45)

    for tick in ax4.xaxis.get_majorticklabels():
        tick.set_horizontalalignment('right')

if __name__ == '__main__':
    plot_cell_72()
