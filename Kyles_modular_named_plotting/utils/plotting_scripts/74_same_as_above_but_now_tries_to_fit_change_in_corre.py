import matplotlib.pyplot as plt
import numpy as np

def plot_cell_74():
    # Reset globals
    D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

    n_pairs = len(session_idx_pairs)

    PS_RESPONSE_THRESH = 0.1

    exemplar_pair_idx = 1

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

    direct_responses = [[] for _ in range(n_pairs)]
    slopes = np.zeros((n_fit_types, n_pairs,))
    log10_rsquares = np.zeros((n_fit_types, n_pairs,))
    log10_pvalues = np.zeros((n_fit_types, n_pairs,))

    full_fit_data_x = [[] for _ in range(n_fit_types)]
    full_fit_data_y = [[] for _ in range(n_fit_types)]

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]

        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

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

            data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
            data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)

            n_neurons = data_1['resp_ps'].shape[0]
            n_groups = data_1['resp_ps'].shape[1]

            indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
                ps_stats_params, data_1, data_2, verbose=False
            )

            flat_direct_filter = dir_mask_weighted.flatten() > 0
            flat_indirect_filter = indir_mask_weighted.flatten() > 0

    #     sum_dir_resp_1 = np.empty((n_groups,))
    #     sum_dir_resp_1[:] = np.nan

    #     mean_dir_resp_1 = sum_dir_resp_1.copy()
    #     sum_indir_resp_1 = sum_dir_resp_1.copy()
    #     mean_indir_resp_1 = sum_dir_resp_1.copy()
    #     sum_dir_resp_2 = sum_dir_resp_1.copy()
    #     mean_dir_resp_2 = sum_dir_resp_1.copy()
    #     sum_indir_resp_2 = sum_dir_resp_1.copy()
    #     mean_indir_resp_2 = sum_dir_resp_1.copy()

    #     for group_idx in range(n_groups):
    #         group_mask_dir = dir_mask_weighted[:, group_idx] > 0
    #         if group_mask_dir.shape[0] > 0: # No neuron catch
    #             sum_dir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_dir, group_idx])
    #             mean_dir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_dir, group_idx])
    #             sum_dir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_dir, group_idx])
    #             mean_dir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_dir, group_idx])

    #         group_mask_indir = indir_mask_weighted[:, group_idx] > 0
    #         if group_mask_indir.shape[0] > 0: # No neuron catch
    #             sum_indir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_indir, group_idx])
    #             mean_indir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_indir, group_idx])
    #             sum_indir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_indir, group_idx])
    #             mean_indir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_indir, group_idx])
            if correlation_mode == 'all':
                pairwsie_corrs_1 = data_dict['data']['trace_corr'][day_1_idx]
                pairwsie_corrs_2 = data_dict['data']['trace_corr'][day_2_idx]
                # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
                pairwsie_corrs_1 = np.where(np.isnan(pairwsie_corrs_1), 0., pairwsie_corrs_1)
                pairwsie_corrs_2 = np.where(np.isnan(pairwsie_corrs_2), 0., pairwsie_corrs_2)
            elif correlation_mode == 'pre+post':
                pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'])
                pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'])
            elif correlation_mode == 'pre':
                pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(-2, 0))
                pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(-2, 0))
            elif correlation_mode == 'post':
                pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(0, 10))
                pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(0, 10))
            else:
                raise ValueError()

            if direct_response_mode == 'group_membership': # No response weighting
                group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted)
                group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted)
            elif direct_response_mode == 'ps_resp': # PS response weighting (sum of both to prevent spurious correlations)
    #             group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * data_1['resp_ps'])
    #             group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * data_2['resp_ps']) 
                group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * (data_1['resp_ps'] + data_2['resp_ps']))
                group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * (data_1['resp_ps'] + data_2['resp_ps'])) 
            elif direct_response_mode == 'pr_resp_thresh': # Threshold response weighting on average of responses
                threshold_mask = (1/2 * (data_1['resp_ps'] + data_2['resp_ps'])) > PS_RESPONSE_THRESH
                group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * threshold_mask) 
                group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * threshold_mask) 
            else:
                raise ValueError()

    # 
    #                 # More careful ways of combining over direct neurons
    #                 group_cors_1 = np.zeros((n_neurons, n_groups,))
    #                 group_cors_2 = np.zeros((n_neurons, n_groups,))
    #                 for group_idx in range(n_groups):
    #                     group_mask = dir_mask_weighted[:, group_idx] > 0 # Mask over neurons
    # #                         resp_ps_weight_1 = data_1['resp_ps']
    #                     resp_ps_weight_1 = data_1['resp_ps'] + data_2['resp_ps']
    #                     resp_ps_weight_1 = resp_ps_weight_1 - np.mean(resp_ps_weight_1[group_mask, group_idx], keepdims=True)
    #                     resp_ps_weight_1 = resp_ps_weight_1 / np.std(resp_ps_weight_1[group_mask, group_idx], keepdims=True)
    # #                         resp_ps_weight_2 = data_2['resp_ps']
    #                     resp_ps_weight_2 = data_1['resp_ps'] + data_2['resp_ps']
    #                     resp_ps_weight_2 = resp_ps_weight_2 - np.mean(resp_ps_weight_2[group_mask, group_idx], keepdims=True)
    #                     resp_ps_weight_2 = resp_ps_weight_2 / np.std(resp_ps_weight_2[group_mask, group_idx], keepdims=True)

    #                     group_cors_1[:, group_idx] = np.matmul(
    #                         pairwsie_corrs_1[:, group_mask], resp_ps_weight_1[group_mask, group_idx]
    #                     )
    #                     group_cors_2[:, group_idx] = np.matmul( 
    #                         pairwsie_corrs_2[:, group_mask], resp_ps_weight_2[group_mask, group_idx]
    #                     ) 

            # Control for what counts as photostimulation response
            resp_ps_plot_1 = data_1['resp_ps'] # Default is just the photostimulation response
    #         resp_ps_plot_1 = resp_ps_1 / mean_dir_resp_1[np.newaxis, :]  # Normalize responses to a group by mean photostimulation size
            resp_ps_plot_2 = data_2['resp_ps'] # Default
    #         resp_ps_plot_2 = resp_ps_2 / mean_dir_resp_2[np.newaxis, :]

            change_in_cors = group_cors_2.flatten() - group_cors_1.flatten()
            change_in_resp_ps = resp_ps_plot_2.flatten() - resp_ps_plot_1.flatten()

            ax_fit = None 
            if pair_idx == exemplar_pair_idx:
                ax1.scatter(change_in_cors[flat_indirect_filter],
                            change_in_resp_ps[flat_indirect_filter],
                            color=c_vals_l[fit_type_idx % 9], alpha=0.05, marker='.')

                ax_fit = ax1

            slope, intercept, rvalue, pvalue, se = add_regression_line(
                change_in_cors[flat_indirect_filter],
                change_in_resp_ps[flat_indirect_filter],      
                ax=ax_fit,
            )

            full_fit_data_x[fit_type_idx].append(change_in_cors[flat_indirect_filter])
            full_fit_data_y[fit_type_idx].append(change_in_resp_ps[flat_indirect_filter])

            print(' Pair idx {} r^2: {:.1e}'.format(pair_idx, rvalue**2))

            slopes[fit_type_idx, pair_idx] = slope
            log10_rsquares[fit_type_idx, pair_idx] = np.log10(rvalue**2)
            log10_pvalues[fit_type_idx, pair_idx] = np.log10(pvalue)

        # When done scanning over the various fit types

    ax1.set_xlabel('Correlation to group')
    ax1.set_ylabel('PS response')

    ax1.legend()

    # Reset globals               
    D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

if __name__ == '__main__':
    plot_cell_74()
