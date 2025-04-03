import matplotlib.pyplot as plt
import numpy as np

def plot_cell_26():
    from sklearn.decomposition import PCA
    import copy

    direct_response_mode = 'ps_resp' # ps_resp, pr_resp_thresh
    direct_input_mode = 'average' # average, average_equal_sessions, minimum
    weight_type = 'minimum_rsquared' # None, minimum_rsquared
    correlation_mode = 'all' # all, pre+post, pre, post 

    PS_RESPONSE_THRESH = 0.1 

    ps_stats_params = {
        'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently

        'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
        'n_direct_predictors': 1,
        'direct_predictor_intercept_fit': False,
        'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
        # shuffle_indirect_events
        'validation_types': (),
    }

    ps_stats_params = default_ps_stats_params(ps_stats_params)

    if ps_stats_params['validation_types'] != ():
        print('Using validation:', ps_stats_params['validation_types'])


    connectivity_metrics = (
        'delta_pairwise_corr_x',
        'delta_tuning_x',
        'mean_tuning_x',
        'delta_trial_resp_x',
        'mean_trial_resp_x',
        'delta_post_resp_x',
        'mean_post_resp_x',
        'delta_pre_resp_x',
        'mean_pre_resp_x',
    #     'laser_resp_x',
    #     'laser_resp_mag_x',
    #     'mask_counts_x',

        'raw_delta_cc_y',
        'delta_tuning_y',
        'mean_tuning_y',
        'delta_trial_resp_y',
        'mean_trial_resp_y',
        'delta_post_resp_y',
        'mean_post_resp_y',
        'delta_pre_resp_y',
        'mean_pre_resp_y',
    #     'delta_cc_y',
    #     'delta_cc_mag_y',
    #     'mask_counts_y',
    )

    plot_pair = ('delta_pairwise_corr_x', 'raw_delta_cc_y',)
    # plot_pair = ('laser_resp_x', 'raw_delta_cc_y',)

    exemplar_pair_idx = 0

    fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(8, 3)) # Consistency of predictions across days
    fig4, (ax4, ax4p) = plt.subplots(1, 2, figsize=(8, 3)) # Consistency of fit parameters
    fig2, (ax2, ax2p) = plt.subplots(1, 2, figsize=(8, 3)) # Delta correlation fit
    fig3, (ax3, ax3p, ax3pp) = plt.subplots(1, 3, figsize=(10, 4)) # Improvement of fits, and resulting slopes
    fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4)) # Individual fits and aggregate fits

    n_pairs = len(session_idx_pairs)

    old_method_ps = np.zeros((n_pairs,))
    new_method_ps = np.zeros((n_pairs,))
    old_method_slopes = np.zeros((n_pairs,))
    new_method_slopes = np.zeros((n_pairs,))
    old_method_r_squareds = np.zeros((n_pairs,))
    new_method_r_squareds = np.zeros((n_pairs,))

    r_squareds_1 = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
    r_squareds_2 = [[] for _ in range(n_pairs)]

    delta_group_corrs_all = []
    delta_indirect_predictions_all = []
    indirect_weights_all = []

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):

        day_1_idx = session_idx_pair[0]
        day_2_idx = session_idx_pair[1]

        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

        print('Pair {} - Sessions {} and {} - Mouse {}'.format(
            pair_idx, day_1_idx, day_2_idx, data_dict['data']['mouse'][day_2_idx]
        )) 

        data_to_extract = ('d_ps', 'trial_start_fs', 'resp_ps', 'resp_ps_pred',)
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)

        d_ps_1 = data_1['d_ps']
        d_ps_2 = data_2['d_ps']

        n_groups = data_1['resp_ps'].shape[-1]

        # data_to_extract = ('trial_start_fs',)
        if correlation_mode == 'all':
            pairwise_corrs_1 = data_dict['data']['trace_corr'][day_1_idx]
            pairwise_corrs_2 = data_dict['data']['trace_corr'][day_2_idx]
            # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
            pairwise_corrs_1 = np.where(np.isnan(pairwise_corrs_1), 0., pairwise_corrs_1)
            pairwise_corrs_2 = np.where(np.isnan(pairwise_corrs_2), 0., pairwise_corrs_2)
        elif correlation_mode == 'pre+post':
            pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'])
            pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'])
        elif correlation_mode == 'pre':
            pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(-2, 0))
            pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(-2, 0))
        elif correlation_mode == 'post':
            pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(0, 10))
            pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(0, 10))
        else:
            raise ValueError()

        delta_corrs = [] # Filled as we iterate across groups
        delta_indirect_resp_ps = []
        delta_indirect_predictions = []

        for group_idx in range(n_groups):
            direct_idxs = np.where(np.logical_and(
                d_ps_1[:, group_idx] < D_DIRECT, 
                d_ps_2[:, group_idx] < D_DIRECT
            ))[0]
            indirect_idxs = np.where(np.logical_and(
                np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
                np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
            ))[0]

            r_squareds_1[pair_idx].append(data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])
            r_squareds_2[pair_idx].append(data_2['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])

            if pair_idx == exemplar_pair_idx:
                indirect_predictions_1 = np.sum(data_1['resp_ps_pred_extras']['params'][group_idx], axis=-1)
                indirect_predictions_2 = np.sum(data_2['resp_ps_pred_extras']['params'][group_idx], axis=-1)

                ax1.scatter(data_1['resp_ps'][indirect_idxs, group_idx], data_2['resp_ps'][indirect_idxs, group_idx], marker='.', color=c_vals[2], alpha=0.1)
                ax1p.scatter(indirect_predictions_1, indirect_predictions_2, marker='.', color=c_vals[2], alpha=0.1)

            COMBINE_CONST = 1/2

            raise NotImplementedError('Should implement new way of computing direct response here.')
            if direct_response_mode == 'group_membership': # No response weighting
                group_corrs_1 = np.sum(pairwise_corrs_1[:, direct_idxs], axis=-1)
                group_corrs_2 = np.sum(pairwise_corrs_2[:, direct_idxs], axis=-1)
            elif direct_response_mode == 'ps_resp': # PS response weighting (sum of both to prevent spurious correlations)
                if direct_input_mode in ('average',):
                    direct_input = COMBINE_CONST * (data_1['resp_ps'][direct_idxs, group_idx] + data_2['resp_ps'][direct_idxs, group_idx])
                elif direct_input_mode in ('minimum',): # Separately computes direct input, then minimum mag of both
                    raise NotImplementedError()
                else:
                    raise ValueError('Direct_input_mode {} not recognized'.format(direct_input_mode))
                group_corrs_1 = np.matmul(pairwise_corrs_1[:, direct_idxs], direct_input)
                group_corrs_2 = np.matmul(pairwise_corrs_2[:, direct_idxs], direct_input) 
            elif direct_response_mode == 'ps_resp_thresh': # Threshold response weighting on average of responses
    #             threshold_mask = (1/2 * (resp_ps_1[direct_idxs, group_idx]  + resp_ps_2[direct_idxs, group_idx])) > PS_RESPONSE_THRESH
                threshold_mask = np.logical_and(data_1['resp_ps'][direct_idxs, group_idx] > PS_RESPONSE_THRESH, data_2['resp_ps'][direct_idxs, group_idx] > PS_RESPONSE_THRESH)
                group_corrs_1 = np.matmul(pairwise_corrs_1[:, direct_idxs], threshold_mask) 
                group_corrs_2 = np.matmul(pairwise_corrs_2[:, direct_idxs], threshold_mask) 
            else:
                raise ValueError()

            if plot_pair[0] in ('delta_pairwise_corr_x',):
                delta_group_corr = (group_corrs_2 - group_corrs_1)[indirect_idxs]
    #             if group_idx == 0: print('Shuffling correlation within groups!')
    #             np.random.shuffle(delta_group_corr) # Turn this one for validation, does indirect identity actually mattter?
            elif plot_pair[0] in ('laser_resp_x',):
                assert direct_response_mode == 'ps_resp' # direct_input not defined otherwise, can update above code to fix
                delta_group_corr = np.nansum(direct_input) * np.ones((indirect_idxs.shape[0],))
            else:
                raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))

    #         if group_idx == 0: print('Shuffling delta correlation within groups!')
    #         np.random.shuffle(delta_group_corr) # Turn this one for validation, does indirect identity actually mattter?        
            delta_corrs.append(delta_group_corr)

            if plot_pair[1] in ('raw_delta_cc_y',):
                delta_indirect_resp_ps.append(data_2['resp_ps'][indirect_idxs, group_idx] - data_1['resp_ps'][indirect_idxs, group_idx])
                delta_indirect_predictions.append(
                    data_2['resp_ps_pred'][indirect_idxs, group_idx] - data_1['resp_ps_pred'][indirect_idxs, group_idx]
                )
            else:
                raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))


            ### Gets average direct input ###
            # Note this way of doing determines average input for each event THEN averages over
            # events of both sessions. This does not necessarily yield the same result as 
            # averaging over events in each session first and then determining the input because 
            # of how we treat nans and uneven event counts


            ### Uses average direct input to predict photostimulation response ###
            # indirect_predictions_1 = photostim_predict(indirect_params_1, direct_input, ps_stats_params)
            # indirect_predictions_2 = photostim_predict(indirect_params_2, direct_input, ps_stats_params)
            # delta_indirect_predictions.append(indirect_predictions_2 - indirect_predictions_1)

            if pair_idx == exemplar_pair_idx:
                slope_idx = 1 if ps_stats_params['direct_predictor_intercept_fit'] else 0
                ax4.scatter(data_1['resp_ps_pred_extras']['params'][group_idx][:, slope_idx], 
                            data_2['resp_ps_pred_extras']['params'][group_idx][:, slope_idx], 
                            color=c_vals[3], marker='.', alpha=0.3)

                if ps_stats_params['direct_predictor_intercept_fit']:
                    ax4p.scatter(data_1['resp_ps_pred_extras']['params'][group_idx][:, 0], 
                                 data_2['resp_ps_pred_extras']['params'][group_idx][:, 0], 
                                 color=c_vals[3], marker='.', alpha=0.3)

            # print('indirect_predictions old:', indirect_predictions_1)
            # print('indirect_predictions new:', data_1['resp_ps_pred'][indirect_idxs, group_idx])
            # print('r_squareds old:', fit_extras_1['r_squareds'])
            # print('r_squareds new:', data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])
            # print(sdfdsfsdfsd)

        delta_corrs = np.concatenate(delta_corrs, axis=0)
        delta_indirect_resp_ps = np.concatenate(delta_indirect_resp_ps, axis=0)
        delta_indirect_predictions = np.concatenate(delta_indirect_predictions, axis=0)

        r_squareds_1[pair_idx] = np.concatenate(r_squareds_1[pair_idx], axis=0)
        r_squareds_2[pair_idx] = np.concatenate(r_squareds_2[pair_idx], axis=0)

        ax_plot, axp_plot = None, None

        if pair_idx == exemplar_pair_idx:
            ax_plot, axp_plot = ax2, ax2p
            ax2.scatter(delta_corrs, delta_indirect_resp_ps, color=c_vals_l[0], alpha=0.05, marker='.')
            ax2p.scatter(delta_corrs, delta_indirect_predictions, color=c_vals_l[0], alpha=0.05, marker='.')

        if weight_type in (None,):
            weights = None
        elif weight_type in ('minimum_rsquared',):
            # Minimum with nan still yields nan
            weights = np.minimum(r_squareds_1[pair_idx], r_squareds_2[pair_idx])
        else:
            raise ValueError('Weight type {} not recognized'.format(weight_type))

        slope_o, _, rvalue_o, pvalue_o, _ = add_regression_line(delta_corrs, delta_indirect_resp_ps, ax=ax_plot, color=c_vals[0], zorder=5)

    #     print('delta_corrs', delta_corrs.shape, 'nan_count', np.sum(np.isnan(delta_corrs)))
    #     print('delta_indirect_predictions', delta_indirect_predictions.shape, 'nan_count', np.sum(np.isnan(delta_indirect_predictions)))
    #     print('weights', weights.shape, 'nan_count', np.sum(np.isnan(weights)))
    #     indirect_predictions = np.where(np.isnan(indirect_predictions), 0., indirect_predictions)
    #     weights = np.where(np.isnan(weights), 0., weights)    
        slope_n, _, rvalue_n, pvalue_n, _ = add_regression_line(
            delta_corrs, delta_indirect_predictions, weights=weights,
            ax=axp_plot, color=c_vals[0], zorder=5
        )

        delta_group_corrs_all.append(delta_corrs)
        delta_indirect_predictions_all.append(delta_indirect_predictions)
        indirect_weights_all.append(weights)

        old_method_ps[pair_idx] = pvalue_o
        new_method_ps[pair_idx] = pvalue_n
        old_method_slopes[pair_idx] = slope_o
        new_method_slopes[pair_idx] = slope_n
        old_method_r_squareds[pair_idx] = rvalue_o**2
        new_method_r_squareds[pair_idx] = rvalue_n**2

    for ax in (ax1, ax1p, ax4, ax4p):
        ax.axhline(0.0, color='k', zorder=5, linestyle='dashed')
        ax.axvline(0.0, color='k', zorder=5, linestyle='dashed')

    for ax in (ax2, ax2p,):
        ax.legend()
        ax.set_xlabel('Delta correlation (direct_1 + direct_2 weight)')

    ax1.set_xlabel('Day 1 Indirect resp. (old method)')
    ax1.set_ylabel('Day 2 Indirect resp. (old method)')
    ax1p.set_xlabel('Day 1 indirect pred.')
    ax1p.set_ylabel('Day 2 indirect pred.')

    ax2.set_ylabel('Delta indirect (old method)')
    ax2p.set_ylabel('Delta indirect (interpolation method)')

    ax4.set_xlabel('Day 1 slope')
    ax4.set_ylabel('Day 2 slope')
    ax4p.set_xlabel('Day 1 intercept')
    ax4p.set_ylabel('Day 2 intercept')

    old_new_pvalues = np.concatenate(
        (old_method_ps[:, np.newaxis], new_method_ps[:, np.newaxis]), axis=-1
    )
    old_new_r_squareds = np.concatenate(
        (old_method_r_squareds[:, np.newaxis], new_method_r_squareds[:, np.newaxis]), axis=-1
    )

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):

        day_1_idx = session_idx_pair[0]
        day_2_idx = session_idx_pair[1]

        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

        print('Mouse {} - Sessions {} to {}, -log10(p): {:.2f} to {:.2f}'.format(
            data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx, 
            -np.log10(old_new_pvalues[pair_idx, 0]), -np.log10(old_new_pvalues[pair_idx, 1])
        ))

        ax3.plot(-np.log10(old_new_pvalues[pair_idx]), color=c_vals[PAIR_COLORS[pair_idx]], marker='.')
        ax3.plot(-np.log10(old_new_pvalues[pair_idx, 0]), color=c_vals_l[PAIR_COLORS[pair_idx]], marker='.')
        ax3pp.plot(old_new_r_squareds[pair_idx], color=c_vals[PAIR_COLORS[pair_idx]], marker='.')
        ax3pp.plot(old_new_r_squareds[pair_idx, 0], color=c_vals_l[PAIR_COLORS[pair_idx]], marker='.')

    ax3p.scatter(new_method_slopes, -np.log10(new_method_ps), 
                 c=[c_vals[pair_color] for pair_color in PAIR_COLORS[:n_pairs]])
    ax3p.scatter(old_method_slopes, -np.log10(old_method_ps), 
                 c=[c_vals_l[pair_color] for pair_color in PAIR_COLORS[:n_pairs]], zorder=-3, marker='.')

    for ax in (ax3, ax3p,):
        ax.axhline(2, color='lightgrey', zorder=-5, linestyle='dashed')
        ax.set_ylabel('-log10(pvalue)')

    ax3.set_xticks((0, 1))
    ax3.set_xticklabels(('No interpolation', 'Interpolation'))
    ax3p.set_xlabel('Slope')
    ax3p.axvline(0., color='lightgrey', zorder=-5, linestyle='dashed')

    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        _ = add_regression_line(
            delta_group_corrs_all[pair_idx], delta_indirect_predictions_all[pair_idx], 
            weights=indirect_weights_all[pair_idx], label=None,
            ax=ax5, color=c_vals[PAIR_COLORS[pair_idx]], zorder=0
        )

    _ = add_regression_line(
        np.concatenate(delta_group_corrs_all, axis=0), np.concatenate(delta_indirect_predictions_all, axis=0), 
        weights=np.concatenate(indirect_weights_all, axis=0),
        ax=ax5, color='k', zorder=5
    )    
    ax5.legend()
    ax5.set_xlabel('Delta correlation (direct_1 + direct_2 weight)')
    ax5.set_ylabel('Delta indirect (interpolation method)')

if __name__ == '__main__':
    plot_cell_26()
