from utils.helper_functions1 import default_ps_stats_params
from utils.helper_functions1 import get_data_dict

def run():
    # Cell 24
from sklearn.decomposition import PCA
import copy

session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

exemplar_session_idx = 6 # 11

weight_type = 'rsquared' # None, rsquared

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1,
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum 
    
    # shuffle_indirect_events
    'validation_types': ('shuffle_indirect_events',),
}

if ps_stats_params['validation_types'] != ():
    print('Using validation:', ps_stats_params['validation_types'])

connectivity_metrics = (
    'pairwise_corr_x',
    'tuning_x',
    'trial_resp_x',
    'post_resp_x',
    'pre_resp_x',
#     'cc_x',
#     'cc_mag_x',
#     'mask_counts_x',
    
    'raw_cc_y',
    'tuning_y',
    'trial_resp_y',
    'post_resp_y',
    'pre_resp_y',
#     'cc_y',
#     'cc_mag_y',
#     'mask_counts_y',
)

plot_pair = ('pairwise_corr_x', 'raw_cc_y',)
# plot_pair = ('cc_x', 'raw_cc_y',)

ps_stats_params = default_ps_stats_params(ps_stats_params)

fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(8, 3))
fig2, (ax2, ax2p) = plt.subplots(1, 2, figsize=(8, 3))
fig6, ((ax6, ax6p,), (ax6pp, ax6ppp),) = plt.subplots(2, 2, figsize=(9, 6))
fig3, (ax3, ax3p, ax3pp) = plt.subplots(1, 3, figsize=(10, 4)) # Improvement of fits, and resulting slopes
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4))
fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4)) # Individual fits and aggregate fits

n_sessions = len(session_idxs)

old_method_ps = np.zeros((n_sessions,))
new_method_ps = np.zeros((n_sessions,))
old_method_slopes = np.zeros((n_sessions,))
new_method_slopes = np.zeros((n_sessions,))
old_method_r_squareds = np.zeros((n_sessions,))
new_method_r_squareds = np.zeros((n_sessions,))

r_squareds = [[] for _ in range(n_sessions)] # Separate this out into sessions to plot by mouse

group_corrs_all = []
indirect_predictions_all = []
indirect_weights_all = []

for session_idx_idx, session_idx in enumerate(session_idxs):

    print('Session idx {}'.format(session_idx))
    day_1_idx = session_idx
    
    data_to_extract = ('d_ps', 'trial_start_fs', 'resp_ps', 'resp_ps_pred',)
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)

    # ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
    # ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
#     print('FAKE!')
#     ps_fs = get_fake_ps_data(session_idx)
    
    # n_ps_times = data_1['resp_ps'].shape[0]
    n_neurons = data_1['resp_ps'].shape[0]
    n_groups = data_1['resp_ps'].shape[1]

    # d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    # d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)
    d_ps = data_1['d_ps']

    # resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    #     ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
    # )

    # resp_ps_events = resp_ps_extras['resp_ps_events']

    pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
    # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
    pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)

    direct_idxs_flat = np.where(d_ps.flatten() < D_DIRECT)[0]
    indirect_idxs_flat = np.where(np.logical_and(d_ps.flatten() > D_NEAR, d_ps.flatten() < D_FAR))[0]
    
    group_corrs = [] # Filled as we iterate across groups
    indirect_resp_ps = []
    indirect_predictions = []
    
    # Filled as we iterate across group for exemplar session only
    flat_direct_resp_ps = []
    flat_indirect_pred_params = []
    flat_indirect_resp_ps_pred = []
    flat_indirect_resp_ps = []
    flat_indirect_pred_r_squareds = []
    
    for group_idx in range(n_groups):
        direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
        indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]

        # dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        # indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        
        # direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
        #     dir_resp_ps_events, ps_stats_params,
        # )
        # indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
        #     dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
        #     ps_stats_params, verbose=False, return_extras=True
        # )
        
        # r_squareds[session_idx_idx].append(fit_extras['r_squareds'])
        r_squareds_group = data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx]
#         if group_idx == 0: print('Shuffling r^2 within groups!')
#         np.random.shuffle(r_squareds_group) # Validate r^2 weighting by shuffling within a group
        r_squareds[session_idx_idx].append(r_squareds_group)

        if plot_pair[1] in ('raw_cc_y',):
            indirect_resp_ps.append(data_1['resp_ps'][indirect_idxs, group_idx])
            indirect_predictions.append(data_1['resp_ps_pred'][indirect_idxs, group_idx])
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
#         sum_direct_resp_ps = np.nansum(resp_ps[direct_idxs, group_idx]) # Avg. over events, then neurons
#         sum_direct_resp_ps =  np.nanmean(np.nansum(dir_resp_ps_events, axis=0)) # Over neurons, then avg events
        
        ### Gets average direct input ###
        # Note this way of doing determines average input for each event THEN averages over
        # events. This does not necessarily yield the same result as averaging over events
        # first then determining the input because of how we treat nans.
        
        # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
        # direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
        # direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)
        
        if plot_pair[0] in ('pairwise_corr_x',):
            group_corr = np.matmul(pairwsie_corrs[:, direct_idxs], data_1['resp_ps_pred'][direct_idxs, group_idx])[indirect_idxs]
#             if group_idx == 0: print('Shuffling correlation within groups!')
#             np.random.shuffle(group_corr) # Turn this one for validation, does indirect identity actually mattter?
        elif plot_pair[0] in ('cc_x',):
            group_corr = np.nansum(data_1['resp_ps_pred'][direct_idxs, group_idx]) * np.ones((indirect_idxs.shape[0],))
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
        group_corrs.append(group_corr)
        
        if exemplar_session_idx == session_idx:
            slope_idx = 0
            
            n_indirect = data_1['resp_ps_pred_extras']['params'][group_idx].shape[0]
            flat_direct_resp_ps.append(np.nansum(data_1['resp_ps'][direct_idxs, group_idx]) * np.ones((n_indirect,)))
            flat_indirect_pred_params.append(data_1['resp_ps_pred_extras']['params'][group_idx][:, slope_idx])
            flat_indirect_resp_ps_pred.append(data_1['resp_ps_pred'][indirect_idxs, group_idx])
            flat_indirect_resp_ps.append(data_1['resp_ps'][indirect_idxs, group_idx])
            flat_indirect_pred_r_squareds.append(data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])

    group_corrs = np.concatenate(group_corrs, axis=0) # Filled as we iterate across groups
    indirect_resp_ps = np.concatenate(indirect_resp_ps, axis=0)
    indirect_predictions = np.concatenate(indirect_predictions, axis=0)
    
    ax_fit, axp_fit = None, None
    if exemplar_session_idx == session_idx:
        indirect_d_ps = d_ps.flatten()[indirect_idxs_flat]
        
        ax1.scatter(indirect_d_ps, indirect_resp_ps, marker='.', color='k', alpha=0.1)
        ax1p.scatter(indirect_d_ps, indirect_predictions, marker='.', color='k', alpha=0.1)
        
        ax_fit, axp_fit = ax2, ax2p
        ax2.scatter(group_corrs, indirect_resp_ps, color=c_vals_l[0], alpha=0.05, marker='.')
        ax2p.scatter(group_corrs, indirect_predictions, color=c_vals_l[0], alpha=0.05, marker='.')
        
        # Plots of fit trends for a given session
        for ax, y_plot in zip(
            (ax6, ax6p, ax6pp, ax6ppp),
            (flat_indirect_pred_params, flat_indirect_resp_ps_pred, flat_indirect_resp_ps, flat_indirect_pred_r_squareds,)
        ):
            for group_idx in range(len(y_plot)):
                ax.scatter(flat_direct_resp_ps[group_idx], y_plot[group_idx], alpha=0.3, color=c_vals_l[1], marker='.')
                ax.scatter(np.nanmean(flat_direct_resp_ps[group_idx]), np.nanmean(y_plot[group_idx]), color=c_vals[1], marker='.', zorder=3)
            _ = add_regression_line(
                np.concatenate(flat_direct_resp_ps, axis=0), np.concatenate(y_plot, axis=0), ax=ax, color=c_vals_d[1], zorder=5
            )
    
    r_squareds_session = np.concatenate(r_squareds[session_idx_idx], axis=0)
#     print('Shuffling r^2 across a single session!')
#     np.random.shuffle(r_squareds_session) # Validate r^2 weighting by shuffling across a session
    r_squareds[session_idx_idx] = r_squareds_session
    
    if weight_type in (None,):
        weights = None
    elif weight_type in ('rsquared',):
        weights = np.copy(r_squareds[session_idx_idx])
    else:
        raise ValueError('Weight type {} not recognized'.format(weight_type))
    
    slope_o, _, rvalue_o, pvalue_o, _ = add_regression_line(group_corrs, indirect_resp_ps, ax=ax_fit, color=c_vals[0], zorder=5)
    
#     indirect_predictions = np.where(np.isnan(indirect_predictions), 0., indirect_predictions)
#     weights = np.where(np.isnan(weights), 0., weights)
    slope_n, _, rvalue_n, pvalue_n, _ = add_regression_line(
        group_corrs, indirect_predictions, weights=weights,
        ax=axp_fit, color=c_vals[0], zorder=5
    )
    
    group_corrs_all.append(group_corrs)
    indirect_predictions_all.append(indirect_predictions)
    indirect_weights_all.append(weights)
    
    old_method_ps[session_idx_idx] = pvalue_o
    new_method_ps[session_idx_idx] = pvalue_n
    old_method_slopes[session_idx_idx] = slope_o
    new_method_slopes[session_idx_idx] = slope_n
    old_method_r_squareds[session_idx_idx] = rvalue_o**2
    new_method_r_squareds[session_idx_idx] = rvalue_n**2

for ax in (ax1, ax1p,):
    ax.set_xlabel('Distance from PS (um)')
ax1.set_ylabel('Indirect PS response (old)')
ax1p.set_ylabel('Indirect PS response prediction')

for ax in (ax2, ax2p,):
    ax.set_xlabel('Correlation with PS group')

ax2.set_ylabel('Indirect PS response (old)')
ax2p.set_ylabel('Indirect PS response prediction')
ax2.legend()
ax2p.legend()

for ax in (ax1, ax1p):
    ax.axhline(0.0, color=c_vals[0], zorder=5, linestyle='dashed')
    
for ax in (ax6, ax6p, ax6pp, ax6ppp):
    ax.set_xlabel('Tot. direct resp_ps')
    ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
    ax.legend()
    
ax6.set_ylabel('Indirect slope')
ax6p.set_ylabel('Indirect resp_ps_pred')
ax6pp.set_ylabel('Indirect resp_ps')
ax6ppp.set_ylabel('Indirect r^2')
    
    
MIN_LOG10_P = -300
old_method_ps = np.where(old_method_ps == 0., 10**MIN_LOG10_P, old_method_ps)
new_method_ps = np.where(new_method_ps == 0., 10**MIN_LOG10_P, new_method_ps)

old_new_pvalues = np.concatenate(
    (old_method_ps[:, np.newaxis], new_method_ps[:, np.newaxis]), axis=-1
)
old_new_r_squareds = np.concatenate(
    (old_method_r_squareds[:, np.newaxis], new_method_r_squareds[:, np.newaxis]), axis=-1
)

for session_idx_idx, session_idx in enumerate(session_idxs):
    
    print('Mouse {} - Session {}, -log10(p): {:.1f} to {:.1f}'.format(
        data_dict['data']['mouse'][session_idx], session_idx, 
        -np.log10(old_new_pvalues[session_idx_idx, 0]), -np.log10(old_new_pvalues[session_idx_idx, 1])
    ))
    
    ax3.plot(-np.log10(old_new_pvalues[session_idx_idx]), color=c_vals[SESSION_COLORS[session_idx_idx]], marker='.')
    ax3.plot(-np.log10(old_new_pvalues[session_idx_idx, 0]), color=c_vals_l[SESSION_COLORS[session_idx_idx]], marker='.')
    
    ax3pp.plot(old_new_r_squareds[session_idx_idx], color=c_vals[SESSION_COLORS[session_idx_idx]], marker='.')
    ax3pp.plot(old_new_r_squareds[session_idx_idx, 0], color=c_vals_l[SESSION_COLORS[session_idx_idx]], marker='.')
    
ax3p.scatter(new_method_slopes, -np.log10(new_method_ps), 
             c=[c_vals[session_color] for session_color in SESSION_COLORS[:n_sessions]], zorder=5)
ax3p.scatter(old_method_slopes, -np.log10(old_method_ps), 
             c=[c_vals_l[session_color] for session_color in SESSION_COLORS[:n_sessions]], marker='.')

for ax in (ax3, ax3pp,):
    ax.set_xticks((0, 1))
    ax.set_xticklabels(('No interpolation', 'Interpolation'))
for ax in (ax3, ax3p, ax3pp,):
    ax.axhline(0.0, color='lightgrey', zorder=-5)

ax3.set_ylabel('-log10(pvalue)')
ax3p.set_xlabel('slope')
ax3pp.set_ylabel('$r^2$')
ax3pp.set_ylim((0., 0.4))

ax3p.axvline(0.0, color='lightgrey', zorder=-5)

for session_idx_idx, session_idx in enumerate(session_idxs):
    
    ax4.hist(r_squareds[session_idx_idx] + SESSION_COLORS[session_idx_idx], bins=20, 
             color=c_vals[SESSION_COLORS[session_idx_idx]], alpha=0.2)
    
ax4.set_xticks((0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6))
ax4.set_xticklabels((0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 1.0))

for sep in (0, 1, 2, 3, 4, 5, 6):
    ax4.axvline(sep, color='lightgrey', zorder=-5, linestyle='dashed')
    
ax4.set_xlabel('r^2 of fits')
ax4.set_ylabel('n_indirect')

for session_idx_idx, session_idx in enumerate(session_idxs):
    _ = add_regression_line(
        group_corrs_all[session_idx_idx], indirect_predictions_all[session_idx_idx], 
        weights=indirect_weights_all[session_idx_idx], label=None,
        ax=ax5, color=c_vals[SESSION_COLORS[session_idx_idx]], zorder=0
    )
    
slope, _, rvalue, pvalue, _ = add_regression_line(
    np.concatenate(group_corrs_all, axis=0), np.concatenate(indirect_predictions_all, axis=0), 
    weights=np.concatenate(indirect_weights_all, axis=0),
    ax=ax5, color='k', zorder=5
)    
ax5.legend()
ax5.set_xlabel('Correlation with PS group')
ax5.set_ylabel('Indirect PS response prediction')

print('Aggregate fit slope:', slope)

    print("✅ weighting_and_session_selection ran successfully.")
if __name__ == '__main__':
    run()