from utils.helper_functions1 import default_ps_stats_params
from utils.helper_functions1 import get_data_dict

import pickle
def run():
    try:
        with open('outputs/data_dict.pkl', 'rb') as f:
            data_dict = get_data_dict()
    except FileNotFoundError:
        print('No saved data_dict found.')
    # Cell 14
from sklearn.decomposition import PCA
import copy

session_idx = 11 # 11
shuffle_events = True # Shuffle indirect events relvative to direct

ps_stats_params = {
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum
}
ps_stats_params = default_ps_stats_params(ps_stats_params)
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
# print('FAKE!')
# ps_fs = get_fake_ps_data(session_idx)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
)

resp_ps_events = resp_ps_extras['resp_ps_events']

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4,))
fig8, ((ax8, ax8p,), (ax8pp, ax8ppp,)) = plt.subplots(2, 2, figsize=(12, 8,))
fig7, (ax7, ax7p, ax7pp) = plt.subplots(1, 3, figsize=(7, 4,), gridspec_kw={'width_ratios': [10., 1., 3.]})
fig9, (ax9, ax9p) = plt.subplots(1, 2, figsize=(12, 4,))
fig6, ax6 = plt.subplots(1, 1, figsize=(6, 4))

# For each photostim event, sees how indirect responses are related to the direct response
exemplar_group_idx = 0 # 0, 5
exemplar_neuron_idx = 35

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]
    
    dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
    indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
    
#     if shuffle_events: # Shuffle indirect events, so they are not synced with direct events
#         indir_resp_ps_events = shuffle_along_axis(indir_resp_ps_events, axis=-1) # Shuffle along event axis
        
#     keep_event_idxs = np.where(~np.any(np.isnan(dir_resp_ps_events), axis=0))[0]
#     dir_resp_ps_events = dir_resp_ps_events[:, keep_event_idxs]
#     indir_resp_ps_events = indir_resp_ps_events[:, keep_event_idxs]
    
    n_direct = dir_resp_ps_events.shape[0]
    n_indirect = indir_resp_ps_events.shape[0]
    
    ### Plot and fit all indirect responses ###
    # (n_neurons, n_events,) -> (n_dir, n_events,) -> (n_events,)
    sum_dir_resp_ps_events = np.nansum(dir_resp_ps_events, axis=0, keepdims=True)
    sum_dir_resp_ps_events = np.repeat(sum_dir_resp_ps_events, n_indirect, axis=0)
    
    plot_ax = ax5 if exemplar_group_idx == group_idx else None
    
    slope, _, rvalue, pvalue, _ = add_regression_line(
        sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(), 
        fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=plot_ax, color='k', zorder=5
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
    )
    
    group_event_slope[group_idx] = slope
    group_event_rsquared[group_idx] = rvalue**2
    
    if exemplar_group_idx == group_idx:
        ax5.scatter(sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(),
                marker='.', alpha=0.3, color=c_vals_l[0])
        
        # Plot a single direct neuron example
        ax5.scatter(sum_dir_resp_ps_events[exemplar_neuron_idx, :], indir_resp_ps_events[exemplar_neuron_idx, :],
                    marker='.', zorder=1, color=c_vals[1])
        _ = add_regression_line(
            sum_dir_resp_ps_events[exemplar_neuron_idx, :], indir_resp_ps_events[exemplar_neuron_idx, :], 
            fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[1], zorder=3, linestyle='dotted'
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        )
        
        ax5.scatter( # Also plot mean responses
            np.nansum(resp_ps[direct_idxs, group_idx]) * np.ones(indirect_idxs.shape[0]),
            resp_ps[indirect_idxs, group_idx], marker='.', alpha=0.3, color=c_vals_d[0]
        )
        
        ax5.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax5.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax5.legend()
        ax5.set_xlabel('Sum direct response (group_idx {})'.format(group_idx))
        ax5.set_ylabel('Indirect responses (group_idx {})'.format(group_idx))
        
        # Now compute fit for every neuron using direct sum as explainer
        ps_stats_params_copy = copy.deepcopy(ps_stats_params)
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        ps_stats_params_copy['direct_predictor_mode'] = 'sum'
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        ps_stats_params_copy['n_direct_predictors'] = 1
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        
        direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params_copy, return_extras=True,
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        )
        
        if shuffle_events:
            plot_iterator = zip(
                (indir_resp_ps_events, shuffle_along_axis(indir_resp_ps_events, axis=-1),), # Shuffle along event axis
                (c_vals[0], 'grey',), # All neuron color
                (c_vals[1], c_vals[3],), # CN color
                (0, -5,), # zorder shift
            )
        else:
            plot_iterator = zip(
                (indir_resp_ps_events,), 
                (c_vals[0],), # All neuron color
                (c_vals[1],), # CN color
                (0,), # zorder shift
            )
            
        slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        
        for indir_resp_ps_events_it, neuron_color, cn_color, zorder_shift in plot_iterator:
        
            indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
                dir_resp_ps_events, indir_resp_ps_events_it, direct_predictors, direct_shift,
                ps_stats_params_copy, verbose=True, return_extras=True,
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
            )

    #         ax8.scatter(indirect_params[:, slope_idx], -np.log10(indirect_pvalues[:, slope_idx]), marker='.', color=c_vals[2])
            ax8.scatter(indirect_params[:, slope_idx], fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
    #         ax8.scatter(indirect_params[exemplar_neuron_idx, slope_idx], -np.log10(indirect_pvalues[exemplar_neuron_idx, slope_idx]), marker='o', color=c_vals[3])
            ax8.scatter(indirect_params[exemplar_neuron_idx, slope_idx], fit_extras['r_squareds'][exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
            ax8.axhline(np.mean(fit_extras['r_squareds']), color=neuron_color, linestyle='dashed', zorder=-5+zorder_shift)
            
            direct_input = np.nanmean(sum_dir_resp_ps_events)
            indirect_prediction = photostim_predict(indirect_params, direct_input, ps_stats_params_copy)
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)

            ax8p.scatter(resp_ps[indirect_idxs, group_idx], indirect_prediction, marker='.', color=neuron_color, zorder=zorder_shift)
            ax8p.scatter(resp_ps[indirect_idxs, group_idx][exemplar_neuron_idx], indirect_prediction[exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
        
            ax8pp.scatter(np.nanstd(indir_resp_ps_events_it, axis=-1), fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
            ax8pp.scatter(np.nanstd(indir_resp_ps_events_it, axis=-1)[exemplar_neuron_idx], fit_extras['r_squareds'][exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
            
            ax8ppp.scatter(np.nanmean(indir_resp_ps_events_it, axis=-1) /np.nanstd(indir_resp_ps_events_it, axis=-1), 
                           fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
            
        ax8.set_xlabel('Direct-indirect slope')
#         ax8.set_ylabel('-log10(p)')
        ax8.set_ylabel('r^2')
        ax8.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8p.set_xlabel('Mean indirect response')
        ax8p.set_ylabel('Interpolated indirect response')
        ax8p.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8p.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8pp.set_xlabel('Std indirect resp. (over events)')
        ax8pp.set_ylabel('r^2')
        ax8pp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8pp.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8ppp.set_xlabel('Mean/Std indirect resp. (over events)')
        ax8ppp.set_ylabel('r^2')
        ax8ppp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8ppp.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
#         ax8pp.scatter(indirect_params[:, slope_idx], marker='.', color=c_vals_l[0])
        
        direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params, return_extras=True,
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        )
        indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
            dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
            ps_stats_params, verbose=True, return_extras=True,
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        )
        
        # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
        direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
        direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)
        indirect_prediction = photostim_predict(indirect_params, direct_input, ps_stats_params)
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        
        # Show diversity in direct stimulations
        max_val = np.nanmax(np.abs(np.array(resp_ps_events[group_idx])[direct_idxs, :]))
        ax7.matshow(np.array(resp_ps_events[group_idx])[direct_idxs, :], 
                    vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        ax7p.matshow(np.nanmean(np.array(resp_ps_events[group_idx])[direct_idxs, :], axis=-1, keepdims=True), 
                     vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        ax7pp.matshow(direct_predictors.T,
                      vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        
        ax7p.set_xticks([])
        ax7pp.set_xticks([])
        ax7.set_xlabel('Event idx (group_idx {})'.format(group_idx))
        ax7p.set_xlabel('Mean')
        ax7.set_ylabel('Dir. neuron idx (group_idx {})'.format(group_idx))
        ax7pp.set_xlabel('Pred. dirs.')
    
        slope_idx = 1 if ps_stats_params['direct_predictor_intercept_fit'] else 0
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
        
        for direct_max_idx in range(ps_stats_params['n_direct_predictors']):
            if ps_stats_params['direct_predictor_mode'] == 'top_mags': # neurons with largest L2 mag across events
    with open('outputs/ps_stats_params.pkl', 'wb') as f:
        pickle.dump(ps_stats_params, f)
                label = 'Max Neuron {}'.format(direct_max_idx)
            elif ps_stats_params['direct_predictor_mode'] in ('top_devs', 'top_devs_center'): 
                label = 'Dev dir {}'.format(direct_max_idx)
            elif ps_stats_params['direct_predictor_mode'] in ('sum',):
                label = None
#             ax9.scatter(neuron_slopes[:, direct_max_idx], -np.log10(neuron_pvalues)[:, direct_max_idx], 
#                         marker='.', color=c_vals[direct_max_idx], label=label)
            ax9.scatter(indirect_params[:, slope_idx+direct_max_idx], -np.log10(indirect_pvalues)[:, slope_idx+direct_max_idx], 
                        marker='.', color=c_vals[direct_max_idx], label=label)
            ax9p.scatter(indirect_params[:, slope_idx+direct_max_idx], fit_extras['r_squareds'], 
                         marker='.', color=c_vals[direct_max_idx], label=label)
                                 
        for ax in (ax9, ax9p,):
            ax.set_xlabel('Direct-indirect parameter')
            ax.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.legend()
                                 
        ax9.set_ylabel('-log10(p)')
        ax9p.set_ylabel('r^2')
        

ax6.scatter(group_event_rsquared, group_event_slope, color=c_vals[1], marker='.')

ax6.set_xlabel('$r**2$ of group\'s event resp_ps')
ax6.set_ylabel('slope of group\'s event resp_ps')

ax6.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
ax6.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
ax6.axvline(1.0, color='lightgrey', zorder=-5, linestyle='dashed')

    print("✅ setup_analysis ran successfully.")
if __name__ == '__main__':
    run()