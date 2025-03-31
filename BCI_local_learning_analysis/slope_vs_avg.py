from sklearn.decomposition import PCA
import copy

session_idx = 12 # 11
shuffle_events = True # Shuffle indirect events relvative to direct

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

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
)

resp_ps_events = resp_ps_extras['resp_ps_events']


# For each photostim event, sees how indirect responses are related to the direct response
exemplar_group_idx = 8 # 0, 5
exemplar_neuron_idx = 35

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > -10, d_ps[:, group_idx] < 2000))[0]
    
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
    )
    
    group_event_slope[group_idx] = slope
    group_event_rsquared[group_idx] = rvalue**2
    
    
    
    
    
    
    if exemplar_group_idx == group_idx:
    # Find predictors for this group
        ps_stats_params_copy = copy.deepcopy(ps_stats_params)
        ps_stats_params_copy['direct_predictor_mode'] = 'sum'
        ps_stats_params_copy['n_direct_predictors'] = 1

        direct_predictors, direct_shift, _ = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params_copy, return_extras=True
        )

        # Run the fit
        indirect_params, _, _ = fit_photostim_variation(
            dir_resp_ps_events,
            indir_resp_ps_events,
            direct_predictors,
            direct_shift,
            ps_stats_params_copy
        )

        # Get per-neuron slope and mean
        slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
        slopes = indirect_params[:, slope_idx]
        mean_indir_response = np.nanmean(indir_resp_ps_events, axis=1)

        # Mask invalid values
        valid_mask = ~np.isnan(slopes) & ~np.isnan(mean_indir_response)
        slopes = slopes[valid_mask]
        mean_indir_response = mean_indir_response[valid_mask]

        # Plot
        plt.figure(figsize=(6, 5))
        plt.scatter(mean_indir_response, slopes, alpha=0.7)
        plt.xlabel('Mean response of indirect neuron')
        plt.ylabel('Fitted slope from direct predictor')
        plt.title(f'Group {group_idx}: Slope vs. Mean response')
        plt.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()

  
        