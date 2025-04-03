# Cell 28
fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))

for pair_idx, ax5 in zip(range(n_pairs), ax5s.flatten()):

# pair_idx = 10

    r_squared_1_pair = r_squareds_1[pair_idx]
    r_squared_2_pair = r_squareds_2[pair_idx]

    ax5.scatter(
        r_squared_1_pair, r_squared_2_pair, color=c_vals[PAIR_COLORS[pair_idx]], marker = '.', alpha=0.3
    )
    add_identity(ax5, color='k', zorder=5, linestyle='dashed')
    _ = add_regression_line(r_squared_1_pair, r_squared_2_pair, ax=ax5, color=c_vals_d[PAIR_COLORS[pair_idx]], 
                            zorder=5, fit_intercept=False)

    ax5.set_xlim((-0.05, 1.05))
    ax5.set_ylim((-0.05, 1.05))
    ax5.axhline(0.1, color='lightgrey', zorder=5)
    ax5.axvline(0.1, color='lightgrey', zorder=5)
    ax5.legend()

# Cell 43
n_cm_x = len(connectivity_metrics_xs)
n_cm_y = len(connectivity_metrics_ys)

bs_ps = np.ones((n_cm_x, n_cm_y, n_bootstrap))

fig1, ax1s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # -log10(p-values)
fig2, ax2s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # parameters

bar_locs = np.array((0.,))

for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):  
            
#         all_ps = -1 * np.log10(all_ps)
#         if max(all_ps) > max_p_for_this_x:
#             max_p_for_this_x = max(all_ps)
        
        frac_pos = np.sum(bs_params[cm_x_idx, cm_y_idx, :] > 0) / n_bootstrap
        ax1s[cm_x_idx, cm_y_idx].scatter(np.array((0., 1.)), np.array((frac_pos, 1 - frac_pos,)), color='k')
#         ax1s[cm_x_idx, cm_y_idx].errorbar(
#             bar_locs[point_idx], all_params[point_idx], yerr=all_stderrs[point_idx], 
#             color=bar_colors[point_idx], linestyle='None'
#         )
        ax2s[cm_x_idx, cm_y_idx].scatter(bar_locs, np.nanmean(bs_params[cm_x_idx, cm_y_idx, :]), color='k', marker='_')
        ax2s[cm_x_idx, cm_y_idx].errorbar(
            bar_locs, np.nanmean(bs_params[cm_x_idx, cm_y_idx, :]), 
            yerr=np.nanstd(bs_params[cm_x_idx, cm_y_idx, :]), 
            color='k', linestyle='None'
        )
        
        ax1s[cm_x_idx, cm_y_idx].set_ylim((-0.1, 1.1))
        ax1s[cm_x_idx, cm_y_idx].axhline(0., color='grey', zorder=-5, linewidth=1.0)
        ax1s[cm_x_idx, cm_y_idx].axhline(.5, color='grey', zorder=-5, linewidth=1.0)
        ax1s[cm_x_idx, cm_y_idx].axhline(1., color='grey', zorder=-5, linewidth=1.0)
        ax2s[cm_x_idx, cm_y_idx].axhline(0., color='grey', zorder=-5, linewidth=1.0)
        
        for axs in (ax1s, ax2s,):
            axs[cm_x_idx, cm_y_idx].set_xticks(())
            if cm_x_idx == n_cm_x - 1:
                axs[cm_x_idx, cm_y_idx].set_xlabel(connectivity_metrics_y, fontsize=8)
            if cm_y_idx == 0:
                axs[cm_x_idx, cm_y_idx].set_ylabel(connectivity_metrics_x, fontsize=8)

# Cell 93
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}

exemplar_pair_idx = 0

n_pairs = len(session_idx_pairs)

differences = []
differences_vector = []
prs_direct_1 = []
prs_direct_2 = []
prs_direct_1_2 = []
n_direct_for_pr = []

n_direct = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
n_events_1 = [[] for _ in range(n_pairs)] 
n_events_2 = [[] for _ in range(n_pairs)]

percent_dir_nans_1 = [[] for _ in range(n_pairs)] # Gets data on how many entries are nans to justify how we're treating them
percent_dir_nans_2 = [[] for _ in range(n_pairs)] 

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]
    
    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    print('Pair {} - Sessions {} and {} - Mouse {}.'.format(
        pair_idx, day_1_idx, day_2_idx, data_dict['data']['mouse'][day_2_idx]
    )) 

    ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx] # This is matlab indexed so always need a -1 here
    ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_1 = data_dict['data']['x'][day_1_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, ps_fs_1.shape[1],)
    resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
        ps_fs_1, ps_events_group_idxs_1, d_ps_1, ps_stats_params,
    )
    resp_ps_events_1 = resp_ps_extras_1['resp_ps_events']

    ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx] # This is matlab indexed so always need a -1 here
    ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_2 = data_dict['data']['x'][day_2_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_2 = unflatted_neurons_by_groups(d_ps_flat_2, ps_fs_2.shape[1],)
    resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
        ps_fs_2, ps_events_group_idxs_2, d_ps_2, ps_stats_params,
    )
    resp_ps_events_2 = resp_ps_extras_2['resp_ps_events']

    n_groups = int(np.max(ps_events_group_idxs_1)) # +1 already accounted for because MatLab indexing
    
    sum_dir_resp_ps_1 = np.zeros((n_groups,))
    sum_dir_resp_ps_2 = np.zeros((n_groups,))
    
    for group_idx in range(n_groups):
        direct_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] < D_DIRECT, d_ps_2[:, group_idx] < D_DIRECT))[0]
        indirect_idxs = np.where(np.logical_and(
            np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
            np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
        ))[0]
        
        # Mean response-based metrics
        dir_resp_ps_1 = resp_ps_1[direct_idxs, group_idx] # (n_direct,)
        indir_resp_ps_1 = resp_ps_1[indirect_idxs, group_idx] # (n_indirect,)

        dir_resp_ps_2 = resp_ps_2[direct_idxs, group_idx] # (n_direct,)
        indir_rresp_ps_2 = resp_ps_2[indirect_idxs, group_idx] # (n_indirect,)
        
        sum_dir_resp_ps_1[group_idx] = np.nansum(dir_resp_ps_1)
        sum_dir_resp_ps_2[group_idx] = np.nansum(dir_resp_ps_2)
        
        differences_vector.append(
            np.linalg.norm(dir_resp_ps_1 - dir_resp_ps_2) /
            (1/2 * (np.linalg.norm(dir_resp_ps_1) + np.linalg.norm(dir_resp_ps_2)))
        )
        
        # Event-based metrics
        dir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        percent_dir_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=0) / dir_resp_ps_events_1.shape[0]
        )
#         dir_resp_ps_events_1 = np.where(np.isnan(dir_resp_ps_events_1), 0., dir_resp_ps_events_1) # Fill nans with 0s
#         keep_event_idxs_1 = np.where(~np.any(np.isnan(dir_resp_ps_events_1), axis=0))[0]
#         dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
        
        dir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        percent_dir_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=0) / dir_resp_ps_events_2.shape[0]
        )
#         dir_resp_ps_events_2 = np.where(np.isnan(dir_resp_ps_events_2), 0., dir_resp_ps_events_2) # Fill nans with 0s
#         keep_event_idxs_2 = np.where(~np.any(np.isnan(dir_resp_ps_events_2), axis=0))[0]
#         dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        
        ax1.matshow(np.isnan(dir_resp_ps_events_1))
        ax1.set_xlabel('Event idx')
        ax1.set_ylabel('Direct neuron idx')
        
        print(percent_dir_nans_1[pair_idx])
        
        print(dsfsdfsd)

# Cell 106
session_idx = 20 # 9, 15
load_behav = True
load_data = False

# Load the corresponding behavioral for each paired session
if load_behav:
    behav_idx = session_idx_to_behav_idx[session_idx]
    print('Loading behav from: {}'.format(mypath + behav[behav_idx]))
    data_dict_behav = scipy.io.loadmat(mypath + behav[behav_idx])

# Load the corresponding data
if load_data: 
    data_idx = session_idx_to_data_idx[session_idx]
    print('Loading data from: {}'.format(mypath + data[data_idx]))
    data_photo = scipy.io.loadmat(mypath + data[data_idx])

# Cell 108
SAMPLE_RATE = 20 # Hz
T_START = -2 # Seconds, time relative to trial start where trial_start_fs begin
TS_POST = (0, 10) # Seconds, time points to include for post-trial start average
TS_PRE 

idxs_post = np.arange(
int((TS_POST[0] - T_START) * SAMPLE_RATE), int((TS_POST[1] - T_START) * SAMPLE_RATE),
)
idxs_pre = np.arange(
    int((TS_PRE[0] - T_START) * SAMPLE_RATE), int((TS_PRE[1] - T_START) * SAMPLE_RATE),
)

# Cell 111
idxs_post = np.arange(
int((TS_POST[0] - T_START) * SAMPLE_RATE), int((TS_POST[1] - T_START) * SAMPLE_RATE),
)
idxs_pre = np.arange(
    int((TS_PRE[0] - T_START) * SAMPLE_RATE), int((TS_PRE[1] - T_START) * SAMPLE_RATE),
)

fit_changes = True

tuning, trial_resp, pre, post, ts_extras  = compute_trial_start_metrics(
    data_dict['data']['F'][session_idx_idx], (idxs_pre, idxs_post), mean_mode='time_first',
    fit_changes=fit_changes
)

tuning_n, trial_resp_n, pre_n, post_n, ts_extras_n  = compute_trial_start_metrics(
    trial_start_dff, (idxs_pre, idxs_post), mean_mode='time_first',
    fit_changes=fit_changes
)

