from utils.helper_functions1 import default_ps_stats_params
from utils.helper_functions1 import get_data_dict

def run():
    # Cell 18
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}

ps_stats_params = default_ps_stats_params(ps_stats_params)

exemplar_pair_idx = 6 # 6: (11, 12)
exemplar_group_idx = 5 # 0, 5
exemplar_neuron_idx = 10

n_pairs = len(session_idx_pairs)

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4,))
fig7, (ax7, ax7p, ax8, ax8p) = plt.subplots(1, 4, figsize=(12, 4,), gridspec_kw={'width_ratios': [10., 1., 10., 1.]})
fig1, (ax1, ax1p, ax1pp) = plt.subplots(1, 3, figsize=(10, 4))

differences = []
differences_vector = []
prs_direct_1 = []
prs_direct_2 = []
prs_direct_1_2 = []
n_direct_for_pr = []

n_direct = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
n_events_1 = [[] for _ in range(n_pairs)] 
n_events_2 = [[] for _ in range(n_pairs)]

# Gets data on how many entries are nans across groups justify if we should just toss out entire events
percent_dir_nans_1 = [[] for _ in range(n_pairs)]
percent_dir_nans_2 = [[] for _ in range(n_pairs)] 

# Gets data on how often a given direct neuron is reported as a nan across all events to see if we should toss out entire neurons
percent_event_nans_1 = [[] for _ in range(n_pairs)]
percent_event_nans_2 = [[] for _ in range(n_pairs)] 

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
    
    if day_1_idx == 7:
        print(np.array(resp_ps_events_1[0]).shape)

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

        dir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        
        # Nan handling
        percent_dir_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=0) / dir_resp_ps_events_1.shape[0]
        )
        percent_event_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=1) / dir_resp_ps_events_1.shape[1]   
        )
        percent_dir_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=0) / dir_resp_ps_events_2.shape[0]
        )
        percent_event_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=1) / dir_resp_ps_events_1.shape[1]   
        )
        # Eliminate events with all nans
        keep_event_idxs_1 = np.where(~np.all(np.isnan(dir_resp_ps_events_1), axis=0))[0]
        dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
        indir_resp_ps_events_1 = indir_resp_ps_events_1[:, keep_event_idxs_1]
        keep_event_idxs_2 = np.where(~np.all(np.isnan(dir_resp_ps_events_2), axis=0))[0]
        dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        indir_resp_ps_events_2 = indir_resp_ps_events_2[:, keep_event_idxs_2]
        if ps_stats_params['direct_predictor_nan_mode'] in ('ignore_nans',):
            dir_resp_ps_events_1 = np.where(np.isnan(dir_resp_ps_events_1), 0., dir_resp_ps_events_1) # Fill nans with 0s
            dir_resp_ps_events_2 = np.where(np.isnan(dir_resp_ps_events_2), 0., dir_resp_ps_events_2) # Fill nans with 0s
        elif ps_stats_params['direct_predictor_nan_mode'] in ('eliminate_events',):
            keep_event_idxs_1 = np.where(~np.any(np.isnan(dir_resp_ps_events_1), axis=0))[0]
            dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
            keep_event_idxs_2 = np.where(~np.any(np.isnan(dir_resp_ps_events_2), axis=0))[0]
            dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        
        # No events left
        if dir_resp_ps_events_1.shape[1] == 0 or dir_resp_ps_events_2.shape[1] == 0:
            continue
        # No neurons
        if dir_resp_ps_events_1.shape[0] == 0 or dir_resp_ps_events_2.shape[0] == 0:
            continue
        
        pca_1 = PCA()
        pca_1.fit(dir_resp_ps_events_1.T)
        prs_direct_1.append(participation_ratio_vector(pca_1.explained_variance_))
        
        pca_2 = PCA()
        pca_2.fit(dir_resp_ps_events_2.T)
        prs_direct_2.append(participation_ratio_vector(pca_2.explained_variance_))
        
        pca_1_2 = PCA()
        pca_1_2.fit(np.concatenate((dir_resp_ps_events_1, dir_resp_ps_events_2,), axis=-1).T)
        prs_direct_1_2.append(participation_ratio_vector(pca_1_2.explained_variance_))
        
        n_direct_for_pr.append(dir_resp_ps_events_1.shape[0])
        
        n_direct[pair_idx].append(dir_resp_ps_events_1.shape[0])
        n_events_1[pair_idx].append(dir_resp_ps_events_1.shape[-1])
        n_events_2[pair_idx].append(dir_resp_ps_events_2.shape[-1])
        
        if pair_idx == exemplar_pair_idx and group_idx == exemplar_group_idx:
            
            n_indirect = indir_resp_ps_events_1.shape[0]
            sum_dir_resp_ps_events_1 = np.repeat(np.nansum(dir_resp_ps_events_1, axis=0, keepdims=True), n_indirect, axis=0)
            sum_dir_resp_ps_events_2 = np.repeat(np.nansum(dir_resp_ps_events_2, axis=0, keepdims=True), n_indirect, axis=0)
            
            ax5.scatter(sum_dir_resp_ps_events_1.flatten(), indir_resp_ps_events_1.flatten(),
                        marker='.', alpha=0.3, color=c_vals_l[0])
            ax5.scatter(sum_dir_resp_ps_events_2.flatten(), indir_resp_ps_events_2.flatten(),
                        marker='.', alpha=0.3, color=c_vals_l[1])
            
            ax5.scatter( # Also plot mean responses
                np.nanmean(sum_dir_resp_ps_events_1, axis=-1), np.nanmean(indir_resp_ps_events_1, axis=-1),
                marker='.', alpha=0.3, color=c_vals_d[0]
            )
            ax5.scatter(
                np.nanmean(sum_dir_resp_ps_events_2, axis=-1), np.nanmean(indir_resp_ps_events_2, axis=-1),
                marker='.', alpha=0.3, color=c_vals_d[1]
            )
        
            # Plot a single direct neuron example
            ax5.scatter(sum_dir_resp_ps_events_1[exemplar_neuron_idx, :], indir_resp_ps_events_1[exemplar_neuron_idx, :],
                        marker='.', zorder=1, color=c_vals[0])
            ax5.scatter(sum_dir_resp_ps_events_2[exemplar_neuron_idx, :], indir_resp_ps_events_2[exemplar_neuron_idx, :],
                        marker='.', zorder=1, color=c_vals[1])
            _ = add_regression_line(
                sum_dir_resp_ps_events_1[exemplar_neuron_idx, :], indir_resp_ps_events_1[exemplar_neuron_idx, :], 
                fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[0], zorder=3, linestyle='dotted'
            )
            _ = add_regression_line(
                sum_dir_resp_ps_events_2[exemplar_neuron_idx, :], indir_resp_ps_events_2[exemplar_neuron_idx, :], 
                fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[1], zorder=3, linestyle='dotted'
            )

            ax5.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax5.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax5.legend()
            ax5.set_xlabel('Sum direct response (group_idx {})'.format(group_idx))
            ax5.set_ylabel('Indirect responses (group_idx {})'.format(group_idx))
            
            
            # Show diversity in direct stimulations
            max_val = np.max((np.nanmax(dir_resp_ps_events_1), np.nanmax(dir_resp_ps_events_2),))
            
            ax7.matshow(dir_resp_ps_events_1, vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            ax7p.matshow(np.nanmean(dir_resp_ps_events_1, axis=-1, keepdims=True), 
                         vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            print('Day 1 sum: {:.1e} vs {:.1e}'.format(np.sum(np.nanmean(dir_resp_ps_events_1, axis=-1, keepdims=True)), np.nansum(dir_resp_ps_1)))
            
            ax8.matshow(dir_resp_ps_events_2, vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            ax8p.matshow(np.nanmean(dir_resp_ps_events_2, axis=-1, keepdims=True), 
                         vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            
            print('Day 2 sum: {:.1e} vs {:.1e}'.format(np.sum(np.nanmean(dir_resp_ps_events_2, axis=-1, keepdims=True)), np.nansum(dir_resp_ps_2)))
            
            print('Vec diff: {:.1e}'.format(
                np.linalg.norm(np.nanmean(dir_resp_ps_events_2, axis=-1) - np.nanmean(dir_resp_ps_events_1, axis=-1))/
                (1/2 * np.linalg.norm(np.nanmean(dir_resp_ps_events_1, axis=-1)) + 1/2 * np.linalg.norm(np.nanmean(dir_resp_ps_events_2, axis=-1)))
            ))
            
            ax7.set_xlabel('Event idx (group_idx {}, Day 1)'.format(group_idx))
            ax8.set_xlabel('Event idx (group_idx {}, Day 2)'.format(group_idx))
            ax7p.set_xticks([])
            ax7p.set_xlabel('Mean')
            ax8p.set_xticks([])
            ax8p.set_xlabel('Mean')
            ax7.set_ylabel('Dir. neuron idx (group_idx {})'.format(group_idx))
    
    ax1.scatter(sum_dir_resp_ps_1, sum_dir_resp_ps_2, marker='.', color=c_vals[0], alpha=0.3)
    
    differences.append(
        np.abs(sum_dir_resp_ps_2 - sum_dir_resp_ps_1) / (1/2 * (np.abs(sum_dir_resp_ps_2) + np.abs(sum_dir_resp_ps_1)))
    )

add_identity(ax1, color='lightgrey', zorder=-5, linestyle='dashed')
    
differences = np.concatenate(differences, axis=0)

ax1p.hist(differences, color=c_vals[0], bins=30)
ax1p.axvline(np.mean(differences), color=c_vals_d[0], zorder=5)
ax1pp.hist(np.array(differences_vector), color=c_vals[1], bins=30)
ax1pp.axvline(np.mean(differences_vector), color=c_vals_d[1], zorder=5)

for ax in (ax1p, ax1pp):
    ax.set_xlim((0, 2.,))

ax1.set_xlabel('Day 1 direct response mag.')
ax1.set_ylabel('Day 2 direct response mag.')
ax1p.set_xlabel('Perc. sum diff')
ax1pp.set_xlabel('Perc. vector diff')

fig3, (ax3, ax3p) = plt.subplots(1, 2, figsize=(8, 4))

mouse_offset = 40 # Spaces out plots to see data across mice better

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    ax3.scatter(n_direct[pair_idx], (mouse_offset * PAIR_COLORS[pair_idx]) + np.array(n_events_1[pair_idx]), marker='.', 
                color=c_vals[PAIR_COLORS[pair_idx]], alpha=0.3)
    ax3p.scatter(n_direct[pair_idx], (mouse_offset * PAIR_COLORS[pair_idx]) + np.array(n_events_2[pair_idx]), marker='.', 
                 color=c_vals[PAIR_COLORS[pair_idx]], alpha=0.3)
    
    n_events_reject_1 = np.where(np.array(n_events_1[pair_idx]) < N_MIN_EVENTS)[0].shape[0]
    n_events_reject_2 = np.where(np.array(n_events_2[pair_idx]) < N_MIN_EVENTS)[0].shape[0]
    
    print('Pair idx {} - n_events reject 1: {},\t2: {}'.format(
        pair_idx, n_events_reject_1, n_events_reject_2
    ))

for ax in (ax3, ax3p):
    ax.set_xlabel('n_direct')
   
    ax.set_yticks((0, 20, 40, 60, 80, 100, 120, 140, 160, 180))
    ax.set_yticklabels((None, 20, None, 20, None, 20, None, 20, None, 20))
    
    for sep in (0, 40, 80, 120, 160, 200):
        ax.axhline(sep, color='grey')
        ax.axhline(sep + 10, color='lightgrey', zorder=-5, linestyle='dashed')

ax3.set_ylabel('n_events_1')
ax3p.set_ylabel('n_events_2')

    print("✅ select_exemplars ran successfully.")
if __name__ == '__main__':
    run()