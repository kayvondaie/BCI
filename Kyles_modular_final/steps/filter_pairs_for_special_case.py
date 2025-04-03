from utils.helper_functions1 import get_data_dict

def run():
    # Cell 51
# ps_stats_params['pairwise_corr_type'] = 'trial' # trace, trial

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

delta_correlations = []
delta_tunings = []
posts_1 = []
posts_2 = []
delta_trial_resps = []
day_2_ts_metrics_changes = []
day_2_cn_idxs = []
day_2_dist_to_cn = []

day_1_trial_activities = [] # For determining random CN draws
day_1_tunings = []


for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    print('Pair idx: {}'.format(pair_idx))
    
    day_2_idx = session_idx_pair[1]
    day_1_idx = session_idx_pair[0]

    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 
                       'trial_start_metrics_changes', 'mean_trial_activity']

#     if ps_stats_params['direct_predictor_mode'] is not None:
#         data_to_extract.append('resp_ps_pred')
#         resp_ps_type = 'resp_ps_pred'
#         if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
#             data_to_extract.append('resp_ps')
#             resp_ps_type = 'resp_ps'
#     else:
#         data_to_extract.append('resp_ps')
#         resp_ps_type = 'resp_ps'
        
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
    data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
    
    delta_correlations.append(data_2['pairwise_corr'] - data_1['pairwise_corr'])
    
    print('{} Pair idx {}'.format(data_dict['data']['mouse'][day_2_idx], pair_idx), np.nanmean(data_2['pairwise_corr']))
    delta_tunings.append(data_2['tuning'] - data_1['tuning'])
    delta_trial_resps.append(data_2['trial_resp'] - data_1['trial_resp'])
    
    posts_1.append(data_1['post'])
    posts_2.append(data_2['post'])
    
    day_2_ts_metrics_changes.append(data_2['trial_start_metrics_changes'])
    day_2_cn_idxs.append(data_dict['data']['conditioned_neuron'][day_2_idx] - 1) # -1 to correct for MatLab indexing
    day_2_dist_to_cn.append(data_dict['data']['dist'][day_2_idx])
    
    day_1_trial_activities.append(data_1['mean_trial_activity'])
    day_1_tunings.append(data_1['tuning'])
    
#     for plot_1, plot_2, ax, ax_bounds in zip(plot_1s, plot_2s, axs, axs_bounds):
#         ax.scatter(
#             plot_1, plot_2, color=c_vals[PAIR_COLORS[pair_idx]], marker = '.', alpha=0.3
#         )

#         if min(np.min(plot_1), np.min(plot_2)) > ax_bounds[0]:
#             ax_bounds[0] = min(np.min(plot_1), np.min(plot_2))
#         if max(np.max(plot_1), np.max(plot_2)) > ax_bounds[-1]:
#             ax_bounds[-1] = max(np.max(plot_1), np.max(plot_2))

#         add_identity(ax, color='k', zorder=5, linestyle='dashed')
        
#         print(' id added')

del data_1
del data_2

    print("✅ filter_pairs_for_special_case ran successfully.")
if __name__ == '__main__':
    run()