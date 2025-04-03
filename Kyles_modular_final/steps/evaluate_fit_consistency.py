from utils.helper_functions1 import get_data_dict

def run():
    # Cell 59
# Assumes this just uses the ps_stats_params from above.

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

day_2_ts_metrics_changes = []

day_1_cn_idxs = []
day_2_cn_idxs = []

direct_masks = [] # Same for both days
indirect_masks = []

day_1_resp_pss = []
day_2_resp_pss = []
day_1_resp_ps_preds = []
day_2_resp_ps_preds = []

day_2_dist_to_cn = []

day_1_rsquared_indirects = []
day_2_rsquared_indirects = []
min_rsquared_indirects = []

day_1_trial_activities = [] # For determining random CN draws
day_1_tunings = []

group_day_2_post = []

# delta_corrs = []

# for pair_idx, session_idx_pair in enumerate(session_idx_pairs[0:3]):
for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    print('Pair idx: {}'.format(pair_idx))
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]

    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 'resp_ps', 
                       'resp_ps_pred', 'mean_trial_activity', 'trial_start_metrics_changes',]
        
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
    data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
    
    day_2_ts_metrics_changes.append(data_2['trial_start_metrics_changes'])
    
    day_1_cn_idxs.append(int(data_dict['data']['conditioned_neuron'][day_1_idx]) - 1) # -1 to correct for MatLab indexing
    day_2_cn_idxs.append(int(data_dict['data']['conditioned_neuron'][day_2_idx]) - 1) # -1 to correct for MatLab indexing
    day_2_dist_to_cn.append(data_dict['data']['dist'][day_2_idx])
    
    indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
        ps_stats_params, data_1, data_2, verbose=False
    )
    
    # Anything that does not make it past the mask is automatically a nan
    direct_mask = np.where(dir_mask_weighted > 0, 1., np.nan)
    indirect_mask = np.where(indir_mask_weighted > 0, 1., np.nan)
    
    direct_masks.append(direct_mask)
    indirect_masks.append(indirect_mask)
    
    day_1_resp_pss.append(data_1['resp_ps'])
    day_2_resp_pss.append(data_2['resp_ps'])
    day_1_resp_ps_preds.append(data_1['resp_ps_pred'])
    day_2_resp_ps_preds.append(data_2['resp_ps_pred'])
    
    # avg_directs.append(direct_mask * 1/2 * (data_2['resp_ps'] + data_1['resp_ps']))
    # avg_directs_pred.append(direct_mask * 1/2 * (data_2['resp_ps_pred'] + data_1['resp_ps_pred']))
    
#     day_1_indirects.append(indirect_mask * data_1['resp_ps'])
#     day_2_indirects.append(indirect_mask * data_2['resp_ps'])
    
#     day_1_indirects.append(indirect_mask * data_1['resp_ps'])
#     day_2_indirects.append(indirect_mask * data_2['resp_ps'])
    
#     delta_indirects.append(indirect_mask * (data_2['resp_ps'] - data_1['resp_ps']))
#     delta_indirects_pred.append(indirect_mask * (data_2['resp_ps_pred'] - data_1['resp_ps_pred']))
    
#     # No indirect masking
#     delta_cc_raw.append()
#     delta_cc_pred.append()
    
    day_1_rsquared_indirects.append(indirect_mask * data_1['resp_ps_pred_extras']['r_squareds'])
    day_2_rsquared_indirects.append(indirect_mask * data_2['resp_ps_pred_extras']['r_squareds'])
    min_rsquared_indirects.append(
        indirect_mask *
        np.minimum(data_1['resp_ps_pred_extras']['r_squareds'], data_2['resp_ps_pred_extras']['r_squareds'])
    )
    
    # indirect_counts.append(np.sum(indir_mask_weighted > 0, axis=-1)) # How many times each neuron is indirect
    # print('CN indirect count:', indirect_counts[-1][day_2_cn_idxs[-1]])
    
#     delta_corrs.append(
#         nan_matmul(data_2['pairwise_corr'], direct_mask * data_2['resp_ps']) -
#         nan_matmul(data_1['pairwise_corr'], direct_mask * data_1['resp_ps'])
#     )
    
    # group_day_2_post.append(nan_matmul(data_2['post'], direct_mask * data_2['resp_ps'])) 
    
    # For CN choices
    day_1_trial_activities.append(data_1['mean_trial_activity'])
    day_1_tunings.append(data_1['tuning'])
    
del data_1
del data_2

    print("✅ evaluate_fit_consistency ran successfully.")
if __name__ == '__main__':
    run()