from utils.helper_functions1 import get_data_dict

def run():
    # Cell 49
# ps_stats_params['pairwise_corr_type'] = 'trial' # trace, trial

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))
    
session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

print('Evaluating {} sessions...'.format(len(session_idxs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

cn_idxs = []
avg_directs = []
avg_directs_pred = []
avg_indirects = []
avg_indirects_pred = []

n_sessions = len(session_idxs)
    
for session_idx_idx, session_idx in enumerate(session_idxs):
    day_1_idx = session_idx
    print('Session idx: {}'.format(session_idx))
    
    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 'resp_ps', 
                       'resp_ps_pred', 'mean_trial_activity', 'trial_start_metrics_changes',]

    data_1 = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)
    
    cn_idxs.append(data_dict['data']['conditioned_neuron'][session_idx] - 1) # -1 to correct for MatLab indexing
    
    # Anything that does not make it past the mask is automatically a nan
    direct_mask = np.where(data_1['dir_mask'] > 0, 1., np.nan)
    indirect_mask = np.where(data_1['indir_mask'] > 0, 1., np.nan)
    
    avg_directs.append(direct_mask * data_1['resp_ps'])
    avg_directs_pred.append(direct_mask * data_1['resp_ps_pred'])
    
    avg_indirects.append(indirect_mask * data_1['resp_ps'])
    avg_indirects_pred.append(indirect_mask * data_1['resp_ps_pred'])
    
    indirect_counts = np.sum(indir_mask_weighted > 0, axis=-1)
    print('CN indirect count:', indirect_counts[cn_idxs[-1]])

del data_1

    print("✅ find_valid_pairs_again ran successfully.")
if __name__ == '__main__':
    run()