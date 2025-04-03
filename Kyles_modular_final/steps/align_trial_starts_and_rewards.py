from utils.helper_functions1 import get_data_dict

def run():
    # Cell 110
dff = data_dict_behav['df_closedLoop']

trial_start_idxs = np.where(data_dict_behav['trial_start'][0, :] == 1)[0]
rew_idxs = np.where(data_dict_behav['rew'][0, :] == 1)[0]
n_trial_start = trial_start_idxs.shape[0]

ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_idxs = np.arange(
    int(ts_range[0] * SAMPLE_RATE), int(ts_range[1] * SAMPLE_RATE),
) - 1 # - 1 keeps these in sync
n_ts_idxs = ts_idxs.shape[0]

n_neurons = dff.shape[1]

trial_start_dff = np.empty((n_ts_idxs, n_neurons, n_trial_start))
trial_start_dff[:] = np.nan

# Pads dff with nans so we don't need to trim start and end
nan_pad = np.empty((n_ts_idxs, n_neurons,))
nan_pad[:] = np.nan

print(dff.shape)

dff = np.concatenate((nan_pad, dff, nan_pad), axis=0)

print(dff.shape)

for trial_start_idx_idx, trial_start_idx in enumerate(trial_start_idxs):
    
    rel_trial_start_idx = trial_start_idx + n_ts_idxs # Accounts for nan padding
    
#     print(rel_trial_start_idx)
#     print(next_rel_trial_start_idx)
#     print(ts_idxs[:10])
    
    trial_start_dff[:, :, trial_start_idx_idx] = dff[rel_trial_start_idx + ts_idxs, :]
#     if trial_start_idx_idx < n_trial_start - 1:
#         next_rel_trial_start_idx = trial_start_idxs[trial_start_idx_idx + 1] + n_ts_idxs 
#         trial_start_dff[next_rel_trial_start_idx - rel_trial_start_idx:, :, trial_start_idx_idx] = np.nan
    
# This is a hack that just copies tha nan statistics from one to the other
trial_start_dff = np.where(np.isnan(data_dict['data']['F'][session_idx_idx]), np.nan, trial_start_dff)

    print("✅ align_trial_starts_and_rewards ran successfully.")
if __name__ == '__main__':
    run()