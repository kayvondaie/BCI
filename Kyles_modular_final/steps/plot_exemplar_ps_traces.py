from utils.helper_functions1 import get_data_dict

def run():
    # Cell 94
day_1_idx = 1
group_idx = 0
seq = data_dict['data']['seq'][day_1_idx]
raw_resp_ps = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
sq = (seq - 1).astype(np.int32) # -1 to account for Matlab indexing

d_ps_flat_1 = data_dict['data']['x'][day_1_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, ps_fs_1.shape[1],)

group_trial_idxs = np.where(sq == group_idx)[0] # All ps event indexes that match group index
assert group_trial_idxs.shape[0] > 0
        
direct_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] < D_DIRECT, d_ps_2[:, group_idx] < D_DIRECT))[0]
    
    
# Mean over time first, then PS events
raw_resp_ps_by_group = raw_resp_ps[:, :, group_trial_idxs] # (ps_times, n_neurons, n_group_events,)    

pre_resp_ps_events = np.nanmean(raw_resp_ps_by_group[IDXS_PRE_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,) 
post_resp_ps_events = np.nanmean(raw_resp_ps_by_group[IDXS_POST_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,)
baseline_resp_ps_events = np.nanmean(pre_resp_ps_events, axis=-1) # (n_neurons, n_group_events,) -> (n_neurons,) 

resp_ps_events = (post_resp_ps_events - pre_resp_ps_events) / baseline_resp_ps_events[:, np.newaxis] # (n_neurons, n_group_events,) 

fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(9, 4))
        
ax1.matshow(np.isnan(resp_ps_events[direct_idxs, :]))
ax1.set_xlabel('Event idx')
ax1.set_ylabel('Direct neuron idx')

event_idx = 1
direct_neuron_idx1 = 5
direct_neuron_idx2 = 6

ax1.scatter(event_idx, direct_neuron_idx1, color=c_vals[1], marker='o')
ax1.scatter(event_idx, direct_neuron_idx2, color=c_vals[0], marker='o')

print('Direct idx: {}'.format(direct_neuron_idx1), raw_resp_ps[:, direct_idxs[direct_neuron_idx1], group_trial_idxs[event_idx]])
print('Direct idx: {}'.format(direct_neuron_idx2), raw_resp_ps[:, direct_idxs[direct_neuron_idx2], group_trial_idxs[event_idx]])

ax1p.plot(raw_resp_ps[:, direct_idxs[direct_neuron_idx1], group_trial_idxs[event_idx]], color=c_vals[1])
ax1p.plot(raw_resp_ps[:, direct_idxs[direct_neuron_idx2], group_trial_idxs[event_idx]], color=c_vals[0])

print('Direct idx: {} - Neuron idx {}'.format(direct_neuron_idx1, direct_idxs[direct_neuron_idx1]))
print('Direct idx: {} - Neuron idx {}'.format(direct_neuron_idx2, direct_idxs[direct_neuron_idx2]))

    print("✅ plot_exemplar_ps_traces ran successfully.")
if __name__ == '__main__':
    run()