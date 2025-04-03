from utils.helper_functions1 import default_ps_stats_params
from utils.helper_functions1 import get_data_dict

def run():
    # Cell 91
exemplar_session_idx = 1 # 11

weight_type = 'rsquared' # None, rsquared

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'top_mags', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 4,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

session_idx = 1

print('Session idx {}'.format(session_idx))

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False,
)

resp_ps_events = resp_ps_extras['resp_ps_events']

pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
# Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)

direct_idxs_flat = np.where(d_ps.flatten() < D_DIRECT)[0]
indirect_idxs_flat = np.where(np.logical_and(d_ps.flatten() > D_NEAR, d_ps.flatten() < D_FAR))[0]

group_corrs = [] # Filled as we iterate across groups
indirect_resp_ps = []
indirect_predictions = []

r_squareds = []

direct_predictors_all = [] # will be (n_groups, n_direct_predictors, n_indirect)

if ps_stats_params['direct_predictor_mode'] in ('top_mags',):
    direct_to_indirect_params = [[[] for _ in range(n_neurons)] for _ in range(n_neurons)] 

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]

    dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
    indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)

    direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
        dir_resp_ps_events, ps_stats_params,
    )
    
    direct_predictors_all.append(direct_predictors)
    
    indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
        dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
        ps_stats_params, verbose=False, return_extras=True
    )
    
    r_squareds.append(fit_extras['r_squareds'])
    indirect_resp_ps.append(resp_ps[indirect_idxs, group_idx])
#         sum_direct_resp_ps = np.nansum(resp_ps[direct_idxs, group_idx]) # Avg. over events, then neurons
#         sum_direct_resp_ps =  np.nanmean(np.nansum(dir_resp_ps_events, axis=0)) # Over neurons, then avg events

    ### Gets average direct input ###
    # Note this way of doing determines average input for each event THEN averages over
    # events. This does not necessarily yield the same result as averaging over events
    # first then determining the input because of how we treat nans.

    # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
    direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
    direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)

#         print('Sum:', sum_direct_resp_ps)
#         print('Sum events, mean first:', np.nansum(np.nanmean(dir_resp_ps_events, axis=-1)))
#         print('Sum events, sum first:', np.nanmean(np.nansum(dir_resp_ps_events, axis=0)))
#         print('Matmul:', direct_input)

#         if ps_stats_params['direct_predictor_intercept_fit']:
#             indirect_prediction = indirect_params[:, 0] + indirect_params[:, 1] * sum_direct_resp_ps
#         else:
#             indirect_prediction = indirect_params[:, 0] * sum_direct_resp_ps

#         print('Man pred', indirect_prediction[:3])

    ### Uses average direct input to predict photostimulation response ###
    indirect_predictions.append(photostim_predict(indirect_params, direct_input, ps_stats_params))

    group_corrs.append(np.matmul(pairwsie_corrs[:, direct_idxs], resp_ps[direct_idxs, group_idx])[indirect_idxs])
#         group_corrs_norm = np.matmul(pairwsie_corrs[:, direct_idxs], resp_ps[direct_idxs, group_idx] / np.sum(resp_ps[direct_idxs, group_idx]))[indirect_idxs]
    
    if ps_stats_params['direct_predictor_mode'] in ('top_mags',):
        top_mag_neurons = direct_idxs[np.where(direct_predictors > 0)[-1]]
        
        for direct_idx, top_mag_neuron_idx in enumerate(top_mag_neurons):
            for indirect_idx, indirect_neuron_idx in enumerate(indirect_idxs):
                direct_to_indirect_params[top_mag_neuron_idx][indirect_neuron_idx].append(
                    indirect_params[indirect_idx, direct_idx]
                )
    
    
group_corrs = np.concatenate(group_corrs, axis=0) # Filled as we iterate across groups
indirect_resp_ps = np.concatenate(indirect_resp_ps, axis=0)
indirect_predictions = np.concatenate(indirect_predictions, axis=0)
    
r_squareds = np.concatenate(r_squareds, axis=0)
    
if weight_type in (None,):
    weights = None
elif weight_type in ('rsquared',):
    weights = np.copy(r_squareds[session_idx_idx])
else:
    raise ValueError('Weight type {} not recognized'.format(weight_type))


fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

n_entries = np.zeros((n_neurons, n_neurons,))

MIN_COUNT = 5
count = 0

for direct_idx in range(n_neurons):
    for indirect_idx in range(n_neurons):
        entry = direct_to_indirect_params[direct_idx][indirect_idx]
        n_entries[direct_idx, indirect_idx] = len(entry)
        
        if len(entry) >= MIN_COUNT:
            ax2.scatter(count * np.ones((len(entry),)), entry, color=c_vals_l[0], marker='.', zorder=-4)
            ax2.errorbar(count, np.mean(entry), np.std(entry), color=c_vals[0], marker='_')
            
            
            count+=1
#             print(direct_to_indirect_params[direct_idx][indirect_idx])
ax2.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')

ax2.set_ylabel('Direct -> Indirect Parameter')
ax2.set_xlabel('Entries with >= {} fits.'.format(MIN_COUNT))

    print("✅ plot_exemplar_session ran successfully.")
if __name__ == '__main__':
    run()