from utils.helper_functions1 import get_data_dict

def run():
    # Cell 115
session_idx = 18

ps_stats_params = {
    'pairwise_corr_type': 'behav_full', # trace, trial, pre, post, behav_full, behav_start, behav_end
}

neuron_corrs = get_correlation_from_behav(session_idx, ps_stats_params)
trace_corr = data_dict['data']['trace_corr'][session_idx]

print(neuron_corrs.shape)
print(trace_corr.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.scatter(neuron_corrs.flatten(), trace_corr.flatten(), marker='.', color='k', alpha=0.005)

ax1.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
ax1.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

ax1.set_xlabel('Pairwise corr from corrcoef(df_closedLoop)')
ax1.set_ylabel('Pairwise corr from trace_corr')

ps_stats_params['pairwise_corr_type'] = 'behav_start'
start_corrs = get_correlation_from_behav(session_idx, ps_stats_params)
ps_stats_params['pairwise_corr_type'] = 'behav_end'
end_corrs = get_correlation_from_behav(session_idx, ps_stats_params)

ax2.scatter(start_corrs.flatten(), end_corrs.flatten(), marker='.', color='k', alpha=0.005)

ax2.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
ax2.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

ax2.set_xlabel('Pairwise corr start')
ax2.set_ylabel('Pairwise corr end')

    print("✅ compute_behavioral_correlations ran successfully.")
if __name__ == '__main__':
    run()