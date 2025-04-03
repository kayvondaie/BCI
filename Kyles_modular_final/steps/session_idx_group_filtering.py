from utils.helper_functions1 import get_data_dict

def run():
    # Cell 102
ps_stats_params['connectivity_metrics'] = (
    'trial_resp_x',
    'trial_resp_y',
)

ps_stats_params['plot_pairs'] = (
    ('trial_resp_x', 'trial_resp_y'),
)


session_idxs = get_unique_sessions(session_idx_pairs, verbose=False)

print('Evaluating {} session idxs...'.format(len(session_idxs)))

records = get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict, verbose=False
)

full_fits = scan_over_connectivity_pairs(ps_stats_params, records, verbose=False)

print('P value result:  \t', full_fits[('trial_resp_x', 'trial_resp_y')]['pvalue'])

    print("✅ session_idx_group_filtering ran successfully.")
if __name__ == '__main__':
    run()