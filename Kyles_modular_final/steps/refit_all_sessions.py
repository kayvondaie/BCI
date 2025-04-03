from utils.helper_functions1 import get_data_dict

def run():
    # Cell 70
# First just isolate desired session pairs and define metrics
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

connectivity_metrics = None

ps_stats_params = {
    'trial_average_mode': 'trials_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': None,
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

    print("✅ refit_all_sessions ran successfully.")
if __name__ == '__main__':
    run()