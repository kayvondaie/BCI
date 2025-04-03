from utils.helper_functions1 import get_data_dict

import pickle
def run():
    try:
        with open('outputs/data_dict.pkl', 'rb') as f:
            data_dict = get_data_dict()
    except FileNotFoundError:
        print('No saved data_dict found.')
    try:
        with open('outputs/ps_stats_params.pkl', 'rb') as f:
            ps_stats_params = pickle.load(f)
    except FileNotFoundError:
        print('No saved ps_stats_params found.')
    # Cell 16
ps_stats_params = {
    'trial_average_mode': 'time_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

    print("✅ find_valid_pairs ran successfully.")
if __name__ == '__main__':
    run()