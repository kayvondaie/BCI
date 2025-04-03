from utils.helper_functions1 import get_data_dict

def run():
    # Cell 113
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

trial_idx = 40
neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][:10, neuron_idx, trial_idx], color='r', label='data_dict F')
ax.plot(trial_start_dff[:10, neuron_idx, trial_idx], color='b', label='my trial_start aligned')
ax.legend()

    print("✅ plot_exemplar_trace ran successfully.")
if __name__ == '__main__':
    run()