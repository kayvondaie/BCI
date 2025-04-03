from utils.helper_functions1 import get_data_dict

def run():
    # Cell 114
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][40:, neuron_idx, 0], color='r')
ax.plot(data_dict_behav['df_closedLoop'][:300, neuron_idx], color='b')

scale = data_dict['data']['F'][session_idx_idx][40, neuron_idx, 0] / data_dict_behav['df_closedLoop'][0, neuron_idx]

# ax.plot(scale * data_dict_behav['df_closedLoop'][:300, neuron_idx], color='g')

    print("✅ compare_behav_and_raw_traces ran successfully.")
if __name__ == '__main__':
    run()