import matplotlib.pyplot as plt
import numpy as np

def plot_cell_114():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    neuron_idx = 5

    ax.plot(data_dict['data']['F'][session_idx_idx][40:, neuron_idx, 0], color='r')
    ax.plot(data_dict_behav['df_closedLoop'][:300, neuron_idx], color='b')

    scale = data_dict['data']['F'][session_idx_idx][40, neuron_idx, 0] / data_dict_behav['df_closedLoop'][0, neuron_idx]

    # ax.plot(scale * data_dict_behav['df_closedLoop'][:300, neuron_idx], color='g')

if __name__ == '__main__':
    plot_cell_114()
