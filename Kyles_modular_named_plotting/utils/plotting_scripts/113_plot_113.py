import matplotlib.pyplot as plt
import numpy as np

def plot_cell_113():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    trial_idx = 40
    neuron_idx = 5

    ax.plot(data_dict['data']['F'][session_idx_idx][:10, neuron_idx, trial_idx], color='r', label='data_dict F')
    ax.plot(trial_start_dff[:10, neuron_idx, trial_idx], color='b', label='my trial_start aligned')
    ax.legend()

if __name__ == '__main__':
    plot_cell_113()
