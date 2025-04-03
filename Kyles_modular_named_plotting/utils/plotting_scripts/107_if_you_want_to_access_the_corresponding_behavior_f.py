import matplotlib.pyplot as plt
import numpy as np

def plot_cell_107():
    # Plot some exemplar profiles
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    start_idx = 0
    end_idx = 1000

    ax.plot(data_dict_behav['trial_start'][0, start_idx:end_idx], color=c_vals[0], label='trial start')
    ax.plot(data_dict_behav['rew'][0, start_idx:end_idx], color=c_vals[1], label='rew')
    ax.plot(data_dict_behav['vel'][0, start_idx:end_idx], color=c_vals_l[2], zorder=-5, label='vel')
    ax.plot(data_dict_behav['thr'][0, start_idx:end_idx], color=c_vals[3], label='thr')

    ax.set_xlabel('Time idx')
    ax.set_ylabel('Value')

    ax.legend()

if __name__ == '__main__':
    plot_cell_107()
