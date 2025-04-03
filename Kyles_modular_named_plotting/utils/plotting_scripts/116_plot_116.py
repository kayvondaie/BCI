import matplotlib.pyplot as plt
import numpy as np

def plot_cell_116():
    # Some Moore-Penrose inversion sanity checks to ensure shifting to neuron -> neuron 
    # causal connectivity makes sense

    n_neurons = 3
    n_groups = 4

    dir_resp_ps = np.zeros((n_neurons, n_groups,))

    dir_resp_ps[0, 0] = 1.0
    dir_resp_ps[1, 0] = 1.0

    dir_resp_ps[1, 1] = 1.0
    dir_resp_ps[2, 1] = 1.0

    dir_resp_ps[0, 2] = 1.0
    dir_resp_ps[2, 2] = 1.0

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    max_val = np.nanmax(np.abs(dir_resp_ps))
    ax1.matshow(dir_resp_ps, vmax=max_val, vmin=-max_val, cmap='bwr')
    for (i, j), z in np.ndenumerate(dir_resp_ps):
        ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax1.set_xlabel('Event idx')
    ax1.set_ylabel('Neuron idx')

    inverse = np.linalg.pinv(dir_resp_ps)

    max_val = np.nanmax(np.abs(inverse))
    ax2.matshow(inverse, vmax=max_val, vmin=-max_val, cmap='bwr')
    for (i, j), z in np.ndenumerate(inverse):
        ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax2.set_xlabel('Neuron idx')
    ax2.set_ylabel('Event idx')

if __name__ == '__main__':
    plot_cell_116()
