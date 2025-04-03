import matplotlib.pyplot as plt
import numpy as np

def plot_cell_52():
    n_pairs = len(delta_tunings)

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 3))
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 3))
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))

    positions = [0, 0.5, 1, 1.75, 2.25, 2.75, 3.5, 4.0, 4.75, 5.25, 6.0,]

    # ax1.violinplot(delta_tunings, showmeans=True)

    for plot_idx, (ax, data,) in enumerate(zip((ax1, ax2, ax3), (delta_tunings, delta_trial_resps, delta_correlations))):

        data = [data_session.flatten() for data_session in data] # Flatten if the measure is multi-dimensional

        bplot = ax.boxplot(data, positions=positions, notch=True, patch_artist=True, sym='.', widths=0.4)
        for patch, color_idx in zip(bplot['boxes'], PAIR_COLORS[:n_pairs]):
            patch.set_facecolor(c_vals_l[color_idx])
            patch.set_edgecolor(c_vals[color_idx])

        ax.axhline(0.0, color='grey', linestyle='dashed', zorder=-5)

    ax3.set_ylabel('Delta Correlations')
    ax1.set_ylabel('Delta Tuning')
    ax2.set_ylabel('Delta Trial Resp.')

if __name__ == '__main__':
    plot_cell_52()
