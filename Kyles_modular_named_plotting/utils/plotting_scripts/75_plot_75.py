import matplotlib.pyplot as plt
import numpy as np

def plot_cell_75():
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))
    fit_type_names = []
    color_idxs = (0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,)

    assert len(color_idxs) == n_fit_types


    for fit_type_idx, fit_type in enumerate(fit_types):

        fit_type_names.append(fit_type[0])

        ax3.scatter(fit_type_idx * np.ones((n_pairs,)), log10_rsquares[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
        ax3.scatter(fit_type_idx, np.mean(log10_rsquares[fit_type_idx]), color='k')

        ax4.scatter(fit_type_idx * np.ones((n_pairs,)), log10_pvalues[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
        ax4.scatter(fit_type_idx, np.mean(log10_pvalues[fit_type_idx]), color='k')

        full_fit_data_x_flat = []
        full_fit_data_y_flat = []
        for pair_idx in range(len(full_fit_data_x[fit_type_idx])):
            full_fit_data_x_flat.extend(full_fit_data_x[fit_type_idx][pair_idx])
            full_fit_data_y_flat.extend(full_fit_data_y[fit_type_idx][pair_idx])

        slope, intercept, rvalue, pvalue, se = add_regression_line(
            full_fit_data_x_flat, full_fit_data_y_flat
        )

        ax4.scatter(fit_type_idx, np.log10(pvalue), color='k', marker='*')

        print(slope)

    ax3.set_ylabel('log10($r^2$) $\Delta$group correlation - $\Delta$PS response')
    ax4.set_ylabel('log10(p) $\Delta$group correlation - $\Delta$PS response')

    ax3.axhline(-1, color='lightgrey', zorder=-5)
    ax3.axhline(-2, color='lightgrey', zorder=-5)
    ax3.axhline(-3, color='lightgrey', zorder=-5)

    ax4.axhline(np.log10(0.1), color='lightgrey', zorder=-5)
    ax4.axhline(np.log10(0.05), color='lightgrey', zorder=-5)
    ax4.axhline(np.log10(0.01), color='lightgrey', zorder=-5)

    for ax in (ax3, ax4):
        ax.set_xticks(np.arange(n_fit_types))

    ax3.set_xticklabels([])
    ax4.set_xticklabels(fit_type_names, rotation=45)

    for tick in ax4.xaxis.get_majorticklabels():
        tick.set_horizontalalignment('right')

if __name__ == '__main__':
    plot_cell_75()
