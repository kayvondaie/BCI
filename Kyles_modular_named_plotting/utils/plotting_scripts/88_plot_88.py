import matplotlib.pyplot as plt
import numpy as np

def plot_cell_88():
    # For a single session, see the relative size of the various contributions to a given A and W 

    connectivity_metric_plot = 'trial_resp_x' # tuning_x, trial_resp_x, post_x
    session_idx_plot = 3

    print('MM:', records[connectivity_metric][session_idx][0][:5])
    print('centered+means:', records[connectivity_metric][session_idx][1][:5] +  records[connectivity_metric][session_idx][2][:5])
    print('MM_centered:', records[connectivity_metric][session_idx][1][:5])
    print('Means:', records[connectivity_metric][session_idx][2][:5])

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))

    for cm_idx, connectivity_metric in enumerate(connectivity_metrics):

        n_sessions = len(records[connectivity_metric])
        centered_over_means = np.zeros((n_sessions,))

        for session_idx in range(n_sessions):

            centered_over_means[session_idx] = np.mean(
                np.abs(records[connectivity_metric][session_idx][1]) / 
                np.abs(records[connectivity_metric][session_idx][2])
            )

            if connectivity_metric == connectivity_metric_plot and session_idx == session_idx_plot:

                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

                ax1.scatter(records[connectivity_metric][session_idx][1], records[connectivity_metric][session_idx][2], 
                            marker='.', color=c_vals[session_idx], alpha=0.3)

                for ax in (ax1,):
                    ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
                    ax.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')

                    ax.set_xlabel(f'{connectivity_metric} - mentered contribution')
                    ax.set_ylabel(f'{connectivity_metric} - mean contribution')

        ax2.scatter(cm_idx * np.ones_like(centered_over_means), centered_over_means, color=c_vals_l[0], marker='.')
        ax2.scatter(cm_idx, np.mean(centered_over_means), color='k', marker='.')

    ax2.set_xticks(np.arange(len(connectivity_metrics)))
    ax2.set_xticklabels(connectivity_metrics, rotation=90)
    ax2.set_yscale('log')
    ax2.axhline(1.0, color='grey', linestyle='dashed', zorder=-10)
    ax2.set_ylabel('Centered/mean ratio')

if __name__ == '__main__':
    plot_cell_88()
