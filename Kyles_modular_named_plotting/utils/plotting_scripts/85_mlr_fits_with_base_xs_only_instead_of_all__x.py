import matplotlib.pyplot as plt
import numpy as np

def plot_cell_85():
    MIN_P = 1e-300
    MAX_PLOT_LOG10P = 100 # 

    X_PERCENTILES = (10, 90) # For fit plots, determines range of x

    bar_locs = np.concatenate((np.array((0,)), np.arange(n_sessions) + 2,), axis=0)
    bar_colors = ['k',]

    if ps_stats_params['ps_analysis_type'] in ('single_session',):
        bar_colors.extend([c_vals[session_color] for session_color in SESSION_COLORS[:n_sessions]]) 
    elif ps_stats_params['ps_analysis_type'] in ('paired',):
        bar_colors.extend([c_vals[pair_color] for pair_color in PAIR_COLORS[:n_sessions]])

    if valid_full_fits is not None:
        bar_locs = np.concatenate((bar_locs, np.array((bar_locs.shape[0]+1,))), axis=0)
        bar_colors.append('grey')

    n_cm_x = len(connectivity_metrics_xs)
    n_cm_y = len(connectivity_metrics_ys)

    session_ps = np.ones((n_cm_x, n_cm_y, n_sessions))
    aggregate_ps = np.ones((n_cm_x, n_cm_y, 1))
    valid_ps = np.ones((n_cm_x, n_cm_y, 1))

    session_params = np.zeros((n_cm_x, n_cm_y, n_sessions))
    aggregate_params = np.zeros((n_cm_x, n_cm_y, 1))
    valid_params = np.zeros((n_cm_x, n_cm_y, 1))
    session_stderrs = np.zeros((n_cm_x, n_cm_y, n_sessions))
    aggregate_stderrs = np.zeros((n_cm_x, n_cm_y, 1))
    valid_stderrs = np.zeros((n_cm_x, n_cm_y, 1))

    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):
        for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
            if connectivity_metrics_x in base_xs:
                print('Skipping base metric {}'.format(connectivity_metrics_x))
                continue
            for session_idx in range(n_sessions):
                session_ps[cm_x_idx, cm_y_idx, session_idx] = (
                    session_fits[connectivity_metrics_y][connectivity_metrics_x][session_idx].pvalues[-1] 
                )
                session_params[cm_x_idx, cm_y_idx, session_idx] = (
                    session_fits[connectivity_metrics_y][connectivity_metrics_x][session_idx].params[-1]
                )
                session_stderrs[cm_x_idx, cm_y_idx, session_idx] = (
                    session_fits[connectivity_metrics_y][connectivity_metrics_x][session_idx].bse[-1]
                )
            aggregate_ps[cm_x_idx, cm_y_idx, 0] = (
                full_fits[connectivity_metrics_y][connectivity_metrics_x].pvalues[-1] # Don't include intercept
            )
            aggregate_params[cm_x_idx, cm_y_idx, 0] = (
                full_fits[connectivity_metrics_y][connectivity_metrics_x].params[-1] # Don't include intercept
            )
            aggregate_stderrs[cm_x_idx, cm_y_idx, 0] = (
                full_fits[connectivity_metrics_y][connectivity_metrics_x].bse[-1] # Don't include intercept
            )


    # Enforce minimum values on ps
    session_ps = np.where(session_ps==0., MIN_P, session_ps)
    aggregate_ps = np.where(aggregate_ps==0., MIN_P, aggregate_ps)

    fig1, ax1s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # -log10(p-values)
    fig2, ax2s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # parameters
    fig3, ax3s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # fits

    specs_string = '\n fit_intercept: {}, standardize_x\'s: {}, standardize_y\'s: {}'.format(
        fit_intercept, standardize_xs, standardize_ys
    )

    fig1.suptitle('-log10(p-values)' + specs_string, fontsize=12)
    fig2.suptitle('Parameters +/- std err' + specs_string, fontsize=12)
    fig3.suptitle('Individual fits' + specs_string, fontsize=12)
    # fig1.tight_layout()

    for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
        max_p_for_this_x = 0.0

        if connectivity_metrics_x in base_xs:
            continue

        for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):

            all_ps = np.concatenate((aggregate_ps[cm_x_idx, cm_y_idx], session_ps[cm_x_idx, cm_y_idx],), axis=-1)
            all_params = np.concatenate((aggregate_params[cm_x_idx, cm_y_idx], session_params[cm_x_idx, cm_y_idx],), axis=-1)
            all_stderrs = np.concatenate((aggregate_stderrs[cm_x_idx, cm_y_idx], session_stderrs[cm_x_idx, cm_y_idx],), axis=-1)

            if valid_full_fits is not None:
                all_ps = np.concatenate((all_ps, valid_ps[cm_x_idx, cm_y_idx],), axis=-1)
                all_params = np.concatenate((all_params, valid_params[cm_x_idx, cm_y_idx],), axis=-1)
                all_stderrs = np.concatenate((all_stderrs, valid_stderrs[cm_x_idx, cm_y_idx],), axis=-1)

            all_ps = -1 * np.log10(all_ps)
            if max(all_ps) > max_p_for_this_x:
                max_p_for_this_x = max(all_ps)

            ax1s[cm_x_idx, cm_y_idx].bar(bar_locs, all_ps, color=bar_colors)
            ax2s[cm_x_idx, cm_y_idx].scatter(bar_locs, all_params, color=bar_colors, marker='_')
            for point_idx in range(bar_locs.shape[0]):
                ax2s[cm_x_idx, cm_y_idx].errorbar(
                    bar_locs[point_idx], all_params[point_idx], yerr=all_stderrs[point_idx], 
                    color=bar_colors[point_idx], linestyle='None'
                )

            # Plot each session's fit
            all_x_vals = []

            for session_idx in range(n_sessions):
                if type(records[connectivity_metrics_x][session_idx]) == list:
                    x_vals = np.concatenate(records[connectivity_metrics_x][session_idx])
                else:
                    x_vals = records[connectivity_metrics_x][session_idx]

                if standardize_xs: # Standardize the x across the session            
                    x_vals = (x_vals - np.nanmean(x_vals)) / np.nanstd(x_vals) 

                all_x_vals.append(x_vals)

                x_range = np.linspace(
                    np.percentile(x_vals, X_PERCENTILES[0]),
                    np.percentile(x_vals, X_PERCENTILES[1]), 10
                )
                y_range = (
                    session_params[cm_x_idx, cm_y_idx, session_idx] * x_range
                )

                if fit_intercept:
                    y_range += session_fits[connectivity_metrics_y][connectivity_metrics_x][session_idx].params[0]

                if ps_stats_params['ps_analysis_type'] in ('single_session',):
                    line_color = c_vals[SESSION_COLORS[session_idx]]
                elif ps_stats_params['ps_analysis_type'] in ('paired',):
                    line_color = c_vals[PAIR_COLORS[session_idx]]  

                ax3s[cm_x_idx, cm_y_idx].plot(x_range, y_range, color=line_color)

            x_range_all = np.linspace(
                    np.percentile(np.concatenate(all_x_vals), X_PERCENTILES[0]),
                    np.percentile(np.concatenate(all_x_vals), X_PERCENTILES[1]), 10
            )
            y_range_all = (
                aggregate_params[cm_x_idx, cm_y_idx] * x_range_all
            )
            if fit_intercept:
                y_range_all += full_fits[connectivity_metrics_y][connectivity_metrics_x].params[0]
            ax3s[cm_x_idx, cm_y_idx].plot(x_range, y_range, color='k')

            ax1s[cm_x_idx, cm_y_idx].axhline(2., color='grey', zorder=-5, linewidth=1.0)
            ax2s[cm_x_idx, cm_y_idx].axhline(0., color='grey', zorder=-5, linewidth=1.0)
            ax3s[cm_x_idx, cm_y_idx].axhline(0.0, color='lightgrey', linestyle='dashed', linewidth=1.0, zorder=-5)
            ax3s[cm_x_idx, cm_y_idx].axvline(0.0, color='lightgrey', linestyle='dashed', linewidth=1.0, zorder=-5)

            for axs in (ax1s, ax2s, ax3s):
                axs[cm_x_idx, cm_y_idx].set_xticks(())
                if cm_x_idx == n_cm_x - 1:
                    axs[cm_x_idx, cm_y_idx].set_xlabel(connectivity_metrics_y, fontsize=8)
                if cm_y_idx == 0:
                    axs[cm_x_idx, cm_y_idx].set_ylabel(connectivity_metrics_x, fontsize=8)

        # Set all p-value axes to be the same range
        max_p_for_this_x = np.where(max_p_for_this_x > MAX_PLOT_LOG10P, MAX_PLOT_LOG10P, max_p_for_this_x)

        for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):
            ax1s[cm_x_idx, cm_y_idx].set_ylim((0.0, 1.1*max_p_for_this_x))
            if cm_y_idx != 0:
                ax1s[cm_x_idx, cm_y_idx].set_yticklabels(())

if __name__ == '__main__':
    plot_cell_85()
