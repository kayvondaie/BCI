# Cell 47
ps_stats_params = {
    'pairwise_corr_type': 'behav_start', # trace, trial, pre, post, behav_full, behav_start, behav_end
    
    ### Fitting photostim parameters
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average_equal_sessions', # average, average_equal_sessions, minimum
    'modify_direct_weights': True,
    'use_only_predictor_weights': False,
}

ps_stats_params = default_ps_stats_params(ps_stats_params)

def get_candidate_random_cns(
    cn_idx, n_neurons, method='random', percent_pass=0.2,
    prev_day_activities=None, prev_day_tuning=None,
):
    """
    Get neuron idxs that are similar to the CN to draw from 
    randomly. Always includes the CN itself.
    
    INPUTS:
    percent_pass: float, percent of neurons that pass criterion. 1.0 = all neurons pass
    """

    if method == 'random': # All neuron idxs can be chosen
        return np.arange(n_neurons)
    elif method in (
        'similar_prev_day_activity', # Only neurons with similar previous day activity
        'similar_prev_day_activity_tuning', # Both tuning and previous activity
    ): 
        
        assert prev_day_activities is not None
        assert prev_day_activities.shape[0] == n_neurons
        
        percent_pass_idx = int(np.ceil(percent_pass * n_neurons))

        if method in ('similar_prev_day_activity',):
            cn_activity = prev_day_activities[cn_idx]
            activity_diff = np.abs(prev_day_activities - cn_activity) # Difference to CN activity
        elif method in ('similar_prev_day_activity_tuning',):
            assert prev_day_tuning is not None
            
            # Standarize both distributions so distances equally weighted
            prev_day_activities = (prev_day_activities - np.nanmean(prev_day_activities)) / np.nanstd(prev_day_activities)
            prev_day_tuning = (prev_day_tuning - np.nanmean(prev_day_tuning)) / np.nanstd(prev_day_tuning)
            
            prev_day_vals = np.concatenate(
                (prev_day_activities[:, np.newaxis], prev_day_tuning[:, np.newaxis],), axis=-1
            )
            
            activity_diff = np.linalg.norm(prev_day_vals - prev_day_vals[cn_idx:cn_idx+1, :], axis=-1)
            
        sort_activity_diff_idxs = np.argsort(activity_diff)

#         print('CN activity at {}:'.format(cn_idx), cn_activity)
#         print('closest activity at {}:'.format(sort_activity_diff_idxs[0]), prev_day_activities[sort_activity_diff_idxs[0]])
#         print('closest activity at {}:'.format(sort_activity_diff_idxs[1]), prev_day_activities[sort_activity_diff_idxs[1]])
#         print(sort_activity_diff_idxs[:percent_pass_idx])

#         print('closest activity at {}:'.format(sort_activity_diff_idxs[0]), prev_day_vals[sort_activity_diff_idxs[0]])
#         print('closest activity at {}:'.format(sort_activity_diff_idxs[1]), prev_day_vals[sort_activity_diff_idxs[1]])
    
        return sort_activity_diff_idxs[:percent_pass_idx]
    else:
        raise ValueError('method {} not recognized'.format(method))

# Cell 82
def get_all_xs(records, connectivity_metrics_xs, standardize_xs=False):
    
    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    n_sessions = len(records[exemplar_key])

    records_x_flat_all = [[] for _ in range(n_sessions)] # Session, connectivity_metrics_x, n_indirect in session

    ### Iteration over all possible things we want to plot against one another ###
    for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):

        for session_idx, session_records_x in enumerate(records[connectivity_metrics_x]):
            exemplar_session_records_y = records[exemplar_key][session_idx]

            session_records_x_flat = []

            # Expand all xs so that they're the same shape 
            for group_idx in range(len(session_records_x)):
                if type(session_records_x) == np.ndarray: # Need to extend x for each y
                    session_records_x_flat.append(
                        session_records_x[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_x) == list: # x already extended
                    session_records_x_flat.append(session_records_x[group_idx])

#             # Concatenate across all groups, to get total number of indirect in session
#             session_records_x_flat = np.concatenate(session_records_x_flat, axis=0)

            if standardize_xs: # Standardize the x across the session            
                temp_session_records_x_flat = np.concatenate(session_records_x_flat, axis=0)
                session_mean = np.nanmean(temp_session_records_x_flat)
                session_std = np.nanstd(temp_session_records_x_flat)
                for group_idx in range(len(session_records_x_flat)): # Now adjust each group
                    session_records_x_flat[group_idx] = (session_records_x_flat[group_idx] - session_mean) / session_std
            
#                 session_records_x_flat = (session_records_x_flat - np.nanmean(session_records_x_flat)) / np.nanstd(session_records_x_flat)

            records_x_flat_all[session_idx].append(session_records_x_flat)

#     for session_idx in range(n_sessions):
        # Flatten across groups, so for each session we have (n_cm_x, n_total_indirect)
#         records_x_flat_all[session_idx] = np.array(records_x_flat_all[session_idx])
        
    return records_x_flat_all

# Cell 89
def fit_and_plot_up(x, y, weights=None, catch_nans=True, ax=None, axes_labels=None, color='k', 
                    linestyle='solid', plot_up_mode='all', keep_scatter=False, bin_plot=True, 
                    equal_count=False, n_bins=10, add_legend=True,
                    idx=None, verbose=False):
        """
        Modified version of Matt's linear regression/bin function, returns the linear fit parameters 
        and optionally plots fits. Can plot in bins too.
        
        INPUTS:
        x : The input x data
        y : the input y data
        weights: Weights of the data for linear regression fit, note weights not yet 
            implemented into bin plots
        ax : optional passed in axis for integration into larger panel arrays
        pair: 2-tuple of name of the pairs in order (x, y)
        color: a color of the associated lines
        keep_scatter: default is to not include the full scatter of the pairs of points
        equal_count: triggers equal number of points in bins, otherwise bins uniformly 
            distributed over x range
        """
        
        assert len(x.shape) == 1
        assert x.shape == y.shape
        
        LOWER_PERC = 1 # the percentile of the smallest bin
        UPPER_PERC = 99 # the percentile of the largest bin
        
        if catch_nans:
            if weights is None:
                nonnan_mask = ~np.isnan(x) * ~np.isnan(y)
            else:
                nonnan_mask = ~np.isnan(x) * ~np.isnan(y) * ~np.isnan(weights)
                weights = np.array(weights)[nonnan_mask]
            x = np.array(x)[nonnan_mask]
            y = np.array(y)[nonnan_mask]
        
        plot_this_fit = False
        
        if weights is None:
            slope, intercept, rvalue, pvalue, se = linregress(x, y)
            res = {
                'slope': slope,
                'intercept': intercept,
                'rsquared': rvalue**2,
                'pvalue': pvalue,
                'se': se,
            }
            
        else:
            X = x[:, np.newaxis]
            Y = y[:, np.newaxis]

            X = sm.add_constant(X)
            wls_model = sm.WLS(Y, X, weights=weights)
            results = wls_model.fit()
            
            res = {
                'slope': results.params[1],
                'intercept': results.params[0],
                'rsquared': results.rsquared,
                'pvalue': results.pvalues[1],
                'se': None,
            } 
        rounded_log10_p = np.min((5, int(np.floor(-np.log10(res['pvalue'])))))
        if verbose:
            print_str = '' if idx is None else '{} -'.format(idx)
            print_str += '{}\tvs. {}\tp={:.2e}\t{}'.format(
                axes_labels[0], axes_labels[1], res['pvalue'], '*' * rounded_log10_p
            )
            print(print_str)
        
        if plot_up_mode in ('all',):
            plot_this_fit = True
        elif plot_up_mode in ('significant',):
            if res['pvalue'] < 1e-2:
                plot_this_fit = True
        elif plot_up_mode not in (None,):
            raise ValueError('Plot_up_mode {} not recoginized.'.format(plot_up_mode))
        
        if plot_this_fit:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6,4))
            
            if keep_scatter:
                ax.scatter(x, y, c=color, alpha = 0.1)
            

            if equal_count:
                perc_bin = np.linspace(LOWER_PERC, UPPER_PERC, n_bins)
                bins = [np.percentile(x, perc) for perc in perc_bin]
            else:
                bins = np.linspace(np.percentile(x, UPPER_PERC), np.percentile(x, LOWER_PERC), n_bins)
            #xbincenter = bins[1::]-np.diff(bins)[0]/2
            meaninbin = []
            stdmeaninbin = []
            countbin = []
            xbincenter = []

            for i in range(n_bins-1):
                meaninbin.append(np.mean(y[np.logical_and(x>bins[i], x<bins[i+1])]))
                stdmeaninbin.append(np.std(y[np.logical_and(x>bins[i], x<bins[i+1])]))
                countbin.append(np.sum(np.logical_and(x>bins[i], x<bins[i+1])))
                xbincenter.append(np.mean(x[np.logical_and(x > bins[i], x < bins[i+1])]))
            
            if bin_plot:
                ax.scatter(xbincenter, meaninbin, color=color, s=25)
                ax.errorbar(
                    xbincenter, meaninbin, np.array(stdmeaninbin) / np.sqrt(countbin), 
                    color=color, lw = 3
                )
            
            fitx = np.array(xbincenter).ravel()
            ax.plot(
                fitx, res['intercept'] + res['slope'] * fitx, color, linestyle=linestyle,
                label='p = {:.2e}, $r^2$ = {:.2f}'.format(res['pvalue'], res['rsquared'])
            )
            
            ax.set_xlabel(axes_labels[0])
            ax.set_ylabel(axes_labels[1])

            ax.spines[['right', 'top']].set_visible(False)
            if add_legend:
                ax.legend(loc='best')

            ax.axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
            ax.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    
        return res, ax

