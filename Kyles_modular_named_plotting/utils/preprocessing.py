# Cell 5
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',] # Normal
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',] # Light
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',] # Dark


def add_regression_line(x, y, fit_intercept=True, weights=None, ax=None, color='red', 
                        zorder=0, catch_nans=True, linestyle='solid', add_bins=False, 
                        label=''):
    """
    Simple linear regression function, used for both fitting a linear regression
    and optionally plotting it. Note that this draws the line slightly
    differently than Matt's plotting function.
    """
    
    if catch_nans:
        if weights is None:
            nonnan_mask = ~np.isnan(x) * ~np.isnan(y)
        else:
            nonnan_mask = ~np.isnan(x) * ~np.isnan(y) * ~np.isnan(weights)
            weights = np.array(weights)[nonnan_mask]
        x = np.array(x)[nonnan_mask]
        y = np.array(y)[nonnan_mask]
    
    x_plot = np.linspace(np.min(x), np.max(x), 10)
    
    if weights is None and fit_intercept: # linregress can't turn off intercept fit
#         reg = LinearRegression
#         reg.fit(x, y)
#         y_plot = reg.coef_ * x_plot + reg.intercept_
        slope, intercept, rvalue, pvalue, se = linregress(x, y)
    else:
        X = x[:, np.newaxis]
        Y = y[:, np.newaxis]
        
        slope_idx = 0
        if fit_intercept:
            X = sm.add_constant(X)
            slope_idx += 1 # Intercept is 0th now
        
        if weights is None:
            fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
        else:
            fit_model = sm.WLS(Y, X, weights=weights, missing='none') # nans should be caught above
        results = fit_model.fit()
        
        slope = results.params[slope_idx]
        intercept = 0. if not fit_intercept else results.params[0]
        rvalue = np.sqrt(results.rsquared)
        pvalue = results.pvalues[slope_idx]
        se = None
        
    if ax is not None:
        
        if label == '':
            label = 'p: {:.2e}, $r^2$: {:.2e}'.format(pvalue, rvalue**2)
        
        y_plot = slope * x_plot + intercept
        ax.plot(x_plot, y_plot, color=color, zorder=zorder, label=label, linestyle=linestyle)
        
        if add_bins:
            add_bin_plot(x, y, n_bins, ax=ax, color=color, zorder=zorder-1, 
                         alpha=alpha)

    return slope, intercept, rvalue, pvalue, se

def add_bin_plot(x, y, n_bins=10, ax=None, color='red', zorder=0, alpha=1.0):
    """
    Bin plot function. Equal spaced bins, unequal number in each bin
    """
    x_data_sort_idxs = np.argsort(x)
    x_data = x[x_data_sort_idxs]
    y_data = y[x_data_sort_idxs]

    assert x_data.shape == y_data.shape

    size = x_data.shape[0]

    sep = (size / float(n_bins)) * (np.arange(n_bins) + 1)
    bin_idxs = sep.searchsorted(np.arange(size))

    bin_x = [[] for _ in range(n_bins)]
    bin_y = [[] for _ in range(n_bins)]

    for idx, (x, y) in enumerate(zip(x_data, y_data)):
        bin_x[bin_idxs[idx]].append(x)
        bin_y[bin_idxs[idx]].append(y)

    bin_x_means = np.zeros((n_bins,))
    bin_x_stds = np.zeros((n_bins,))
    bin_y_means = np.zeros((n_bins,))
    bin_y_stds = np.zeros((n_bins,))

    for bin_idx in range(n_bins):
        bin_x_means[bin_idx] = np.mean(bin_x[bin_idx])
        bin_x_stds[bin_idx] = np.std(bin_x[bin_idx]) / np.sqrt(len(bin_x[bin_idx]))
        bin_y_means[bin_idx] = np.mean(bin_y[bin_idx])
        bin_y_stds[bin_idx] = np.std(bin_y[bin_idx]) / np.sqrt(len(bin_y[bin_idx]))


    ax.errorbar(bin_x_means, bin_y_means, yerr=bin_y_stds, color=color, zorder=zorder, marker='.', alpha=alpha)
    
def participation_ratio_vector(C, axis=None):
    """Computes the participation ratio of a vector of variances."""
    return np.sum(C, axis=axis) ** 2 / np.sum(C*C, axis=axis)

def shuffle_along_axis(a, axis):
    """ 
    Shuffles array along a given axis. Shuffles each axis differently. 
    
    E.g. a[:, 0] = [1, 2], shuffle_along_axis(a, axis=0)[:, 0] = [2, 1]
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def add_identity(axes, *line_args, **line_kwargs):
    """
    Adds an identity line to the given plot
    """
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def visualize_dict(d, lvl=0):
    """
    Quickly look at the structure of the data to help.
    """
    for k in sorted(d):
        if lvl == 0 and k == sorted(d)[0]:
            print('{:<25} {:<15} {:<10}'.format('KEY','LEVEL','TYPE/Example'))
            print('-'*79)

        indent = '  '*lvl # indent the table to visualise hierarchy
        t = str(type(d[k]))

        print("{:<25} {:<15} {:<10}".format(indent+str(k),lvl,t))

        if type(d[k])==dict:
            # visualise THAT dictionary with +1 indent
            visualize_dict(d[k],lvl+1)
            
        elif type(d[k])==list:
            if type(d[k][1])==list:
                t = str(np.array(d[k][1][0]).shape) + ' x ' + str(len(d[k][1]))
            else:
                t = str(np.array(d[k][1]).shape) 
                if t == '()':
                    t = d[k][1]
            indent = '  '*(lvl+1)
            try:
                print("{:<25} {:<15} {:<10}".format(indent,'',t))
            except:
                print("{:<25} {:<15} {:<10}".format(indent,'','None'))

# visualize_dict(data_dict)

# Cell 9
import warnings
from scipy.stats import ttest_1samp

def unflatted_neurons_by_groups(flat_data, n_neurons):
    """ (n_neurons * n_groups,) -> (n_neurons, n_groups,) """
    assert flat_data.shape[0] % n_neurons == 0
    n_groups = int(flat_data.shape[0] / n_neurons) 

    if UNFLATTED_MODE == 0: # Matches Matlab reshape
        return flat_data.reshape((n_groups, n_neurons)).T 
    elif UNFLATTED_MODE == 1:
        return flat_data.reshape((n_neurons, n_groups))

def compute_resp_ps_mask_prevs(raw_resp_ps, seq, X, ps_stats_params, normalization_mode='pre', 
                               return_extras=False):
    """
    This code produces photostim responses when we want to
    omit certain neuron's causal connectivity responses if they were 
    recently stimulated by the laser. These neurons are omitted by setting
    their corresponding photostimulation response to all nans, so they are ignored in when 
    computing the photostimulation response.
    
    Equivalent to Kayvon's 'mask_prev_trial_decay_fun' function but 
    called on sessions individually.
    
    INPUTS:
    - raw_resp_ps: photostim responses, (ps_time_idx, neuron_idx, ps_event_idx,)
      -  Note this array can contain a lot of nan entries. Some of these nans are used for padding, but
         many of them also represent faulty laser data. For some neurons and events, all entries will be
         nan. This can happen for all neurons in a given event (representing the laster 'tripping') but
         also for individual neurons during an event. See below for different options for handling nans.
    - seq: (ps_event_idx,): sequence of which group is indexed.
        - Note by default this uses MATLAB indexing so we need to convert it to python indexing (via -1)
    - X: (n_neurons, n_groups) distances to nearest photostim group, used to determine direct neurons
    - ps_stats_params['resp_ps_n_trials_back_mask']: number of trials back to omit. When 0 equivalent,
        to data_dict's y.
    - ps_stats_params['resp_ps_average_mode']: how to compute the average resp_ps, due to the presence of
        nans the average over trials and time do not commute
    - normalization_mode: 
        - None: just post - pre
        - pre: (post - pre) / pre

    OUTPUTS:
    - resp_ps: (n_neurons, n_groups,)
    - resp_ps_extras: 
        - 'resp_ps_events': Same as resp_ps, but computed for each event individually. This is shape
          (n_groups,) (n_neurons, n_group_events,). Note some entries can be nan for the reasons 
          stated above.

    """

    n_ps_times = raw_resp_ps.shape[0]
    n_neurons = raw_resp_ps.shape[1]
    n_ps_events = raw_resp_ps.shape[2]
    sq = (seq - 1).astype(np.int32) # -1 to account for Matlab indexing
    n_groups = np.max(sq) + 1
    
    n_trials_back = ps_stats_params['resp_ps_n_trials_back_mask']
    
    for ps_trial_idx in range(1, n_ps_events): # For each photostim event (skip first since no previous)
        # Account for early ps_trial_idx where trials back may not exist
        if ps_trial_idx < n_trials_back: # (if n_trial_back = 1, this does nothing)
            n_trials_to_look_back = np.copy(ps_trial_idx)
        else:
            n_trials_to_look_back = np.copy(n_trials_back)

        retro_direct_idxs = []
        for retro_trial_idx in range(1, n_trials_to_look_back + 1): # Inclusive of n_trials_to_look_back
            retro_direct_idxs.extend(np.where(X[:, sq[ps_trial_idx - retro_trial_idx]] < D_DIRECT)[0]) 

        if retro_direct_idxs != []:
            retro_direct_idxs = np.array(retro_direct_idxs)
            raw_resp_ps[:, retro_direct_idxs, ps_trial_idx] = np.nan # Fills the corresponding responses with nans for this event
    
    raw_resp_ps_by_group = [None for _ in range(n_groups)] # still raw data, now just separated into groups
    raw_resp_ps_mean = np.zeros((n_neurons, n_groups, n_ps_times,)) # mean over all group events, but keeps time axis
    
    # Below metrics use resp_ps, (post - pre) / baseline, computed for each event
    resp_ps = np.zeros((n_neurons, n_groups,)) # mean over all group events and post - pre
    resp_ps_sem = np.zeros_like(resp_ps) # sem over all group events
    resp_ps_pvalues = np.zeros_like(resp_ps) if return_extras else None # mean over all group events
    resp_ps_events = [] # (n_groups,) (n_neurons, n_group_events,) 
    
    for group_idx in range(n_groups): # Recompute amplitude for each photostimulation group
        group_trial_idxs = np.where(sq == group_idx)[0] # All ps event indexes that match group index
        assert group_trial_idxs.shape[0] > 0
        
        raw_resp_ps_by_group[group_idx] = raw_resp_ps[:, :, group_trial_idxs] # (ps_times, n_neurons, n_group_events,)    
        # Mean over PS events first
        with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
            warnings.simplefilter('ignore', category=RuntimeWarning)
            raw_resp_ps_mean[:, group_idx, :] = np.nanmean(raw_resp_ps[:, :, group_trial_idxs], axis=2).T # (ps_times, n_neurons, n_ps_events,) -> (ps_times, n_neurons,)
        
        if ps_stats_params['resp_ps_average_mode'] == 'time_first':
            # Mean over time first, then PS events
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                pre_resp_ps_events = np.nanmean(raw_resp_ps_by_group[group_idx][IDXS_PRE_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,) 
                post_resp_ps_events = np.nanmean(raw_resp_ps_by_group[group_idx][IDXS_POST_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,)

            # Gets normalized pre-response by taking mean across all events for a given group, similar to how dF/F is normalized by longer time scale
            if normalization_mode in ('pre',):
#                 if np.any(np.all(np.isnan(pre_resp_ps_events), axis=0)):
#                     print('All nans along n_neurons axis:', np.where(np.all(np.isnan(pre_resp_ps_events), axis=0))[0])
#                 if np.any(np.all(np.isnan(pre_resp_ps_events), axis=1)):
#                     print('All nans along n_groups axis:', np.where(np.all(np.isnan(pre_resp_ps_events), axis=1))[0])
                
                # Some neurons are nans across all events for a few sessions, suppress warning and just 
                # replace baseline with 1 since it will be nan anyway from numerator
                with warnings.catch_warnings(): 
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    baseline_resp_ps_events = np.nanmean(pre_resp_ps_events, axis=-1) # (n_neurons, n_group_events,) -> (n_neurons,)     
            
                baseline_resp_ps_events = np.where(np.isnan(baseline_resp_ps_events), 1., baseline_resp_ps_events) 
                resp_ps_events.append((post_resp_ps_events - pre_resp_ps_events) / baseline_resp_ps_events[:, np.newaxis]) # (n_neurons, n_group_events,) 
            elif normalization_mode in (None,):
                resp_ps_events.append(post_resp_ps_events - pre_resp_ps_events) # (n_neurons, n_group_events,) 
            else:
                raise ValueError('Normalization mode {} not recognized.'.format(normalization_mode))
            # Now get statistics over the distinct PS events
            with warnings.catch_warnings(): # Just returns nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                resp_ps[:, group_idx] = np.nanmean(resp_ps_events[group_idx], axis=-1) # (n_neurons, n_group_events,) -> (n_neurons,) 
                resp_ps_sem[:, group_idx] = (
                    np.nanstd(resp_ps_events[group_idx], axis=-1) /
                    np.sqrt(np.sum(~np.isnan(resp_ps_events[group_idx]), axis=-1)) # sqrt(# of not-nans)
                )
            if return_extras: 
                # Adds quite a bit of compute time, so only compute if needed
                _, resp_ps_pvalues[:, group_idx] = ttest_1samp(resp_ps_events[group_idx], 0, axis=-1, nan_policy='omit')
        elif ps_stats_params['resp_ps_average_mode'] == 'trials_first':
            # Mean over PS events first (already one above), then times
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                pre_ps_responses = np.nanmean(raw_resp_ps_mean[:, group_idx, IDXS_PRE_PS], axis=-1) # (n_neurons, ps_times,) -> (n_neurons,) 
                post_ps_responses = np.nanmean(raw_resp_ps_mean[:, group_idx, IDXS_POST_PS], axis=-1) # (n_neurons, ps_times,) -> (n_neurons,) 
                
            # Gets normalized pre-response by taking mean across all events for a given group, similar to how dF/F is normalized by longer time scale
            if normalization_mode in ('pre',):
                resp_ps[:, group_idx] = (post_ps_responses - pre_ps_responses) / pre_ps_responses # (n_neurons,) 
            elif normalization_mode in (None,):
                resp_ps[:, group_idx] = post_ps_responses - pre_ps_responses # (n_neurons,) 
            else:
                raise ValueError('Normalization mode {} not recognized.'.format(normalization_mode))
            
            # Doesn't compute these in this setting
            resp_ps_events.append(None)
            resp_ps_sem[:, group_idx] = np.nan
            resp_ps_pvalues = None
            
        else:
            raise ValueError('resp_ps_average_mode {} not recognized'.format(ps_stats_params['resp_ps_average_mode']))

    resp_ps_extras = {
        'raw_resp_ps_by_group': raw_resp_ps_by_group,
        'raw_resp_ps_mean': raw_resp_ps_mean,

        'resp_ps_events': resp_ps_events,
        'resp_ps_sem': resp_ps_sem,
        'resp_ps_pvalues': resp_ps_pvalues,
    }
        
    return resp_ps, resp_ps_extras

def find_valid_ps_pairs(ps_stats_params, sessions_to_include, data_dict, 
                        verbose=False,):
    """
    Determine which session pairs to evaluate. Rejection sessions if 
    they are not a valid pair, if they do not have photostimulation 
    distances, or if their photostimulation distances are too dissimilar.
    
    INPUTS:
    - sessions_to_include: list/array
    - data_dict: loaded data file
    
    """               
    session_pairs = []
    
    for session_idx in sessions_to_include:
        day_2_idx = session_idx + 1 
        day_1_idx = session_idx
        if data_dict['data']['mouse'][day_2_idx] != data_dict['data']['mouse'][day_1_idx]:
            continue
            
        data_to_extract = ('d_ps',)
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)       

        # Sometimes there is no photostim distance
        if data_1['d_ps'] is None or data_2['d_ps'] is None:
            print('Skipping mouse {}, session_idxs {} and {}, no photostim distances.'.format(
                data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx
            ))
            continue

        assert data_1['d_ps'].shape[0] == data_2['d_ps'].shape[0] # Need to have same number of neurons 

        # Skip any session pairs where ps is too dissimilar
        d_ps_corrcoef = np.corrcoef(data_1['d_ps'].flatten(), data_2['d_ps'].flatten())[0, 1]
        if d_ps_corrcoef < 0.6: # In practice this could be much higher, all passes > 0.99
            print('Skipping mouse {}, sessions {} to {}, photostim distance corr: {:.2f}'.format(
                data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx, d_ps_corrcoef
            ))
            continue

        ### At this point the pair of sessions pass criterion for further analysis ###
        session_pairs.append((int(day_1_idx), int(day_2_idx),))
            
    return session_pairs

def get_unique_sessions(session_idx_pairs, verbose=False):
    """
    Simple function to extract unique sessions in list of pairs.
    E.g. [(1, 2), (2, 3)] -> [1, 2, 3]
    """
    unique_sessions_idxs = []
    for pair_idxs in session_idx_pairs:
        day_1_idx, day_2_idx = pair_idxs 
        if day_1_idx not in unique_sessions_idxs:
            unique_sessions_idxs.append(day_1_idx)
        if day_2_idx not in unique_sessions_idxs:
            unique_sessions_idxs.append(day_2_idx)
    
    if verbose:
        print('Found {} unique sessions...'.format(len(unique_sessions_idxs)))
    return unique_sessions_idxs

def extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=None):
    """ 
    Wrapper function to extract various useful session data specified by data_to_extract dict.
    
    Note that extracting some session data such as the 'resp_ps_pred' requires
    we know the paired session (in order to properly determine the direct predictors).

    """
    extracted_data = {}

    # Always extract this to get some useful parameters for below
    trial_start_fs = data_dict['data']['F'][session_idx] # (trial_time_idx, neuron_idx, trial_idx)
    n_neurons = trial_start_fs.shape[1]

    for extract_data in data_to_extract:
        if 'd_ps' == extract_data:
            d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
            if d_ps_flat is None: # Some sessions don't have photostim distances
                extracted_data['d_ps'] = None
            else:
                extracted_data['d_ps'] = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)
        elif 'resp_ps' == extract_data or 'fake_resp_ps' == extract_data:
            if ps_stats_params['resp_ps_n_trials_back_mask'] > 0:  # Need to recompute
                assert 'd_ps' in extracted_data # Need to have loaded this to compute (always put d_ps before resp_ps)
                assert 'd_ps' != None

                ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
                
                if 'fake_resp_ps' == extract_data:
                    ps_fs = get_fake_ps_data(session_idx)
                else:
                    ps_fs = data_dict['data']['Fstim'][session_idx] 

                # Recalculate ps responses from data with omissions for recent stimulation
                # (ignore extra returns here, could change this in future)
                resp_ps, _ = compute_resp_ps_mask_prevs(
                    np.copy(ps_fs), ps_events_group_idxs, extracted_data['d_ps'],
                    ps_stats_params, return_extras=False
                )

                if np.sum(np.isnan(resp_ps)): # Zeros out nans
                    resp_ps[np.isnan(resp_ps)] = 0. # (n_neurons, n_groups)

                extracted_data['resp_ps'] = resp_ps
            else:
                resp_ps_flat = data_dict['data']['y'][session_idx] # mean response of neuron to a given photostimulation group (n_groups x n_neurons)
                extracted_data['resp_ps'] = unflatted_neurons_by_groups(resp_ps_flat, n_neurons,)
        elif 'resp_ps_pred' == extract_data or 'fake_resp_ps_pred' == extract_data:
            fake_ps_data = True if 'fake_resp_ps_pred' == extract_data else False 
            extracted_data['resp_ps_pred'], extracted_data['resp_ps_pred_extras'] = get_resp_ps_pred(
                session_idx, data_dict, ps_stats_params, paired_session_idx=paired_session_idx, fake_ps_data=fake_ps_data
            )
        elif 'trial_start_fs' == extract_data: # Don't normally need this unless recomputing metrics from raw responses
            extracted_data['trial_start_fs'] = trial_start_fs
        elif 'trial_start_metrics' == extract_data:
            idxs_post = np.arange(
            int((TS_POST[0] - T_START) * SAMPLE_RATE), int((TS_POST[1] - T_START) * SAMPLE_RATE),
            )
            idxs_pre = np.arange(
                int((TS_PRE[0] - T_START) * SAMPLE_RATE), int((TS_PRE[1] - T_START) * SAMPLE_RATE),
            )
            
            fit_changes = True if 'trial_start_metrics_changes' in data_to_extract else False

            tuning, trial_resp, pre, post, ts_extras  = compute_trial_start_metrics(
                trial_start_fs, (idxs_pre, idxs_post), mean_mode=ps_stats_params['trial_average_mode'],
                fit_changes=fit_changes
            )
            extracted_data['tuning'] = tuning
            extracted_data['trial_resp'] = trial_resp
            extracted_data['pre'] = pre
            extracted_data['post'] = post
            
            extracted_data['trial_start_metrics_changes'] = ts_extras
            
        elif 'd_masks' == extract_data:
            assert 'd_ps' in extracted_data # Need to have loaded this to compute (always put d_ps before resp_ps)
            assert 'd_ps' != None
            
            extracted_data['indir_mask'] = (extracted_data['d_ps'] > D_NEAR) * (extracted_data['d_ps'] < D_FAR)  # (n_neurons, n_groups) indirect mask, all neurons that are far enough to not be stimulated, but close enough to be influenced
            extracted_data['dir_mask'] = (extracted_data['d_ps'] < D_DIRECT)               # (n_neurons, n_groups) direct mask          
        elif extract_data == 'pairwise_corr':
            if ps_stats_params['pairwise_corr_type'] in ('trace',):
                extracted_data['pairwise_corr'] = data_dict['data']['trace_corr'][session_idx] # (n_neurons, n_neurons)       
            elif ps_stats_params['pairwise_corr_type'] in ('trial',):
                extracted_data['pairwise_corr'] = compute_cross_corrs_special(trial_start_fs, ts_trial=(-2, 10))
            elif ps_stats_params['pairwise_corr_type'] in ('pre',):
                extracted_data['pairwise_corr'] = compute_cross_corrs_special(trial_start_fs, ts_trial=(-2, 0))
            elif ps_stats_params['pairwise_corr_type'] in ('post',):
                extracted_data['pairwise_corr'] = compute_cross_corrs_special(trial_start_fs, ts_trial=(0, 10))
            elif ps_stats_params['pairwise_corr_type'] in ('behav_full', 'behav_start', 'behav_end',):
                extracted_data['pairwise_corr'] = get_correlation_from_behav(
                    session_idx, ps_stats_params, verbose=False
                )
            else:
                raise NotImplementedError('pairwise_corr_type {} not recognized.'.format(ps_stats_params['pairwise_corr_type']))
            
            # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
            extracted_data['pairwise_corr'] = np.where(np.isnan(extracted_data['pairwise_corr']), 0., extracted_data['pairwise_corr'])
        elif extract_data == 'mean_trial_activity':
            # Average of dF/F across all trials in the session
            F = data_dict['data']['F'][session_idx]
            # Transpose here so order is (n_trial_times x n_trials, n_neurons)
            F = np.transpose(F, (2, 0, 1)).reshape((-1, F.shape[1])) 

            extracted_data['mean_trial_activity'] = np.nanmean(F, axis=0) # Mean over trials and trial times        
        elif extract_data not in ('trial_start_metrics_changes',):
            raise ValueError('Data extraction {} not recognized'.format(extract_data))
        
    return extracted_data

def compute_trial_start_metrics(
    F, idxs, mean_mode='time_first', trial_response_mode='pre+post', fit_changes=False):
    """
    Computes responses aligned to trial start like tuning and 
    mean trial responses.
    
    INPUTS:
    F: Trial aligned fluorescences (n_trial_time, n_neurons, n_trials)
    idxs = (idxs_pre, idxs_post)
        idxs_pre: trial time idxs corresponding to pre-trial start
        idxs_post: trial time idxs corresponding to post-trial start
    mean_mode: time_first or trials_first
        time_first: Take mean over time, then trials (equal trial weight)
        trials_first: Take mean over trials, then time (upweights long trials)
    trial_response_mode: pre+post or even
        pre+post: Same as tuning, but adds pre and post. Upweights pre since its generally fewer time steps
        even: Even weighting over all trial time steps
    """
    
    idxs_pre, idxs_post = idxs
    
    ts_extras = {}

    # Equivalent to old way, just clearer
    if mean_mode == 'trials_first': # Mean over trials first, then mean over time. Upweights long trials
        if fit_changes:
            raise ValueError('Changes not well defined here since computes mean over trial first.')
        
        f = np.nanmean(F[:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=2) # Mean across trials
        f_mean_pre_trial = np.nanmean(f[idxs_pre, :], axis=0) # Mean across pre-trial times (trial_time_idx, neuron_idx,) -> (neuron_idx,)
        f_mean_post_trial = np.nanmean(f[idxs_post, :], axis=0) # Mean across post-trial times (trial_time_idx, neuron_idx,) -> (neuron_idx,)        
        
        if trial_response_mode not in ('pre+post',):
            raise NotImplementedError('Need to add even weighting here.')
        
        return (
            f_mean_post_trial - f_mean_pre_trial, # Tuning
            f_mean_post_trial + f_mean_pre_trial, # Trial response
            f_mean_pre_trial,                     # Pre
            f_mean_post_trial,                    # Post
            ts_extras,
        )

    elif mean_mode == 'time_first': # Mean over each trial's times first, then mean over trials
        with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
            warnings.simplefilter('ignore', category=RuntimeWarning)
            f_pre_trial = np.nanmean(F[idxs_pre, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across pre-trial times
            f_post_trial = np.nanmean(F[idxs_post, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across post-trial times
            
        # Important to take sum/difference before nanmean, so if pre-/post-trial response is nan, tuning for trial is nan
        if trial_response_mode in ('pre+post',):
            trial_response = f_post_trial + f_pre_trial
        elif trial_response_mode in ('even',):
            idxs_trial = np.concatenate((idxs_pre, idxs_post,), axis=0)
            trial_response = np.nanmean(F[idxs_trial, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0)
        else:
            raise ValueError('trial_response_mode {} not recognized!'.format(trial_response_mode))
        
        if fit_changes:
            
            max_trials = min(N_MAX_TRIALS, f_post_trial.shape[-1]) # Session could have less than N_MAX_TRIALS
            trial_idxs = np.arange(N_MIN_TRIALS, max_trials)
            
            for ts_metric, ts_metric_name in zip(
                (f_post_trial - f_pre_trial, trial_response, f_pre_trial, f_post_trial),
                ('tuning', 'trial_resp', 'pre', 'post',)
            ):
                neuron_slopes = np.zeros((ts_metric.shape[0],))
                neuron_intercepts = np.zeros((ts_metric.shape[0],))
                for neuron_idx in range(ts_metric.shape[0]):
                    nonnan_mask = ~np.isnan(ts_metric[neuron_idx, :]) # Non-nan trials
                    
                    neuron_slopes[neuron_idx], neuron_intercepts[neuron_idx], rvalue, pvalue, se = linregress(
                        trial_idxs[nonnan_mask], ts_metric[neuron_idx, nonnan_mask]
                    )
        
                ts_extras[ts_metric_name] = {'slope': neuron_slopes, 'intercept': neuron_intercepts,}
        
        return ( # Performs mean across trials in all these (n_neurons, n_trials,) -> (n_neurons,)
                np.nanmean(f_post_trial - f_pre_trial, axis=-1), # Tuning
                np.nanmean(trial_response, axis=-1),             # Trial response
                np.nanmean(f_pre_trial, axis=-1),                # Pre
                np.nanmean(f_post_trial, axis=-1),               # Post
                ts_extras,
            )
    else:
        raise ValueError('Mean mode {} not recognized.'.format(mean_mode))
    
def get_dir_indir_masks(ps_stats_params, data_1, data_2, verbose=False):
    """
    For a given pair of sessions, produces weighted masks for both 
    sessions. How this is done depents on ps_stats_params['mask_mode'],
    but generally this results in using the same mask for both sessions.
    
    Also has functionality for normalizing the masks by number of 
    neurons that get past the mask, if desired.
    """
    
    # Easier to just unpack these at this point
    dir_mask_1 = data_1['dir_mask']
    indir_mask_1 = data_1['indir_mask']
    dir_mask_2 = data_2['dir_mask']
    indir_mask_2 = data_2['indir_mask']
    
    if ps_stats_params['mask_mode'] in ('constant',): # Same mask for both sessions considered
        
        # Needs to be indirect/direct for BOTH sessions (* functions as AND)
        indir_mask_weighted = (indir_mask_1 * indir_mask_2).astype(np.float32) 
        dir_mask_weighted = (dir_mask_1 * dir_mask_2).astype(np.float32)

        if verbose:
            print('Constant mask:')
            print(' indir mask - kept {:.0f} eliminated {:.0f}'.format(
                np.sum(indir_mask_weighted), np.sum(np.logical_xor(indir_mask_1, indir_mask_2))
            ))
            print(' dir mask - kept {:.0f} eliminated {:.0f}'.format(
                np.sum(dir_mask_weighted), np.sum(np.logical_xor(dir_mask_1, dir_mask_2))
            ))

        if ps_stats_params['normalize_masks']: # Divide each mask by the number of nonzero neurons in each group
            raise NotImplementedError('Double check this isnt depricated, havent used for a while.')
            indir_mask_weighted = indir_mask_weighted / np.sum(indir_mask_weighted, axis=0, keepdims=True)
            dir_mask_weighted = dir_mask_weighted / np.sum(dir_mask_weighted, axis=0, keepdims=True)
    
    elif ps_stats_params['mask_mode'] in ('kayvon_match',): # Just use Day 2
        indir_mask_weighted = indir_mask_2.astype(np.float32)
        dir_mask_weighted = dir_mask_2.astype(np.float32)
        
        if ps_stats_params['normalize_masks']:
            raise NotImplementedError('Double check this isnt depricated, havent used for a while.')
        if (ps_stats_params['x_over_neuron_mode'] not in ('matrix_multiply', 'matrix_multiply_sanity',) or
            ps_stats_params['y_over_neuron_mode'] not in ('matrix_multiply', 'matrix_multiply_sanity',)):
            raise NotImplementedError('Kayvon doesnt consider these so doesnt make sense')
    elif ps_stats_params['mask_mode'] in ('each_day',): # Individual distance masks for each session
        raise NotImplementedError('Double check this isnt depricated, havent used for a while.')
        # These may be used for metrics later but not here, just a weighted sum of two masks
        indir_mask_weighted = 1/2 * (indir_mask_1.astype(np.float32) + indir_mask_2.astype(np.float32))
        dir_mask_weighted = 1/2 * (dir_mask_1.astype(np.float32) + dir_mask_2.astype(np.float32))
        if ps_stats_params['normalize_masks']:
            raise NotImplementedError('Need to decide if we want to do this individually or not')
        if (ps_stats_params['x_over_neuron_mode'] not in ('matrix_multiply',) or
            ps_stats_params['y_over_neuron_mode'] not in ('matrix_multiply',)):
            raise NotImplementedError('Evaluate_over_neurons needs to be modified to account for different masks for different days.')
    else:
        raise ValueError('Mask mode {} not recognized.'.format(ps_stats_params['mask_mode']))
   
    return indir_mask_weighted, dir_mask_weighted

def compute_cross_corrs_special(F, ts_trial=(-2, 10)):
    """
    Cross correlation computation over special times to compare against entire session.

    INPUTS:
    F: Trial aligned fluorescences (n_trial_time, n_neurons, n_trials)
    """
    n_trial_times = F.shape[0]
    n_neurons = F.shape[1]
    n_trials = F.shape[2]

    idxs_trial = np.arange(
        int((ts_trial[0] - T_START) * SAMPLE_RATE), int((ts_trial[1] - T_START) * SAMPLE_RATE),
    )

    n_max_trials = np.min((n_trials, N_MAX_TRIALS)) # Might not have N_MAX_TRIALS trials to scan over
    trial_corrs = np.zeros((n_max_trials - N_MIN_TRIALS, n_neurons, n_neurons))

    for trial_idx in range(N_MIN_TRIALS, n_max_trials): # Compute mean over each trial
        # Select down to relevant times 
        trial_times_fs = F[idxs_trial, :, trial_idx]
        # Remove nan times (use first neuron to find nans)
        trial_times_fs = trial_times_fs[~np.isnan(trial_times_fs[:, 0]), :]

        if trial_times_fs.shape[0] == 0: # Skip this trial
            trial_corrs[trial_idx - N_MIN_TRIALS] = np.nan
        else:
            trial_corrs[trial_idx - N_MIN_TRIALS] = np.corrcoef(trial_times_fs.T)

    return np.nanmean(trial_corrs, axis=0) # Mean over trials

def get_correlation_from_behav(session_idx, ps_stats_params, verbose=False):
    behav_idx = SESSION_IDX_TO_BEHAV_IDX[session_idx]
    if verbose:
        print('Loading behav from: {}'.format(BEHAV_DATA_PATH + BEHAV_FILES[behav_idx]))
    data_dict_behav = scipy.io.loadmat(BEHAV_DATA_PATH + BEHAV_FILES[behav_idx])
    
    raw_df_trace = data_dict_behav['df_closedLoop']
    
    if ps_stats_params['pairwise_corr_type'] == 'behav_full':
        df_trace = raw_df_trace
    elif ps_stats_params['pairwise_corr_type'] == 'behav_start':
        split_idx = int(np.floor(raw_df_trace.shape[0] / 2))
        df_trace = raw_df_trace[:split_idx, :]
    elif ps_stats_params['pairwise_corr_type'] == 'behav_end':
        split_idx = int(np.floor(raw_df_trace.shape[0] / 2))
        df_trace = raw_df_trace[split_idx:, :]
        
    # Suppress warnings when df_trace contains nans/inf since corrcoef just returns nans as expected
    # session_idx = 18 and neuron_idx = 382 has 500-ish nans and 23487 infs, no other known cases
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        corrs = np.corrcoef(df_trace.T)
        
        # Set nans corrs to 0.
        corrs = np.where(np.isnan(corrs), 0., corrs)
    
    return corrs

def get_fake_ps_data(session_idx, data_type='trial_start_data', verbose=False):
    """
    For a given session, generate data that has the same structure as the photostimulation
    data (i.e. (ps_times, n_neurons, n_ps_events,)) but from times where no photostimulation
    is actually occuring. This data is used to validate various metrics against true 
    photostimulation data. Generally used to replace 'FStim'.
    
    Note because even when not photostimulating we expect to capture some causal connectivity
    between neurons, this data may still yield significant trends, but should not be nearly
    as significant as true photostimulation data.
    
    """
    
    def get_random_ps_data(data, n_steps_per_ps, data_type='trial_start_data'):
        if data_type == 'trial_start_data': # Random trial then random non-nan time
            trial_idx = np.random.randint(data.shape[-1])
            neuron_idx = np.random.randint(data.shape[1])
            
            nan_idxs_found = False
            
            while not nan_idxs_found:
                non_nan_idxs = np.where(~np.isnan(data[:, neuron_idx, trial_idx]))[0]
                
                # This can trigger an error if the particular neuron has all nans for the event, 
                # if this occurs just redraws another trial/neuron compo
                try:
                    start_idx = np.random.randint( # Ensures this isn't keeping a bunch of nans
                        np.min(non_nan_idxs), np.max(non_nan_idxs) - n_steps_per_ps
                    )
                    nan_idxs_found = True
                except ValueError as err:
                    if 'low >= high' in str(err):
                        if verbose:
                            print('All nans at trial {} neuron {}, getting new trial and neuron'.format(trial_idx, neuron_idx))
                        trial_idx = np.random.randint(data.shape[-1])
                        neuron_idx = np.random.randint(data.shape[1])
                    else:
                        raise ValueError('Another error: {} occured')

            return data[start_idx:start_idx+n_steps_per_ps, :, trial_idx]
        else:
            raise NotImplementedError('data_type not recognized.')

    ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
    n_events = ps_fs.shape[-1]
    if data_type == 'trial_start_data': # Random trial then random non-nan time
        trial_start_fs = data_dict['data']['F'][session_idx]

    fake_ps_fs = np.zeros_like(ps_fs)
    ps_resp_bounds = (np.min(IDXS_PRE_PS), np.max(IDXS_POST_PS)) # Finds bounds that need to filled for each PS with fake data
    n_steps_per_ps = ps_resp_bounds[1] - ps_resp_bounds[0] # Length of each fake data

    for ps_event_idx in range(n_events):
        fake_ps_fs[ps_resp_bounds[0]:ps_resp_bounds[1], :, ps_event_idx] = get_random_ps_data(
            trial_start_fs, n_steps_per_ps, data_type=data_type
        )

    # Copies nan statistics of true PS data
    fake_ps_fs = np.where(np.isnan(ps_fs), ps_fs, fake_ps_fs)
    
    return fake_ps_fs

# Cell 11
import sys
from sklearn.decomposition import PCA

def get_resp_ps_pred(session_idx, data_dict, ps_stats_params, paired_session_idx=None, fake_ps_data=False, verbose=False):
    """
    Wrapper function for the full determination of predicted photostimulation responses. Contains three
    primary steps:
    1. Find the direct predictors
    2. Fits the indirect responses from the direct predictors
    3. Gets the predicted indirect response
    
    Note that this code uses day_1 and day_2 session_idx notation, but the sessions may not necessarily
    be 'Day 1' and 'Day 2' (in some cases, day 1 could be after what is called day 2). Additionally, this
    code extracts some data that might have already been extracted, so could be made more efficient.
    
    INPUTS:
    - fake_ps_data: validation test on fake photostim data. Does everything the same but replaces resp_ps
      with fake photostim data to test if photostimulation is actually doing anything.

    OUTPUTS:
    - resp_ps_pred: (n_neurons, n_groups,)
        - For a group's indirect neurons, contains the predictions from the fit
        - For a group's direct neurons, just contains the corresponding direct_input
        - Otherwise, all entires are nans.
    - extras:
        - 'r_squareds': (n_neurons, n_groups,)
        - 'params': list of size (n_groups,), with each entry of size (n_indirect, n_params,)

    """
    day_1_idx = session_idx

    if paired_session_idx == None:
        has_paired_session = False
    else:
        has_paired_session = True
        day_2_idx = paired_session_idx
    
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    if fake_ps_data:
        ps_fs_1 = get_fake_ps_data(day_1_idx)
    else:
        ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_1 = unflatted_neurons_by_groups(data_dict['data']['x'][day_1_idx], ps_fs_1.shape[1],) # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
        ps_fs_1, data_dict['data']['seq'][day_1_idx], d_ps_1, ps_stats_params,
    )
    resp_ps_events_1 = resp_ps_extras_1['resp_ps_events']

    if has_paired_session:
        if fake_ps_data:
            ps_fs_2 = get_fake_ps_data(day_2_idx)
        else:
            ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)
        d_ps_2 = unflatted_neurons_by_groups(data_dict['data']['x'][day_2_idx], ps_fs_2.shape[1],) # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
        resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
            ps_fs_2, data_dict['data']['seq'][day_2_idx], d_ps_2, ps_stats_params,
        )
        resp_ps_events_2 = resp_ps_extras_2['resp_ps_events']

    n_neurons = ps_fs_1.shape[1]
    n_groups = d_ps_1.shape[1] # +1 already accounted for because MatLab indexing
    
    resp_ps_pred = np.empty((n_neurons, n_groups,))
    resp_ps_pred[:] = np.nan
    r_squareds = np.copy(resp_ps_pred)

    indirect_params_all = []

    for group_idx in range(n_groups):
        
        # Determine direct and indirect masks
        if has_paired_session: 
            direct_idxs = np.where(np.logical_and( # Needs to be considered direct in both sessions
                d_ps_1[:, group_idx] < D_DIRECT, 
                d_ps_2[:, group_idx] < D_DIRECT
            ))[0]
            indirect_idxs = np.where(np.logical_and( # Needs to be considered indirect in both sessions
                np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
                np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
            ))[0]
        else:
            direct_idxs = np.where(d_ps_1[:, group_idx] < D_DIRECT)[0]
            indirect_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR))[0]

        ### 1. Determine direct predictors ###
        dir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        
        if 'shuffle_indirect_events' in ps_stats_params['validation_types']:
            indir_resp_ps_events_1 = shuffle_along_axis(indir_resp_ps_events_1, axis=-1)
        
        if has_paired_session: 
            dir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[direct_idxs, :] # (n_direct, n_events,)
            
            # Concatenate over events to determine direct predictors
            dir_resp_ps_events_1_2 = np.concatenate((dir_resp_ps_events_1, dir_resp_ps_events_2), axis=-1)
            direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
                dir_resp_ps_events_1_2, ps_stats_params, verbose=verbose,
                n_events_sessions=(dir_resp_ps_events_1.shape[-1], dir_resp_ps_events_2.shape[-1]),
            )
        else:
            direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
                dir_resp_ps_events_1, ps_stats_params, verbose=verbose
            )

        ### 2. Fit the indirect photostimulation response ###
        indirect_params_1, indirect_pvalues_1, fit_extras_1 = fit_photostim_variation(
            dir_resp_ps_events_1, indir_resp_ps_events_1, direct_predictors, direct_shift,
            ps_stats_params, verbose=verbose, return_extras=True
        )   
    
        ### 3. Gets average direct input, then use this to determine indirect predictions ###
        ###    Also determine what the direct photostim response should be.               ###
        
        if ps_stats_params['direct_input_mode'] in ('average',):
            if has_paired_session: 
                dir_resp_ps_events = dir_resp_ps_events_1_2
            else:
                dir_resp_ps_events = dir_resp_ps_events_1
                
            # Note here we determine the average input for each event THEN average over
            # events of both sessions. This does not necessarily yield the same result as 
            # averaging over events in each session first and then determining the input because 
            # of how we treat nans and uneven event counts   
            # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
            direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,) avg over all events
        
            # Optional direct weights shifting for fit
            if ps_stats_params['modify_direct_weights']:
                if ps_stats_params['direct_predictor_mode'] in ('sum',):
                    with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        direct_input_1 = np.nanmean(direct_predictors_events, axis=-1)
                    direct_predictor_idx = 0
                    magnitude_scale = direct_input / direct_input_1[direct_predictor_idx]
                    resp_ps_pred[direct_idxs, group_idx] = magnitude_scale * resp_ps_1[direct_idxs, group_idx]
                else:
                    raise NotImplementedError('Modification not yet implemented for type {}'.format(ps_stats_params['direct_predictor_mode']))
            else:
                # Note the direct weights are generally averaged over in A metrics, so just return the raw values back and handle evaluation at mean later explicitly
                resp_ps_pred[direct_idxs, group_idx] = resp_ps_1[direct_idxs, group_idx]
        
        elif ps_stats_params['direct_input_mode'] in ('average_equal_sessions',): # Average over sessions first, then put them together
            assert has_paired_session
            direct_predictors_events_1 = nan_matmul(direct_predictors, dir_resp_ps_events_1)
            direct_predictors_events_2 = nan_matmul(direct_predictors, dir_resp_ps_events_2)
            
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                direct_input_1 = np.nanmean(direct_predictors_events_1, axis=-1, keepdims=True) # (n_direct_predictors, 1)
                direct_input_2 = np.nanmean(direct_predictors_events_2, axis=-1, keepdims=True) # (n_direct_predictors, 1)
                direct_input = np.nanmean(np.concatenate((direct_input_1, direct_input_2), axis=-1)) # (n_direct_predictors, 2) -> (n_direct_predictors,)
            
            # Optional direct weights shifting for fit
            if ps_stats_params['modify_direct_weights']:
                if ps_stats_params['direct_predictor_mode'] in ('sum',):
                    direct_predictor_idx = 0
                    magnitude_scale = direct_input / direct_input_1[direct_predictor_idx, 0]
                    resp_ps_pred[direct_idxs, group_idx] = magnitude_scale * resp_ps_1[direct_idxs, group_idx]
                else:
                    raise NotImplementedError('Modification not yet implemented for type {}'.format(ps_stats_params['direct_predictor_mode']))
            else:
                # Note the direct weights are generally averaged over in A metrics, so just return the raw values back and handle evaluation at mean later explicitly
                resp_ps_pred[direct_idxs, group_idx] = resp_ps_1[direct_idxs, group_idx]
            
            
        elif ps_stats_params['direct_input_mode'] in ('ones',):
            direct_input = np.ones((direct_predictors.shape[0],))
            
#             # Normalize the direct inputs so that they sum to 1
#             resp_ps_pred[direct_idxs, group_idx] = resp_ps_1[direct_idxs, group_idx] / np.nansum(resp_ps_1[direct_idxs, group_idx])
            resp_ps_pred[direct_idxs, group_idx] = resp_ps_1[direct_idxs, group_idx]
        elif ps_stats_params['direct_input_mode'] in ('minimum',): # Separately computes direct input, then minimum mag of both
            assert has_paired_session
            raise NotImplementedError()
        else:
            raise ValueError('Direct_input_mode {} not recognized'.format(ps_stats_params['direct_input_mode']))
    
        
        # Uses average direct input to predict photostimulation response
        indirect_predictions_1 = photostim_predict(indirect_params_1, direct_input, ps_stats_params, verbose=verbose)

        resp_ps_pred[indirect_idxs, group_idx] = indirect_predictions_1

        r_squareds[indirect_idxs, group_idx] = fit_extras_1['r_squareds']
        indirect_params_all.append(indirect_params_1)

    extras = {
        'r_squareds': r_squareds,
        'params': indirect_params_all,
    }

    return resp_ps_pred, extras

def find_photostim_variation_predictors(
    direct_resp_ps_events, ps_stats_params, n_events_sessions=None, 
    return_extras=False, verbose=False):
    """
    For a single group in either a single photostimulation session 
    or a pair of sessions, finds the predictor directions in the direct 
    neuron space of the given group. The `direct predictors' are then 
    used to fit the indirect responses.

    We fit indirect responses to `direct predictors' and not just the
    direct neurons because there are often a comparable number of direct
    neurons to events, and thus fitting each indirect response would be
    ill-conditioned (i.e. if there are more direct neurons than events,
    we could perfectly fit each indirect neuron variation).
    
    INPUTS:
    - direct_resp_ps_events: (n_direct, n_events)
        - Note this can have nan entries due to bad laser data and also nan masking of previously
          directly stimulated neurons. When all direct neurons have nan entries for an event, the
          event is auotmatically removed. When there are only some direct neurons that are nans, 
          the handling of these nan entries is controlled by the 'direct_predictor_nan_mode' option, 
          see below
    - ps_stats_params['direct_predictor_mode']: various ways of determining the direct predictors
        - sum: just use the total direct response via summing over all direct neurons
        - top_devs: 
        - top_devs_center:
    - ps_stats_params['n_direct_predictors']: number of direct predictors that we will use to 
        fit the indirect repsonses. This number can't be too large or the fitting becomes ill-
        conditioned.
    - ps_stats_params['direct_predictor_intercept_fit']: bool, whether or not to fit intercept
    - ps_stats_params['direct_predictor_nan_mode']: Controls how nans are treated in events when 
        only a few direct neurons are nans.
        - ignore_nans: the nan entries are treated as 0s, which affects a given direct neuron's 
            statistics. This allows one to keep the maximum number of events.
        - eliminate_events: any event with a nan entry is completely eliminated, including non-nan
            entries. This removes a substantial amount of events, but does not affect neuron 
            statistics.
    - n_events_sessions: when session events are concatenated together, this is an optional tuple
        of the number of events in each indivdiual session. Used to reject sessions that have too
        few events
    
    OUTPUTS:
    - direct_predictors: (n_direct_predictors, n_direct)
        - Direct predictors can be all nans and thus effectively ignored for a few cases. This 
          includes either the event or direct neuron counts being too low (including after 
          eliminating events due to nans) or eigendecompositions not converging.
    - extras: information from direct_predictor fitting
    
    """
    
    n_direct = direct_resp_ps_events.shape[0]
    n_events = direct_resp_ps_events.shape[-1]
    
    intercept_fit = ps_stats_params['direct_predictor_intercept_fit']
    n_direct_predictors = ps_stats_params['n_direct_predictors']
    
#     direct_predictors = np.zeros((n_direct_predictors, n_direct,)) # Default of 0s
    direct_predictors = np.empty((n_direct_predictors, n_direct,)) 
    direct_predictors[:] = np.nan # default value, overriden unless early exit from too few direct/events
    direct_shift = np.zeros((n_direct, 1,))
    
    # Extra return defaults, overriden below
    variation_explained = None
    direct_pr = None
    pr_ratio = None

    determine_direct_predictors = True

    if n_direct < N_MIN_DIRECT or n_events < N_MIN_EVENTS: # Note this catches n_direct = 0 cases
        determine_direct_predictors = False
        if verbose:
            print('  find pred - Too few events or direct, skipping group. (n_direct, n_events):', direct_resp_ps_events.shape)
    if n_events_sessions is not None:
        n_events_1, n_events_2 = n_events_sessions
        if n_events_1 < N_MIN_EVENTS or n_events_2 < N_MIN_EVENTS:
            determine_direct_predictors = False
            if verbose:
                print('  find pred - Too few in individual sessions. (n_events_1, n_events_2):', n_events_1, n_events_2)
    
    if determine_direct_predictors: # Determine direct_predictors if early stop conditions haven't been met yet
        if ps_stats_params['direct_predictor_mode'] in ('sum',):
            if ps_stats_params['n_direct_predictors'] != 1:
                raise ValueError('For direct_predictor_mode sum, n_direct_predictors = 1 by definition.')
            direct_predictors = np.ones((1, n_direct))
        elif ps_stats_params['direct_predictor_mode'] in ('top_mags',):
            max_direct_idxs = np.argsort(np.nansum(direct_resp_ps_events**2, axis=-1))[::-1] # sum across events

            direct_predictors = np.zeros((n_direct_predictors, n_direct,)) # Override nans default
            # Keeping only the neurons with the largest responsing via 1-hot matrix
            if n_direct_predictors > n_direct: # This catches special case when n_direct is smaller than n_direct_predictors
                print('More direct predictors than direct neurons, extra entries will just be zeros. n_direct = {}'.format(n_direct))
                n_nonzero = np.copy(n_direct)
            else:
                n_nonzero = np.copy(n_direct_predictors)
                
            for direct_predictor_idx in range(n_nonzero): # Set corresponding direct element to 1.0
                direct_predictors[direct_predictor_idx, max_direct_idxs[direct_predictor_idx]] = 1.0
                
        elif ps_stats_params['direct_predictor_mode'] in ('top_devs', 'top_devs_center'):
            # Always eliminate an event that has all nans across all direct neurons
            keep_event_idxs = np.where(~np.all(np.isnan(direct_resp_ps_events), axis=0))[0]
            direct_resp_ps_events = direct_resp_ps_events[:, keep_event_idxs]
            
            if ps_stats_params['direct_predictor_nan_mode'] in ('ignore_nans',):
                # Just replace nans with 0s
                direct_resp_ps_events = np.where(np.isnan(direct_resp_ps_events), 0.0, direct_resp_ps_events)
            elif ps_stats_params['direct_predictor_nan_mode'] in ('eliminate_events',):
                # Nan masking across events, see if event has any nans across all direct neurons
                keep_event_idxs = np.where(~np.any(np.isnan(direct_resp_ps_events), axis=0))[0]
                direct_resp_ps_events = direct_resp_ps_events[:, keep_event_idxs]
            
            n_events = direct_resp_ps_events.shape[-1] # Update this because some event could have been eliminated
    #         direct_events_pca = PCA()
    #         direct_events_pca.fit(direct_resp_ps_events.T)
            if n_events < N_MIN_EVENTS: # Check to see if updated events have too few now
                print('  find pred - After nan masking, too few events. (n_direct, n_events):', direct_resp_ps_events.shape)
            else:
                if ps_stats_params['direct_predictor_mode'] in ('top_devs_center',):
                    direct_shift = np.mean(direct_resp_ps_events, axis=-1, keepdims=True)
                    direct_resp_ps_events -= direct_shift

                sum_of_squared_deviations =  np.matmul(direct_resp_ps_events, direct_resp_ps_events.T) / (n_events - 1) # (n_direct, n_direct)
        #         sum_of_squared_deviations = np.cov(direct_resp_ps_events.T, rowvar=False) # Only equivalent to above if centered

                try: # Very rarely, eigenvalue does not converge so need to catch error
                    evals, evecs = np.linalg.eigh(sum_of_squared_deviations) # eigh since symmetric
                    sort_idxs = np.argsort(evals)[::-1] # largest to smallest
                    evecs = evecs[:, sort_idxs]
                    evals = evals[sort_idxs]

                    if return_extras:
                        variation_explained = evals / np.sum(evals)
                        direct_pr = participation_ratio_vector(variation_explained)
                        pr_ratio = direct_pr / n_direct

                        print('Direct_PR: {:.1f}, ratio: {:.2f}'.format(direct_pr, pr_ratio))

                except np.linalg.LinAlgError as err:
                    if 'Eigenvalues did not converge' in str(err):
                        print('Eigenvalues did not converge, setting direct_predictors to zeros. Dev shape:', sum_of_squared_deviations.shape)
                        evecs = np.zeros_like(sum_of_squared_deviations)
                        evals = np.zeros((sum_of_squared_deviations.shape[0]))

                        variation_explained = np.nan * evals
                        direct_pr = np.nan
                        pr_ratio = np.nan
                    else:
                        raise ValueError('Another error: {} occured')        

                direct_predictors = evecs[:, :n_direct_predictors].T

                # This is a catch if number of direct predictors is greater than number of direct,
                # in which case evals will be too small, just fills the rest of it with zeros
                if direct_predictors.shape[0] < n_direct_predictors:
                    print('Too many direct predictors for n_direct, filling with zeros. Direct_predictors shape', direct_predictors.shape)
                    direct_predictors_temp = np.zeros(((n_direct_predictors, n_direct)))
                    direct_predictors_temp[:direct_predictors.shape[0], :] = direct_predictors

                    direct_predictors = np.copy(direct_predictors_temp)
            
        else:
            raise ValueError('direct_predictor_mode {} not recongized!'.format(
                ps_stats_params['direct_predictor_mode']
            ))
        
    if return_extras:
        extras = {
            'variation_explained': variation_explained,
            'pr_ratio': pr_ratio,
        }
    else:
        extras = None
        
    return direct_predictors, direct_shift, extras

def nan_matmul(x, y, mode='zeros', verbose=False):
    """
    A matmul that accounts for nans, setting individual entries in rows/columns to 0s.
    
    If all columns or rows are nans, also sets corresponding elements to be nans.
    Note: np.nansum() defaults to 0. for all nans, so different behavior.
    
    INPUTS:
    x shape (a, b)
    y shape (b, c)
    
    OUTPUT:
    z shape (a, c)
    """
        
    # Just sets corresponding elements to zero
    x_zeros = np.where(np.isnan(x), 0., x)
    y_zeros = np.where(np.isnan(y), 0., y)
    
    result = np.matmul(x_zeros, y_zeros)

    # Now see if any elements should be set to nans because all corresponding elements are nans
    if np.any(np.all(np.isnan(x), axis=-1)): # Checks if all along b are nans
        if verbose: print('Warning: Found rows that are all nans, setting elements to nan.')
        
        if len(result.shape) == 2:
            result[np.where(np.all(np.isnan(x), axis=-1))[0], :] = np.nan
        elif len(result.shape) == 1:
            result[np.where(np.all(np.isnan(x), axis=-1))[0]] = np.nan
    if np.any(np.all(np.isnan(y), axis=0)): # Checks if all along b are nans
        if verbose: print('Warning: Found columns that are all nans, setting elements to nan.')
#         raise ValueError('One or more column is all nans:', np.where(np.all(np.isnan(y), axis=0))[0])
        if len(result.shape) == 2:
            result[:, np.where(np.all(np.isnan(x), axis=-1))[0]] = np.nan
        elif len(result.shape) == 1:
            result[np.where(np.all(np.isnan(y), axis=0))[0]] = np.nan
     
    return result

def fit_photostim_variation(
    direct_resp_ps_events, indirect_resp_ps_events, direct_predictors, direct_shift, ps_stats_params,
    return_extras=False, verbose=False,
):
    """
    Fits the variation over a single photostimulation group. First this uses the direct_predictor directions
    to convert the direct neuron responses into direct predictors for every event (with special nan
    handling, see below). Given these direct predictors, does an OLS fit to each indirect neuron and
    returns the parameters of the fit so they can be used for prediction later.
    
    INPUTS:
    - direct_resp_ps_events: (n_direct, n_events,)
        - Can have nan entries because of laser tripping, bad data, or n_trials_back masking. Any event
          with all nan entries for direct neurons is automatically eliminated. 
    - indirect_resp_ps_events: (n_indirect, n_events,)
        - Can have nan entries because of laser tripping, bad data, or n_trials_back masking. Events that 
          are elimianted for direct are similarly eliminated here
    - direct_predictors: (n_direct_predictors, n_direct,)
        - Note this can be all nans for a given groups
    - direct_shift: (n_direct, 1,)
    - ps_stats_params
    
    - nan_mode: 
        - ignore_nans: effectively sets nan elements to zero, minimal elimination of valid data
        - eliminate_events: removes any event that has a nan for any direct neuron, can eliminate
            a lot of data

    OUTPUTS:
    - indirect_params: (n_indirect, n_params,)
        - Parameters can be all nans and thus ignored if direct_predictors is has any nans
    - indirect_pvalues: (n_indirect, n_params,)
    - extras:
        - conditioning_number: conditioning number of the direct predictors over events, used to 
            determine if fit is justified
        - r_squareds: (n_indirect,)
    """
    
    n_direct = direct_resp_ps_events.shape[0]
    n_indirect = indirect_resp_ps_events.shape[0]
    
    assert direct_resp_ps_events.shape[-1] == indirect_resp_ps_events.shape[-1]
    
    n_events = direct_resp_ps_events.shape[-1]
    n_direct_predictors = direct_predictors.shape[0]
    
    if ps_stats_params['direct_predictor_intercept_fit']:
        n_params = n_direct_predictors + 1
    else:
        n_params = n_direct_predictors
        
#     indirect_params = np.zeros((n_indirect, n_params,))
#     indirect_pvalues = np.ones((n_indirect, n_params,)) # 1s to not cause errors
#     indirect_rsquareds = np.zeros((n_indirect,))
    
    # Default values, filled below if fit is performed 
    indirect_params = np.empty((n_indirect, n_params,))
    indirect_params[:] = np.nan
    indirect_pvalues = np.copy(indirect_params)
    indirect_rsquareds = np.empty((n_indirect,))
    indirect_rsquareds[:] = np.nan
    conditioning_number = np.nan
    
    fit_indirect = True
    
    if np.isnan(direct_predictors).any(): # Case where direct predictors was ignored so fitting should be as well
        if verbose: print('  fit - direct_predictors has nans, skipping fit')
        fit_indirect = False
    if n_direct == 0: # Case where there are no direct neurons
        if verbose: print('  fit - n_direct = 0, skipping fit')
        fit_indirect = False

    if fit_indirect:
        ### First, conststruct the direct predictors ###
        if ps_stats_params['direct_predictor_mode'] in ('top_mags',): # Slightly different nan masking in this case
            # Recovers nonzero idxs from direct_predictors
            direct_idxs = []
            for direct_predictor_idx in range(n_direct_predictors):
                direct_idxs.append(np.where(direct_predictors[direct_predictor_idx] > 0)[0])
            direct_idxs = np.concatenate(direct_idxs, axis=0)

            # Any nans that are in direct_resp_ps_events get carried through then ignored in the fit below.
            direct_predictors_events = (direct_resp_ps_events - direct_shift)[direct_idxs, :]
        else:
            # Always eliminate any events that are entirely nans
            keep_event_idxs = np.where(~np.all(np.isnan(direct_resp_ps_events), axis=0))[0]
            direct_resp_ps_events = direct_resp_ps_events[:, keep_event_idxs]
            indirect_resp_ps_events = indirect_resp_ps_events[:, keep_event_idxs]

            # Determines what to do with the remaining events that are only partially nans
            if ps_stats_params['direct_predictor_nan_mode'] in ('ignore_nans',):
                direct_predictors_events = nan_matmul(direct_predictors, direct_resp_ps_events - direct_shift)
            elif ps_stats_params['direct_predictor_nan_mode'] in ('eliminate_events',):
                 # Nan masking across events, see if event has any nans across all direct neurons
                keep_event_idxs = np.where(~np.any(np.isnan(direct_resp_ps_events), axis=0))[0]
                direct_resp_ps_events = direct_resp_ps_events[:, keep_event_idxs]
                indirect_resp_ps_events = indirect_resp_ps_events[:, keep_event_idxs]

                if verbose:
                    print(' Keeping {}/{} events due to nans'.format(keep_event_idxs.shape[0], n_events))

                n_events = direct_resp_ps_events.shape[-1] # Update this because some event could have been eliminated

                # All nans removed, so can use usual matrix multiply
                direct_predictors_events = np.matmul(direct_predictors, direct_resp_ps_events - direct_shift)

        if return_extras:
            try:
                conditioning_number = np.linalg.cond(direct_predictors_events)
            except np.linalg.LinAlgError as err:
                print('n_dir:', n_direct)
                print('direct_resp_ps_events shape:', direct_resp_ps_events.shape)
                if 'SVD did not converge' in str(err):
                    conditioning_number = np.inf
                else:
                    raise ValueError('Another error: {} occured'.format(str(err)))
#         print('Conditioning number: {:.2f}'.format(conditioning_number))
        
        ### Next, fit each indirect based on the direct predictors ### 
        for indirect_idx in range(n_indirect):

            indirect_resp_ps_neuron = indirect_resp_ps_events[indirect_idx:indirect_idx+1, :] # (1, n_events)

            if ps_stats_params['direct_predictor_intercept_fit']:
                fit_X = sm.add_constant(direct_predictors_events.T)
            else:
                fit_X = direct_predictors_events.T

            # Y: (n_events, 1)
            # X: (n_events, n_direct_predictors)
            ols_model = sm.OLS(indirect_resp_ps_neuron.T, fit_X, missing='drop') # drop nans
            ols_fit = ols_model.fit()

            if ols_fit.params.shape[0] < n_params: # In the case n_direct < n_direct_predictors, doesn't produce enough params
                indirect_params[indirect_idx, :ols_fit.params.shape[0]] = ols_fit.params
                indirect_pvalues[indirect_idx, :ols_fit.params.shape[0]] = ols_fit.pvalues
            else:
                indirect_params[indirect_idx] = ols_fit.params
                indirect_pvalues[indirect_idx] = ols_fit.pvalues
                indirect_rsquareds[indirect_idx] = ols_fit.rsquared

    if return_extras:
        extras = {
            'conditioning_number': conditioning_number,
            'r_squareds': indirect_rsquareds,
        }
    else:
        extras = None
    
    return indirect_params, indirect_pvalues, extras

def photostim_predict(indirect_params, direct_input, ps_stats_params, verbose=False):
    """
    Given fit indirect_params and a direct_input for a given group, yields a prediction of the 
    photostimulation response. Note we don't use the built in predict for the OLS fits above 
    because we want to do this for all indirect neurons at once, so we just manually do the 
    matrix multiplication.
    
    INPUTS:
    - indirect_params: (n_indirect, n_params,)
        - Note: this could be all nans for a group if the fit was not valid because the
          direct_predictors were all nans (i.e. if too few direct or events). In this case
          just returns all nans which will be ignored in fits.
    - direct_input: scalar, (n_direct_predictors,) OR (n_samples, n_direct_predictors,)
    
    OUTPUTS:
    - indirect_predictions: (n_indirect,)
    """
    
    if np.isnan(indirect_params).any(): # Early exit condition if indirect_params are nans
        if verbose: print('  predict - indirect_params has nans, skipping predict')
        indirect_predictions = np.empty((indirect_params.shape[0],))
        indirect_predictions[:] = np.nan
        return indirect_predictions
    
    if len(direct_input.shape) == 0: # 0d input case
        n_samples = 1
        n_direct_predictors = 1
        direct_input = np.array((direct_input,)) # Makes 1d
    elif len(direct_input.shape) == 1: # 1d input case
        n_samples = 1
        n_direct_predictors = direct_input.shape[0]
    elif len(direct_input.shape) == 2:
        n_samples = direct_input.shape[0]
        n_direct_predictors = direct_input.shape[1]
    else:
        raise ValueError('direct_input shape {} not recognized'.format(direct_input.shape))
    
    assert ps_stats_params['n_direct_predictors'] == n_direct_predictors
    
    if ps_stats_params['direct_predictor_intercept_fit']:
        if len(direct_input.shape) == 1:
            # (n_direct_predictors,) -> (1+n_direct_predictors,) = (n_params,)
            direct_input = np.concatenate((np.array((1,)), direct_input,), axis=0)
        elif len(direct_input.shape) == 2:
            # (n_samples, n_direct_predictors,) -> (n_samples, 1+n_direct_predictors,)
            direct_input = np.concatenate((np.ones((n_samples, 1)), direct_input,), axis=-1)
     
    if len(direct_input.shape) == 1:
        return np.matmul(indirect_params, direct_input) # (n_indirect,) <- (n_indirect, n_params,) x (n_params,) 
    elif len(direct_input.shape) == 2:
        raise NotImplementedError('Multiple sample return not yet implemented')

