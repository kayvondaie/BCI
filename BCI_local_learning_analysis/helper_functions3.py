import warnings
from scipy.stats import ttest_1samp
from helper_functions1 import *
from helper_functions2 import *
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