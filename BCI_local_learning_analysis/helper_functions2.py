import sys
from sklearn.decomposition import PCA
from helper_functions1 import *  # or specific functions you use
from helper_functions3 import *  # or specific functions you use
import numpy as np
import statsmodels.api as sm


from collections import defaultdict
from datetime import date, timedelta

N_MIN_TRIALS = 0
N_MAX_TRIALS = 40             # used to sub-select tuning over the first N trials
D_NEAR = 30 #20                   # Distance from photostim (microns) to include as minumum distance for indirect
D_FAR = 100                    # Distance from photostim (microns) to include as maximum distance for indirect
D_DIRECT = 20                 # Distance from photostim (microns) if less than considered targeted by the photostim
# D_DIRECT = 30                 # Distance from photostim (microns) if less than considered targeted by the photostim

# Whether to do rows or columns first for MatLab unflattening, 0 makes 5c significant
# MatLab and numpy have different resahpe conventions, mode 0 seems to make them equivalent
UNFLATTED_MODE = 0   

### Parameters that determine what is pre- and post- trial-start response
SAMPLE_RATE = 20 # Hz
T_START = -2 # Seconds, time relative to trial start where trial_start_fs begin
TS_POST = (0, 10) # Seconds, time points to include for post-trial start average
TS_PRE = (-2, -1) # Seconds, time points to include for pre-trial start average

### PS Parameters ### (justified by looking at aggregate PS responses)
IDXS_PRE_PS = np.arange(0, 5)   # Indexes in FStim to consider pre-photostim response (~250 ms)
IDXS_PS = (5, 6, 7,)            # Indexes in FStim to consider photostim response (~150 ms of PS)
IDXS_POST_PS = np.arange(8, 16) # Indexes in FStim to consider post-photostim response (~400 ms)
# IDXS_PRE_PS = np.arange(0, 4)   # Indexes in FStim to consider pre-photostim response (~200 ms)
# IDXS_PS = (4, 5, 6, 7, 8,)      # Indexes in FStim to consider photostim response (~250 ms of PS)
# IDXS_POST_PS = np.arange(9, 16) # Indexes in FStim to consider post-photostim response (~350 ms)

### PS Fitting Parameters ###
N_MIN_EVENTS = 10
N_MIN_DIRECT = 1 # CHANGE - 5 

### Plot color conventions ###
PAIR_COLORS = (0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5) # Colored by mice (inclues (20, 21) pair)
SESSION_COLORS = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5) # Colored by mice (inclues (20, 21) pair)

MIN_P_VALUE = 1e-300

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