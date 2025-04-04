import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'BCI_local_learning_analysis'))
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.stats as stats
import io
from helper_functions1 import *
from helper_functions2 import *

try:
    import mat73
except:
    !pip install mat73
    import mat73

mypath = 'H:/My Drive/Learning rules/BCI_data/combined_new_old_060524.mat'
data_dict = mat73.loadmat(mypath)


# %% Cell 5: Various helper functions and nice colors. - add_regression_line: fit OLS or WLS and optionally plot ...
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

# %% Cell 7: ### Set global parameters Extracts maps that allow one to connect combined data to raw data files
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
    
def default_ps_stats_params(ps_stats_params={}):
    """ 
    Sets unused keys to their defaults 
    
    KEYS:
    - trial_average_mode: Whether to average over trials or time first when computing 
        trial average responses (to compute things like tuning). These do not necessarily 
        commute because trials can last different lengths of time. Averaging over time
        first equally weights all trials, while averaging over trials upweights longer trials.
    - resp_ps_average_mode: Whether to average over events or time first when computing
        photostimulation responses. These do not necessarily commute because of the 
        presence of nans in the data, but overall minor effect.
    - resp_ps_n_trials_back_mask: Determines the number of events after a neuron is 
        directly photostimulated where it should be omitted from other photostim 
        response calculations.
    - mask_mode: How the direct and indirect photostimulation masks are created across
        pairs of sessions. Determines whether each session uses its own mask or if both
        sessions use the same mask.
    - normalize_masks: 
    - neuron_metrics_adjust:
    - pairwise_corr_type: 
    
    
    """
    
    ps_stats_params_default = {
        'trial_average_mode': 'time_first', # trials_first, time_first
        'resp_ps_average_mode': 'time_first', # trials_first, time_first
        'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
        'mask_mode': 'constant', # constant, each_day, kayvon_match
        'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
        'neuron_metrics_adjust': None, # None, normalize, standardize
        'pairwise_corr_type': 'trace', # trace, trial, pre, post, behav_full, behav_start, behav_end

        ### Various ways of computing the over_neuron modes (this can be a tuple to do multiple)
        # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, 
        # matrix_multiply_standardized
        'x_over_neuron_mode': 'matrix_multiply_sanity',
        'y_over_neuron_mode': 'matrix_multiply_sanity',

        ### Plotting/fitting parameters
        'fit_individual_sessions': False, # Fit each individual session too
        'connectivity_metrics': [],
        'plot_pairs': [],
        'group_weights_type': None, # None, direct_resp, direct_resp_norm_sessions
        'indirect_weight_type': None, # None, rsquared, minimum_rsquared
        'use_only_predictor_weights': False, # Validation case where only weights are used
 
        ### Fitting photostim parameters
        'direct_predictor_mode': None, # None sum, top_mags, top_devs, top_devs_center
        'n_direct_predictors': 0,
        'direct_predictor_intercept_fit': False,
        'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
        'direct_input_mode': 'average', # average, ones, minimum
        'modify_direct_weights': False, 
        'validation_types': (),
    }
    
    for key in ps_stats_params_default.keys():
        if key not in ps_stats_params.keys():
            ps_stats_params[key] = ps_stats_params_default[key]
            
    return ps_stats_params

def get_behav_and_data_maps(mypath, verbose=False):
    """
    Establishes maps to corresponding behav and data files.
    
    INPUTS:
    - mypath: Path to directory where data is stored
    
    OUTPTUS:
    - behav: list of behav file names
    - data: list of data file names
    - maps: Dict of various maps connecting data files by date/mouse
        - session_idx_to_behav_idx
        - session_idx_to_data_idx
        - behav_idx_to_data_idx
        - behav_idx_to_session_idx
        - behav_idx_to_next_behav_idx 
    
    """

    # Some files in directory are behavioral and some not, so separate into two lists
    # of all the file names
    behav = []
    data = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in onlyfiles:
        if 'behav' in f:
            behav.append(f)
        else:
            data.append(f)

    data = sorted(data)
    behav = sorted(behav)

    # Now create some maps between files that allows one to find different
    # types of files that correspond to the same session/paired sessions

    behav_idx_to_data_idx = np.zeros([len(behav)],dtype=int)
    behav_idx_to_session_idx = defaultdict(lambda: 'none')
    behav_idx_to_next_behav_idx = defaultdict(lambda: 'none')

    session_idx_to_behav_idx = {}

    for behav_idx in range(len(behav)):
        f = behav[behav_idx].split("_")
        mouse = f[2]
        day = f[3].split('.')[0] # remove .mat
        for data_idx in range(len(data)): # Checks to find a data that matches behavior
            compare = data[data_idx].split('_')
            mousec = compare[1]
            dayc = compare[2].split('.')[0] # remove .mat
            if dayc == day and mousec == mouse:
                behav_idx_to_data_idx[behav_idx] = int(data_idx)

                found_match = False

                for session_idx_idx in range(len(data_dict['data']['session'])):
                    daystim = data_dict['data']['session'][session_idx_idx]
                    mousestim = data_dict['data']['mouse'][session_idx_idx]
                    if day == daystim and mouse == mousestim:
                        behav_idx_to_session_idx[behav_idx] = int(session_idx_idx)
                        session_idx_to_behav_idx[int(session_idx_idx)] = behav_idx
                        found_match = True
                
                if verbose:
                    if found_match:
                        print('Mouse {}, day {} - match found at session_idx {}'.format(
                            mouse, day, behav_idx_to_session_idx[behav_idx]
                        ))
                    else:
                        print('Mouse {}, day {} - no match '.format(mouse, day))

        for behav_idx2 in range(len(behav)):
            compare = behav[behav_idx2].split('_')
            mousec = compare[2]
            dayc = compare[3].split('.')[0] # remove .mat
            if mousec == mouse:
                date1 = date(int('20'+day[4::]), int(day[0:2]), int(day[2:4]))
                date2 = date1 + timedelta(days=1)
                checkdate2 = date(int('20'+dayc[4::]), int(dayc[0:2]), int(dayc[2:4]))
                if date2 == checkdate2:
                    behav_idx_to_next_behav_idx[behav_idx] = int(behav_idx2)

    session_idx_to_data_idx = {}                

    for data_idx in range(len(data)):
        compare = data[data_idx].split('_')
        mousec = compare[1]
        dayc = compare[2].split('.')[0] # remove .mat
        for session_idx_idx in range(len(data_dict['data']['session'])):
            daystim = data_dict['data']['session'][session_idx_idx]
            mousestim = data_dict['data']['mouse'][session_idx_idx]
            if dayc == daystim and mousec == mousestim:
                session_idx_to_data_idx[int(session_idx_idx)] = data_idx
    
    if verbose:
        print('There are', len(behav), 'behavior sessions with', len(data), 'data files.')
        print('There are',len(behav_idx_to_next_behav_idx),'pairs of days.')
        print(len(behav_idx_to_session_idx),'of them have corresponding photostim.')
        temp = [np.logical_and(
            behav_idx_to_session_idx[behav_idx_to_next_behav_idx[behav_idx]] != 'none', 
            behav_idx_to_next_behav_idx[behav_idx] != 'none'
        ) for behav_idx in range(len(behav))]
        print(np.sum(temp), 'days which have a photostim the next day')

    return behav, data, {
        'session_idx_to_behav_idx': session_idx_to_behav_idx,
        'session_idx_to_data_idx': session_idx_to_data_idx,
        'behav_idx_to_data_idx': behav_idx_to_data_idx,
        'behav_idx_to_session_idx': behav_idx_to_session_idx,
        'behav_idx_to_next_behav_idx': behav_idx_to_next_behav_idx, 
    }
        
# mypath = '/data/bci_oct24_upload/'
mypath = '/data/bci_data/'

BEHAV_FILES, DATA_FILES, DATA_MAPS = get_behav_and_data_maps(mypath, verbose=False)
SESSION_IDX_TO_BEHAV_IDX = DATA_MAPS['session_idx_to_behav_idx']
SESSION_IDX_TO_DATA_IDX = DATA_MAPS['session_idx_to_data_idx']
print('Maps to behav and data files loaded!')

# %% Cell 9: ### Helper functions Helper functions for evaluating pairs of sessionsHelper functions for evaluatin...
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

# %% Cell 11: ### Fit PS functions Functions for fitting the variation in photostimulation response.  - get_resp_p...
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

# %% Cell 12: Code starts with: indirect_params.shape
indirect_params.shape

# %% Cell 15: ### Exemplar plots This generates some exemplar plots showing various ways of doing interpolation on...
from sklearn.decomposition import PCA
import copy

session_idx = 11 # 11
shuffle_events = True # Shuffle indirect events relvative to direct

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
# print('FAKE!')
# ps_fs = get_fake_ps_data(session_idx)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
)

resp_ps_events = resp_ps_extras['resp_ps_events']

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4,))
fig8, ((ax8, ax8p,), (ax8pp, ax8ppp,)) = plt.subplots(2, 2, figsize=(12, 8,))
fig7, (ax7, ax7p, ax7pp) = plt.subplots(1, 3, figsize=(7, 4,), gridspec_kw={'width_ratios': [10., 1., 3.]})
fig9, (ax9, ax9p) = plt.subplots(1, 2, figsize=(12, 4,))
fig6, ax6 = plt.subplots(1, 1, figsize=(6, 4))

# For each photostim event, sees how indirect responses are related to the direct response
exemplar_group_idx = 0 # 0, 5
exemplar_neuron_idx = 35

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]
    
    dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
    indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
    
#     if shuffle_events: # Shuffle indirect events, so they are not synced with direct events
#         indir_resp_ps_events = shuffle_along_axis(indir_resp_ps_events, axis=-1) # Shuffle along event axis
        
#     keep_event_idxs = np.where(~np.any(np.isnan(dir_resp_ps_events), axis=0))[0]
#     dir_resp_ps_events = dir_resp_ps_events[:, keep_event_idxs]
#     indir_resp_ps_events = indir_resp_ps_events[:, keep_event_idxs]
    
    n_direct = dir_resp_ps_events.shape[0]
    n_indirect = indir_resp_ps_events.shape[0]
    
    ### Plot and fit all indirect responses ###
    # (n_neurons, n_events,) -> (n_dir, n_events,) -> (n_events,)
    sum_dir_resp_ps_events = np.nansum(dir_resp_ps_events, axis=0, keepdims=True)
    sum_dir_resp_ps_events = np.repeat(sum_dir_resp_ps_events, n_indirect, axis=0)
    
    plot_ax = ax5 if exemplar_group_idx == group_idx else None
    
    slope, _, rvalue, pvalue, _ = add_regression_line(
        sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(), 
        fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=plot_ax, color='k', zorder=5
    )
    
    group_event_slope[group_idx] = slope
    group_event_rsquared[group_idx] = rvalue**2
    
    if exemplar_group_idx == group_idx:
        ax5.scatter(sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(),
                marker='.', alpha=0.3, color=c_vals_l[0])
        
        # Plot a single direct neuron example
        ax5.scatter(sum_dir_resp_ps_events[exemplar_neuron_idx, :], indir_resp_ps_events[exemplar_neuron_idx, :],
                    marker='.', zorder=1, color=c_vals[1])
        _ = add_regression_line(
            sum_dir_resp_ps_events[exemplar_neuron_idx, :], indir_resp_ps_events[exemplar_neuron_idx, :], 
            fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[1], zorder=3, linestyle='dotted'
        )
        
        ax5.scatter( # Also plot mean responses
            np.nansum(resp_ps[direct_idxs, group_idx]) * np.ones(indirect_idxs.shape[0]),
            resp_ps[indirect_idxs, group_idx], marker='.', alpha=0.3, color=c_vals_d[0]
        )
        
        ax5.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax5.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax5.legend()
        ax5.set_xlabel('Sum direct response (group_idx {})'.format(group_idx))
        ax5.set_ylabel('Indirect responses (group_idx {})'.format(group_idx))
        
        # Now compute fit for every neuron using direct sum as explainer
        ps_stats_params_copy = copy.deepcopy(ps_stats_params)
        ps_stats_params_copy['direct_predictor_mode'] = 'sum'
        ps_stats_params_copy['n_direct_predictors'] = 1
        
        direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params_copy, return_extras=True,
        )
        
        if shuffle_events:
            plot_iterator = zip(
                (indir_resp_ps_events, shuffle_along_axis(indir_resp_ps_events, axis=-1),), # Shuffle along event axis
                (c_vals[0], 'grey',), # All neuron color
                (c_vals[1], c_vals[3],), # CN color
                (0, -5,), # zorder shift
            )
        else:
            plot_iterator = zip(
                (indir_resp_ps_events,), 
                (c_vals[0],), # All neuron color
                (c_vals[1],), # CN color
                (0,), # zorder shift
            )
            
        slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
        
        for indir_resp_ps_events_it, neuron_color, cn_color, zorder_shift in plot_iterator:
        
            indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
                dir_resp_ps_events, indir_resp_ps_events_it, direct_predictors, direct_shift,
                ps_stats_params_copy, verbose=True, return_extras=True,
            )

    #         ax8.scatter(indirect_params[:, slope_idx], -np.log10(indirect_pvalues[:, slope_idx]), marker='.', color=c_vals[2])
            ax8.scatter(indirect_params[:, slope_idx], fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
    #         ax8.scatter(indirect_params[exemplar_neuron_idx, slope_idx], -np.log10(indirect_pvalues[exemplar_neuron_idx, slope_idx]), marker='o', color=c_vals[3])
            ax8.scatter(indirect_params[exemplar_neuron_idx, slope_idx], fit_extras['r_squareds'][exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
            ax8.axhline(np.mean(fit_extras['r_squareds']), color=neuron_color, linestyle='dashed', zorder=-5+zorder_shift)
            
            direct_input = np.nanmean(sum_dir_resp_ps_events)
            indirect_prediction = photostim_predict(indirect_params, direct_input, ps_stats_params_copy)

            ax8p.scatter(resp_ps[indirect_idxs, group_idx], indirect_prediction, marker='.', color=neuron_color, zorder=zorder_shift)
            ax8p.scatter(resp_ps[indirect_idxs, group_idx][exemplar_neuron_idx], indirect_prediction[exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
        
            ax8pp.scatter(np.nanstd(indir_resp_ps_events_it, axis=-1), fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
            ax8pp.scatter(np.nanstd(indir_resp_ps_events_it, axis=-1)[exemplar_neuron_idx], fit_extras['r_squareds'][exemplar_neuron_idx], marker='o', color=cn_color, zorder=zorder_shift)
            
            ax8ppp.scatter(np.nanmean(indir_resp_ps_events_it, axis=-1) /np.nanstd(indir_resp_ps_events_it, axis=-1), 
                           fit_extras['r_squareds'], marker='.', color=neuron_color, zorder=zorder_shift)
            
        ax8.set_xlabel('Direct-indirect slope')
#         ax8.set_ylabel('-log10(p)')
        ax8.set_ylabel('r^2')
        ax8.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8p.set_xlabel('Mean indirect response')
        ax8p.set_ylabel('Interpolated indirect response')
        ax8p.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8p.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8pp.set_xlabel('Std indirect resp. (over events)')
        ax8pp.set_ylabel('r^2')
        ax8pp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8pp.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
        ax8ppp.set_xlabel('Mean/Std indirect resp. (over events)')
        ax8ppp.set_ylabel('r^2')
        ax8ppp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        ax8ppp.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
        
#         ax8pp.scatter(indirect_params[:, slope_idx], marker='.', color=c_vals_l[0])
        
        direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params, return_extras=True,
        )
        indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
            dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
            ps_stats_params, verbose=True, return_extras=True,
        )
        
        # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
        direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
        direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)
        indirect_prediction = photostim_predict(indirect_params, direct_input, ps_stats_params)
        
        # Show diversity in direct stimulations
        max_val = np.nanmax(np.abs(np.array(resp_ps_events[group_idx])[direct_idxs, :]))
        ax7.matshow(np.array(resp_ps_events[group_idx])[direct_idxs, :], 
                    vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        ax7p.matshow(np.nanmean(np.array(resp_ps_events[group_idx])[direct_idxs, :], axis=-1, keepdims=True), 
                     vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        ax7pp.matshow(direct_predictors.T,
                      vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
        
        ax7p.set_xticks([])
        ax7pp.set_xticks([])
        ax7.set_xlabel('Event idx (group_idx {})'.format(group_idx))
        ax7p.set_xlabel('Mean')
        ax7.set_ylabel('Dir. neuron idx (group_idx {})'.format(group_idx))
        ax7pp.set_xlabel('Pred. dirs.')
    
        slope_idx = 1 if ps_stats_params['direct_predictor_intercept_fit'] else 0
        
        for direct_max_idx in range(ps_stats_params['n_direct_predictors']):
            if ps_stats_params['direct_predictor_mode'] == 'top_mags': # neurons with largest L2 mag across events
                label = 'Max Neuron {}'.format(direct_max_idx)
            elif ps_stats_params['direct_predictor_mode'] in ('top_devs', 'top_devs_center'): 
                label = 'Dev dir {}'.format(direct_max_idx)
            elif ps_stats_params['direct_predictor_mode'] in ('sum',):
                label = None
#             ax9.scatter(neuron_slopes[:, direct_max_idx], -np.log10(neuron_pvalues)[:, direct_max_idx], 
#                         marker='.', color=c_vals[direct_max_idx], label=label)
            ax9.scatter(indirect_params[:, slope_idx+direct_max_idx], -np.log10(indirect_pvalues)[:, slope_idx+direct_max_idx], 
                        marker='.', color=c_vals[direct_max_idx], label=label)
            ax9p.scatter(indirect_params[:, slope_idx+direct_max_idx], fit_extras['r_squareds'], 
                         marker='.', color=c_vals[direct_max_idx], label=label)
                                 
        for ax in (ax9, ax9p,):
            ax.set_xlabel('Direct-indirect parameter')
            ax.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.legend()
                                 
        ax9.set_ylabel('-log10(p)')
        ax9p.set_ylabel('r^2')
        

ax6.scatter(group_event_rsquared, group_event_slope, color=c_vals[1], marker='.')

ax6.set_xlabel('$r**2$ of group\'s event resp_ps')
ax6.set_ylabel('slope of group\'s event resp_ps')

ax6.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
ax6.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
ax6.axvline(1.0, color='lightgrey', zorder=-5, linestyle='dashed')


# %% Cell 16: Code starts with: print(dir_resp_ps_events.shape)
print(dir_resp_ps_events.shape)
print(indir_resp_ps_events.shape)
print(np.sum(np.isnan(indirect_params)==1))

# %% Cell 17: Code starts with: # Inspect data_1['resp_ps']
# Inspect data_1['resp_ps']
resp_ps = data_1['resp_ps']

print("Type:", type(resp_ps))
print("Shape:", resp_ps.shape)

# Show a small preview of the data
print("\nFirst 3 entries:")
print(resp_ps[:3])

# Check for number of dimensions in one entry
print("\nType and shape of a single entry (if accessible):")
try:
    example = resp_ps[0, 0]  # Try accessing one cell
    print("Type:", type(example))
    if hasattr(example, 'shape'):
        print("Shape:", example.shape)
    else:
        print("Value:", example)
except Exception as e:
    print("Access error:", e)

# Check for presence of NaNs
print("\nAny NaNs:", np.isnan(resp_ps).any())


# %% Cell 18: Code starts with: import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Pick the stim group
group_idx = 0  # Change this if needed

# Get distances and select indirect neurons
d_ps = data_1['d_ps']
indirect_idxs = np.where((d_ps[:, group_idx] > D_NEAR) & (d_ps[:, group_idx] < D_FAR))[0]

# Mean response of each indirect neuron to this stim group
indir_mean_response = data_1['resp_ps'][indirect_idxs, group_idx]  # shape (n_indirect,)

# Filter valid entries (exclude NaNs if any)
valid_mask = ~np.isnan(indir_mean_response)
slopes = indirect_params[valid_mask, 0]
mean_response = indir_mean_response[valid_mask]

# Plot slope vs. mean response
plt.figure(figsize=(6, 5))
plt.scatter(mean_response, slopes, alpha=0.7)
plt.xlabel('Mean photostim response (indirect neuron)', fontsize=12)
plt.ylabel('Fitted slope from direct predictor', fontsize=12)
plt.title(f'Group {group_idx}: Indirect neuron response vs. slope', fontsize=14)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()


# %% Cell 19: Code starts with: from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import copy

session_idx = 12 # 11
shuffle_events = True # Shuffle indirect events relvative to direct

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
# print('FAKE!')
# ps_fs = get_fake_ps_data(session_idx)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
)

resp_ps_events = resp_ps_extras['resp_ps_events']


# For each photostim event, sees how indirect responses are related to the direct response
exemplar_group_idx = 8 # 0, 5
exemplar_neuron_idx = 35

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > -10, d_ps[:, group_idx] < 2000))[0]
    
    dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
    indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
    
#     if shuffle_events: # Shuffle indirect events, so they are not synced with direct events
#         indir_resp_ps_events = shuffle_along_axis(indir_resp_ps_events, axis=-1) # Shuffle along event axis
        
#     keep_event_idxs = np.where(~np.any(np.isnan(dir_resp_ps_events), axis=0))[0]
#     dir_resp_ps_events = dir_resp_ps_events[:, keep_event_idxs]
#     indir_resp_ps_events = indir_resp_ps_events[:, keep_event_idxs]
    
    n_direct = dir_resp_ps_events.shape[0]
    n_indirect = indir_resp_ps_events.shape[0]
    
    ### Plot and fit all indirect responses ###
    # (n_neurons, n_events,) -> (n_dir, n_events,) -> (n_events,)
    sum_dir_resp_ps_events = np.nansum(dir_resp_ps_events, axis=0, keepdims=True)
    sum_dir_resp_ps_events = np.repeat(sum_dir_resp_ps_events, n_indirect, axis=0)
    
    plot_ax = ax5 if exemplar_group_idx == group_idx else None
    
    slope, _, rvalue, pvalue, _ = add_regression_line(
        sum_dir_resp_ps_events.flatten(), indir_resp_ps_events.flatten(), 
        fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=plot_ax, color='k', zorder=5
    )
    
    group_event_slope[group_idx] = slope
    group_event_rsquared[group_idx] = rvalue**2
    
    
    
    
    
    
    if exemplar_group_idx == group_idx:
    # Find predictors for this group
        ps_stats_params_copy = copy.deepcopy(ps_stats_params)
        ps_stats_params_copy['direct_predictor_mode'] = 'sum'
        ps_stats_params_copy['n_direct_predictors'] = 1

        direct_predictors, direct_shift, _ = find_photostim_variation_predictors(
            dir_resp_ps_events, ps_stats_params_copy, return_extras=True
        )

        # Run the fit
        indirect_params, _, _ = fit_photostim_variation(
            dir_resp_ps_events,
            indir_resp_ps_events,
            direct_predictors,
            direct_shift,
            ps_stats_params_copy
        )

        # Get per-neuron slope and mean
        slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
        slopes = indirect_params[:, slope_idx]
        mean_indir_response = np.nanmean(indir_resp_ps_events, axis=1)

        # Mask invalid values
        valid_mask = ~np.isnan(slopes) & ~np.isnan(mean_indir_response)
        slopes = slopes[valid_mask]
        mean_indir_response = mean_indir_response[valid_mask]

        # Plot
        plt.figure(figsize=(6, 5))
        plt.scatter(mean_indir_response, slopes, alpha=0.7)
        plt.xlabel('Mean response of indirect neuron')
        plt.ylabel('Fitted slope from direct predictor')
        plt.title(f'Group {group_idx}: Slope vs. Mean response')
        plt.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()

  
        

# %% Cell 20: Code starts with: print(mean_indir_response.shape)
print(mean_indir_response.shape)
print(d_ps.shape)
print(indirect_params.shape)
len(indirect_idxs)
plt.subplot(121)
plt.scatter(d_ps[indirect_idxs,exemplar_group_idx],slopes)
plt.subplot(122)
plt.scatter(d_ps[indirect_idxs,exemplar_group_idx],mean_indir_response)

# %% Cell 22: ### Many session fitting  Now use the fitting metrics across several sessions and session pairs to c...
ps_stats_params = {
    'trial_average_mode': 'time_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

# %% Cell 24: For pairs of sessions, evaluate dissimilarity of photostimulation to justify needing to control for ...
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}

ps_stats_params = default_ps_stats_params(ps_stats_params)

exemplar_pair_idx = 6 # 6: (11, 12)
exemplar_group_idx = 5 # 0, 5
exemplar_neuron_idx = 10

n_pairs = len(session_idx_pairs)

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4,))
fig7, (ax7, ax7p, ax8, ax8p) = plt.subplots(1, 4, figsize=(12, 4,), gridspec_kw={'width_ratios': [10., 1., 10., 1.]})
fig1, (ax1, ax1p, ax1pp) = plt.subplots(1, 3, figsize=(10, 4))

differences = []
differences_vector = []
prs_direct_1 = []
prs_direct_2 = []
prs_direct_1_2 = []
n_direct_for_pr = []

n_direct = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
n_events_1 = [[] for _ in range(n_pairs)] 
n_events_2 = [[] for _ in range(n_pairs)]

# Gets data on how many entries are nans across groups justify if we should just toss out entire events
percent_dir_nans_1 = [[] for _ in range(n_pairs)]
percent_dir_nans_2 = [[] for _ in range(n_pairs)] 

# Gets data on how often a given direct neuron is reported as a nan across all events to see if we should toss out entire neurons
percent_event_nans_1 = [[] for _ in range(n_pairs)]
percent_event_nans_2 = [[] for _ in range(n_pairs)] 

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]
    
    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    print('Pair {} - Sessions {} and {} - Mouse {}.'.format(
        pair_idx, day_1_idx, day_2_idx, data_dict['data']['mouse'][day_2_idx]
    )) 

    ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx] # This is matlab indexed so always need a -1 here
    ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_1 = data_dict['data']['x'][day_1_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, ps_fs_1.shape[1],)
    resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
        ps_fs_1, ps_events_group_idxs_1, d_ps_1, ps_stats_params,
    )
    resp_ps_events_1 = resp_ps_extras_1['resp_ps_events']
    
    if day_1_idx == 7:
        print(np.array(resp_ps_events_1[0]).shape)

    ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx] # This is matlab indexed so always need a -1 here
    ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_2 = data_dict['data']['x'][day_2_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_2 = unflatted_neurons_by_groups(d_ps_flat_2, ps_fs_2.shape[1],)
    resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
        ps_fs_2, ps_events_group_idxs_2, d_ps_2, ps_stats_params,
    )
    resp_ps_events_2 = resp_ps_extras_2['resp_ps_events']

    n_groups = int(np.max(ps_events_group_idxs_1)) # +1 already accounted for because MatLab indexing
    
    sum_dir_resp_ps_1 = np.zeros((n_groups,))
    sum_dir_resp_ps_2 = np.zeros((n_groups,))
    
    for group_idx in range(n_groups):
        direct_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] < D_DIRECT, d_ps_2[:, group_idx] < D_DIRECT))[0]
        indirect_idxs = np.where(np.logical_and(
            np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
            np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
        ))[0]
        
        # Mean response-based metrics
        dir_resp_ps_1 = resp_ps_1[direct_idxs, group_idx] # (n_direct,)
        indir_resp_ps_1 = resp_ps_1[indirect_idxs, group_idx] # (n_indirect,)

        dir_resp_ps_2 = resp_ps_2[direct_idxs, group_idx] # (n_direct,)
        indir_rresp_ps_2 = resp_ps_2[indirect_idxs, group_idx] # (n_indirect,)
        
        sum_dir_resp_ps_1[group_idx] = np.nansum(dir_resp_ps_1)
        sum_dir_resp_ps_2[group_idx] = np.nansum(dir_resp_ps_2)
        
        differences_vector.append(
            np.linalg.norm(dir_resp_ps_1 - dir_resp_ps_2) /
            (1/2 * (np.linalg.norm(dir_resp_ps_1) + np.linalg.norm(dir_resp_ps_2)))
        )
        
        # Event-based metrics
        dir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)

        dir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        
        # Nan handling
        percent_dir_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=0) / dir_resp_ps_events_1.shape[0]
        )
        percent_event_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=1) / dir_resp_ps_events_1.shape[1]   
        )
        percent_dir_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=0) / dir_resp_ps_events_2.shape[0]
        )
        percent_event_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=1) / dir_resp_ps_events_1.shape[1]   
        )
        # Eliminate events with all nans
        keep_event_idxs_1 = np.where(~np.all(np.isnan(dir_resp_ps_events_1), axis=0))[0]
        dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
        indir_resp_ps_events_1 = indir_resp_ps_events_1[:, keep_event_idxs_1]
        keep_event_idxs_2 = np.where(~np.all(np.isnan(dir_resp_ps_events_2), axis=0))[0]
        dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        indir_resp_ps_events_2 = indir_resp_ps_events_2[:, keep_event_idxs_2]
        if ps_stats_params['direct_predictor_nan_mode'] in ('ignore_nans',):
            dir_resp_ps_events_1 = np.where(np.isnan(dir_resp_ps_events_1), 0., dir_resp_ps_events_1) # Fill nans with 0s
            dir_resp_ps_events_2 = np.where(np.isnan(dir_resp_ps_events_2), 0., dir_resp_ps_events_2) # Fill nans with 0s
        elif ps_stats_params['direct_predictor_nan_mode'] in ('eliminate_events',):
            keep_event_idxs_1 = np.where(~np.any(np.isnan(dir_resp_ps_events_1), axis=0))[0]
            dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
            keep_event_idxs_2 = np.where(~np.any(np.isnan(dir_resp_ps_events_2), axis=0))[0]
            dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        
        # No events left
        if dir_resp_ps_events_1.shape[1] == 0 or dir_resp_ps_events_2.shape[1] == 0:
            continue
        # No neurons
        if dir_resp_ps_events_1.shape[0] == 0 or dir_resp_ps_events_2.shape[0] == 0:
            continue
        
        pca_1 = PCA()
        pca_1.fit(dir_resp_ps_events_1.T)
        prs_direct_1.append(participation_ratio_vector(pca_1.explained_variance_))
        
        pca_2 = PCA()
        pca_2.fit(dir_resp_ps_events_2.T)
        prs_direct_2.append(participation_ratio_vector(pca_2.explained_variance_))
        
        pca_1_2 = PCA()
        pca_1_2.fit(np.concatenate((dir_resp_ps_events_1, dir_resp_ps_events_2,), axis=-1).T)
        prs_direct_1_2.append(participation_ratio_vector(pca_1_2.explained_variance_))
        
        n_direct_for_pr.append(dir_resp_ps_events_1.shape[0])
        
        n_direct[pair_idx].append(dir_resp_ps_events_1.shape[0])
        n_events_1[pair_idx].append(dir_resp_ps_events_1.shape[-1])
        n_events_2[pair_idx].append(dir_resp_ps_events_2.shape[-1])
        
        if pair_idx == exemplar_pair_idx and group_idx == exemplar_group_idx:
            
            n_indirect = indir_resp_ps_events_1.shape[0]
            sum_dir_resp_ps_events_1 = np.repeat(np.nansum(dir_resp_ps_events_1, axis=0, keepdims=True), n_indirect, axis=0)
            sum_dir_resp_ps_events_2 = np.repeat(np.nansum(dir_resp_ps_events_2, axis=0, keepdims=True), n_indirect, axis=0)
            
            ax5.scatter(sum_dir_resp_ps_events_1.flatten(), indir_resp_ps_events_1.flatten(),
                        marker='.', alpha=0.3, color=c_vals_l[0])
            ax5.scatter(sum_dir_resp_ps_events_2.flatten(), indir_resp_ps_events_2.flatten(),
                        marker='.', alpha=0.3, color=c_vals_l[1])
            
            ax5.scatter( # Also plot mean responses
                np.nanmean(sum_dir_resp_ps_events_1, axis=-1), np.nanmean(indir_resp_ps_events_1, axis=-1),
                marker='.', alpha=0.3, color=c_vals_d[0]
            )
            ax5.scatter(
                np.nanmean(sum_dir_resp_ps_events_2, axis=-1), np.nanmean(indir_resp_ps_events_2, axis=-1),
                marker='.', alpha=0.3, color=c_vals_d[1]
            )
        
            # Plot a single direct neuron example
            ax5.scatter(sum_dir_resp_ps_events_1[exemplar_neuron_idx, :], indir_resp_ps_events_1[exemplar_neuron_idx, :],
                        marker='.', zorder=1, color=c_vals[0])
            ax5.scatter(sum_dir_resp_ps_events_2[exemplar_neuron_idx, :], indir_resp_ps_events_2[exemplar_neuron_idx, :],
                        marker='.', zorder=1, color=c_vals[1])
            _ = add_regression_line(
                sum_dir_resp_ps_events_1[exemplar_neuron_idx, :], indir_resp_ps_events_1[exemplar_neuron_idx, :], 
                fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[0], zorder=3, linestyle='dotted'
            )
            _ = add_regression_line(
                sum_dir_resp_ps_events_2[exemplar_neuron_idx, :], indir_resp_ps_events_2[exemplar_neuron_idx, :], 
                fit_intercept=ps_stats_params['direct_predictor_intercept_fit'], ax=ax5, color=c_vals[1], zorder=3, linestyle='dotted'
            )

            ax5.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax5.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax5.legend()
            ax5.set_xlabel('Sum direct response (group_idx {})'.format(group_idx))
            ax5.set_ylabel('Indirect responses (group_idx {})'.format(group_idx))
            
            
            # Show diversity in direct stimulations
            max_val = np.max((np.nanmax(dir_resp_ps_events_1), np.nanmax(dir_resp_ps_events_2),))
            
            ax7.matshow(dir_resp_ps_events_1, vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            ax7p.matshow(np.nanmean(dir_resp_ps_events_1, axis=-1, keepdims=True), 
                         vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            print('Day 1 sum: {:.1e} vs {:.1e}'.format(np.sum(np.nanmean(dir_resp_ps_events_1, axis=-1, keepdims=True)), np.nansum(dir_resp_ps_1)))
            
            ax8.matshow(dir_resp_ps_events_2, vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            ax8p.matshow(np.nanmean(dir_resp_ps_events_2, axis=-1, keepdims=True), 
                         vmax=max_val, vmin=-max_val, cmap='bwr', aspect='auto')
            
            print('Day 2 sum: {:.1e} vs {:.1e}'.format(np.sum(np.nanmean(dir_resp_ps_events_2, axis=-1, keepdims=True)), np.nansum(dir_resp_ps_2)))
            
            print('Vec diff: {:.1e}'.format(
                np.linalg.norm(np.nanmean(dir_resp_ps_events_2, axis=-1) - np.nanmean(dir_resp_ps_events_1, axis=-1))/
                (1/2 * np.linalg.norm(np.nanmean(dir_resp_ps_events_1, axis=-1)) + 1/2 * np.linalg.norm(np.nanmean(dir_resp_ps_events_2, axis=-1)))
            ))
            
            ax7.set_xlabel('Event idx (group_idx {}, Day 1)'.format(group_idx))
            ax8.set_xlabel('Event idx (group_idx {}, Day 2)'.format(group_idx))
            ax7p.set_xticks([])
            ax7p.set_xlabel('Mean')
            ax8p.set_xticks([])
            ax8p.set_xlabel('Mean')
            ax7.set_ylabel('Dir. neuron idx (group_idx {})'.format(group_idx))
    
    ax1.scatter(sum_dir_resp_ps_1, sum_dir_resp_ps_2, marker='.', color=c_vals[0], alpha=0.3)
    
    differences.append(
        np.abs(sum_dir_resp_ps_2 - sum_dir_resp_ps_1) / (1/2 * (np.abs(sum_dir_resp_ps_2) + np.abs(sum_dir_resp_ps_1)))
    )

add_identity(ax1, color='lightgrey', zorder=-5, linestyle='dashed')
    
differences = np.concatenate(differences, axis=0)

ax1p.hist(differences, color=c_vals[0], bins=30)
ax1p.axvline(np.mean(differences), color=c_vals_d[0], zorder=5)
ax1pp.hist(np.array(differences_vector), color=c_vals[1], bins=30)
ax1pp.axvline(np.mean(differences_vector), color=c_vals_d[1], zorder=5)

for ax in (ax1p, ax1pp):
    ax.set_xlim((0, 2.,))

ax1.set_xlabel('Day 1 direct response mag.')
ax1.set_ylabel('Day 2 direct response mag.')
ax1p.set_xlabel('Perc. sum diff')
ax1pp.set_xlabel('Perc. vector diff')

fig3, (ax3, ax3p) = plt.subplots(1, 2, figsize=(8, 4))

mouse_offset = 40 # Spaces out plots to see data across mice better

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    ax3.scatter(n_direct[pair_idx], (mouse_offset * PAIR_COLORS[pair_idx]) + np.array(n_events_1[pair_idx]), marker='.', 
                color=c_vals[PAIR_COLORS[pair_idx]], alpha=0.3)
    ax3p.scatter(n_direct[pair_idx], (mouse_offset * PAIR_COLORS[pair_idx]) + np.array(n_events_2[pair_idx]), marker='.', 
                 color=c_vals[PAIR_COLORS[pair_idx]], alpha=0.3)
    
    n_events_reject_1 = np.where(np.array(n_events_1[pair_idx]) < N_MIN_EVENTS)[0].shape[0]
    n_events_reject_2 = np.where(np.array(n_events_2[pair_idx]) < N_MIN_EVENTS)[0].shape[0]
    
    print('Pair idx {} - n_events reject 1: {},\t2: {}'.format(
        pair_idx, n_events_reject_1, n_events_reject_2
    ))

for ax in (ax3, ax3p):
    ax.set_xlabel('n_direct')
   
    ax.set_yticks((0, 20, 40, 60, 80, 100, 120, 140, 160, 180))
    ax.set_yticklabels((None, 20, None, 20, None, 20, None, 20, None, 20))
    
    for sep in (0, 40, 80, 120, 160, 200):
        ax.axhline(sep, color='grey')
        ax.axhline(sep + 10, color='lightgrey', zorder=-5, linestyle='dashed')

ax3.set_ylabel('n_events_1')
ax3p.set_ylabel('n_events_2')

# %% Cell 26: Separate out this plot because code ocean is having problems outputting both plots above.
fig2, (ax2pp, ax2, ax2p) = plt.subplots(1, 3, figsize=(10, 4))

n_direct_for_pr = np.array(n_direct_for_pr)
prs_direct_1_2 = np.array(prs_direct_1_2)
prs_direct_1 = np.array(prs_direct_1)
prs_direct_2 = np.array(prs_direct_2)

ax2pp.scatter(prs_direct_1_2, n_direct_for_pr, marker='.', color=c_vals[4], alpha=0.3, label='Day 1+2')
ax2pp.scatter(prs_direct_1, n_direct_for_pr, marker='.', color=c_vals[2], alpha=0.3, label='Day 1')
ax2pp.scatter(prs_direct_2, n_direct_for_pr, marker='.', color=c_vals[3], alpha=0.3, label='Day 2')

add_identity(ax2pp, color='k', zorder=5)

ax2pp.set_xlabel('Direct participation ratio')
ax2pp.set_ylabel('# Direct')
ax2pp.legend()

ax2pp.set_xlim((0, 1.05 * np.max(n_direct_for_pr)))
ax2pp.set_ylim((0, 1.05 * np.max(n_direct_for_pr)))

_, bins, _ = ax2.hist(prs_direct_1_2, bins=30, color=c_vals[4], alpha=0.3, label='Day 1+2')
_ = ax2.hist(prs_direct_1, bins=bins, color=c_vals[2], alpha=0.3, label='Day 1')
_ = ax2.hist(prs_direct_2, bins=bins, color=c_vals[3], alpha=0.3, label='Day 2')
ax2.axvline(np.nanmean(prs_direct_1_2), color=c_vals_d[4], zorder=5)
ax2.axvline(np.nanmean(prs_direct_1), color=c_vals_d[2], zorder=5)
ax2.axvline(np.nanmean(prs_direct_2), color=c_vals_d[3], zorder=5)

ax2.set_xlabel('Direct participation ratio')
ax2.legend()

pr_ratio = prs_direct_1_2 / (1/2. * (np.array(prs_direct_1) + np.array(prs_direct_1)))

ax2p.hist(pr_ratio, bins=40, color=c_vals[4])
ax2p.axvline(np.nanmean(pr_ratio), color=c_vals_d[4], zorder=5)
ax2p.axvline(1.0, color='k', zorder=5)

ax2p.set_xlabel('Relative PR size (combined/individual)')

# %% Cell 28: Some additional nan statistics.
n_bins = 20
perc_bins = np.linspace(0, 1, n_bins)

fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))

for pair_idx, ax5 in zip(range(n_pairs), ax5s.flatten()):
    
    percent_nans_1 = np.concatenate(percent_dir_nans_1[pair_idx], axis=0)
    percent_nans_2 = np.concatenate(percent_dir_nans_2[pair_idx], axis=0)
    percent_nans = np.concatenate((percent_nans_1, percent_nans_1), axis=0)

    ax5.hist(percent_nans, bins=perc_bins, color=c_vals[PAIR_COLORS[pair_idx]],)
    
fig6, ax6s = plt.subplots(2, 6, figsize=(24, 8))

for pair_idx, ax6 in zip(range(n_pairs), ax6s.flatten()):
    
    percent_nans_1 = np.concatenate(percent_event_nans_1[pair_idx], axis=0)
    percent_nans_2 = np.concatenate(percent_event_nans_2[pair_idx], axis=0)
    percent_nans = np.concatenate((percent_nans_1, percent_nans_1), axis=0)

    ax6.hist(percent_nans, bins=perc_bins, color=c_vals[PAIR_COLORS[pair_idx]],)

fig5.suptitle('Percent of direct neurons in an events that are nans (n_trials_back={})'.format(
     ps_stats_params['resp_ps_n_trials_back_mask']
), fontsize=20)

fig6.suptitle('Percent a events that a direct neuron is nans (n_trials_back={})'.format(
     ps_stats_params['resp_ps_n_trials_back_mask']
), fontsize=20)

# %% Cell 30: #### Single session Use the fit parameters in single-sessions measures.
from sklearn.decomposition import PCA
import copy

session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

exemplar_session_idx = 6 # 11

weight_type = 'rsquared' # None, rsquared

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1,
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, ones, minimum 
    
    # shuffle_indirect_events
    'validation_types': ('shuffle_indirect_events',),
}

if ps_stats_params['validation_types'] != ():
    print('Using validation:', ps_stats_params['validation_types'])

connectivity_metrics = (
    'pairwise_corr_x',
    'tuning_x',
    'trial_resp_x',
    'post_resp_x',
    'pre_resp_x',
#     'cc_x',
#     'cc_mag_x',
#     'mask_counts_x',
    
    'raw_cc_y',
    'tuning_y',
    'trial_resp_y',
    'post_resp_y',
    'pre_resp_y',
#     'cc_y',
#     'cc_mag_y',
#     'mask_counts_y',
)

plot_pair = ('pairwise_corr_x', 'raw_cc_y',)
# plot_pair = ('cc_x', 'raw_cc_y',)

ps_stats_params = default_ps_stats_params(ps_stats_params)

fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(8, 3))
fig2, (ax2, ax2p) = plt.subplots(1, 2, figsize=(8, 3))
fig6, ((ax6, ax6p,), (ax6pp, ax6ppp),) = plt.subplots(2, 2, figsize=(9, 6))
fig3, (ax3, ax3p, ax3pp) = plt.subplots(1, 3, figsize=(10, 4)) # Improvement of fits, and resulting slopes
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4))
fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4)) # Individual fits and aggregate fits

n_sessions = len(session_idxs)

old_method_ps = np.zeros((n_sessions,))
new_method_ps = np.zeros((n_sessions,))
old_method_slopes = np.zeros((n_sessions,))
new_method_slopes = np.zeros((n_sessions,))
old_method_r_squareds = np.zeros((n_sessions,))
new_method_r_squareds = np.zeros((n_sessions,))

r_squareds = [[] for _ in range(n_sessions)] # Separate this out into sessions to plot by mouse

group_corrs_all = []
indirect_predictions_all = []
indirect_weights_all = []

for session_idx_idx, session_idx in enumerate(session_idxs):

    print('Session idx {}'.format(session_idx))
    day_1_idx = session_idx
    
    data_to_extract = ('d_ps', 'trial_start_fs', 'resp_ps', 'resp_ps_pred',)
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)

    # ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
    # ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
#     print('FAKE!')
#     ps_fs = get_fake_ps_data(session_idx)
    
    # n_ps_times = data_1['resp_ps'].shape[0]
    n_neurons = data_1['resp_ps'].shape[0]
    n_groups = data_1['resp_ps'].shape[1]

    # d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    # d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)
    d_ps = data_1['d_ps']

    # resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    #     ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False
    # )

    # resp_ps_events = resp_ps_extras['resp_ps_events']

    pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
    # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
    pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)

    direct_idxs_flat = np.where(d_ps.flatten() < D_DIRECT)[0]
    indirect_idxs_flat = np.where(np.logical_and(d_ps.flatten() > D_NEAR, d_ps.flatten() < D_FAR))[0]
    
    group_corrs = [] # Filled as we iterate across groups
    indirect_resp_ps = []
    indirect_predictions = []
    
    # Filled as we iterate across group for exemplar session only
    flat_direct_resp_ps = []
    flat_indirect_pred_params = []
    flat_indirect_resp_ps_pred = []
    flat_indirect_resp_ps = []
    flat_indirect_pred_r_squareds = []
    
    for group_idx in range(n_groups):
        direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
        indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]

        # dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        # indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        
        # direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
        #     dir_resp_ps_events, ps_stats_params,
        # )
        # indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
        #     dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
        #     ps_stats_params, verbose=False, return_extras=True
        # )
        
        # r_squareds[session_idx_idx].append(fit_extras['r_squareds'])
        r_squareds_group = data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx]
#         if group_idx == 0: print('Shuffling r^2 within groups!')
#         np.random.shuffle(r_squareds_group) # Validate r^2 weighting by shuffling within a group
        r_squareds[session_idx_idx].append(r_squareds_group)

        if plot_pair[1] in ('raw_cc_y',):
            indirect_resp_ps.append(data_1['resp_ps'][indirect_idxs, group_idx])
            indirect_predictions.append(data_1['resp_ps_pred'][indirect_idxs, group_idx])
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
#         sum_direct_resp_ps = np.nansum(resp_ps[direct_idxs, group_idx]) # Avg. over events, then neurons
#         sum_direct_resp_ps =  np.nanmean(np.nansum(dir_resp_ps_events, axis=0)) # Over neurons, then avg events
        
        ### Gets average direct input ###
        # Note this way of doing determines average input for each event THEN averages over
        # events. This does not necessarily yield the same result as averaging over events
        # first then determining the input because of how we treat nans.
        
        # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
        # direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
        # direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)
        
        if plot_pair[0] in ('pairwise_corr_x',):
            group_corr = np.matmul(pairwsie_corrs[:, direct_idxs], data_1['resp_ps_pred'][direct_idxs, group_idx])[indirect_idxs]
#             if group_idx == 0: print('Shuffling correlation within groups!')
#             np.random.shuffle(group_corr) # Turn this one for validation, does indirect identity actually mattter?
        elif plot_pair[0] in ('cc_x',):
            group_corr = np.nansum(data_1['resp_ps_pred'][direct_idxs, group_idx]) * np.ones((indirect_idxs.shape[0],))
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
        group_corrs.append(group_corr)
        
        if exemplar_session_idx == session_idx:
            slope_idx = 0
            
            n_indirect = data_1['resp_ps_pred_extras']['params'][group_idx].shape[0]
            flat_direct_resp_ps.append(np.nansum(data_1['resp_ps'][direct_idxs, group_idx]) * np.ones((n_indirect,)))
            flat_indirect_pred_params.append(data_1['resp_ps_pred_extras']['params'][group_idx][:, slope_idx])
            flat_indirect_resp_ps_pred.append(data_1['resp_ps_pred'][indirect_idxs, group_idx])
            flat_indirect_resp_ps.append(data_1['resp_ps'][indirect_idxs, group_idx])
            flat_indirect_pred_r_squareds.append(data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])

    group_corrs = np.concatenate(group_corrs, axis=0) # Filled as we iterate across groups
    indirect_resp_ps = np.concatenate(indirect_resp_ps, axis=0)
    indirect_predictions = np.concatenate(indirect_predictions, axis=0)
    
    ax_fit, axp_fit = None, None
    if exemplar_session_idx == session_idx:
        indirect_d_ps = d_ps.flatten()[indirect_idxs_flat]
        
        ax1.scatter(indirect_d_ps, indirect_resp_ps, marker='.', color='k', alpha=0.1)
        ax1p.scatter(indirect_d_ps, indirect_predictions, marker='.', color='k', alpha=0.1)
        
        ax_fit, axp_fit = ax2, ax2p
        ax2.scatter(group_corrs, indirect_resp_ps, color=c_vals_l[0], alpha=0.05, marker='.')
        ax2p.scatter(group_corrs, indirect_predictions, color=c_vals_l[0], alpha=0.05, marker='.')
        
        # Plots of fit trends for a given session
        for ax, y_plot in zip(
            (ax6, ax6p, ax6pp, ax6ppp),
            (flat_indirect_pred_params, flat_indirect_resp_ps_pred, flat_indirect_resp_ps, flat_indirect_pred_r_squareds,)
        ):
            for group_idx in range(len(y_plot)):
                ax.scatter(flat_direct_resp_ps[group_idx], y_plot[group_idx], alpha=0.3, color=c_vals_l[1], marker='.')
                ax.scatter(np.nanmean(flat_direct_resp_ps[group_idx]), np.nanmean(y_plot[group_idx]), color=c_vals[1], marker='.', zorder=3)
            _ = add_regression_line(
                np.concatenate(flat_direct_resp_ps, axis=0), np.concatenate(y_plot, axis=0), ax=ax, color=c_vals_d[1], zorder=5
            )
    
    r_squareds_session = np.concatenate(r_squareds[session_idx_idx], axis=0)
#     print('Shuffling r^2 across a single session!')
#     np.random.shuffle(r_squareds_session) # Validate r^2 weighting by shuffling across a session
    r_squareds[session_idx_idx] = r_squareds_session
    
    if weight_type in (None,):
        weights = None
    elif weight_type in ('rsquared',):
        weights = np.copy(r_squareds[session_idx_idx])
    else:
        raise ValueError('Weight type {} not recognized'.format(weight_type))
    
    slope_o, _, rvalue_o, pvalue_o, _ = add_regression_line(group_corrs, indirect_resp_ps, ax=ax_fit, color=c_vals[0], zorder=5)
    
#     indirect_predictions = np.where(np.isnan(indirect_predictions), 0., indirect_predictions)
#     weights = np.where(np.isnan(weights), 0., weights)
    slope_n, _, rvalue_n, pvalue_n, _ = add_regression_line(
        group_corrs, indirect_predictions, weights=weights,
        ax=axp_fit, color=c_vals[0], zorder=5
    )
    
    group_corrs_all.append(group_corrs)
    indirect_predictions_all.append(indirect_predictions)
    indirect_weights_all.append(weights)
    
    old_method_ps[session_idx_idx] = pvalue_o
    new_method_ps[session_idx_idx] = pvalue_n
    old_method_slopes[session_idx_idx] = slope_o
    new_method_slopes[session_idx_idx] = slope_n
    old_method_r_squareds[session_idx_idx] = rvalue_o**2
    new_method_r_squareds[session_idx_idx] = rvalue_n**2

for ax in (ax1, ax1p,):
    ax.set_xlabel('Distance from PS (um)')
ax1.set_ylabel('Indirect PS response (old)')
ax1p.set_ylabel('Indirect PS response prediction')

for ax in (ax2, ax2p,):
    ax.set_xlabel('Correlation with PS group')

ax2.set_ylabel('Indirect PS response (old)')
ax2p.set_ylabel('Indirect PS response prediction')
ax2.legend()
ax2p.legend()

for ax in (ax1, ax1p):
    ax.axhline(0.0, color=c_vals[0], zorder=5, linestyle='dashed')
    
for ax in (ax6, ax6p, ax6pp, ax6ppp):
    ax.set_xlabel('Tot. direct resp_ps')
    ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
    ax.legend()
    
ax6.set_ylabel('Indirect slope')
ax6p.set_ylabel('Indirect resp_ps_pred')
ax6pp.set_ylabel('Indirect resp_ps')
ax6ppp.set_ylabel('Indirect r^2')
    
    
MIN_LOG10_P = -300
old_method_ps = np.where(old_method_ps == 0., 10**MIN_LOG10_P, old_method_ps)
new_method_ps = np.where(new_method_ps == 0., 10**MIN_LOG10_P, new_method_ps)

old_new_pvalues = np.concatenate(
    (old_method_ps[:, np.newaxis], new_method_ps[:, np.newaxis]), axis=-1
)
old_new_r_squareds = np.concatenate(
    (old_method_r_squareds[:, np.newaxis], new_method_r_squareds[:, np.newaxis]), axis=-1
)

for session_idx_idx, session_idx in enumerate(session_idxs):
    
    print('Mouse {} - Session {}, -log10(p): {:.1f} to {:.1f}'.format(
        data_dict['data']['mouse'][session_idx], session_idx, 
        -np.log10(old_new_pvalues[session_idx_idx, 0]), -np.log10(old_new_pvalues[session_idx_idx, 1])
    ))
    
    ax3.plot(-np.log10(old_new_pvalues[session_idx_idx]), color=c_vals[SESSION_COLORS[session_idx_idx]], marker='.')
    ax3.plot(-np.log10(old_new_pvalues[session_idx_idx, 0]), color=c_vals_l[SESSION_COLORS[session_idx_idx]], marker='.')
    
    ax3pp.plot(old_new_r_squareds[session_idx_idx], color=c_vals[SESSION_COLORS[session_idx_idx]], marker='.')
    ax3pp.plot(old_new_r_squareds[session_idx_idx, 0], color=c_vals_l[SESSION_COLORS[session_idx_idx]], marker='.')
    
ax3p.scatter(new_method_slopes, -np.log10(new_method_ps), 
             c=[c_vals[session_color] for session_color in SESSION_COLORS[:n_sessions]], zorder=5)
ax3p.scatter(old_method_slopes, -np.log10(old_method_ps), 
             c=[c_vals_l[session_color] for session_color in SESSION_COLORS[:n_sessions]], marker='.')

for ax in (ax3, ax3pp,):
    ax.set_xticks((0, 1))
    ax.set_xticklabels(('No interpolation', 'Interpolation'))
for ax in (ax3, ax3p, ax3pp,):
    ax.axhline(0.0, color='lightgrey', zorder=-5)

ax3.set_ylabel('-log10(pvalue)')
ax3p.set_xlabel('slope')
ax3pp.set_ylabel('$r^2$')
ax3pp.set_ylim((0., 0.4))

ax3p.axvline(0.0, color='lightgrey', zorder=-5)

for session_idx_idx, session_idx in enumerate(session_idxs):
    
    ax4.hist(r_squareds[session_idx_idx] + SESSION_COLORS[session_idx_idx], bins=20, 
             color=c_vals[SESSION_COLORS[session_idx_idx]], alpha=0.2)
    
ax4.set_xticks((0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6))
ax4.set_xticklabels((0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 1.0))

for sep in (0, 1, 2, 3, 4, 5, 6):
    ax4.axvline(sep, color='lightgrey', zorder=-5, linestyle='dashed')
    
ax4.set_xlabel('r^2 of fits')
ax4.set_ylabel('n_indirect')

for session_idx_idx, session_idx in enumerate(session_idxs):
    _ = add_regression_line(
        group_corrs_all[session_idx_idx], indirect_predictions_all[session_idx_idx], 
        weights=indirect_weights_all[session_idx_idx], label=None,
        ax=ax5, color=c_vals[SESSION_COLORS[session_idx_idx]], zorder=0
    )
    
slope, _, rvalue, pvalue, _ = add_regression_line(
    np.concatenate(group_corrs_all, axis=0), np.concatenate(indirect_predictions_all, axis=0), 
    weights=np.concatenate(indirect_weights_all, axis=0),
    ax=ax5, color='k', zorder=5
)    
ax5.legend()
ax5.set_xlabel('Correlation with PS group')
ax5.set_ylabel('Indirect PS response prediction')

print('Aggregate fit slope:', slope)

# %% Cell 32: #### Paired sessions Same as above but for paired sessions. First find valid session pairs
from sklearn.decomposition import PCA
import copy

direct_response_mode = 'ps_resp' # ps_resp, pr_resp_thresh
direct_input_mode = 'average' # average, average_equal_sessions, minimum
weight_type = 'minimum_rsquared' # None, minimum_rsquared
correlation_mode = 'all' # all, pre+post, pre, post 

PS_RESPONSE_THRESH = 0.1 

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    # shuffle_indirect_events
    'validation_types': (),
}

ps_stats_params = default_ps_stats_params(ps_stats_params)

if ps_stats_params['validation_types'] != ():
    print('Using validation:', ps_stats_params['validation_types'])
    

connectivity_metrics = (
    'delta_pairwise_corr_x',
    'delta_tuning_x',
    'mean_tuning_x',
    'delta_trial_resp_x',
    'mean_trial_resp_x',
    'delta_post_resp_x',
    'mean_post_resp_x',
    'delta_pre_resp_x',
    'mean_pre_resp_x',
#     'laser_resp_x',
#     'laser_resp_mag_x',
#     'mask_counts_x',
    
    'raw_delta_cc_y',
    'delta_tuning_y',
    'mean_tuning_y',
    'delta_trial_resp_y',
    'mean_trial_resp_y',
    'delta_post_resp_y',
    'mean_post_resp_y',
    'delta_pre_resp_y',
    'mean_pre_resp_y',
#     'delta_cc_y',
#     'delta_cc_mag_y',
#     'mask_counts_y',
)

plot_pair = ('delta_pairwise_corr_x', 'raw_delta_cc_y',)
# plot_pair = ('laser_resp_x', 'raw_delta_cc_y',)

exemplar_pair_idx = 0

fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(8, 3)) # Consistency of predictions across days
fig4, (ax4, ax4p) = plt.subplots(1, 2, figsize=(8, 3)) # Consistency of fit parameters
fig2, (ax2, ax2p) = plt.subplots(1, 2, figsize=(8, 3)) # Delta correlation fit
fig3, (ax3, ax3p, ax3pp) = plt.subplots(1, 3, figsize=(10, 4)) # Improvement of fits, and resulting slopes
fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4)) # Individual fits and aggregate fits

n_pairs = len(session_idx_pairs)

old_method_ps = np.zeros((n_pairs,))
new_method_ps = np.zeros((n_pairs,))
old_method_slopes = np.zeros((n_pairs,))
new_method_slopes = np.zeros((n_pairs,))
old_method_r_squareds = np.zeros((n_pairs,))
new_method_r_squareds = np.zeros((n_pairs,))

r_squareds_1 = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
r_squareds_2 = [[] for _ in range(n_pairs)]

delta_group_corrs_all = []
delta_indirect_predictions_all = []
indirect_weights_all = []

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]
    
    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    print('Pair {} - Sessions {} and {} - Mouse {}'.format(
        pair_idx, day_1_idx, day_2_idx, data_dict['data']['mouse'][day_2_idx]
    )) 

    data_to_extract = ('d_ps', 'trial_start_fs', 'resp_ps', 'resp_ps_pred',)
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
    data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)

    d_ps_1 = data_1['d_ps']
    d_ps_2 = data_2['d_ps']

    n_groups = data_1['resp_ps'].shape[-1]
    
    # data_to_extract = ('trial_start_fs',)
    if correlation_mode == 'all':
        pairwise_corrs_1 = data_dict['data']['trace_corr'][day_1_idx]
        pairwise_corrs_2 = data_dict['data']['trace_corr'][day_2_idx]
        # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
        pairwise_corrs_1 = np.where(np.isnan(pairwise_corrs_1), 0., pairwise_corrs_1)
        pairwise_corrs_2 = np.where(np.isnan(pairwise_corrs_2), 0., pairwise_corrs_2)
    elif correlation_mode == 'pre+post':
        pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'])
        pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'])
    elif correlation_mode == 'pre':
        pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(-2, 0))
        pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(-2, 0))
    elif correlation_mode == 'post':
        pairwise_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(0, 10))
        pairwise_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(0, 10))
    else:
        raise ValueError()
    
    delta_corrs = [] # Filled as we iterate across groups
    delta_indirect_resp_ps = []
    delta_indirect_predictions = []

    for group_idx in range(n_groups):
        direct_idxs = np.where(np.logical_and(
            d_ps_1[:, group_idx] < D_DIRECT, 
            d_ps_2[:, group_idx] < D_DIRECT
        ))[0]
        indirect_idxs = np.where(np.logical_and(
            np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
            np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
        ))[0]

        r_squareds_1[pair_idx].append(data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])
        r_squareds_2[pair_idx].append(data_2['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])
        
        if pair_idx == exemplar_pair_idx:
            indirect_predictions_1 = np.sum(data_1['resp_ps_pred_extras']['params'][group_idx], axis=-1)
            indirect_predictions_2 = np.sum(data_2['resp_ps_pred_extras']['params'][group_idx], axis=-1)
                                       
            ax1.scatter(data_1['resp_ps'][indirect_idxs, group_idx], data_2['resp_ps'][indirect_idxs, group_idx], marker='.', color=c_vals[2], alpha=0.1)
            ax1p.scatter(indirect_predictions_1, indirect_predictions_2, marker='.', color=c_vals[2], alpha=0.1)

        COMBINE_CONST = 1/2

        raise NotImplementedError('Should implement new way of computing direct response here.')
        if direct_response_mode == 'group_membership': # No response weighting
            group_corrs_1 = np.sum(pairwise_corrs_1[:, direct_idxs], axis=-1)
            group_corrs_2 = np.sum(pairwise_corrs_2[:, direct_idxs], axis=-1)
        elif direct_response_mode == 'ps_resp': # PS response weighting (sum of both to prevent spurious correlations)
            if direct_input_mode in ('average',):
                direct_input = COMBINE_CONST * (data_1['resp_ps'][direct_idxs, group_idx] + data_2['resp_ps'][direct_idxs, group_idx])
            elif direct_input_mode in ('minimum',): # Separately computes direct input, then minimum mag of both
                raise NotImplementedError()
            else:
                raise ValueError('Direct_input_mode {} not recognized'.format(direct_input_mode))
            group_corrs_1 = np.matmul(pairwise_corrs_1[:, direct_idxs], direct_input)
            group_corrs_2 = np.matmul(pairwise_corrs_2[:, direct_idxs], direct_input) 
        elif direct_response_mode == 'ps_resp_thresh': # Threshold response weighting on average of responses
#             threshold_mask = (1/2 * (resp_ps_1[direct_idxs, group_idx]  + resp_ps_2[direct_idxs, group_idx])) > PS_RESPONSE_THRESH
            threshold_mask = np.logical_and(data_1['resp_ps'][direct_idxs, group_idx] > PS_RESPONSE_THRESH, data_2['resp_ps'][direct_idxs, group_idx] > PS_RESPONSE_THRESH)
            group_corrs_1 = np.matmul(pairwise_corrs_1[:, direct_idxs], threshold_mask) 
            group_corrs_2 = np.matmul(pairwise_corrs_2[:, direct_idxs], threshold_mask) 
        else:
            raise ValueError()
        
        if plot_pair[0] in ('delta_pairwise_corr_x',):
            delta_group_corr = (group_corrs_2 - group_corrs_1)[indirect_idxs]
#             if group_idx == 0: print('Shuffling correlation within groups!')
#             np.random.shuffle(delta_group_corr) # Turn this one for validation, does indirect identity actually mattter?
        elif plot_pair[0] in ('laser_resp_x',):
            assert direct_response_mode == 'ps_resp' # direct_input not defined otherwise, can update above code to fix
            delta_group_corr = np.nansum(direct_input) * np.ones((indirect_idxs.shape[0],))
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
        
#         if group_idx == 0: print('Shuffling delta correlation within groups!')
#         np.random.shuffle(delta_group_corr) # Turn this one for validation, does indirect identity actually mattter?        
        delta_corrs.append(delta_group_corr)
        
        if plot_pair[1] in ('raw_delta_cc_y',):
            delta_indirect_resp_ps.append(data_2['resp_ps'][indirect_idxs, group_idx] - data_1['resp_ps'][indirect_idxs, group_idx])
            delta_indirect_predictions.append(
                data_2['resp_ps_pred'][indirect_idxs, group_idx] - data_1['resp_ps_pred'][indirect_idxs, group_idx]
            )
        else:
            raise NotImplementedError('Plot pair {} not recognized.'.format(plot_pair[1]))
        
        
        ### Gets average direct input ###
        # Note this way of doing determines average input for each event THEN averages over
        # events of both sessions. This does not necessarily yield the same result as 
        # averaging over events in each session first and then determining the input because 
        # of how we treat nans and uneven event counts

        
        ### Uses average direct input to predict photostimulation response ###
        # indirect_predictions_1 = photostim_predict(indirect_params_1, direct_input, ps_stats_params)
        # indirect_predictions_2 = photostim_predict(indirect_params_2, direct_input, ps_stats_params)
        # delta_indirect_predictions.append(indirect_predictions_2 - indirect_predictions_1)
        
        if pair_idx == exemplar_pair_idx:
            slope_idx = 1 if ps_stats_params['direct_predictor_intercept_fit'] else 0
            ax4.scatter(data_1['resp_ps_pred_extras']['params'][group_idx][:, slope_idx], 
                        data_2['resp_ps_pred_extras']['params'][group_idx][:, slope_idx], 
                        color=c_vals[3], marker='.', alpha=0.3)

            if ps_stats_params['direct_predictor_intercept_fit']:
                ax4p.scatter(data_1['resp_ps_pred_extras']['params'][group_idx][:, 0], 
                             data_2['resp_ps_pred_extras']['params'][group_idx][:, 0], 
                             color=c_vals[3], marker='.', alpha=0.3)

        # print('indirect_predictions old:', indirect_predictions_1)
        # print('indirect_predictions new:', data_1['resp_ps_pred'][indirect_idxs, group_idx])
        # print('r_squareds old:', fit_extras_1['r_squareds'])
        # print('r_squareds new:', data_1['resp_ps_pred_extras']['r_squareds'][indirect_idxs, group_idx])
        # print(sdfdsfsdfsd)

    delta_corrs = np.concatenate(delta_corrs, axis=0)
    delta_indirect_resp_ps = np.concatenate(delta_indirect_resp_ps, axis=0)
    delta_indirect_predictions = np.concatenate(delta_indirect_predictions, axis=0)
    
    r_squareds_1[pair_idx] = np.concatenate(r_squareds_1[pair_idx], axis=0)
    r_squareds_2[pair_idx] = np.concatenate(r_squareds_2[pair_idx], axis=0)
    
    ax_plot, axp_plot = None, None
    
    if pair_idx == exemplar_pair_idx:
        ax_plot, axp_plot = ax2, ax2p
        ax2.scatter(delta_corrs, delta_indirect_resp_ps, color=c_vals_l[0], alpha=0.05, marker='.')
        ax2p.scatter(delta_corrs, delta_indirect_predictions, color=c_vals_l[0], alpha=0.05, marker='.')
    
    if weight_type in (None,):
        weights = None
    elif weight_type in ('minimum_rsquared',):
        # Minimum with nan still yields nan
        weights = np.minimum(r_squareds_1[pair_idx], r_squareds_2[pair_idx])
    else:
        raise ValueError('Weight type {} not recognized'.format(weight_type))
    
    slope_o, _, rvalue_o, pvalue_o, _ = add_regression_line(delta_corrs, delta_indirect_resp_ps, ax=ax_plot, color=c_vals[0], zorder=5)
    
#     print('delta_corrs', delta_corrs.shape, 'nan_count', np.sum(np.isnan(delta_corrs)))
#     print('delta_indirect_predictions', delta_indirect_predictions.shape, 'nan_count', np.sum(np.isnan(delta_indirect_predictions)))
#     print('weights', weights.shape, 'nan_count', np.sum(np.isnan(weights)))
#     indirect_predictions = np.where(np.isnan(indirect_predictions), 0., indirect_predictions)
#     weights = np.where(np.isnan(weights), 0., weights)    
    slope_n, _, rvalue_n, pvalue_n, _ = add_regression_line(
        delta_corrs, delta_indirect_predictions, weights=weights,
        ax=axp_plot, color=c_vals[0], zorder=5
    )
    
    delta_group_corrs_all.append(delta_corrs)
    delta_indirect_predictions_all.append(delta_indirect_predictions)
    indirect_weights_all.append(weights)
    
    old_method_ps[pair_idx] = pvalue_o
    new_method_ps[pair_idx] = pvalue_n
    old_method_slopes[pair_idx] = slope_o
    new_method_slopes[pair_idx] = slope_n
    old_method_r_squareds[pair_idx] = rvalue_o**2
    new_method_r_squareds[pair_idx] = rvalue_n**2

for ax in (ax1, ax1p, ax4, ax4p):
    ax.axhline(0.0, color='k', zorder=5, linestyle='dashed')
    ax.axvline(0.0, color='k', zorder=5, linestyle='dashed')

for ax in (ax2, ax2p,):
    ax.legend()
    ax.set_xlabel('Delta correlation (direct_1 + direct_2 weight)')

ax1.set_xlabel('Day 1 Indirect resp. (old method)')
ax1.set_ylabel('Day 2 Indirect resp. (old method)')
ax1p.set_xlabel('Day 1 indirect pred.')
ax1p.set_ylabel('Day 2 indirect pred.')

ax2.set_ylabel('Delta indirect (old method)')
ax2p.set_ylabel('Delta indirect (interpolation method)')

ax4.set_xlabel('Day 1 slope')
ax4.set_ylabel('Day 2 slope')
ax4p.set_xlabel('Day 1 intercept')
ax4p.set_ylabel('Day 2 intercept')

old_new_pvalues = np.concatenate(
    (old_method_ps[:, np.newaxis], new_method_ps[:, np.newaxis]), axis=-1
)
old_new_r_squareds = np.concatenate(
    (old_method_r_squareds[:, np.newaxis], new_method_r_squareds[:, np.newaxis]), axis=-1
)

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]
    
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    print('Mouse {} - Sessions {} to {}, -log10(p): {:.2f} to {:.2f}'.format(
        data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx, 
        -np.log10(old_new_pvalues[pair_idx, 0]), -np.log10(old_new_pvalues[pair_idx, 1])
    ))
    
    ax3.plot(-np.log10(old_new_pvalues[pair_idx]), color=c_vals[PAIR_COLORS[pair_idx]], marker='.')
    ax3.plot(-np.log10(old_new_pvalues[pair_idx, 0]), color=c_vals_l[PAIR_COLORS[pair_idx]], marker='.')
    ax3pp.plot(old_new_r_squareds[pair_idx], color=c_vals[PAIR_COLORS[pair_idx]], marker='.')
    ax3pp.plot(old_new_r_squareds[pair_idx, 0], color=c_vals_l[PAIR_COLORS[pair_idx]], marker='.')

ax3p.scatter(new_method_slopes, -np.log10(new_method_ps), 
             c=[c_vals[pair_color] for pair_color in PAIR_COLORS[:n_pairs]])
ax3p.scatter(old_method_slopes, -np.log10(old_method_ps), 
             c=[c_vals_l[pair_color] for pair_color in PAIR_COLORS[:n_pairs]], zorder=-3, marker='.')
    
for ax in (ax3, ax3p,):
    ax.axhline(2, color='lightgrey', zorder=-5, linestyle='dashed')
    ax.set_ylabel('-log10(pvalue)')
    
ax3.set_xticks((0, 1))
ax3.set_xticklabels(('No interpolation', 'Interpolation'))
ax3p.set_xlabel('Slope')
ax3p.axvline(0., color='lightgrey', zorder=-5, linestyle='dashed')

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    _ = add_regression_line(
        delta_group_corrs_all[pair_idx], delta_indirect_predictions_all[pair_idx], 
        weights=indirect_weights_all[pair_idx], label=None,
        ax=ax5, color=c_vals[PAIR_COLORS[pair_idx]], zorder=0
    )
    
_ = add_regression_line(
    np.concatenate(delta_group_corrs_all, axis=0), np.concatenate(delta_indirect_predictions_all, axis=0), 
    weights=np.concatenate(indirect_weights_all, axis=0),
    ax=ax5, color='k', zorder=5
)    
ax5.legend()
ax5.set_xlabel('Delta correlation (direct_1 + direct_2 weight)')
ax5.set_ylabel('Delta indirect (interpolation method)')

# %% Cell 34: Consistency of fits across pairs
fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))

for pair_idx, ax5 in zip(range(n_pairs), ax5s.flatten()):

# pair_idx = 10

    r_squared_1_pair = r_squareds_1[pair_idx]
    r_squared_2_pair = r_squareds_2[pair_idx]

    ax5.scatter(
        r_squared_1_pair, r_squared_2_pair, color=c_vals[PAIR_COLORS[pair_idx]], marker = '.', alpha=0.3
    )
    add_identity(ax5, color='k', zorder=5, linestyle='dashed')
    _ = add_regression_line(r_squared_1_pair, r_squared_2_pair, ax=ax5, color=c_vals_d[PAIR_COLORS[pair_idx]], 
                            zorder=5, fit_intercept=False)

    ax5.set_xlim((-0.05, 1.05))
    ax5.set_ylim((-0.05, 1.05))
    ax5.axhline(0.1, color='lightgrey', zorder=5)
    ax5.axvline(0.1, color='lightgrey', zorder=5)
    ax5.legend()

# %% Cell 37: Some helper functions for the below code
def eval_over_neurons(
    neuron_metric_1, neuron_cc_1, neuron_metric_2, neuron_cc_2, ps_stats_params, weights=None,
    over_neuron_mode='matrix_multiply_sanity', debug=False, second_weights=None, keep_days_separate=False,
):
    """
    Various ways of computing "A" or "W" for a given group OR group/indirect neuron
    combination (e.g. in computing correlation). 
    
    An example usage would be the computation of 5c, where the y-axis's W, for a 
    given group, is given by the matrix multiplication of delta tuning and 
    indirect change in photostimulation.

    This function gives added functionality to decompose various contributions 
    to said matrix multiplication including things like doing mean subtraction
    or compuating the pearson correlation instead of matrix multiplication. All
    of these still function as taking something that is shape (n_neurons,) and 
    another quantity that is (n_neurons, n_groups,) to something that is 
    (n_groups,)

    Note that many of these measures require being more careful than simply doing
    a matrix multiplication, so the masks for each group are done by hand
    rather than relying on zeros in the neuron_cc_1 to properly mask things.
    
    Note this function can return NaNs for certain elements and they will be omitted
    by a nan mask later.
    
    Finally note that over_neuron_mode can be a list of modes, in which case it 
    iterates over the various modes
    
    INPUTS:
    - neuron_metric_1: string OR np.array of shape: (n_neurons,) OR (n_neurons2, n_neurons)
    - neuron_cc_1: the weight metric (n_neurons, n_groups,)
    - neuron_metric_2: None OR string OR np.array of shape: (n_neurons,) OR (n_neurons2, n_neurons)
    - neuron_cc_2: None OR the weight metric (n_neurons, n_groups,)
    - over_neuron_mode: string or list of strings
        - matrix_multiply: usual way of doing matrix multiplication that relies
            on zeros to mask groups
        - matrix_multiply_sanity: sanity check to make sure everything is done 
            correctly, should yield same results as matrix-multiply, but 
            eliminates group members by hand
        - means: isolate mean contribution to matrix_multiply_sanity
        - matrix_multiply_centered: isolate mean subtraction contribution to matrix_multiply_sanity
    - second_weights: optional additional weights to mask n_neurons2 if neuron_metric_1 is 2d
        - Generally used to properly select down to indirect neurons
    """
    
    if ps_stats_params['ps_analysis_type'] == 'single_session': # Doesnt make sense to have two days in this setting
        assert neuron_metric_2 is None
        assert neuron_cc_2 is None
        assert ~keep_days_separate
    elif ps_stats_params['ps_analysis_type'] == 'paired': # Doesnt make sense to have two days in this setting
        assert neuron_metric_2 is not None
        assert neuron_cc_2 is not None
        if type(neuron_metric_1) != str:
            assert np.all(neuron_metric_1.shape == neuron_metric_2.shape)
        assert np.all(neuron_cc_1.shape == neuron_cc_2.shape)
    
    group_vals_all = []
    if type(over_neuron_mode) == str: # Default case, creates single element tuple to iterate over
        on_modes = (over_neuron_mode,)
    else:
        on_modes = over_neuron_mode
        
    neuron_metric_1, neuron_cc_1, weights = validation_tests(
        neuron_metric_1, neuron_cc_1, ps_stats_params, weights=weights, second_weights=second_weights,
        on_modes=on_modes
    )
    
    if type(neuron_metric_1) != str:
        # These work when neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons)
        if ps_stats_params['neuron_metrics_adjust'] in ('normalize',):
            raise NotImplementedError('Update to work with two neuron_metrics')
            neuron_metric_1 /= np.linalg.norm(neuron_metric_1, axis=-1, keepdims=True)
        elif ps_stats_params['neuron_metrics_adjust'] in ('standardize',):
            raise NotImplementedError('Update to work with two neuron_metrics')
            neuron_metric_1 = (
                (neuron_metric_1 - np.nanmean(neuron_metric_1, axis=-1, keepdims=True)) / 
                np.nanstd(neuron_metric_1, axis=-1, keepdims=True)
            )
        elif ps_stats_params['neuron_metrics_adjust'] != None:
            raise NotImplementedError('neuron_metrics_adjust {} not recognized.'.format(ps_stats_params['neuron_metrics_adjust']))
        
    for on_mode in on_modes: 
        # This is the slow way of doing things but treats things more carefully than just explicit matrix multiplication
        if on_mode in ('pearson', 'means', 'matrix_multiply_centered', 
                       'matrix_multiply_standardized', 'neuron_centered_conn_standardized'):
            raise ValueError('Depricated in new eval_over_neurons function')
        if on_mode in ('matrix_multiply_sanity', 'product_nc',) or type(neuron_metric_1) == str:
            assert weights is not None
            
            weights_mask = weights > 0 # (n_neurons, n_groups,)
            
            n_groups = neuron_cc_1.shape[-1]
            
            if type(neuron_metric_1) == str:
                if neuron_metric_1 in ('raw_nc',): # Do not collapse over indirect neurons
                    group_vals = [] # (n_groups, n_indirect_neurons)
                else:
                    group_vals = np.zeros((n_groups,)) # (n_groups,)
            elif on_mode in ('product_nc',): # Metrics that do not collapse over indirect neurons
                group_vals = [] # (n_groups, n_indirect_neurons)
            else:
                if len(neuron_metric_1.shape) == 1: # (n_neurons,)
                    group_vals = np.zeros((n_groups,))
                elif len(neuron_metric_1.shape) == 2: # For when things like correlation are used, (n_neurons2, n_neurons,)
                    group_vals = []
                    assert second_weights is not None
                    second_weights_mask = second_weights > 0 # (n_neurons, n_groups)

            for group_idx in range(n_groups):
                
                masked_neuron_cc_1 = neuron_cc_1[weights_mask[:, group_idx], group_idx]
                if neuron_cc_2 is not None:
                    masked_neuron_cc_2 = neuron_cc_2[weights_mask[:, group_idx], group_idx]

                if masked_neuron_cc_1.shape[0]== 0: # Default values if no neurons make it past the mask
                    if type(group_vals) == list: 
                        if len(neuron_metric_1.shape) == 1: # (n_neurons = 0)
                            group_vals.append(np.array(()))
                        elif len(neuron_metric_1.shape) == 2: # (n_neurons2,)
                            masked_neuron_metric_1 = neuron_metric_1[second_weights_mask[:, group_idx], :]
                            group_vals.append(np.nan * np.ones(masked_neuron_metric_1.shape[0]))
                    else:
                        group_vals[group_idx] = np.nan
                    continue

                if type(neuron_metric_1) == str: # Special sum, all are independent of over_neuron_mode
                    if neuron_cc_2 is not None: # Distinct ways of combining don't matter in these cases
                        masked_neuron_cc = masked_neuron_cc_1 + masked_neuron_cc_2
                    else:
                        masked_neuron_cc = np.copy(masked_neuron_cc_1)
                    
                    if neuron_metric_1 in ('raw_nc',):
                        assert on_mode in ('product_nc',)
                        group_vals.append(masked_neuron_cc) # n_indirect
                    elif neuron_metric_1 in ('sum',):
                        group_vals[group_idx] = np.sum(masked_neuron_cc)
                    elif neuron_metric_1 in ('abs_sum',):
                        group_vals[group_idx] = np.sum(np.abs(masked_neuron_cc))
                    elif neuron_metric_1 in ('mask_sum',): # Zeros eliminated in sum
                        group_vals[group_idx] = np.sum(weights[:, group_idx])
                else:
                    # Handle masking for the various possible shapes of the neuron metric
                    if len(neuron_metric_1.shape) == 1: # (n_neurons,)
                        masked_neuron_metric_1 = neuron_metric_1[weights_mask[:, group_idx]]
                        if neuron_metric_2 is not None:
                            masked_neuron_metric_2 = neuron_metric_2[weights_mask[:, group_idx]]
                    elif len(neuron_metric_1.shape) == 2: # (n_neurons2, n_neurons,)
                        if on_mode not in ('matrix_multiply_sanity',):
                            NotImplementedError('neuron_metric_1 shape not implemented for on_mode:', on_mode)
                        masked_neuron_metric_1 = neuron_metric_1[:, weights_mask[:, group_idx]]
                        masked_neuron_metric_1 = masked_neuron_metric_1[second_weights_mask[:, group_idx], :]
                        if neuron_metric_2 is not None:
                            masked_neuron_metric_2 = neuron_metric_2[:, weights_mask[:, group_idx]]
                            masked_neuron_metric_2 = masked_neuron_metric_2[second_weights_mask[:, group_idx], :]
                    else:
                        raise NotImplementedError('neuron_metric_1 shape not recognized:', neuron_metric_1.shape)                
                        
                    if on_mode in ('matrix_multiply_sanity',): # Sanity check that this yields same as mm above
                        if len(masked_neuron_metric_1.shape) == 1: # yields scale
                            if ps_stats_params['ps_analysis_type'] == 'single_session':
                                group_vals[group_idx] = np.dot(masked_neuron_metric_1, masked_neuron_cc_1)
                            elif ps_stats_params['ps_analysis_type'] == 'paired':
                                if not keep_days_separate:
                                    group_vals[group_idx] = np.dot( # Combine values between days first, then product
                                        masked_neuron_metric_1 + masked_neuron_metric_2,
                                        masked_neuron_cc_1 + masked_neuron_cc_2
                                    )
                                else:
                                    group_vals[group_idx] = ( # Product first, then combine between days
                                        np.dot(masked_neuron_metric_1, masked_neuron_cc_1) +
                                        np.dot(masked_neuron_metric_2, masked_neuron_cc_2)
                                    )
                        elif len(masked_neuron_metric_1.shape) == 2: # yields (n_neurons2,)
                            if ps_stats_params['ps_analysis_type'] == 'single_session':
                                group_vals.append(np.matmul(masked_neuron_metric_1, masked_neuron_cc_1))
                            elif ps_stats_params['ps_analysis_type'] == 'paired':
                                if not keep_days_separate:
                                    group_vals.append(np.matmul( # Combine values between days first, then product
                                        masked_neuron_metric_1 + masked_neuron_metric_2,
                                        masked_neuron_cc_1 + masked_neuron_cc_2
                                    ))
                                else:
                                    group_vals.append( # Product first, then combine between days
                                        np.matmul(masked_neuron_metric_1, masked_neuron_cc_1) +
                                        np.matmul(masked_neuron_metric_2, masked_neuron_cc_2)
                                    )
                    elif on_mode in ('product_nc',):
                        if ps_stats_params['ps_analysis_type'] == 'single_session':
                            group_vals.append(masked_neuron_metric_1 * masked_neuron_cc_1) # n_indirect
                        elif ps_stats_params['ps_analysis_type'] == 'paired':
                            if not keep_days_separate: # Combine values between days first, then product
                                group_vals.append(
                                    (masked_neuron_metric_1 + masked_neuron_metric_2) *
                                    (masked_neuron_cc_1 + masked_neuron_cc_2)
                                )
                            else:
                                group_vals.append( # Product first, then combine between days
                                    (masked_neuron_metric_1 * masked_neuron_cc_1) +
                                    (masked_neuron_metric_2 * masked_neuron_cc_2)
                                )
                                
            group_vals_all.append(group_vals)
        elif on_mode == 'matrix_multiply':
            # Mask sum is already incorporated into neuron_cc_1 by setting 
            # corresponding elements to zero
            if weights is not None:
                raise ValueError('Weights dont make sense here.')

            group_vals_all.append(np.matmul(neuron_metric_1.T, neuron_cc_1))
        else:
            raise ValueError('Over_neuron_mode {} not recognized.'.format(on_mode))
            
    if type(over_neuron_mode) == str: # Default case, returns first element in list
        return group_vals_all[0]
    else: # Retruns entire list instead
        return group_vals_all
    
def validation_tests(
    neuron_metric_1, neuron_cc_1, ps_stats_params, weights=None,
    second_weights=None, on_modes=(),
):
    if 'matrix_multiply' in on_modes:
        raise NotImplementedError('Validation tests not yet implemented for on_modes:', on_modes)
    
    weights_mask = weights > 0 # (n_neurons, n_groups,)
    if 'shuffle_X_within_session' in ps_stats_params['validation_types']: # Shuffle the neuron metric
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            # Neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons), so always shuffle along -1th axis
            neuron_metric_1 = shuffle_along_axis(neuron_metric_1, axis=-1)
    if 'mean_X_within_session' in ps_stats_params['validation_types']:
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            # Neuron metric is either (n_neurons,) OR (n_neurons2, n_neurons), so take mean along -1th axis
            neuron_metric_1 = np.nanmean(neuron_metric_1, axis=-1, keepdims=True) * np.ones_like(neuron_metric_1)
    if 'shuffle_X_within_group' in ps_stats_params['validation_types']: # Shuffle current group
        raise NotImplementedError('Update to work with two neuron_metrics')
        if type(neuron_metric_1) != str: # Skip if string, because neuron_metric_1 does nothing for these types
            raise NotImplementedError()
            temp_neuron_metric_1 = np.copy(neuron_metric_1)
    
    return neuron_metric_1, neuron_cc_1, weights

# %% Cell 39: ### Fit functions Functions for taking various raw "A" and "W" metrics and fitting results.Functions...
MAX_P_LOG10 = 5 # Maximum significance to plot for various plots below

def scan_over_connectivity_pairs(ps_stats_params, records, exemplar_session_idx=None, verbose=False):
    """
    Scans over various pairs of connectivity metrics that have been
    collected in records. This is used to look for significant trends
    for both single sessions and session pairs.
    
    If ps_stats_params['plot_pairs'] is None, this does a brute force
    scan over all possible pair combinations for all connectivity 
    metrics that end in '_x' or '_y'.
    """
    
    # This can either be session pairs or individual sessions
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])

    full_fits = {} # Dict of fits across all sessions
    session_fits = {} # Dict of fits to individual sessions (optional)
    
    # Create an exhaustive list of all possible fit pairs from connectivity_metrics
    plot_pairs_significance_plot = False
    if ps_stats_params['plot_pairs'] is None:
        print('Generating all possible combinations of plot_pairs...')
        connectivity_metrics_xs = []
        connectivity_metrics_ys = []
        
        for key in ps_stats_params['connectivity_metrics']:
            if key in ('mask_counts_x', 'mask_counts_y',):
                continue # Don't add these ones to the scan
            
            if key[-2:] == '_x':
                connectivity_metrics_xs.append(key)
            elif key[-2:] == '_y':
                connectivity_metrics_ys.append(key)
            elif key[-2:] not in ('_w',): # Things like weights are not included in this scan
                print('Ending of {} is {}, not recognized.'.format(key, key[-2:]))
        plot_pairs = list(itertools.product(connectivity_metrics_xs, connectivity_metrics_ys))
        plot_pairs_idxs = list(itertools.product(
            np.arange(len(connectivity_metrics_xs)), np.arange(len(connectivity_metrics_ys))
        ))
        
        ps_stats_params['plot_pairs'] = plot_pairs
        
        plot_pairs_xs = np.zeros((len(connectivity_metrics_xs), len(connectivity_metrics_ys),))
        plot_pairs_ys = np.zeros_like(plot_pairs_xs)
        plot_pairs_ps = np.zeros_like(plot_pairs_xs)
        plot_pairs_slopes = np.zeros_like(plot_pairs_xs)
        plot_pairs_r_squareds = np.zeros_like(plot_pairs_xs)
        
        plot_pairs_significance_plot = True
    
    plot_pairs = ps_stats_params['plot_pairs']
    
    ### Iteration over all possible things we want to plot against one another ###
    for plot_pairs_idx, pair in enumerate(plot_pairs):
        
        session_fits[pair] = [] # Only fillted in if fitting individual sessions
        
        # For the given pair, flattens records across sessions to plot all at once. 
        # Note this is not the most efficient way to do this because it
        # will have to flatten the same thing many times, but was the 
        # easiest to code up.
        records_x_flat = []
        records_y_flat = []
        
        records_w_flat = []
        
        if ps_stats_params['plot_up_mode'] in ('all',):
            fig, (ax, axp) = plt.subplots(1, 2, figsize=(12, 4))
        elif ps_stats_params['plot_up_mode'] == None:
            ax, axp = None, None
        else:
            raise NotImplementedError('plot_up_mode {} not recognized or depricated'.format(ps_stats_params['plot_up_mode']))
        
        # Somtimes x and y will be different shapes, so this code handles casting them to same shape
        for session_idx, (session_records_x, session_records_y) in enumerate(zip(records[pair[0]], records[pair[1]])):
            if type(session_records_x) == np.ndarray and type(session_records_y) == np.ndarray: # No modification needed
                session_records_x_flat = session_records_x
                session_records_y_flat = session_records_y
            elif type(session_records_x) == list or type(session_records_y) == list: # either x or y for each indirect, may need to extend the other
                session_records_x_flat = []
                session_records_y_flat = []
                for group_idx in range(len(session_records_y)):
                    if type(session_records_x) == np.ndarray: # Need to extend x for each y
                        session_records_x_flat.append(
                            session_records_x[group_idx] * np.ones((len(session_records_y[group_idx]),))
                        )
                        session_records_y_flat.append(session_records_y[group_idx])
                    elif type(session_records_y) == np.ndarray: # Need to extend y for each x
                        session_records_x_flat.append(session_records_x[group_idx])
                        session_records_y_flat.append(
                            session_records_y[group_idx] * np.ones((len(session_records_x[group_idx]),))
                        )
                    elif type(session_records_x) == list and type(session_records_y) == list: # x and y already extended
                        session_records_x_flat.append(session_records_x[group_idx])
                        session_records_y_flat.append(session_records_y[group_idx])
                    else:
                        raise NotImplementedError('This combination shouldnt occur')
                    
                    if 'shuffle_A_within_group' in ps_stats_params['validation_types']: # Shuffle most recent group
                        if plot_pairs_idx==0 and session_idx == 0 and group_idx ==0: print('Shuffling A within group!')
                        np.random.shuffle(session_records_y_flat[-1])
                    
                session_records_x_flat = np.concatenate(session_records_x_flat, axis=0)
                session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)
                
                if ps_stats_params['indirect_weight_type'] is not None: # Does the same for weights
                    session_records_w_flat = []
                    for group_idx in range(len(session_records_y)):
                        session_records_w_flat.append(records['indirect_weights_w'][session_idx][group_idx])
                    session_records_w_flat = np.concatenate(session_records_w_flat, axis=0)
            else:
                raise NotImplementedError('This combination shouldnt occur')
            
            # Append session records to across-session collection
            records_x_flat.append(session_records_x_flat)
            records_y_flat.append(session_records_y_flat)
            if ps_stats_params['indirect_weight_type'] is not None:
                records_w_flat.append(session_records_w_flat)
                
            if ps_stats_params['fit_individual_sessions']:
            
                nan_mask = np.where(~np.isnan(session_records_x_flat) * ~np.isnan(session_records_y_flat))[0]
                session_records_x_flat = session_records_x_flat[nan_mask]
                session_records_y_flat = session_records_y_flat[nan_mask]
                if ps_stats_params['indirect_weight_type'] is not None:
                    session_weights = session_records_w_flat[nan_mask]
                else:
                    session_weights = None

                if ps_stats_params['ps_analysis_type'] in ('single_session',):
                    session_color = c_vals[SESSION_COLORS[session_idx]]
                elif ps_stats_params['ps_analysis_type'] in ('paired',):
                    session_color = c_vals[PAIR_COLORS[session_idx]]
                
                session_fit = {}
                session_fit['slope'], _, session_fit['rvalue'], session_fit['pvalue'], _ = add_regression_line(
                    session_records_x_flat, session_records_y_flat, weights=session_weights, ax=ax, 
                    color=session_color, label=None,
                )
                session_fit['pvalue'] = np.where(session_fit['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, session_fit['pvalue'])
                
                session_fits[pair].append(session_fit)
                
                if axp is not None:
                    axp.scatter(session_fit['slope'], -np.log10(session_fit['pvalue']), marker='.', color=session_color)
                    
                if exemplar_session_idx is not None: # Exemplar session plot of fit
                    if session_idx == exemplar_session_idx:
                        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
                        
                        ax2.scatter(
                            session_records_x_flat, session_records_y_flat, color=session_color, alpha=0.05,
                            marker='.'
                        )
                        add_regression_line(
                            session_records_x_flat, session_records_y_flat, weights=session_weights, ax=ax2, 
                            color=session_color,
                        )
                        ax2.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
                        ax2.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')

                        ax2.set_xlabel(pair[0])
                        ax2.set_ylabel(pair[1])
                        ax2.legend()
            
        records_x_flat = np.concatenate(records_x_flat, axis=0)
        records_y_flat = np.concatenate(records_y_flat, axis=0)
        
        # Catch nans in records that can result from invalid group masks
        nonnan_mask = np.where(~np.isnan(records_x_flat) * ~np.isnan(records_y_flat))[0]
        records_x_flat = records_x_flat[nonnan_mask]
        records_y_flat = records_y_flat[nonnan_mask]
        
        if ps_stats_params['normalize_masks']: # Trivial relation in this setting
            if pair[0] == 'mask_counts_x' and pair[1] == 'mask_counts_y':
                continue
        
        if ps_stats_params['indirect_weight_type'] is not None: 
            records_w_flat = np.concatenate(records_w_flat, axis=0)
            records_w_flat = records_w_flat[nonnan_mask]
            weights = records_w_flat
        elif ps_stats_params['group_weights_type'] in ('direct_resp', 'direct_resp_norm_sessions',):
            raise NotImplemntedError('This is depricated now, need to update to new name conventions \
                                      and the fact that flat records is not computed ahead of time.')
            
            if ps_stats_params['ps_analysis_type'] == 'single_session':
                weight_key = 'cc_x'  # Total direct response for each group
            elif ps_stats_params['ps_analysis_type'] == 'paired':
                # Construct a special weight vector
                flat_records['laser_resp_special_x'] = []
                for session_cc_x, session_laser_resp in zip(records['cc_x'], records['laser_resp_x']):
                    # 0. if either session's cc_x < 0., otherwise mean laser resp.
                    for group_idx in range(session_cc_x[0].shape[0]):
                        if session_cc_x[0][group_idx] < 0. or session_cc_x[1][group_idx] < 0.:
                            flat_records['laser_resp_special_x'].append(0.0)
                        else:
                            flat_records['laser_resp_special_x'].append(session_laser_resp[group_idx])
                flat_records['laser_resp_special_x'] = np.array(flat_records['laser_resp_special_x'])
                weight_key = 'laser_resp_special_x' 
#                 weight_key = 'laser_resp_x'  # Mean direct response across pair for each group 
            else:
                raise ValueError('No weight key specified.')
            
            weights = np.where(flat_records[weight_key] > 0., flat_records[weight_key], 0.) # Zero out negatives
            if ps_stats_params['group_weights_type'] in ('direct_resp_norm_sessions',):
                raise NotImplementedError()
            # Apply same nan mask used above 
            weights = weights[nonnan_mask]
        elif ps_stats_params['group_weights_type'] == None and ps_stats_params['indirect_weight_type'] == None:
            weights = None
        else:
            raise NotImplementedError('Group weights type {} not recognized.'.format(ps_stats_params['group_weights_type']))
        
        aggregate_fit = {}
        aggregate_fit['slope'], _, aggregate_fit['rvalue'], aggregate_fit['pvalue'], _ = add_regression_line(
            records_x_flat, records_y_flat, weights=weights, ax=ax, color='k', label='',
        )
        aggregate_fit['pvalue'] = np.where(aggregate_fit['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, aggregate_fit['pvalue'])
        
        if verbose:
            print_str = '{} - {}\tvs. {}\tp={:.2e}\t{}'.format(
                plot_pairs_idx, pair[0], pair[1], aggregate_fit['pvalue'], 
                '*' * np.min((MAX_P_LOG10, int(-np.log10(aggregate_fit['pvalue'])),))
            )
            print(print_str)

        full_fits[pair] = aggregate_fit
        
        if ax is not None:
            ax.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            ax.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            
            ax.set_xlabel(pair[0])
            ax.set_ylabel(pair[1])
            ax.legend()
        if axp is not None:
            axp.scatter(aggregate_fit['slope'], -np.log10(aggregate_fit['pvalue']), marker='o', color='k')
            
            axp.axvline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')
            axp.axhline(2.0, color='lightgrey', zorder=-5, linestyle='dashed')
            
            axp.set_xlabel('fit slope')
            axp.set_ylabel('fit -log10(p)')
        
        if plot_pairs_significance_plot:
            pair_idxs = plot_pairs_idxs[plot_pairs_idx]
            plot_pairs_xs[pair_idxs[0], pair_idxs[1]] = pair_idxs[0]
            plot_pairs_ys[pair_idxs[0], pair_idxs[1]] = pair_idxs[1]
            plot_pairs_ps[pair_idxs[0], pair_idxs[1]] = aggregate_fit['pvalue'] 
            plot_pairs_slopes[pair_idxs[0], pair_idxs[1]] = aggregate_fit['slope']
            plot_pairs_r_squareds[pair_idxs[0], pair_idxs[1]] = aggregate_fit['rvalue']**2
    
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    if plot_pairs_significance_plot:
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Avoids errors where p_values are too low to be meaningful
        plot_pairs_ps = np.where(plot_pairs_ps < MIN_P_VALUE, MIN_P_VALUE, plot_pairs_ps)
        
        plot_pairs_ps_log10 = np.where(
            -1 * np.log10(plot_pairs_ps) > MAX_P_LOG10, MAX_P_LOG10, -1 * np.log10(plot_pairs_ps)
        )
        # Color plot_piar_ps by sign of slope
        plot_pairs_ps_log10 = np.sign(plot_pairs_slopes) * plot_pairs_ps_log10
        
        # Transpose needed since matshow has first index run vertically by default, but first
        # index is xs so want it to run horizontally
        ax.matshow(plot_pairs_ps_log10.T, cmap='bwr', vmin=-MAX_P_LOG10, vmax=MAX_P_LOG10)
        axp.matshow(plot_pairs_r_squareds.T, cmap='Reds', vmin=0)
        
        for (i, j), z in np.ndenumerate(plot_pairs_ps): # Above transpose accounted for here too
            ax.text(i, j, '({:0.0e})'.format(z), ha='center', va='center')
        for (i, j), z in np.ndenumerate(plot_pairs_r_squareds): # Above transpose accounted for here too
            axp.text(i, j, '({:0.1e})'.format(z), ha='center', va='center')
        
#         # Old plots that use dots
#         plot_pairs_colors = ['k' if s > 0 else 'r' for s in plot_pairs_slopes.flatten()]

#         ax.scatter(plot_pairs_xs.flatten(), plot_pairs_ys.flatten(), s=50*plot_pairs_ps_log10.flatten(),
#                    color=plot_pairs_colors)
        for ax_plot in (ax, axp,):
            ax_plot.set_xticks(np.arange(len(connectivity_metrics_xs)))
            ax_plot.set_xticklabels(connectivity_metrics_xs, rotation=90)
            ax_plot.set_yticks(np.arange(len(connectivity_metrics_ys)))
            ax_plot.set_yticklabels(connectivity_metrics_ys) 
        
    if ps_stats_params['fit_individual_sessions']:
        for plot_pairs_idx, pair in enumerate(plot_pairs):
            
            full_fits[pair]['pvalue'] = np.where(full_fits[pair]['pvalue'] < MIN_P_VALUE, MIN_P_VALUE, full_fits[pair]['pvalue'])
            full_p_val = np.min((-1 * np.log10(full_fits[pair]['pvalue']), MAX_P_LOG10))
            ax1.scatter(plot_pairs_idx, full_p_val, color='k', marker='_')
            
            ax2.scatter(plot_pairs_idx, full_fits[pair]['slope'] / full_fits[pair]['slope'], color='k', marker='_')
            
            pair_ps = np.zeros((n_sessions,))
            pair_slopes = np.zeros((n_sessions,))
            
            for session_idx in range(n_sessions):
                if ps_stats_params['ps_analysis_type'] in ('single_session',):
                    session_color = c_vals[SESSION_COLORS[session_idx]]
                elif ps_stats_params['ps_analysis_type'] in ('paired',):
                    session_color = c_vals[PAIR_COLORS[session_idx]]
                
                session_p_val = np.min((-1 * np.log10(session_fits[pair][session_idx]['pvalue']), MAX_P_LOG10))
                ax1.scatter(plot_pairs_idx, session_p_val, color=session_color, zorder=-5,
                            alpha=0.5, marker='.')
                pair_ps[session_idx] = session_p_val
                
                # Normalize slopes by full fit slope to see consistency
                pair_slopes[session_idx] = session_fits[pair][session_idx]['slope'] / full_fits[pair]['slope']
                ax2.scatter(plot_pairs_idx, pair_slopes[session_idx], color=session_color, 
                            zorder=-5, alpha=0.5, marker='.')
                
            ax1.scatter(plot_pairs_idx, pair_ps.mean(), color='k', zorder=-1, marker='.')
            ax2.scatter(plot_pairs_idx, pair_slopes.mean(), color='k', zorder=-1, marker='.')
            ax1.errorbar(plot_pairs_idx, pair_ps.mean(), pair_ps.std() / np.sqrt(n_sessions), 
                         color='k', zorder=-2, marker='.', alpha=0.8)
            ax2.errorbar(plot_pairs_idx, pair_slopes.mean(), pair_slopes.std() / np.sqrt(n_sessions), 
                         color='k', zorder=-2, marker='.', alpha=0.8)
            
        ax1.set_xlabel('Plot_pairs')
        ax1.set_ylabel('-log10(ps)')
        
        ax2.set_xlabel('Plot_pairs')
        ax2.set_ylabel('Norm. slope')
        
        ax2.set_ylim((-2, 2))
        ax2.axhline(1.0, color='grey', linestyle='dashed', zorder=-10)
        ax2.axhline(0.0, color='grey', linestyle='dashed', zorder=-10)
        
        for ax in (ax1, ax2,):
            ax.set_xticks(np.arange(len(plot_pairs)))
            ax.set_xticklabels([None if x % 5 else x for x in range(len(plot_pairs))]) # Label every 5th
            for val in range(0, len(plot_pairs), 5):
                ax.axvline(val, color='lightgrey', alpha=0.5, zorder=-20)
            
        fig2.show()
                
    return full_fits, session_fits, ps_stats_params

def enumerate_plot_pairs(ps_stats_params):
    """
    Given a list of all possible connectivity_metrics, separates them into independent/dependent variables.
    """
    connectivity_metrics_xs = []
    connectivity_metrics_ys = []

    for key in ps_stats_params['connectivity_metrics']:
        if key in ('mask_counts_x', 'mask_counts_y',):
            continue # Don't add these ones to the scan

        if key[-2:] == '_x':
            connectivity_metrics_xs.append(key)
        elif key[-2:] == '_y':
            connectivity_metrics_ys.append(key)
        elif key[-2:] not in ('_w',): # Things like weights are not included in this scan
            print('Ending of {} is {}, not recognized.'.format(key, key[-2:]))
            
    return connectivity_metrics_xs, connectivity_metrics_ys

def get_all_xs(records, connectivity_metrics_xs, standardize_xs=False):
    """
    Get all x predictors that are going to be used later for MLR fits.
    Output is in a form that can be used for full fit and for hierarchical
    bootstrapping too.
    
    OUTPUTS:
    - records_x_flat_all: list of lists of numpy arrays
        - First list idx: session_idx
        - Second list idx: group_idx
        - Each numpy array is shape (n_cm_x, n_indirect) for the given session/group
    """
    
    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    n_sessions = len(records[exemplar_key])

    records_x_flat_all = [] # session_idx, group_idx, (n_cm_x, n_indirect)

    ### Iteration over all possible things we want to plot against one another ###
    for session_idx in range(n_sessions):
        
        exemplar_session_records_y = records[exemplar_key][session_idx]
        n_groups = len(exemplar_session_records_y)
        session_records_x_flat = [[] for group_idx in range(n_groups)] # group_idx, (n_cm_x, n_indirect)
        
        for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):

            session_records_x = records[connectivity_metrics_x][session_idx]

            # Expand all xs so that they're the same shape 
            for group_idx in range(len(session_records_x)):
                if type(session_records_x) == np.ndarray: # Need to extend x for each y
                    session_records_x_flat[group_idx].append(
                        session_records_x[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_x) == list: # x already extended
                    session_records_x_flat[group_idx].append(session_records_x[group_idx])

        
        # Now for each group, stack the different possible cm_x
        for group_idx in range(len(session_records_x)):
            try:
                session_records_x_flat[group_idx] = np.array(session_records_x_flat[group_idx]) # (n_cm_x, n_indirect)
            except ValueError:
                print('session', session_idx, 'group', group_idx)
                for val_idx, val in enumerate(session_records_x_flat[group_idx]):
                    print(val_idx, len(val))
                
                print(connectivity_metrics_xs)
                
                print(session_records_x_flat[group_idx])
                print(sdfdsfds)
    
        if standardize_xs: # Standardize the x across the session  
            temp_session_records_x_flat = np.concatenate(session_records_x_flat, axis=-1) # (n_cm_x, n_total_indirect)
            session_mean = np.nanmean(temp_session_records_x_flat, axis=-1, keepdims=True) # (n_cm_x, 1)
            session_std = np.nanstd(temp_session_records_x_flat, axis=-1, keepdims=True) # (n_cm_x, 1)
            
            for group_idx in range(len(session_records_x_flat)): # Now adjust each group
                session_records_x_flat[group_idx] = (session_records_x_flat[group_idx] - session_mean) / session_std

        records_x_flat_all.append(session_records_x_flat)
        
    return records_x_flat_all

def fit_all_xs_at_once(records, ps_stats_params, fit_intercept=True, standardize_xs=False,
                       standardize_ys=False, bootstrap_type=None, n_bootstrap=1, 
                       verbose=False):
    """
    Fit all _xs at once for each _y using multiple linear regression. 
    
    INPUTS:
    
    - bootstrap_type: None, hierarcy
    - n_bootstrap: Number of bootstraps to run
    
    OUTPUTS:
    - full_fits: Dict with key for each connectivity_metrics_y, containing the full LS fit
        - e.g. full_fits[connectivity_metrics_y]
    - session_fits: Same as above, with each dict key containing a list of sessions
        - e.g. full_fits[connectivity_metrics_y][session_idx]
    """

    def full_fit(records_x_all, records_y_all, records_w_all=[]):
        """ This is sepearated into a function becauses its repeated when bootstrapping """
        
        records_x_all_temp = [None for _ in range(len(records_x_all))]
        records_y_all_temp = [None for _ in range(len(records_y_all))]
        records_w_all_temp = [None for _ in range(len(records_y_all))]
        
        # For each session, need to flatten across groups
        for session_idx in range(len(records_y_flat)):
            records_x_all_temp[session_idx] = np.concatenate(records_x_all[session_idx], axis=-1)
            records_y_all_temp[session_idx] = np.concatenate(records_y_all[session_idx], axis=0)
            if ps_stats_params['indirect_weight_type'] is not None: 
                records_w_all_temp[session_idx] = np.concatenate(records_w_all[session_idx], axis=0)
        
        # Now that all sessions are collected, concatenate sessions together
        records_x_all_temp = np.concatenate(records_x_all_temp, axis=-1) # Temporary concat to fit everything
        records_y_all_temp = np.concatenate(records_y_all_temp, axis=0)

        nonnan_mask = np.where(np.all(~np.isnan(records_x_all_temp), axis=0) * ~np.isnan(records_y_all_temp))[0]

        records_x_all_temp = records_x_all_temp[:, nonnan_mask]
        records_y_all_temp = records_y_all_temp[nonnan_mask]
        if ps_stats_params['indirect_weight_type'] is not None: 
            weights = np.concatenate(records_w_all_temp, axis=0)[nonnan_mask]
        elif ps_stats_params['group_weights_type'] == None and ps_stats_params['indirect_weight_type'] == None:
            weights = None
        else:
            raise NotImplementedError('Group weights type {} not recognized.'.format(ps_stats_params['group_weights_type']))

        X = records_x_all_temp.T # (n_regressors, n_neurons) -> (n_neurons, n_regressors)
        Y = records_y_all_temp[:, np.newaxis]

        if fit_intercept:
            X = sm.add_constant(X)

        if weights is None:
            fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
        else:
            fit_model = sm.WLS(Y, X, weights=weights, missing='drop') # nans should be caught above
        
        return fit_model.fit()
    
    param_idx_offset = 1 if fit_intercept else 0

    # This can either be session pairs or individual sessions
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])

    full_fits = {} # Dict of fits across all sessions
    session_fits = {} # Dict of fits to individual sessions (optional)

    if standardize_xs and verbose:
        print('Standardizing _x\'s!')
    if standardize_ys and verbose:
        print('Standardizing _y\'s!')

    # Create lists of all _x's and _y's from connectivity_metrics
    if ps_stats_params['plot_pairs'] is not None:
        raise NotImplementedError('All possible pairs is hard coded for now.')

    # Get list of all possible 
    connectivity_metrics_xs, connectivity_metrics_ys = enumerate_plot_pairs(ps_stats_params)

    # Get all _x's used to fit the _y's below
    records_x_flat_all = get_all_xs(records, connectivity_metrics_xs, standardize_xs=standardize_xs)
    
    if bootstrap_type is None:
        bs_params = None
    else:
        bs_params = np.zeros((len(connectivity_metrics_xs), len(connectivity_metrics_ys), n_bootstrap))

    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    # Now use the _x to predict each possible _y  
    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):

        session_fits[connectivity_metrics_y] = [] # Initialize, only filled in if fitting individual sessions

        records_y_flat = []
        records_w_flat = []

        for session_idx, session_records_y in enumerate(records[connectivity_metrics_y]):
            exemplar_session_records_y = records[exemplar_key][session_idx]

            # Expand all ys so that they're the same shape 
            session_records_y_flat = []
            for group_idx in range(len(session_records_y)):
                if type(session_records_y) == np.ndarray: # Need to extend y for each x
                    session_records_y_flat.append(
                        session_records_y[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_y) == list: # y already extended
                    session_records_y_flat.append(session_records_y[group_idx])

#             # Concatenate across all groups, to get total number of indirect in session
#             session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)

            if standardize_ys:
                temp_session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)
                session_mean = np.nanmean(temp_session_records_y_flat)
                session_std = np.nanstd(temp_session_records_y_flat)
                for group_idx in range(len(session_records_y_flat)): # Now adjust each group
                    session_records_y_flat[group_idx] = (session_records_y_flat[group_idx] - session_mean) / session_std

            if ps_stats_params['indirect_weight_type'] is not None: # Does the same for weights
                session_records_w_flat = []
                for group_idx in range(len(session_records_y)): # Already extended
                    session_records_w_flat.append(records['indirect_weights_w'][session_idx][group_idx])
#                 session_records_w_flat = np.concatenate(session_records_w_flat, axis=0) # Across groups

            # Append session records to across-session collection
            records_y_flat.append(session_records_y_flat)
            if ps_stats_params['indirect_weight_type'] is not None:
                records_w_flat.append(session_records_w_flat)

            if ps_stats_params['fit_individual_sessions']:
                
                # Temporary concat, leave separated into groups for potential bootstrapping later on
                session_records_x_flat_temp = np.concatenate(records_x_flat_all[session_idx], axis=-1) # Concat across groups
                session_records_y_flat_temp = np.concatenate(session_records_y_flat, axis=0)
                
                nonnan_mask = np.where(np.all(~np.isnan(session_records_x_flat_temp), axis=0) * ~np.isnan(session_records_y_flat_temp))[0]
                session_records_x_flat_temp = session_records_x_flat_temp[:, nonnan_mask]
                session_records_y_flat_temp = session_records_y_flat_temp[nonnan_mask]
                if ps_stats_params['indirect_weight_type'] is not None:
                    session_records_w_flat_temp = np.concatenate(session_records_w_flat, axis=0)
                    session_weights = session_records_w_flat_temp[nonnan_mask]
                else:
                    session_weights = None

                session_fit = {}

                X = session_records_x_flat_temp.T # (n_regressors, n_neurons) -> (n_neurons, n_regressors)
                Y = session_records_y_flat_temp[:, np.newaxis]

                if fit_intercept:
                    X = sm.add_constant(X)

                if session_weights is None:
                    fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
                else:
                    fit_model = sm.WLS(Y, X, weights=session_weights, missing='none') # nans should be caught above
                session_results = fit_model.fit()

                session_fits[connectivity_metrics_y].append(session_results)

        # Fit on all data at once
        full_fits[connectivity_metrics_y] = full_fit(
            records_x_flat_all, records_y_flat, records_w_flat
        )
        
        # Bootstrap fittings
        if bootstrap_type is not None:    
            for bs_idx in range(n_bootstrap):
    
                bs_session_idxs, bs_group_idxs = get_hier_bootstrap_shuffle(
                    records, bootstrap_groups=bootstrap_groups, seed=BASE_SEED+bs_idx
                )

                bs_records_x_all = []
                bs_records_y_all = []
                bs_records_w_all = []

                for session_idx, bs_session_idx in enumerate(bs_session_idxs):
                    bs_records_x_all.append(
                           [records_x_flat_all[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                    )
                    bs_records_y_all.append(
                           [records_y_flat[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                    )
                    if ps_stats_params['indirect_weight_type'] is not None:
                        bs_records_w_all.append(
                               [records_w_flat[bs_session_idx][bs_group_idx] for bs_group_idx in bs_group_idxs[session_idx]]
                        )

                bs_fit = full_fit(
                    bs_records_x_all, bs_records_y_all, bs_records_w_all
                )
                bs_params[:, cm_y_idx, bs_idx] = bs_fit.params[param_idx_offset:] # Don't include intercept
            
    return full_fits, session_fits, (connectivity_metrics_xs, connectivity_metrics_ys), bs_params

N_GROUPS = 100 # Hardcoding this in for now, since it is true for all sessions we are interested in

def get_hier_bootstrap_shuffle(records, bootstrap_groups=True, seed=None):
    """
    Seed is used here to have uniform bootstrap shuffles across all fits.
    
    First chooses same number of mice from full list of mice, with recplacement.
    Given mice, creates list of all sessions from the chosen mice (if the same 
    mouse is chosen twice, its sessions will be in the twice). Then chooses
    
    
    """
    if seed is not None:
        np.random.seed(seed)
    
    # This code could be made more efficient if we didn't need to reproduce these every time
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])
    mouse_idx_to_session_idx_map = []
    mouse_idx_to_mouse_map = []
    mouse_idx = 0
    for session_idx in range(n_sessions):
        if records['mice'][session_idx] not in mouse_idx_to_mouse_map: # New mouse
            mouse_idx_to_session_idx_map.append([]) 
            mouse_idx_to_mouse_map.append(records['mice'][session_idx])
            mouse_idx += 1
        mouse_idx_to_session_idx_map[mouse_idx_to_mouse_map.index(records['mice'][session_idx])].append(session_idx)
    
    n_mice = len(mouse_idx_to_mouse_map)

    bs_mouse_idxs = np.random.choice(np.arange(n_mice), n_mice, replace=True)
    bs_possible_session_idxs = [] # This might not be the same total number of session idxs, so sample below
    for mouse_idx in bs_mouse_idxs: 
        bs_possible_session_idxs.extend(mouse_idx_to_session_idx_map[mouse_idx])
    
    bs_session_idxs = np.random.choice(bs_possible_session_idxs, n_sessions, replace=True)
    
    bs_group_idxs = [] # For each session, which groups to include
    
    for session_idx in bs_session_idxs:
        groups_idxs = np.arange(N_GROUPS)
        if bootstrap_groups:
            bs_group_idxs.append(np.random.choice(groups_idxs, N_GROUPS, replace=True))
        else:
            bs_group_idxs.append(groups_idxs)
            
    return bs_session_idxs, bs_group_idxs

# %% Cell 41: ### Single session Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Fun...
def get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict, 
    verbose=False,
):
    """
    Looks over several possible neuron-specific metrics to understand
    what explains causal connectivity for a SINGLE session.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'mice': [],
        'session_idxs': [],
        'ps_CC': [],
    }
    for key in ps_stats_params['connectivity_metrics']:
        records[key] = []
    
    n_sessions = len(session_idxs)
    
    for session_idx_idx, session_idx in enumerate(session_idxs):
        day_1_idx = session_idx
            
        records['session_idxs'].append(day_1_idx)
        records['mice'].append(data_dict['data']['mouse'][day_1_idx])
        
        data_to_extract = ['d_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',]
        if 'pairwise_corr_x' in ps_stats_params['connectivity_metrics']: 
            data_to_extract.append('pairwise_corr')
        
        if ps_stats_params['direct_predictor_mode'] is not None:
            data_to_extract.append('resp_ps_pred')
            resp_ps_type = 'resp_ps_pred'
            if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
                data_to_extract.append('resp_ps')
                resp_ps_type = 'resp_ps'
        else:
            data_to_extract.append('resp_ps')
            resp_ps_type = 'resp_ps'
        
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        
        # May need to save fit data for weights later on, saves in same format as records
        if ps_stats_params['indirect_weight_type'] is not None:
            assert ps_stats_params['direct_predictor_mode'] is not None # Doesn't make sense if we're not fitting
            
            if 'indirect_weights_w' not in records: # Initialize first time through
                records['indirect_weights_w'] = []
            group_vals_weights = [] # (n_groups, n_indirect)
            
            indir_mask = data_1['indir_mask'] > 0 # (n_neurons, n_groups,)
            for group_idx in range(indir_mask.shape[-1]):
                r_squareds_1 = data_1['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]
                if ps_stats_params['indirect_weight_type'] in ('rsquared',):
                    group_vals_weights.append(r_squareds_1)
                else:
                    raise NotImplementedError('indirect_weight_type {} not recognized'.format(ps_stats_params['indirect_weight_type']))
            records['indirect_weights_w'].append(group_vals_weights)
        
        for key in ps_stats_params['connectivity_metrics']: # All of these should take (n_neurons) x (n_neurons, n_groups) -> (n_groups)
            second_weights = None # Only needed when _x differs for each direct/indirect pair (like pairwise corr)
            if key[-2:] == '_x': # Metrics that capture laser-stimulated properties
                neuron_cc_1 = data_1[resp_ps_type]
                if key in ('cc_x',):
                    neuron_metric_1 = 'sum'
                elif key in ('pairwise_corr_x',):
                    neuron_metric_1 = data_1['pairwise_corr']
                    second_weights = data_1['indir_mask']
                elif key in ('tuning_x',):
                    neuron_metric_1 = data_1['tuning']
                elif key in ('trial_resp_x',):
                    neuron_metric_1 = data_1['trial_resp']
                elif key in ('post_resp_x',):
                    neuron_metric_1 = data_1['post']
                elif key in ('pre_resp_x',):
                    neuron_metric_1 = data_1['pre']
                elif key in ('cc_mag_x',):
                    neuron_metric_1 = 'abs_sum'
                elif key in ('mask_counts_x',):
                    neuron_metric_1 = 'mask_sum'
#                 group_vals_1 = eval_over_neurons(
#                     neuron_weight_1, data_1[resp_ps_type], ps_stats_params, data_1['dir_mask'],
#                     over_neuron_mode=ps_stats_params['x_over_neuron_mode'],
#                     second_weights=second_weights
#                 )
                
                group_vals_1 = eval_over_neurons(
                    neuron_metric_1, neuron_cc_1, None, None, # Nones are paired session vals 
                    ps_stats_params, data_1['dir_mask'], second_weights=second_weights,
                    over_neuron_mode=ps_stats_params['x_over_neuron_mode'],
                    keep_days_separate=False
                )

            elif key[-2:] == '_y': # Metrics that capture causal connectivity properties
                neuron_cc_1 = data_1[resp_ps_type]
                if key in ('cc_y',):
                    neuron_metric_1 = 'sum'
                elif key in ('tuning_y',):
                    neuron_metric_1 = data_1['tuning']
                elif key in ('trial_resp_y',):
                    neuron_metric_1 = data_1['trial_resp']
                elif key in ('post_resp_y',):
                    neuron_metric_1 = data_1['post']
                elif key in ('pre_resp_y',):
                    neuron_metric_1 = data_1['pre']
                elif key in ('raw_cc_y',):
                    neuron_metric_1 = 'raw_nc'
                elif key in ('cc_mag_y',):
                    neuron_metric_1 = 'abs_sum'
                elif key in ('mask_counts_y',):
                    neuron_metric_1 = 'mask_sum'
#                 group_vals_1 = eval_over_neurons(
#                     neuron_weight_1, data_1[resp_ps_type], ps_stats_params, data_1['indir_mask'],
#                     over_neuron_mode=ps_stats_params['y_over_neuron_mode'],
#                     second_weights=second_weights
#                 )

                group_vals_1 = eval_over_neurons(
                    neuron_metric_1, neuron_cc_1, None, None, # Nones are paired session vals
                    ps_stats_params, data_1['indir_mask'], second_weights=second_weights,
                    over_neuron_mode=ps_stats_params['y_over_neuron_mode'],
                    keep_days_separate=False
                )
                
            else:
                raise NotImplementedError('Connectivity_metric {} not recognized.'.format(key))

            records[key].append(group_vals_1)
    
    return records
            
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

connectivity_metrics = (
    'cc_x',
    'pairwise_corr_x',
#     'tuning_x', 
#     'trial_resp_x', 
    'post_resp_x',
#     'pre_resp_x',
#     'cc_mag_x',
#     'mask_counts_x',
    
    'raw_cc_y',
#     'tuning_y',
#     'trial_resp_y', 
    'post_resp_y',
#     'pre_resp_y',
#     'cc_y',
#     'cc_mag_y',
#     'mask_counts_y',
)

plot_pairs = None # If None, will autofill
# plot_pairs = (
#     ('tuning_x', 'tuning_y'),
#     ('trial_resp_x', 'trial_resp_y'),
# #     ('mask_counts_x', 'mask_counts_y'),
# #     ('mean_tuning_x', 'delta_tuning_y'),
# #     ('mean_tuning_x', 'mean_tuning_y'),
# #     ('mean_tuning_x', 'cc_y'),
# #     ('mean_tuning_x', 'cc_mag_y'),
# #     ('delta_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'mean_trial_resp_y'),
# #     ('delta_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'mean_post_resp_y'),
# #     ('delta_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'mean_pre_resp_y'),
# )

ps_stats_params = {
    'ps_analysis_type': 'single_session',
    'ps_fit_type': 'mlr', # slr, mlr
    
    'neuron_metrics_adjust': None, # None, normalize, standardize
    'pairwise_corr_type': 'behav_full', # trace, trial, pre, post, behav_full, behav_start, behav_end
    
    ### Various ways of computing the over_neuron modes (this can be a tuple to do multiple)
    # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, 
    # matrix_multiply_standardized, neuron_centered_conn_standardized, product_nc
    'x_over_neuron_mode': 'matrix_multiply_sanity', #('matrix_multiply_sanity', 'matrix_multiply_centered', 'means',),
    'y_over_neuron_mode': 'product_nc', #('matrix_multiply_sanity', 'matrix_multiply_centered', 'means',),
    
    ### Fitting photostim parameters
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, average_equal_sessions, minimum
    'modify_direct_weights': True,
    'use_only_predictor_weights': False, # Validation case where only weights are used
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': plot_pairs,
    'group_weights_type': None, # None, direct_resp, direct_resp_norm_sessions
    'indirect_weight_type': 'rsquared', # None, rsquared, minimum_rsquared
    # shuffle_X_within_group, shuffle_X_within_session, shuffle_A_within_group, mean_X_within_session, fake_ps, shuffle_indirect_events
    'validation_types': (), # (), 
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

# if (20, 21) in session_idx_pairs:
#     print('Removing session pair to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

print('Evaluating {} session idxs...'.format(len(session_idxs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])
    
records = get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict,
    verbose=False
)

if ps_stats_params['ps_fit_type'] == 'slr':
    exemplar_session_idx = None
    # Only do full fit if these are single modes, otherwise code not ready to handle
    if type(ps_stats_params['x_over_neuron_mode']) == str and type(ps_stats_params['y_over_neuron_mode']) == str:
        full_fits, session_fits, ps_stats_params = scan_over_connectivity_pairs(
            ps_stats_params, records, exemplar_session_idx=exemplar_session_idx, verbose=True
    )
elif ps_stats_params['ps_fit_type'] == 'mlr':
    print('Records gathered, run additional MLR analysis code below!')

# %% Cell 43: ### Paired sessions Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Fu...
def get_causal_connectivity_metrics_pair(
    ps_stats_params, session_idx_pairs, data_dict, verbose=False,
):
    """
    Looks over several possible neuron-specific metrics to understand
    what explains causal connectivity for a PAIRED sessions.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'mice': [],
        'ps_pair_idx': [],
        'ps_CC': [],
        'cc_x': [], 
    }
    for key in ps_stats_params['connectivity_metrics']:
        records[key] = []
    
    n_pairs = len(session_idx_pairs)
    
    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]
        
        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
            
#         print('{} - Mouse {}, session_idxs {} and {} pass'.format(
#             pair_idx, data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx
#         ))
        records['ps_pair_idx'].append(pair_idx)
        records['mice'].append(data_dict['data']['mouse'][day_1_idx])
        
        data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr']
        
        if ps_stats_params['direct_predictor_mode'] is not None:
            if 'fake_ps' in ps_stats_params['validation_types']:
                raise NotImplementedError()
            
            data_to_extract.append('resp_ps_pred')
            resp_ps_type = 'resp_ps_pred'
            if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
                data_to_extract.append('resp_ps')
                resp_ps_type = 'resp_ps'
        else:
            data_to_extract.append('resp_ps')
            resp_ps_type = 'resp_ps'
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
        
        # Similarity of photostimulation via distances
        records['ps_CC'].append(np.corrcoef(data_1['d_ps'].flatten(), data_2['d_ps'].flatten())[0, 1])

#         delta_tuning = data_2['tuning'] - data_1['tuning']        # change in tuning between days (n_neurons,)
#         mean_tuning = 1/2 * (data_2['tuning'] + data_1['tuning']) # mean tuning across days (n_neurons,)

#         delta_trial_resp = data_2['trial_resp'] - data_1['trial_resp']
#         mean_trial_resp = 1/2 * (data_2['trial_resp'] + data_1['trial_resp'])

#         delta_post_resp = data_2['post'] - data_1['post']
#         mean_post_resp = 1/2 * (data_2['post'] + data_1['post'])

#         delta_pre_resp = data_2['pre'] - data_1['pre']
#         mean_pre_resp = 1/2 * (data_2['pre'] + data_1['pre'])
                                            
        indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
            ps_stats_params, data_1, data_2, verbose=verbose
        )
        
        if ps_stats_params['mask_mode'] in ('constant', 'kayvon_match',): # Same mask for both sessions considered, * functions as AND

            delta_cc = indir_mask_weighted * (data_2[resp_ps_type] - data_1[resp_ps_type])

#             delta_laser_resp = dir_mask_weighted * (data_2[resp_ps_type] - data_1[resp_ps_type])
#             mean_laser_resp = dir_mask_weighted * 1/2 * (data_2[resp_ps_type] + data_1[resp_ps_type])
#             mean_laser_resp = dir_mask_weighted * (data_2[resp_ps_type] + data_1[resp_ps_type]) # No 1/2 for Kayvon's
        elif ps_stats_params['mask_mode'] in ('each_day',): # Individual distance masks for each session
            
            delta_cc = (data_2[resp_ps_type] * data_2['indir_mask'] - data_1[resp_ps_type] * data_1['indir_mask'])        # Change in causal connectivity, (n_neurons, n_groups)

#             delta_laser_resp = (data_2[resp_ps_type] * data_2['dir_mask'] - data_1[resp_ps_type] * data_1['dir_mask'])        # Change in the laser response, (n_neurons, n_groups)
#             mean_laser_resp = 1/2 * (data_2[resp_ps_type] * data_2['dir_mask'] + data_1[resp_ps_type] * data_1['dir_mask'])  # Mean laser response, (n_neurons, n_groups)
        
#         # Always gather raw laser responses for both sessions because this is sometimes used for masking/weighting
#         # Note stores the two pairs sessions as a tuple to stay in sync with other paired metrics
#         records['cc_x'].append((
#             eval_over_neurons('sum', data_1[resp_ps_type], ps_stats_params, dir_mask_weighted),
#             eval_over_neurons('sum', data_2[resp_ps_type], ps_stats_params, dir_mask_weighted),
#         ))
        
        # May need to save fit data for weights later on, saves in same format as other records
        if ps_stats_params['indirect_weight_type'] is not None:
            assert ps_stats_params['direct_predictor_mode'] is not None # Doesn't make sense if we're not fitting
            
            if 'indirect_weights_w' not in records: # Initialize first time through
                records['indirect_weights_w'] = []
            group_vals_weights = [] # (n_groups, n_indirect)
            
            indir_mask = indir_mask_weighted > 0 # (n_neurons, n_groups,)
            for group_idx in range(indir_mask.shape[-1]):
                r_squareds_1 = data_1['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]
                r_squareds_2 = data_2['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]            
                if ps_stats_params['indirect_weight_type'] in ('minimum_rsquared',):
                    group_vals_weights.append(np.minimum(r_squareds_1, r_squareds_2))
                else:
                    raise NotImplementedError('indirect_weight_type {} not recognized'.format(ps_stats_params['indirect_weight_type']))
            records['indirect_weights_w'].append(group_vals_weights)
        
        for key in ps_stats_params['connectivity_metrics']: # All of these should take (n_neurons) x (n_neurons, n_groups) -> (n_groups)
            second_weights = None # Only needed when _x differs for each direct/indirect pair (like pairwise corr)
            if key[-2:] == '_x': # Metrics that capture laser-stimulated properties
                neuron_cc_1 = 1/2 * data_1[resp_ps_type]
                neuron_cc_2 = 1/2 * data_2[resp_ps_type]
#                         resp_weight = delta_laser_resp
                if key in ('delta_laser_x',):
                    neuron_metric_1 = 'sum'
                    neuron_metric_2 = 'sum'
                    # Change to delta metric
                    neuron_cc_1 = -1 * data_1[resp_ps_type]
                    neuron_cc_2 = data_2[resp_ps_type]
                elif key in ('laser_resp_x',):
                    neuron_metric_1 = 'sum'
                    neuron_metric_2 = 'sum'
                elif key in ('delta_pairwise_corr_x',):
                    neuron_metric_1 = -1 * data_1['pairwise_corr']
                    neuron_metric_2 = data_2['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('pairwise_corr_1_x',): # Just corr from Day 1
                    neuron_metric_1 = 1/2 * data_1['pairwise_corr']
                    neuron_metric_2 = 1/2 * data_1['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('pairwise_corr_2_x',): # Just corr from Day 2
                    neuron_metric_1 = 1/2 * data_2['pairwise_corr']
                    neuron_metric_2 = 1/2 * data_2['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('delta_tuning_x',):
                    neuron_metric_1 = -1 * data_1['tuning']
                    neuron_metric_2 = data_2['tuning']
                elif key in ('mean_tuning_x',):
                    neuron_metric_1 = 1/2 * data_1['tuning']
                    neuron_metric_2 = 1/2 * data_2['tuning']
                elif key in ('delta_trial_resp_x',):
                    neuron_metric_1 = -1 * data_1['trial_resp']
                    neuron_metric_2 = data_2['trial_resp']
                elif key in ('mean_trial_resp_x',):
                    neuron_metric_1 = 1/2 * data_1['trial_resp']
                    neuron_metric_2 = 1/2 * data_2['trial_resp']
                elif key in ('delta_post_resp_x',):
                    neuron_metric_1 = -1 * data_1['post']
                    neuron_metric_2 = data_2['post']
                elif key in ('mean_post_resp_x',):
                    neuron_metric_1 = 1/2 * data_1['post']
                    neuron_metric_2 = 1/2 * data_2['post']
                elif key in ('delta_pre_resp_x',):
                    neuron_metric_1 = -1 * data_1['pre']
                    neuron_metric_2 = data_2['pre']
                elif key in ('mean_pre_resp_x',):
                    neuron_metric_1 = 1/2 * data_1['pre']
                    neuron_metric_2 = 1/2 * data_2['pre']
                elif key in ('laser_resp_mag_x',):
                    neuron_metric_1 = 'abs_sum'
                    neuron_metric_2 = 'abs_sum'
                elif key in ('mask_counts_x',):
                    neuron_metric_1 = 'mask_sum'
                    neuron_metric_2 = 'mask_sum'

                group_vals = eval_over_neurons(
                    neuron_metric_1, neuron_cc_1, neuron_metric_2, neuron_cc_2,
                    ps_stats_params, dir_mask_weighted, second_weights=second_weights,
                    over_neuron_mode=ps_stats_params['x_over_neuron_mode'],
                    keep_days_separate=True
                )

            elif key[-2:] == '_y': # Metrics that capture causal connectivity properties
                neuron_cc_1 = -1 * data_1[resp_ps_type]
                neuron_cc_2 = data_2[resp_ps_type]

                if key in ('raw_delta_cc_y',):
                    neuron_metric_1 = 'raw_nc'
                    neuron_metric_2 = 'raw_nc'
                elif key in ('delta_tuning_y',): # Equiv to 5c's W
                    neuron_metric_1 = -1 * data_1['tuning']
                    neuron_metric_2 = data_2['tuning']
                elif key in ('mean_tuning_y',):
                    neuron_metric_1 = 1/2 * data_1['tuning']
                    neuron_metric_2 = 1/2 * data_2['tuning'] 
                elif key in ('delta_trial_resp_y',):
                    neuron_metric_1 = -1 * data_1['trial_resp']
                    neuron_metric_2 = data_2['trial_resp']
                elif key in ('mean_trial_resp_y',):
                    neuron_metric_1 = 1/2 * data_1['trial_resp']
                    neuron_metric_2 = 1/2 * data_2['trial_resp'] 
                elif key in ('delta_post_resp_y',):
                    neuron_metric_1 = -1 * data_1['post']
                    neuron_metric_2 = data_2['post']
                elif key in ('mean_post_resp_y',):
                    neuron_metric_1 = 1/2 * data_1['post']
                    neuron_metric_2 = 1/2 * data_2['post'] 
                elif key in ('delta_pre_resp_y',):
                    neuron_metric_1 = -1 * data_1['pre']
                    neuron_metric_2 = data_2['pre']
                elif key in ('mean_pre_resp_y',):
                    neuron_metric_1 = 1/2 * data_1['pre']
                    neuron_metric_2 = 1/2 * data_2['pre'] 
                elif key in ('delta_cc_y',):
                    neuron_metric_1 = 'sum'
                    neuron_metric_2 = 'sum'
                elif key in ('delta_cc_mag_y',):
                    neuron_metric_1 = 'abs_sum'
                    neuron_metric_2 = 'abs_sum'
                elif key in ('mask_counts_y',):
                    neuron_metric_1 = 'mask_sum'
                    neuron_metric_2 = 'mask_sum'

                group_vals = eval_over_neurons(
                    neuron_metric_1, neuron_cc_1, neuron_metric_2, neuron_cc_2,
                    ps_stats_params, indir_mask_weighted, second_weights=second_weights,
                    over_neuron_mode=ps_stats_params['y_over_neuron_mode'],
                    keep_days_separate=False
                )
            else:
                raise NotImplementedError('Connectivity_metric {} not recognized.'.format(key))

            # Some of these can be nans because a group might not have valid direct/indirect mask
            # Currently case for BCI35 group 52, BCI37, group 11
            records[key].append(group_vals)
        
    return records
                
connectivity_metrics = (
    'delta_laser_x', 
    'laser_resp_x',
    'delta_pairwise_corr_x',
#     'pairwise_corr_1_x',
#     'pairwise_corr_2_x',
#     'delta_tuning_x',
#     'mean_tuning_x',
#     'delta_trial_resp_x',
#     'mean_trial_resp_x',
    'delta_post_resp_x',
    'mean_post_resp_x',
#     'delta_pre_resp_x',
#     'mean_pre_resp_x',
#     'laser_resp_x',
#     'laser_resp_mag_x',
#     'mask_counts_x',
    
    'raw_delta_cc_y',
#     'delta_tuning_y',
#     'mean_tuning_y',
#     'delta_trial_resp_y',
#     'mean_trial_resp_y',
    'delta_post_resp_y',
    'mean_post_resp_y',
#     'delta_pre_resp_y',
#     'mean_pre_resp_y',
#     'delta_cc_y',
#     'delta_cc_mag_y',
#     'mask_counts_y',
)

plot_pairs = None # If None, will autofill
# plot_pairs = (
#     ('delta_tuning_x', 'delta_tuning_y'),
# #     ('mean_trial_resp_x', 'delta_tuning_y'),
# #     ('mask_counts_x', 'mask_counts_y'),
# #     ('mean_tuning_x', 'delta_tuning_y'),
# #     ('mean_tuning_x', 'mean_tuning_y'),
# #     ('mean_tuning_x', 'cc_y'),
# #     ('mean_tuning_x', 'cc_mag_y'),
# #     ('delta_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'mean_trial_resp_y'),
# #     ('delta_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'mean_post_resp_y'),
# #     ('delta_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'mean_pre_resp_y'),
# )

ps_stats_params = {
    'ps_analysis_type': 'paired',
    'ps_fit_type': 'mlr', # slr, mlr
    
    'neuron_metrics_adjust': None, # None, normalize, standardize
    'pairwise_corr_type': 'behav_full', # trace, trial, pre, post, behav_full, behav_start, behav_end
    
    ### Various ways of computing the over_neuron modes for x and y ###
    # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, 
    # matrix_multiply_standardized, neuron_centered_conn_standardized, product_nc
    'x_over_neuron_mode': 'matrix_multiply_sanity',
    'y_over_neuron_mode': 'product_nc',
    
    ### Fitting photostim parameters
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average_equal_sessions', # average, average_equal_sessions, minimum
    'modify_direct_weights': True,
    'use_only_predictor_weights': False, # Validation case where only weights are used
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': plot_pairs,
    'group_weights_type': None, # None, direct_resp, direct_resp_norm_sessions
    'indirect_weight_type': 'minimum_rsquared', # None, rsquared, minimum_rsquared
    # shuffle_X_within_group, shuffle_X_within_session, shuffle_A_within_group, mean_X_within_session, fake_ps, shuffle_indirect_events
    'validation_types': (), # (), 

    'plot_up_mode': 'all', # all, significant, None; what to actually create plots for 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

# print(STOP_HERE_FOR_NOW)
    
records = get_causal_connectivity_metrics_pair(
    ps_stats_params, session_idx_pairs, data_dict, verbose=False,
)

if ps_stats_params['ps_fit_type'] == 'slr':
    exemplar_session_idx = None

    full_fits, session_fits, ps_stats_params = scan_over_connectivity_pairs(
        ps_stats_params, records, exemplar_session_idx=exemplar_session_idx, verbose=True
    )
elif ps_stats_params['ps_fit_type'] == 'mlr':
    print('Records gathered, run additional MLR analysis code below!')

# %% Cell 45: ### MLR fit/plot Further MLR fitting/plotting code for both single and paired sessions Fit raw A and...
bootstrap_type = None #'hierarchy'
n_bootstrap = 100
BASE_SEED = 100 # For bootstrap

fit_intercept = True
standardize_xs = True
standardize_ys = True

full_fits, session_fits, (connectivity_metrics_xs, connectivity_metrics_ys,), bs_params  = fit_all_xs_at_once(
    records, ps_stats_params, fit_intercept=fit_intercept, standardize_xs=standardize_xs,
    standardize_ys=standardize_ys, bootstrap_type=bootstrap_type, n_bootstrap=n_bootstrap, 
    verbose=True
)

print('Done!')

# %% Cell 47: Run this for plotting up results from fit_all_xs_at_once without bootstrapping
MIN_P = 1e-300
MAX_PLOT_LOG10P = 25 # 

if 'valid_full_fits' not in locals():
    valid_full_fits = None

X_PERCENTILES = (10, 90) # For fit plots, determines range of x

param_idx_offset = 1 if fit_intercept else 0
n_sessions = len(records[list(records.keys())[0]])

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
    for session_idx in range(n_sessions):
        session_ps[:, cm_y_idx, session_idx] = (
            session_fits[connectivity_metrics_y][session_idx].pvalues[param_idx_offset:] # Don't include intercept
        )
        session_params[:, cm_y_idx, session_idx] = (
            session_fits[connectivity_metrics_y][session_idx].params[param_idx_offset:] # Don't include intercept
        )
        session_stderrs[:, cm_y_idx, session_idx] = (
            session_fits[connectivity_metrics_y][session_idx].bse[param_idx_offset:] # Don't include intercept
        )
    aggregate_ps[:, cm_y_idx, 0] = (
        full_fits[connectivity_metrics_y].pvalues[param_idx_offset:] # Don't include intercept
    )
    aggregate_params[:, cm_y_idx, 0] = (
        full_fits[connectivity_metrics_y].params[param_idx_offset:] # Don't include intercept
    )
    aggregate_stderrs[:, cm_y_idx, 0] = (
        full_fits[connectivity_metrics_y].bse[param_idx_offset:] # Don't include intercept
    )
    
    if valid_full_fits is not None:
        valid_ps[:, cm_y_idx, 0] = (
            valid_full_fits[connectivity_metrics_y].pvalues[param_idx_offset:] # Don't include intercept
        )
        valid_params[:, cm_y_idx, 0] = (
            valid_full_fits[connectivity_metrics_y].params[param_idx_offset:] # Don't include intercept
        )
        valid_stderrs[:, cm_y_idx, 0] = (
            valid_full_fits[connectivity_metrics_y].bse[param_idx_offset:] # Don't include intercept
        )

# Enforce minimum values on ps
session_ps = np.where(session_ps==0., MIN_P, session_ps)
aggregate_ps = np.where(aggregate_ps==0., MIN_P, aggregate_ps)
valid_ps = np.where(valid_ps==0., MIN_P, valid_ps)
        
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
                np.nanpercentile(x_vals, X_PERCENTILES[0]),
                np.nanpercentile(x_vals, X_PERCENTILES[1]), 10
            )
            y_range = (
                session_params[cm_x_idx, cm_y_idx, session_idx] * x_range
            )
            
            if fit_intercept:
                y_range += session_fits[connectivity_metrics_y][session_idx].params[0]
                
            if ps_stats_params['ps_analysis_type'] in ('single_session',):
                line_color = c_vals[SESSION_COLORS[session_idx]]
            elif ps_stats_params['ps_analysis_type'] in ('paired',):
                line_color = c_vals[PAIR_COLORS[session_idx]]  
            
            ax3s[cm_x_idx, cm_y_idx].plot(x_range, y_range, color=line_color)
        
        x_range_all = np.linspace(
                np.nanpercentile(np.concatenate(all_x_vals), X_PERCENTILES[0]),
                np.nanpercentile(np.concatenate(all_x_vals), X_PERCENTILES[1]), 10
        )        
        y_range_all = (
            aggregate_params[cm_x_idx, cm_y_idx] * x_range_all
        )
        if fit_intercept:
            y_range_all += full_fits[connectivity_metrics_y].params[0]

        ax3s[cm_x_idx, cm_y_idx].plot(x_range_all, y_range_all, color='k')
        
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
        

# %% Cell 49: Run this for an analysis on bootstrapped metrics (not the best plots, work in progress)
n_cm_x = len(connectivity_metrics_xs)
n_cm_y = len(connectivity_metrics_ys)

bs_ps = np.ones((n_cm_x, n_cm_y, n_bootstrap))

fig1, ax1s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # -log10(p-values)
fig2, ax2s = plt.subplots(n_cm_x, n_cm_y, figsize=(12, 12)) # parameters

bar_locs = np.array((0.,))

for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):  
            
#         all_ps = -1 * np.log10(all_ps)
#         if max(all_ps) > max_p_for_this_x:
#             max_p_for_this_x = max(all_ps)
        
        frac_pos = np.sum(bs_params[cm_x_idx, cm_y_idx, :] > 0) / n_bootstrap
        ax1s[cm_x_idx, cm_y_idx].scatter(np.array((0., 1.)), np.array((frac_pos, 1 - frac_pos,)), color='k')
#         ax1s[cm_x_idx, cm_y_idx].errorbar(
#             bar_locs[point_idx], all_params[point_idx], yerr=all_stderrs[point_idx], 
#             color=bar_colors[point_idx], linestyle='None'
#         )
        ax2s[cm_x_idx, cm_y_idx].scatter(bar_locs, np.nanmean(bs_params[cm_x_idx, cm_y_idx, :]), color='k', marker='_')
        ax2s[cm_x_idx, cm_y_idx].errorbar(
            bar_locs, np.nanmean(bs_params[cm_x_idx, cm_y_idx, :]), 
            yerr=np.nanstd(bs_params[cm_x_idx, cm_y_idx, :]), 
            color='k', linestyle='None'
        )
        
        ax1s[cm_x_idx, cm_y_idx].set_ylim((-0.1, 1.1))
        ax1s[cm_x_idx, cm_y_idx].axhline(0., color='grey', zorder=-5, linewidth=1.0)
        ax1s[cm_x_idx, cm_y_idx].axhline(.5, color='grey', zorder=-5, linewidth=1.0)
        ax1s[cm_x_idx, cm_y_idx].axhline(1., color='grey', zorder=-5, linewidth=1.0)
        ax2s[cm_x_idx, cm_y_idx].axhline(0., color='grey', zorder=-5, linewidth=1.0)
        
        for axs in (ax1s, ax2s,):
            axs[cm_x_idx, cm_y_idx].set_xticks(())
            if cm_x_idx == n_cm_x - 1:
                axs[cm_x_idx, cm_y_idx].set_xlabel(connectivity_metrics_y, fontsize=8)
            if cm_y_idx == 0:
                axs[cm_x_idx, cm_y_idx].set_ylabel(connectivity_metrics_x, fontsize=8)

# %% Cell 51: Sometimes want to fit against some validation data, run this to save full_fits as validation
import copy
valid_full_fits = copy.deepcopy(full_fits)
print('Fit assigned as validation.')

# %% Cell 53: # CN specific changes  Initialize some parameters that are used in all tests below. Also function fo...
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

# %% Cell 55: ### Single session
# ps_stats_params['pairwise_corr_type'] = 'trial' # trace, trial

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))
    
session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

print('Evaluating {} sessions...'.format(len(session_idxs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

cn_idxs = []
avg_directs = []
avg_directs_pred = []
avg_indirects = []
avg_indirects_pred = []

n_sessions = len(session_idxs)
    
for session_idx_idx, session_idx in enumerate(session_idxs):
    day_1_idx = session_idx
    print('Session idx: {}'.format(session_idx))
    
    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 'resp_ps', 
                       'resp_ps_pred', 'mean_trial_activity', 'trial_start_metrics_changes',]

    data_1 = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)
    
    cn_idxs.append(data_dict['data']['conditioned_neuron'][session_idx] - 1) # -1 to correct for MatLab indexing
    
    # Anything that does not make it past the mask is automatically a nan
    direct_mask = np.where(data_1['dir_mask'] > 0, 1., np.nan)
    indirect_mask = np.where(data_1['indir_mask'] > 0, 1., np.nan)
    
    avg_directs.append(direct_mask * data_1['resp_ps'])
    avg_directs_pred.append(direct_mask * data_1['resp_ps_pred'])
    
    avg_indirects.append(indirect_mask * data_1['resp_ps'])
    avg_indirects_pred.append(indirect_mask * data_1['resp_ps_pred'])
    
    indirect_counts = np.sum(indir_mask_weighted > 0, axis=-1)
    print('CN indirect count:', indirect_counts[cn_idxs[-1]])

del data_1

# %% Cell 57: ### Paired (no CC) Just plot some paired statistics to check if distributions looks different for ce...
# ps_stats_params['pairwise_corr_type'] = 'trial' # trace, trial

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

delta_correlations = []
delta_tunings = []
posts_1 = []
posts_2 = []
delta_trial_resps = []
day_2_ts_metrics_changes = []
day_2_cn_idxs = []
day_2_dist_to_cn = []

day_1_trial_activities = [] # For determining random CN draws
day_1_tunings = []


for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    print('Pair idx: {}'.format(pair_idx))
    
    day_2_idx = session_idx_pair[1]
    day_1_idx = session_idx_pair[0]

    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 
                       'trial_start_metrics_changes', 'mean_trial_activity']

#     if ps_stats_params['direct_predictor_mode'] is not None:
#         data_to_extract.append('resp_ps_pred')
#         resp_ps_type = 'resp_ps_pred'
#         if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
#             data_to_extract.append('resp_ps')
#             resp_ps_type = 'resp_ps'
#     else:
#         data_to_extract.append('resp_ps')
#         resp_ps_type = 'resp_ps'
        
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
    data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
    
    delta_correlations.append(data_2['pairwise_corr'] - data_1['pairwise_corr'])
    
    print('{} Pair idx {}'.format(data_dict['data']['mouse'][day_2_idx], pair_idx), np.nanmean(data_2['pairwise_corr']))
    delta_tunings.append(data_2['tuning'] - data_1['tuning'])
    delta_trial_resps.append(data_2['trial_resp'] - data_1['trial_resp'])
    
    posts_1.append(data_1['post'])
    posts_2.append(data_2['post'])
    
    day_2_ts_metrics_changes.append(data_2['trial_start_metrics_changes'])
    day_2_cn_idxs.append(data_dict['data']['conditioned_neuron'][day_2_idx] - 1) # -1 to correct for MatLab indexing
    day_2_dist_to_cn.append(data_dict['data']['dist'][day_2_idx])
    
    day_1_trial_activities.append(data_1['mean_trial_activity'])
    day_1_tunings.append(data_1['tuning'])
    
#     for plot_1, plot_2, ax, ax_bounds in zip(plot_1s, plot_2s, axs, axs_bounds):
#         ax.scatter(
#             plot_1, plot_2, color=c_vals[PAIR_COLORS[pair_idx]], marker = '.', alpha=0.3
#         )

#         if min(np.min(plot_1), np.min(plot_2)) > ax_bounds[0]:
#             ax_bounds[0] = min(np.min(plot_1), np.min(plot_2))
#         if max(np.max(plot_1), np.max(plot_2)) > ax_bounds[-1]:
#             ax_bounds[-1] = max(np.max(plot_1), np.max(plot_2))

#         add_identity(ax, color='k', zorder=5, linestyle='dashed')
        
#         print(' id added')

del data_1
del data_2

# %% Cell 58: Code starts with: n_pairs = len(delta_tunings)
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

# %% Cell 59: Code starts with: random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_a...
random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity, similar_prev_day_activity_tuning
random_cn_percent_pass = 0.2

n_pairs = len(day_2_cn_idxs)

fig4, ax4s = plt.subplots(2, 6, figsize=(24, 8))
fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))
fig6, ax6s = plt.subplots(2, 6, figsize=(24, 8))
fig7, ax7s = plt.subplots(2, 6, figsize=(24, 8))

n_random_neurons = 1000
cn_idx_idx = 0

for ts_metric, axs in zip(
    ('tuning', 'trial_resp', 'pre', 'post'),
    (ax4s, ax5s, ax6s, ax7s)
):
    percentiles = np.zeros((n_random_neurons, n_pairs,))
    
    for pair_idx, ax in zip(range(n_pairs), axs.flatten()):
        cn_idx = int(day_2_cn_idxs[pair_idx])
        ts_slopes = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']
        change_sort_idxs = np.argsort(ts_slopes)
        
        n_neurons = ts_slopes.shape[0]
        neuron_idxs = np.arange(n_neurons)
        cn_idx_sort_loc = np.where(change_sort_idxs == cn_idx)[0][0]

        ax.scatter(neuron_idxs, ts_slopes[change_sort_idxs], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
        ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, ts_slopes[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        ax.axhline(0.0, color='lightgrey', zorder=-5)
        
        # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
        neuron_percentiles = np.zeros((n_neurons,))
        for neuron_idx in range(n_neurons):
            neuron_percentiles[neuron_idx] = np.where(change_sort_idxs == neuron_idx)[0][0] / (n_neurons - 1)
            
        percentiles[cn_idx_idx, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th
        
        candidate_random_cns = get_candidate_random_cns(
            cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass,
            prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
        )
        
        for neuron_idx in range(n_random_neurons - 1): 
            percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[np.random.choice(candidate_random_cns)]
        
    # Mean across pairs
    _, bins, _ = axs.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
    sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
    axs.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                              label='CN Mean, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    # Median across pairs
    _, bins, _ = axs.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
    sort_idxs = np.argsort(np.median(percentiles, axis=-1))
    axs.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                               label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

    axs.flatten()[-1].legend()
                                        
fig4.suptitle('Tuning Slopes')
fig5.suptitle('Trial Response Slopes')
fig6.suptitle('Pre Trial Response Slopes')
fig7.suptitle('Post Trial Response Slopes')

# %% Cell 61: Distance to CN versus trial-start (within session) change plots.
random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity, similar_prev_day_activity_tuning
random_cn_percent_pass = 0.2

n_pairs = len(day_2_cn_idxs)

fig4, ax4s = plt.subplots(2, 6, figsize=(24, 8))
fig5, ax5s = plt.subplots(2, 6, figsize=(24, 8))
fig6, ax6s = plt.subplots(2, 6, figsize=(24, 8))
fig7, ax7s = plt.subplots(2, 6, figsize=(24, 8))

n_random_neurons = 1000
cn_idx_idx = 0

for ts_metric, axs in zip(
    ('tuning', 'trial_resp', 'pre', 'post'),
    (ax4s, ax5s, ax6s, ax7s)
):   
    xs = []
    ys = []
    
    for pair_idx, ax in zip(range(n_pairs), axs.flatten()):
        cn_idx = int(day_2_cn_idxs[pair_idx])
        ts_slopes = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']
        dist_to_cn = day_2_dist_to_cn[pair_idx]

        ax.scatter(dist_to_cn, ts_slopes, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
        ax.scatter(dist_to_cn[cn_idx], ts_slopes[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
        slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, ts_slopes, ax=ax, color=c_vals[PAIR_COLORS[pair_idx]])
        
#         ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, ts_slopes[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        ax.axhline(0.0, color='lightgrey', zorder=-5)
        ax.legend()
        
        if pair_idx == 0 or pair_idx == 6:
            ax.set_ylabel('{} slope'.format(ts_metric))
        if pair_idx in (6, 7, 8, 9, 10):
            ax.set_xlabel('Distance to CN (um)')
    
#         axs.flatten()[-1].scatter(ts_slopes[cn_idx], slope, marker='.', color=c_vals[PAIR_COLORS[pair_idx]])
        axs.flatten()[-1].errorbar(ts_slopes[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        xs.append(ts_slopes[cn_idx])
        ys.append(slope)
        
    axs.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    axs.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
    axs.flatten()[-1].set_xlabel('CN change')
    axs.flatten()[-1].set_ylabel('Distance vs. change slope')
    add_regression_line(xs, ys, ax=axs.flatten()[-1], color='k')
                                        
fig4.suptitle('Distance to CN vs. (within-session) tuning slopes')
fig5.suptitle('Distance to CN vs. (within-session) trial response slopes')
fig6.suptitle('Distance to CN vs. (within-session) pre trial response slopes')
fig7.suptitle('Distance to CN vs. (within-session) post trial response slopes')

# %% Cell 63: Distance to CN vs. generic neuron metric plot
random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity, similar_prev_day_activity_tuning
random_cn_percent_pass = 0.2

n_pairs = len(day_2_cn_idxs)
ts_metric = 'post'

fig1, ax1s = plt.subplots(2, 6, figsize=(24, 8))
fig2, ax2s = plt.subplots(2, 6, figsize=(24, 8))

n_random_neurons = 1000
cn_idx_idx = 0

percentiles = np.zeros((n_random_neurons, n_pairs,))

xs = []
ys = []

for pair_idx, ax, axp in zip(range(n_pairs), ax1s.flatten(), ax2s.flatten()):
    cn_idx = int(day_2_cn_idxs[pair_idx])
    
#     neuron_metric = np.nansum(delta_correlations[pair_idx], axis=-1)
    neuron_metric = np.nansum(np.abs(delta_correlations[pair_idx]), axis=-1) # Looks quite significant
#     neuron_metric = nan_matmul(posts_2[pair_idx], delta_correlations[pair_idx]) # Not really significant
    
#     top_changes = posts_2[pair_idx] > np.percentile(posts_2[pair_idx], 66)
#     top_changes = posts_2[pair_idx] < np.percentile(posts_2[pair_idx], 33)
#     neuron_metric = np.nansum(delta_correlations[pair_idx][:, top_changes], axis=-1)
    sort_idxs = np.argsort(neuron_metric)

    n_neurons = neuron_metric.shape[0]
    neuron_idxs = np.arange(n_neurons)
    cn_idx_sort_loc = np.where(sort_idxs == cn_idx)[0][0]

    ax.scatter(neuron_idxs, neuron_metric[sort_idxs], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
    ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, neuron_metric[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
    ax.axhline(0.0, color='lightgrey', zorder=-5)
        
    # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
    neuron_percentiles = np.zeros((n_neurons,))
    for neuron_idx in range(n_neurons):
        neuron_percentiles[neuron_idx] = np.where(sort_idxs == neuron_idx)[0][0] / (n_neurons - 1)

    percentiles[0, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th
    
    candidate_random_cns = get_candidate_random_cns(
        cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass,
        prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
    )
    
    for neuron_idx in range(n_random_neurons - 1): 
        percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[np.random.choice(candidate_random_cns)]
    
    dist_to_cn = day_2_dist_to_cn[pair_idx]
    
    axp.scatter(dist_to_cn, neuron_metric, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
    axp.scatter(dist_to_cn[cn_idx], neuron_metric[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
    slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, neuron_metric, ax=axp, color=c_vals[PAIR_COLORS[pair_idx]])
    axp.legend()
    
    if pair_idx == 0 or pair_idx == 6:
        axp.set_ylabel('Neuron metric')
    if pair_idx in (6, 7, 8, 9, 10):
        axp.set_xlabel('Distance to CN (um)')

#         axs.flatten()[-1].scatter(ts_slopes[cn_idx], slope, marker='.', color=c_vals[PAIR_COLORS[pair_idx]])
    ax2s.flatten()[-1].errorbar(neuron_metric[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
    xs.append(neuron_metric[cn_idx])
    ys.append(slope)
        
ax2s.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
ax2s.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
ax2s.flatten()[-1].set_xlabel('CN neuron metric')
ax2s.flatten()[-1].set_ylabel('Distance vs. neuron metric slope')
add_regression_line(xs, ys, ax=ax2s.flatten()[-1], color='k')
        
# Mean across pairs
_, bins, _ = ax1s.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
ax1s.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                          label='CN Mean, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

# Median across pairs
_, bins, _ = ax1s.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
sort_idxs = np.argsort(np.median(percentiles, axis=-1))
ax1s.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                           label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

ax1s.flatten()[-1].legend()

# %% Cell 65: ### Paired (w CC) Similar to scan above, but actually gets the predicted responses too so takes a lo...
# Assumes this just uses the ps_stats_params from above.

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

day_2_ts_metrics_changes = []

day_1_cn_idxs = []
day_2_cn_idxs = []

direct_masks = [] # Same for both days
indirect_masks = []

day_1_resp_pss = []
day_2_resp_pss = []
day_1_resp_ps_preds = []
day_2_resp_ps_preds = []

day_2_dist_to_cn = []

day_1_rsquared_indirects = []
day_2_rsquared_indirects = []
min_rsquared_indirects = []

day_1_trial_activities = [] # For determining random CN draws
day_1_tunings = []

group_day_2_post = []

# delta_corrs = []

# for pair_idx, session_idx_pair in enumerate(session_idx_pairs[0:3]):
for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    print('Pair idx: {}'.format(pair_idx))
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]

    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

    data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr', 'resp_ps', 
                       'resp_ps_pred', 'mean_trial_activity', 'trial_start_metrics_changes',]
        
    data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
    data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
    
    day_2_ts_metrics_changes.append(data_2['trial_start_metrics_changes'])
    
    day_1_cn_idxs.append(int(data_dict['data']['conditioned_neuron'][day_1_idx]) - 1) # -1 to correct for MatLab indexing
    day_2_cn_idxs.append(int(data_dict['data']['conditioned_neuron'][day_2_idx]) - 1) # -1 to correct for MatLab indexing
    day_2_dist_to_cn.append(data_dict['data']['dist'][day_2_idx])
    
    indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
        ps_stats_params, data_1, data_2, verbose=False
    )
    
    # Anything that does not make it past the mask is automatically a nan
    direct_mask = np.where(dir_mask_weighted > 0, 1., np.nan)
    indirect_mask = np.where(indir_mask_weighted > 0, 1., np.nan)
    
    direct_masks.append(direct_mask)
    indirect_masks.append(indirect_mask)
    
    day_1_resp_pss.append(data_1['resp_ps'])
    day_2_resp_pss.append(data_2['resp_ps'])
    day_1_resp_ps_preds.append(data_1['resp_ps_pred'])
    day_2_resp_ps_preds.append(data_2['resp_ps_pred'])
    
    # avg_directs.append(direct_mask * 1/2 * (data_2['resp_ps'] + data_1['resp_ps']))
    # avg_directs_pred.append(direct_mask * 1/2 * (data_2['resp_ps_pred'] + data_1['resp_ps_pred']))
    
#     day_1_indirects.append(indirect_mask * data_1['resp_ps'])
#     day_2_indirects.append(indirect_mask * data_2['resp_ps'])
    
#     day_1_indirects.append(indirect_mask * data_1['resp_ps'])
#     day_2_indirects.append(indirect_mask * data_2['resp_ps'])
    
#     delta_indirects.append(indirect_mask * (data_2['resp_ps'] - data_1['resp_ps']))
#     delta_indirects_pred.append(indirect_mask * (data_2['resp_ps_pred'] - data_1['resp_ps_pred']))
    
#     # No indirect masking
#     delta_cc_raw.append()
#     delta_cc_pred.append()
    
    day_1_rsquared_indirects.append(indirect_mask * data_1['resp_ps_pred_extras']['r_squareds'])
    day_2_rsquared_indirects.append(indirect_mask * data_2['resp_ps_pred_extras']['r_squareds'])
    min_rsquared_indirects.append(
        indirect_mask *
        np.minimum(data_1['resp_ps_pred_extras']['r_squareds'], data_2['resp_ps_pred_extras']['r_squareds'])
    )
    
    # indirect_counts.append(np.sum(indir_mask_weighted > 0, axis=-1)) # How many times each neuron is indirect
    # print('CN indirect count:', indirect_counts[-1][day_2_cn_idxs[-1]])
    
#     delta_corrs.append(
#         nan_matmul(data_2['pairwise_corr'], direct_mask * data_2['resp_ps']) -
#         nan_matmul(data_1['pairwise_corr'], direct_mask * data_1['resp_ps'])
#     )
    
    # group_day_2_post.append(nan_matmul(data_2['post'], direct_mask * data_2['resp_ps'])) 
    
    # For CN choices
    day_1_trial_activities.append(data_1['mean_trial_activity'])
    day_1_tunings.append(data_1['tuning'])
    
del data_1
del data_2

# %% Cell 67: This is not a CN specific test.
pair_idx = 8

n_bins = 5

group_sort = 'delta_tuning' 
neuron_sort = None # delta_cc, delta_tuning

n_groups = day_2_indirects[pair_idx].shape[-1]
group_bins = np.round(np.linspace(0, n_groups, n_bins+1)).astype(np.int32)

n_neurons = day_2_indirects[pair_idx].shape[0]
neuron_bins = np.round(np.linspace(0, n_neurons, n_bins+1)).astype(np.int32)

print('group_bins:', group_bins)
print('neuron_bins:', neuron_bins)

# direct_resp = direct_masks[pair_idx] * 1/2 * (
#     day_1_resp_pss[pair_idx] + day_2_resp_pss[pair_idx]
# ) 

# direct_resp = direct_masks[pair_idx] * 1/2 * (
#     day_1_resp_ps_preds[pair_idx] + day_2_resp_ps_preds[pair_idx]
# ) 

direct_resp = direct_masks[pair_idx] / np.nansum(direct_masks[pair_idx], axis=0) # Just include all direct, normalizing for count

# indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_pss[pair_idx] - day_1_resp_pss[pair_idx])
# indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_ps_preds[pair_idx] - day_1_resp_ps_preds[pair_idx])
indirect_delta_resp = indirect_masks[pair_idx] * (day_2_resp_ps_preds[pair_idx])

if group_sort == 'delta_tuning':
    # ts_metric = 'tuning'
    ts_metric = 'post'
    # ts_metric = 'trial_resp'
    # ts_metric = 'pre'
    ts_metric_change = day_2_ts_metrics_changes[pair_idx][ts_metric]['slope']
        
    group_metric = nan_matmul(ts_metric_change, direct_resp)
else:
    raise NotImplementedError('group_sort {} not recognized!')

neuron_metric = None
if neuron_sort is not None:
    if neuron_sort == 'delta_cc':
        neuron_metric = np.nanmean(indirect_delta_resp, axis=-1)
    else:
        raise NotImplementedError('neuron_sort {} not recognized!')

# group_day_2_post[pair_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

group_sort_idxs = np.argsort(group_metric)
group_bin_idxs = []

for bin_idx in range(n_bins):
    group_bin_idxs.append(group_sort_idxs[group_bins[bin_idx]:group_bins[bin_idx+1]])
    
neuron_sort_idxs = np.argsort(neuron_metric)
neuron_bin_idxs = []

for bin_idx in range(n_bins):
    neuron_bin_idxs.append(neuron_sort_idxs[neuron_bins[bin_idx]:neuron_bins[bin_idx+1]])

neuron_metric_bin_means = np.zeros((n_bins, n_bins)) # group, neuron_idx

running_group_count = 0

bin_x = []
bin_y_1 = []
bin_y_se_1 = []
bin_y_2 = []
bin_y_se_2 = []

for group_bin_idx in range(n_bins):
    
    n_bin_groups = indirect_delta_resp[:, group_bin_idxs[group_bin_idx]].shape[-1]
    
    bin_x.append(running_group_count + n_bin_groups/2)
    
    # average over groups first
    group_indirect_delta_resp = np.nanmean(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]], axis=0)
    group_indirect_delta_resp_std = (np.nanstd(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]], axis=0) /
                                     np.sqrt(np.nansum(~np.isnan(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)))
    
    ax1.errorbar(np.arange(n_bin_groups) + running_group_count, group_indirect_delta_resp, 
                 yerr=group_indirect_delta_resp_std, linestyle='None',
                 marker='.', color=c_vals_l[group_bin_idx])
    
    bin_y_1.append(np.nanmean(group_indirect_delta_resp)) # Mean across groups in bin
    bin_y_se_1.append(np.nanstd(group_indirect_delta_resp) / np.sqrt(n_bin_groups))
    
    group_indirect_abs_delta_resp = np.nanmean(np.abs(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)
    group_indirect_abs_delta_resp_std = (np.nanstd(np.abs(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0) /
                                         np.sqrt(np.nansum(~np.isnan(indirect_delta_resp[:, group_bin_idxs[group_bin_idx]]), axis=0)))
    
    ax2.errorbar(np.arange(n_bin_groups) + running_group_count, group_indirect_abs_delta_resp, 
                 yerr=group_indirect_abs_delta_resp_std, linestyle='None',
                 marker='.', color=c_vals_l[group_bin_idx])
    
    bin_y_2.append(np.nanmean(group_indirect_abs_delta_resp)) # Mean across groups in bin
    bin_y_se_2.append(np.nanstd(group_indirect_abs_delta_resp) / np.sqrt(n_bin_groups))
    
    running_group_count += n_bin_groups

ax1.errorbar(bin_x, bin_y_1, yerr=bin_y_se_1, color='k', zorder=5)
ax2.errorbar(bin_x, bin_y_2, yerr=bin_y_se_2, color='k', zorder=5)
    
for ax in (ax1, ax2):
    ax.axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)

# %% Cell 68: Code starts with: random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_a...
random_cn_method = 'similar_prev_day_activity_tuning' # random, similar_prev_day_activity
random_cn_percent_pass = 0.2

n_pairs = len(day_2_cn_idxs)
ts_metric = 'post'

fig1, ax1s = plt.subplots(2, 6, figsize=(24, 8))
fig2, ax2s = plt.subplots(2, 6, figsize=(24, 8))

n_random_neurons = 1000

percentiles = np.zeros((n_random_neurons, n_pairs,))
cn_idx_idx = 0

xs = []
ys = []

MIN_DIST = 1000
MIN_GROUPS_INDIRECT = 8

for pair_idx, ax, axp in zip(range(n_pairs), ax1s.flatten(), ax2s.flatten()):
#     cn_idx = int(day_1_cn_idxs[pair_idx])
    cn_idx = int(day_2_cn_idxs[pair_idx])
    
#     neuron_metric = np.nanmean(day_1_indirects[pair_idx], axis=-1) # Negatively skewed, but not as much as Day 2
#     neuron_metric = np.nanmean(day_2_indirects[pair_idx], axis=-1) # Quite negatively skewed
#     neuron_metric = np.nanmean(np.abs(day_2_indirects[pair_idx]), axis=-1) # Kind of positively skewed
#     neuron_metric = np.nanmean(delta_indirects[pair_idx], axis=-1) # Negatively skewed, barely significant
#     neuron_metric = np.nanmean(np.abs(delta_indirects[pair_idx]), axis=-1) # Positively skewed, barely significant
#     neuron_metric = np.nanmean(delta_indirects_pred[pair_idx], axis=-1) # Tiny bit positively skewed
#     neuron_metric = np.nanmean(np.abs(delta_indirects_pred[pair_idx]), axis=-1) # Quite positively skewed
    
#     r_squared_weight = (
#         day_1_rsquared_indirects[pair_idx] / np.nansum(day_1_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
#         np.nansum(day_1_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
#     )
#     r_squared_weight = (
#         day_2_rsquared_indirects[pair_idx] / np.nansum(day_2_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
#         np.nansum(day_2_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
#     )
#     r_squared_weight = (
#         min_rsquared_indirects[pair_idx] / np.nansum(min_rsquared_indirects[pair_idx], axis=-1, keepdims=True) *
#         np.nansum(min_rsquared_indirects[pair_idx] > 0., axis=-1, keepdims=True)
#     )
#     neuron_metric = np.nanmean(r_squared_weight * day_1_indirects_pred[pair_idx], axis=-1)
#     neuron_metric = np.nanmean(r_squared_weight * day_2_indirects_pred[pair_idx], axis=-1)
#     neuron_metric = np.nanmean(r_squared_weight * delta_indirects_pred[pair_idx], axis=-1)
#     neuron_metric = np.nanmean(r_squared_weight *  np.abs(delta_indirects_pred[pair_idx]), axis=-1) # Quite positively skewed
    
    # Enforces minimum indirect counts
    print('CN indirect count: {}'.format(indirect_counts[pair_idx][cn_idx]))
    neuron_metric = np.where(indirect_counts[pair_idx] < MIN_GROUPS_INDIRECT, np.nan, neuron_metric)
    
    assert ~np.isnan(neuron_metric[cn_idx]) # Neuron metric cannot be nan for the CN
     
    sort_idxs = np.argsort(neuron_metric) # Puts nans at the end

    n_neurons = neuron_metric.shape[0]
    n_nonnan_neurons =  np.sum(~np.isnan(neuron_metric))
    print('Pair {}, percent non-nan: {:.2f}'.format(pair_idx, n_nonnan_neurons/n_neurons))
    
    neuron_idxs = np.arange(n_neurons)
    nonnan_neuron_idxs = np.arange(n_nonnan_neurons)
    cn_idx_sort_loc = np.where(sort_idxs == cn_idx)[0][0]
    
    if pair_idx < 11: # Can only plot 11 spots
        ax.scatter(nonnan_neuron_idxs, neuron_metric[sort_idxs][:n_nonnan_neurons], marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]])
        ax.plot((cn_idx_sort_loc, cn_idx_sort_loc), (0, neuron_metric[cn_idx]), marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
        ax.axhline(0.0, color='lightgrey', zorder=-5)

    # Go through all neurons and determine their percentile locations so this doesn't need to be redone every draw
    # If the neuron_metric contains nans, doesn't count said neurons in the percentile computation
    neuron_percentiles = np.zeros((n_neurons,))
    for neuron_idx in range(n_neurons):
        if np.isnan(neuron_metric[neuron_idx]):
            neuron_percentiles[neuron_idx] = np.nan
        else:
            neuron_percentiles[neuron_idx] = np.where(sort_idxs == neuron_idx)[0][0] / (n_nonnan_neurons - 1)

    percentiles[cn_idx_idx, pair_idx] = neuron_percentiles[cn_idx] # cn always 0th

    candidate_random_cns = get_candidate_random_cns(
        cn_idx, n_neurons, method=random_cn_method, percent_pass=random_cn_percent_pass, 
        prev_day_activities=day_1_trial_activities[pair_idx], prev_day_tuning=day_1_tunings[pair_idx]
    )

    for neuron_idx in range(n_random_neurons - 1): 
        random_neuron_idx = np.random.choice(candidate_random_cns)
        while np.isnan(neuron_percentiles[random_neuron_idx]): # Redraw if nan
            random_neuron_idx = np.random.choice(candidate_random_cns)
        percentiles[neuron_idx+1, pair_idx] = neuron_percentiles[random_neuron_idx]
        
    dist_to_cn = day_2_dist_to_cn[pair_idx]
    neuron_metric = np.where(dist_to_cn < MIN_DIST, neuron_metric, np.nan)
    
    if pair_idx < 11: # Can only plot 11 spots
        axp.scatter(dist_to_cn, neuron_metric, marker='.', color=c_vals_l[PAIR_COLORS[pair_idx]], alpha=0.3)
        axp.scatter(dist_to_cn[cn_idx], neuron_metric[cn_idx], marker='o', color=c_vals_d[PAIR_COLORS[pair_idx]])
        slope, intercept, rvalue, pvalue, se = add_regression_line(dist_to_cn, neuron_metric, ax=axp, color=c_vals[PAIR_COLORS[pair_idx]])
        axp.legend()
    
    if pair_idx == 0 or pair_idx == 6:
        axp.set_ylabel('Neuron metric')
    if pair_idx in (6, 7, 8, 9, 10):
        axp.set_xlabel('Distance to CN (um)')

    ax2s.flatten()[-1].errorbar(neuron_metric[cn_idx], slope, yerr=se, marker='o', color=c_vals[PAIR_COLORS[pair_idx]])
    xs.append(neuron_metric[cn_idx])
    ys.append(slope)
        
ax2s.flatten()[-1].axhline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
ax2s.flatten()[-1].axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
ax2s.flatten()[-1].set_xlabel('CN neuron metric')
ax2s.flatten()[-1].set_ylabel('Distance vs. neuron metric slope')
add_regression_line(xs, ys, ax=ax2s.flatten()[-1], color='k')
ax2s.flatten()[-1].legend()
    
# Mean across pairs
_, bins, _ = ax1s.flatten()[-1].hist(np.mean(percentiles, axis=-1), bins=30, color=c_vals_l[1], alpha=0.3)
sort_idxs = np.argsort(np.mean(percentiles, axis=-1))
ax1s.flatten()[-1].axvline(np.mean(percentiles, axis=-1)[cn_idx_idx], color=c_vals[1], zorder=5,
                          label='CN Mean, p={:.3f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

# Median across pairs
_, bins, _ = ax1s.flatten()[-1].hist(np.median(percentiles, axis=-1), bins=30, color=c_vals_l[0], alpha=0.3)
sort_idxs = np.argsort(np.median(percentiles, axis=-1))
ax1s.flatten()[-1].axvline(np.median(percentiles, axis=-1)[cn_idx_idx], color=c_vals[0], zorder=5,
                           label='CN Median, p={:.2f} (greater)'.format(1 - np.where(sort_idxs == cn_idx_idx)[0][0] / (n_random_neurons - 1)))

ax1s.flatten()[-1].legend()

# %% Cell 70: # Photostimulation Sanity Checks For a single session, look at some PS statistics. Generates some ba...
session_idx = 11
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
}
exemplar_neuron_idx = 250

P_VALUE_THRESH = 0.05 # Threshold for significance

normalize_by_pre = True # Normalize by pre response in the plots

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)
print('CN idx:', data_dict['data']['conditioned_neuron'][session_idx] - 1)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=True
)

raw_resp_ps_by_group = resp_ps_extras['raw_resp_ps_by_group']

raw_resp_ps_mean = resp_ps_extras['raw_resp_ps_mean']

resp_ps_sem = resp_ps_extras['resp_ps_sem']
resp_ps_pvalues = resp_ps_extras['resp_ps_pvalues']

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4,))
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4,))
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6,))
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4,))

if normalize_by_pre: # Normalize group mean responses by mean pre-PS response for each neuron separately
    
    for group_idx in range(n_groups): # Do this first or its already been noramlized
        raw_resp_ps_by_group[group_idx] = (
            raw_resp_ps_by_group[group_idx] / 
            np.mean(raw_resp_ps_mean[:, group_idx, IDXS_PRE_PS], axis=-1)[np.newaxis, :, np.newaxis]
        )
        
    raw_resp_ps_mean = raw_resp_ps_mean / np.mean(raw_resp_ps_mean[:, :, IDXS_PRE_PS], axis=-1, keepdims=True)

direct_resps_ps = np.zeros((n_neurons, n_ps_times,))  
indirect_resps_ps = np.zeros((n_neurons, n_ps_times,))  

for neuron_idx in range(n_neurons):
    
    # # Plot raw photostim responses to first few photostimulation events
    # ax1.plot(ps_fs[:, neuron_idx, 1], color=c_vals[0])
    # ax1.plot(ps_fs[:, neuron_idx, 2], color=c_vals[1])
    # ax1.plot(ps_fs[:, neuron_idx, 3], color=c_vals[2])

    direct_idxs = np.where(d_ps[neuron_idx, :] < D_DIRECT)[0]
    indirect_idxs = np.where(d_ps[neuron_idx, :] > D_DIRECT)[0]
    #     indirect_idxs = np.where(d_ps[neuron_idx, :] > D_NEAR and d_ps[neuron_idx, :] < D_FAR)[0]

    direct_resps_ps[neuron_idx] = np.nanmean(raw_resp_ps_mean[neuron_idx, direct_idxs, :], axis=0) # Average over direct groups
    indirect_resps_ps[neuron_idx] = np.nanmean(raw_resp_ps_mean[neuron_idx, indirect_idxs, :], axis=0) # Average over indirect groups

    ax1.plot(direct_resps_ps[neuron_idx], color=c_vals_l[0], alpha=0.3, zorder=-5)
    ax1.plot(indirect_resps_ps[neuron_idx], color=c_vals_l[1], alpha=0.3, zorder=-5)

    # # Plot group averaged photostim responses
    # ax1.plot(raw_resp_ps_mean[neuron_idx, 3, :], color=c_vals[0], marker='.')
    # ax1.plot(raw_resp_ps_mean[neuron_idx, 4, :], color=c_vals[1], marker='.')
    # ax1.plot(raw_resp_ps_mean[neuron_idx, 5, :], color=c_vals[2], marker='.')
    
    if exemplar_neuron_idx == neuron_idx:
                
        ### Plot of indirect responses for a given neuron ###
        
        indirect_mean_resps_ps = raw_resp_ps_mean[neuron_idx, indirect_idxs, :] # (group_idxs, ps_times)

        resps_ps = (np.nanmean(indirect_mean_resps_ps[:, IDXS_POST_PS], axis=-1) -  # Post minus pre
                    np.nanmean(indirect_mean_resps_ps[:, IDXS_PRE_PS], axis=-1))
            
        max_indirect_resp_idxs = np.argsort(resps_ps)[::-1] # Max to min
        
        my_cmap = plt.cm.get_cmap('bwr')
        vmax = np.max(np.abs(resps_ps))
        vmin = -1 * vmax
        
        for max_indirect_resp_idx in max_indirect_resp_idxs[::-1]:
            cmap_color = my_cmap((resps_ps[max_indirect_resp_idx] - vmin) / (vmax - vmin))
            ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idx], color=cmap_color)
            
#         ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idxs[0]], color=c_vals_l[1]) # Max
#         ax2.plot(indirect_mean_resps_ps[max_indirect_resp_idxs[-1]], color=c_vals_l[1]) # Min

        if normalize_by_pre:
            ax2.set_ylabel(f'Normalized Fl. - Neuron {neuron_idx}')
        else:
            ax2.set_ylabel(f'Raw Fl - Neuron {neuron_idx}')
        
        ### Plot of PS responses as a function of distance for a given neuron ###
    
#         # Each element is (ps_times, n_neurons, n_events)
#         raw_resps_ps_neuron = [raw_resp_ps_by_group[group_idx][:, neuron_idx, :] for group_idx in range(n_groups)] 
        
#         resps_ps_all = [[] for _ in range(n_groups)]
#         for group_idx in range(n_groups):
#             for event_idx in range(raw_resps_ps_neuron[group_idx].shape[-1]):
#                 resps_ps_all[group_idx].append(
#                     np.nanmean(raw_resps_ps_neuron[group_idx][IDXS_POST_PS, event_idx], axis=0) - 
#                     np.nanmean(raw_resps_ps_neuron[group_idx][IDXS_PRE_PS, event_idx], axis=0)
#                 )
                
#         resps_ps_means = np.zeros((n_groups,))
#         resps_ps_mses = np.zeros((n_groups,))
#         resps_ps_pvalues = np.zeros((n_groups,))
        
#         for group_idx in range(n_groups):
#             resps_ps_means[group_idx] = np.nanmean(resps_ps_all[group_idx])
#             resps_ps_mses[group_idx] = np.nanstd(resps_ps_all[group_idx]) / np.sqrt(np.sum(~np.isnan(resps_ps_all[group_idx])))
            
#             non_nans = [resps_ps_all[group_idx][idx] for idx in np.where(~np.isnan(resps_ps_all[group_idx]))[0]]
#             _, pvalue = ttest_1samp(non_nans, 0)
#             resps_ps_pvalues[group_idx] = pvalue
            
#         print(resp_ps_sem[neuron_idx, :3])
#         print(resps_ps_mses[:3])
        
#         print(resp_ps_pvalues[neuron_idx, :3])
#         print(resps_ps_pvalues[:3])
        
#         ax3.errorbar(d_ps[neuron_idx, :], resps_ps_means, resps_ps_mses, fmt='.', color=c_vals_l[2], zorder=-1)
        ax3.errorbar(d_ps[neuron_idx, :], resp_ps[neuron_idx], resp_ps_sem[neuron_idx], fmt='.', color=c_vals_l[2], zorder=-1)
        sig_idxs = np.where(resp_ps_pvalues[neuron_idx] < P_VALUE_THRESH)[0]
        ax3.errorbar(d_ps[neuron_idx, sig_idxs], resp_ps[neuron_idx, sig_idxs], resp_ps_sem[neuron_idx, sig_idxs], fmt='.', color=c_vals[2], zorder=-1)
        
        ax3.set_xlabel(f'Distance - Neuron {neuron_idx}')
        ax3.set_ylabel(f'PS Resp. - Neuron {neuron_idx}')
        
        ax3.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=-5)
        ax3.axvline(D_DIRECT, color='lightgrey', linestyle='dashed', zorder=-5)
        ax3.axvline(D_FAR, color='lightgrey', linestyle='dashed', zorder=-5)
        ax3.axhline(0.0, color='k', linestyle='dashed', zorder=-3)
        
# # Plot mean across all groups and neurons
# ax1.plot(np.mean(raw_resp_ps_mean, axis=(0, 1)), color='k', marker='.')

ax1.plot(np.nanmean(direct_resps_ps, axis=0), color=c_vals[0], marker='.', label='direct', zorder=5)
ax1.plot(np.nanmean(indirect_resps_ps, axis=0), color=c_vals[1], marker='.', label='indirect', zorder=5)

for ax in (ax1, ax2):
    # Draw dividing lines between sessions
    ax.axvline(np.max(IDXS_PRE_PS) + 0.5, color='lightgrey', linestyle='dashed', zorder=-5)
    ax.axvline(np.min(IDXS_POST_PS) - 0.5, color='lightgrey', linestyle='dashed', zorder=-5)
    ax.axvline(np.max(IDXS_POST_PS) + 0.5, color='lightgrey', linestyle='dashed', zorder=-5)

    ax.axhline(1.0, color='k', linestyle='dashed', zorder=-3)

    ax.set_xlabel('FStim time point')    
    ax.set_xlim((0, np.max(IDXS_POST_PS)+1))
    
if normalize_by_pre:
    ax1.set_ylabel('Normalized Fl. (by pre-PS resp.)')
else:
    ax1.set_ylabel('Raw Fl')
    
ax1.set_ylim((0.75, 2.25))
ax2.set_ylim((0.8, 1.75))
# ax2.set_ylim((50, 150))

#### Bin plot #####

d_bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
n_bins = len(d_bins) - 1

n_neurons_bins = np.zeros((n_bins,))
n_sig_exc_bins = np.zeros((n_bins,))
n_sig_inh_bins = np.zeros((n_bins,))

for neuron_idx in range(n_neurons):
    bin_idxs = np.digitize(d_ps[neuron_idx], d_bins) - 1
    
    for bin_idx in range(n_bins):
        n_neurons_bins[bin_idx] += np.where(bin_idxs == bin_idx)[0].shape[0]
    
    sig_idxs = np.where(ps_resp_pvals[neuron_idx] < P_VALUE_THRESH)[0]
    
    sig_ds = d_ps[neuron_idx, sig_idxs]
    sig_resp_ps = resp_ps[neuron_idx, sig_idxs]
    
    bin_idxs = np.digitize(sig_ds, d_bins) - 1
    
    for bin_idx in range(n_bins):
        n_sig_exc_bins[bin_idx] += np.where(
            np.logical_and(bin_idxs == bin_idx, sig_resp_ps > 0.)
        )[0].shape[0]
        n_sig_inh_bins[bin_idx] += np.where(
            np.logical_and(bin_idxs == bin_idx, sig_resp_ps < 0.)
        )[0].shape[0]

bin_widths = np.array(d_bins[1:]) - np.array(d_bins[:-1])
bin_locs = d_bins[:-1] + bin_widths / 2
_ = ax4.bar(bin_locs, n_sig_exc_bins / n_neurons_bins, width=bin_widths, color=c_vals[0])
_ = ax4.bar(bin_locs, n_sig_inh_bins / n_neurons_bins, width=bin_widths, color=c_vals[1],
            bottom = n_sig_exc_bins / n_neurons_bins)

ax4.set_xlabel('Distance from neuron to PS group')
ax4.set_ylabel('Percent significant (p < {})'.format(P_VALUE_THRESH))

ax4.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axvline(D_DIRECT, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axvline(D_FAR, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axhline(0.0, color='k', linestyle='dashed', zorder=3)
ax4.axhline(P_VALUE_THRESH / 2, color='k', linestyle='dashed', zorder=3)
ax4.axhline(P_VALUE_THRESH, color='k', linestyle='dashed', zorder=3)

# ax4.plot(n_neurons_bins)
# ax4.scatter(d_ps[neuron_idx, sig_idxs], resp_ps[neuron_idx, sig_idxs], marker='.', color=c_vals[2])


# %% Cell 72: Similar to above cell but runs over all session_idxs rather than a single session. Still not looking...
session_idxs = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19,)
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
}

P_VALUE_THRESH = 0.05 # Threshold for significance

d_ps_all = []
resp_ps_all = []
ps_resp_pvals_all = []

for session_idx in session_idxs:

    print(f'Session idx: {session_idx}')
    ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
    ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)

    n_ps_times = ps_fs.shape[0]
    n_neurons = ps_fs.shape[1]
    n_groups = int(np.max(ps_events_group_idxs))

    d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

    resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
        ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=True
    )
    
    d_ps_all.append(d_ps)
    resp_ps_all.append(resp_ps)
    ps_resp_pvals_all.append(resp_ps_extras['resp_ps_pvalues'])

fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4,))

#### Bin plot #####

d_bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700]
n_bins = len(d_bins) - 1

n_neurons_bins = np.zeros((n_bins,))
n_sig_exc_bins = np.zeros((n_bins,))
n_sig_inh_bins = np.zeros((n_bins,))

for session_idx_idx in range(len(d_ps_all)):
    
    d_ps = d_ps_all[session_idx_idx]
    resp_ps = resp_ps_all[session_idx_idx]
    ps_resp_pvals = ps_resp_pvals_all[session_idx_idx]
    
    n_neurons = d_ps.shape[0]
    for neuron_idx in range(n_neurons):
        bin_idxs = np.digitize(d_ps[neuron_idx], d_bins) - 1

        for bin_idx in range(n_bins):
            n_neurons_bins[bin_idx] += np.where(bin_idxs == bin_idx)[0].shape[0]

        sig_idxs = np.where(ps_resp_pvals[neuron_idx] < P_VALUE_THRESH)[0]

        sig_ds = d_ps[neuron_idx, sig_idxs]
        sig_resp_ps = resp_ps[neuron_idx, sig_idxs]

        bin_idxs = np.digitize(sig_ds, d_bins) - 1

        for bin_idx in range(n_bins):
            n_sig_exc_bins[bin_idx] += np.where(
                np.logical_and(bin_idxs == bin_idx, sig_resp_ps > 0.)
            )[0].shape[0]
            n_sig_inh_bins[bin_idx] += np.where(
                np.logical_and(bin_idxs == bin_idx, sig_resp_ps < 0.)
            )[0].shape[0]

bin_widths = np.array(d_bins[1:]) - np.array(d_bins[:-1])
bin_locs = d_bins[:-1] + bin_widths / 2
_ = ax4.bar(bin_locs, n_sig_exc_bins / n_neurons_bins, width=bin_widths, color=c_vals[0])
_ = ax4.bar(bin_locs, n_sig_inh_bins / n_neurons_bins, width=bin_widths, color=c_vals[1],
            bottom = n_sig_exc_bins / n_neurons_bins)

ax4.set_xlabel('Distance from neuron to PS group')
ax4.set_ylabel('Percent significant (p < {})'.format(P_VALUE_THRESH))

ax4.axvline(0.0, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axvline(D_DIRECT, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axvline(D_FAR, color='lightgrey', linestyle='dashed', zorder=3)
ax4.axhline(0.0, color='k', linestyle='dashed', zorder=3)
ax4.axhline(P_VALUE_THRESH / 2, color='k', linestyle='dashed', zorder=3)
ax4.axhline(P_VALUE_THRESH, color='k', linestyle='dashed', zorder=3)

# ax4.plot(n_neurons_bins)
# ax4.scatter(d_ps[neuron_idx, sig_idxs], resp_ps[neuron_idx, sig_idxs], marker='.', color=c_vals[2])


# %% Cell 74: Some basic analysis on the effects of photostimulation on responses for paired sessions
# def basic_paired_photostim_analysis(
#     ps_stats_params, session_idxs, data_dict, 
#     verbose=False,
# ):
#     """
#     Conducts some basic analysis on photostim data
    
#     INPUTS:
#     session_idx_pairs: list of session pairs
#     data_dict: loaded data file
#     return_dict: compact dictionary for easy saving an futher plotting.
    
#     """
#     records = { # Fill with defaults used no matter what metrics
#         'mice': [],
#         'session_idxs': [],
#         'ps_CC': [],
#     }
    
#     n_sessions = len(session_idxs)
    
#     for session_idx_idx, session_idx in enumerate(session_idxs):

#         records['session_idxs'].append(session_idx)
#         records['mice'].append(data_dict['data']['mouse'][session_idx])
        
#         data_to_extract = ('d_ps', 'resp_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',)
#         data = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)

ps_stats_params = {
    'trial_average_mode': 'trials_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': None,
    'plot_pairs': None,
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

exemplar_pair_idx = 0
fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))

P_VALUE_THRESH = 0.05 # Threshold for significance

# See how consistent direct responses are across pairs
bin_mode = True
d_direct_temps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
#     d_direct_temps = [5, 10, 15, 20, 25, 30, 35, 40]

n_direct_temps = len(d_direct_temps)
if bin_mode: n_direct_temps -= 1 # Skip last
    
d_direct_bins_rsquares = np.zeros((len(session_idx_pairs), n_direct_temps,))
d_direct_bins_slopes = np.zeros((len(session_idx_pairs), n_direct_temps,))

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    day_2_idx = session_idx_pair[1]
    day_1_idx = session_idx_pair[0]
    
    print(f'Pair {pair_idx} - Sessions {day_1_idx} and {day_2_idx}.')
    
    ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx] 
    ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
    ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx]
    ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)
    
#     assert ps_fs_1.shape[0] == ps_fs_2.shape[0] # Same ps_times
    assert ps_fs_1.shape[1] == ps_fs_2.shape[1] # Same number of neurons
    
    n_neurons = ps_fs_1.shape[1]
    n_groups = int(np.max(ps_events_group_idxs_1))

    d_ps_flat_1 = data_dict['data']['x'][day_1_idx]
    d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, n_neurons,)
    d_ps_flat_2 = data_dict['data']['x'][day_2_idx]
    d_ps_2 = unflatted_neurons_by_groups(d_ps_flat_2, n_neurons,)

    resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
        ps_fs_1, ps_events_group_idxs_1, d_ps_1, ps_stats_params, return_extras=False
    )
    resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
        ps_fs_2, ps_events_group_idxs_2, d_ps_2, ps_stats_params, return_extras=False
    )
    
    # Scan over distances
    for d_direct_temp_idx in range(n_direct_temps):
        
        d_direct_temp = d_direct_temps[d_direct_temp_idx]
        
        direct_resp_ps_1 = []
        direct_resp_ps_2 = []
        direct_d_ps_mean = []
        
        for neuron_idx in range(n_neurons):
            
            if bin_mode:
                direct_idxs = np.where(np.logical_and(
                    np.logical_and(d_ps_1[neuron_idx, :] >= d_direct_temp, d_ps_1[neuron_idx, :] < d_direct_temps[d_direct_temp_idx+1]),
                    np.logical_and(d_ps_2[neuron_idx, :] >= d_direct_temp, d_ps_2[neuron_idx, :] < d_direct_temps[d_direct_temp_idx+1]),
                ))[0]
            else:
                direct_idxs = np.where(np.logical_and(
                    d_ps_1[neuron_idx, :] < d_direct_temp, d_ps_2[neuron_idx, :] < d_direct_temp
                ))[0]
            
            # Take distance to be mean between two sessions
            d_ps_mean = np.mean(np.concatenate(
                (d_ps_1[neuron_idx:neuron_idx+1, direct_idxs], d_ps_2[neuron_idx:neuron_idx+1, direct_idxs]), 
                axis=0), axis=0)

            direct_resp_ps_1.append(resp_ps_1[neuron_idx, direct_idxs])
            direct_resp_ps_2.append(resp_ps_2[neuron_idx, direct_idxs])
            direct_d_ps_mean.append(d_ps_mean)

        direct_resp_ps_1 = np.concatenate(direct_resp_ps_1, axis=0)
        direct_resp_ps_2 = np.concatenate(direct_resp_ps_2, axis=0)
        direct_d_ps_mean = np.concatenate(direct_d_ps_mean, axis=0)
        
        ax = None
        if exemplar_pair_idx == pair_idx:
            ax = ax1
            ax1.scatter(direct_resp_ps_1, direct_resp_ps_2, c=direct_d_ps_mean, marker='.', 
                        vmin=0.0, vmax=30, cmap='viridis', alpha=0.5)

        slope, _, rvalue, _, _ = add_regression_line(direct_resp_ps_1, direct_resp_ps_2,  ax=ax, color='k', zorder=1)
        
        d_direct_bins_rsquares[pair_idx, d_direct_temp_idx] = rvalue**2
        d_direct_bins_slopes[pair_idx, d_direct_temp_idx] = slope
#         if bin_mode:
#             print('D_dir {} to {} um - slope {:.2f}\tr*2: {:.2f}'.format(
#                 d_direct_temp, d_direct_temps[d_direct_temp_idx+1], slope, rvalue**2
#             ))
#         else:
#             print('D_dir {} um - slope {:.2f}\tr*2: {:.2f}'.format(d_direct_temp, slope, rvalue**2))

ax1.set_xlabel('Day 1 Direct Resp.')
ax1.set_ylabel('Day 2 Direct Resp.')
# ax1.legend()

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
fig2, ax3 = plt.subplots(1, 1, figsize=(6, 4))

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    if pair_idx > 9:
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    
    ax2.plot(d_direct_bins_rsquares[pair_idx], color=c_vals[pair_idx%9], linestyle=linestyle)
    ax3.plot(d_direct_bins_slopes[pair_idx], color=c_vals[pair_idx%9], linestyle=linestyle)
    
for ax in (ax2, ax3):    
    ax.set_xlabel('Distance to photostim group (um)')
    ax.set_xticks(np.arange(n_direct_temps))
    ax.set_xticklabels((
        '0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40',
    ), rotation=45)
    
    ax.axvline(5.5, color='grey', zorder=-5)
    
ax2.set_ylabel('$r^2$ of Day 1 vs. Day 2')
ax3.set_ylabel('slope of Day 1 vs. Day 2')

ax3.axhline(1.0, color='grey', zorder=-5)

# %% Cell 76: ## Look for simple predictors of photostim, like correlations  Tries to see how well correlations pr...
# First just isolate desired session pairs and define metrics
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

connectivity_metrics = None

ps_stats_params = {
    'trial_average_mode': 'trials_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': None,
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

# %% Cell 78: For single sessions, looks at what is a good predictor of causal connectivity.
# Reset globals
D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

n_sessions = len(session_idxs)

PS_RESPONSE_THRESH = 0.1

exemplar_session_idx = 1

# direct_response_mode: group_membership, ps_resp, pr_resp_thresh
# d_ps_mode: default, small_indir, small_indir_dir
# correlation_mode: all, pre+post, pre, post 
fit_types = ( # (name, (direct_response_mode, d_ps_mode, correlation_mode))
    ('corr - group_membership', ('group_membership', 'default', 'all',)),
    ('corr - response_weight',  ('ps_resp', 'default', 'all',)),
    ('corr - threshold',        ('pr_resp_thresh', 'default', 'all',)),
    ('d_direct = 20',           ('ps_resp', 'small_dir', 'all',)),
    ('d_direct, d_near = 20',   ('ps_resp', 'small_dir_near', 'all',)),
    ('d_direct, d_near = 20, d_far=250',   ('ps_resp', 'small_dir_near_large_far', 'all',)),
    ('d_direct, d_near = 20, d_far=50',   ('ps_resp', 'small_dir_near_far', 'all',)),
    ('corr - pre + post',   ('ps_resp', 'small_dir_near_far', 'pre+post',)),
    ('corr - pre only',   ('ps_resp', 'small_dir_near_far', 'pre',)),
    ('corr - post only',   ('ps_resp', 'small_dir_near_far', 'post',)),
    ('corr - pre+post (default ds)',   ('ps_resp', 'default', 'pre+post',)),
)

n_fit_types = len(fit_types)

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
# fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

direct_responses = [[] for _ in range(n_sessions)]
slopes = np.zeros((n_fit_types, n_sessions,))
log10_rsquares = np.zeros((n_fit_types, n_sessions,))
log10_pvalues = np.zeros((n_fit_types, n_sessions,))

for session_idx_idx, session_idx in enumerate(session_idxs):
    
    data_to_extract = ('d_ps', 'resp_ps', 'trial_start_fs', 'd_masks',)
    
    for fit_type_idx, fit_type in enumerate(fit_types):
        fit_type_name, fit_type_params = fit_type
        direct_response_mode, d_ps_mode, correlation_mode = fit_type_params
        
        if d_ps_mode == 'default':
            D_DIRECT, D_NEAR, D_FAR = 30, 30, 100
        elif d_ps_mode == 'small_dir':
            D_DIRECT, D_NEAR, D_FAR = 20, 30, 100
        elif d_ps_mode == 'small_dir_near':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 100
        elif d_ps_mode == 'small_dir_near_large_far':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 250
        elif d_ps_mode == 'small_dir_near_far':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 50
        else:
            raise ValueError()
        
        data = extract_session_data(session_idx, data_dict, data_to_extract, ps_stats_params)

        n_neurons = data['resp_ps'].shape[0]
        n_groups = data['resp_ps'].shape[1]

        dir_mask_weighted = data['dir_mask']
        indir_mask_weighted = data['indir_mask']

        flat_direct_filter = dir_mask_weighted.flatten() > 0
        flat_indirect_filter = indir_mask_weighted.flatten() > 0

#     sum_dir_resp = np.empty((n_groups,))
#     sum_dir_resp[:] = np.nan
    
#     mean_dir_resp = sum_dir_resp.copy()
#     sum_indir_resp = sum_dir_resp.copy()
#     mean_indir_resp = sum_dir_resp.copy()
    
#     for group_idx in range(n_groups):
#         group_mask_dir = dir_mask_weighted[:, group_idx] > 0
#         if group_mask_dir.shape[0] > 0: # No neuron catch
#             sum_dir_resp[group_idx] = np.sum(data['resp_ps'][group_mask_dir, group_idx])
#             mean_dir_resp[group_idx] = np.mean(data['resp_ps'][group_mask_dir, group_idx])

#             direct_responses[session_idx_idx].append(data['resp_ps'][group_mask_dir, group_idx])

#         group_mask_indir = indir_mask_weighted[:, group_idx] > 0
#         if group_mask_indir.shape[0] > 0: # No neuron catch
#             sum_indir_resp[group_idx] = np.sum(data['resp_ps'][group_mask_indir, group_idx])
#             mean_indir_resp[group_idx] = np.mean(data['resp_ps'][group_mask_indir, group_idx])

#     direct_responses[session_idx_idx] = np.concatenate(direct_responses[session_idx_idx], axis=0)
        
        if correlation_mode == 'all':
            pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
            # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
            pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)
        elif correlation_mode == 'pre+post':
            pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'])
        elif correlation_mode == 'pre':
            pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'], ts_trial=(-2, 0))
        elif correlation_mode == 'post':
            pairwsie_corrs = compute_cross_corrs_special(data['trial_start_fs'], ts_trial=(0, 10))
        else:
            raise ValueError()
        
        if direct_response_mode == 'group_membership': # No response weighting
            group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted)
        elif direct_response_mode == 'ps_resp': # PS response weighting (might introduce spurious correlations)
            group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted * data['resp_ps']) 
        elif direct_response_mode == 'pr_resp_thresh': # Threshold response weighting
            threshold_mask = data['resp_ps'] > PS_RESPONSE_THRESH
            group_cors = np.matmul(pairwsie_corrs, dir_mask_weighted * threshold_mask) 
        else:
            raise ValueError()

        # Control for what counts as photostimulation response
        resp_ps_plot = data['resp_ps'] # Default is just the photostimulation response

        ax_fit = None 
        if session_idx == exemplar_session_idx:
            ax1.scatter(group_cors.flatten()[flat_indirect_filter], 
                        resp_ps_plot.flatten()[flat_indirect_filter], 
                        color=c_vals_l[fit_type_idx % 9], alpha=0.05, marker='.')
            
            ax_fit = ax1

        slope, intercept, rvalue, pvalue, se = add_regression_line(
            group_cors.flatten()[flat_indirect_filter], 
            resp_ps_plot.flatten()[flat_indirect_filter],       
            ax=ax_fit,
        )
        
        print(' Session idx {} r^2: {:.1e}'.format(session_idx, rvalue**2))
        slopes[fit_type_idx, pair_idx] = slope
        log10_rsquares[fit_type_idx, session_idx_idx] = np.log10(rvalue**2)
        log10_pvalues[fit_type_idx, session_idx_idx] = np.log10(pvalue)
    
    # When done scanning over the various fit types
    if session_idx == exemplar_session_idx:
        ax1.set_xlabel('Correlation to group')
        ax1.set_ylabel('PS response')
        
        ax1.legend()
    
# print('Mean rsquared log10: {:.1f} ( = {:.2e})'.format(
#     np.mean(log10_rsquares), 10**np.mean(log10_rsquares)
# ))    

# ax2.violinplot(direct_responses)
# ax2.boxplot(direct_responses, notch=True, sym='.')

# ax2.set_xlabel('Session idx')
# ax2.set_xticks((np.arange(1, n_sessions+1)))
# ax2.set_xticklabels(session_idxs)

# ax2.set_ylabel('Direct PS Response')

# ax2.axhline(0.0, color='grey', zorder=-5)
# ax2.axhline(PS_RESPONSE_THRESH, color=c_vals[1], zorder=-5)

# Reset globals               
D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))
fit_type_names = []
color_idxs = (0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,)

assert len(color_idxs) == n_fit_types

for fit_type_idx, fit_type in enumerate(fit_types):
    
    fit_type_names.append(fit_type[0])
    
    ax3.scatter(fit_type_idx * np.ones((n_sessions,)), log10_rsquares[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
    ax3.scatter(fit_type_idx, np.mean(log10_rsquares[fit_type_idx]), color='k')
    
    ax4.scatter(fit_type_idx * np.ones((n_sessions,)), log10_pvalues[fit_type_idx], color=c_vals[color_idxs[fit_type_idx]], marker='.')
    ax4.scatter(fit_type_idx, np.mean(log10_pvalues[fit_type_idx]), color='k')

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

# %% Cell 80: Same as above but now tries to fit change in correlation between two session to the resulting change...
# Reset globals
D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

n_pairs = len(session_idx_pairs)

PS_RESPONSE_THRESH = 0.1

exemplar_pair_idx = 1

# direct_response_mode: group_membership, ps_resp, pr_resp_thresh
# d_ps_mode: default, small_indir, small_indir_dir
# correlation_mode: all, pre+post, pre, post 
fit_types = ( # (name, (direct_response_mode, d_ps_mode, correlation_mode))
    ('corr - group_membership', ('group_membership', 'default', 'all',)),
    ('corr - response_weight',  ('ps_resp', 'default', 'all',)),
    ('corr - threshold',        ('pr_resp_thresh', 'default', 'all',)),
    ('d_direct = 20',           ('ps_resp', 'small_dir', 'all',)),
    ('d_direct, d_near = 20',   ('ps_resp', 'small_dir_near', 'all',)),
    ('d_direct, d_near = 20, d_far=250',   ('ps_resp', 'small_dir_near_large_far', 'all',)),
    ('d_direct, d_near = 20, d_far=50',   ('ps_resp', 'small_dir_near_far', 'all',)),
    ('corr - pre + post',   ('ps_resp', 'small_dir_near_far', 'pre+post',)),
    ('corr - pre only',   ('ps_resp', 'small_dir_near_far', 'pre',)),
    ('corr - post only',   ('ps_resp', 'small_dir_near_far', 'post',)),
    ('corr - pre+post (default ds)',   ('ps_resp', 'default', 'pre+post',)),
)

n_fit_types = len(fit_types)

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
# fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

direct_responses = [[] for _ in range(n_pairs)]
slopes = np.zeros((n_fit_types, n_pairs,))
log10_rsquares = np.zeros((n_fit_types, n_pairs,))
log10_pvalues = np.zeros((n_fit_types, n_pairs,))

full_fit_data_x = [[] for _ in range(n_fit_types)]
full_fit_data_y = [[] for _ in range(n_fit_types)]

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    day_2_idx = session_idx_pair[1]
    day_1_idx = session_idx_pair[0]

    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]

    data_to_extract = ('d_ps', 'resp_ps', 'trial_start_fs', 'd_masks',)
    
    for fit_type_idx, fit_type in enumerate(fit_types):
        fit_type_name, fit_type_params = fit_type
        direct_response_mode, d_ps_mode, correlation_mode = fit_type_params
        
        if d_ps_mode == 'default':
            D_DIRECT, D_NEAR, D_FAR = 30, 30, 100
        elif d_ps_mode == 'small_dir':
            D_DIRECT, D_NEAR, D_FAR = 20, 30, 100
        elif d_ps_mode == 'small_dir_near':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 100
        elif d_ps_mode == 'small_dir_near_large_far':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 250
        elif d_ps_mode == 'small_dir_near_far':
            D_DIRECT, D_NEAR, D_FAR = 20, 20, 50
        else:
            raise ValueError()
    
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)

        n_neurons = data_1['resp_ps'].shape[0]
        n_groups = data_1['resp_ps'].shape[1]

        indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
            ps_stats_params, data_1, data_2, verbose=False
        )

        flat_direct_filter = dir_mask_weighted.flatten() > 0
        flat_indirect_filter = indir_mask_weighted.flatten() > 0
    
#     sum_dir_resp_1 = np.empty((n_groups,))
#     sum_dir_resp_1[:] = np.nan

#     mean_dir_resp_1 = sum_dir_resp_1.copy()
#     sum_indir_resp_1 = sum_dir_resp_1.copy()
#     mean_indir_resp_1 = sum_dir_resp_1.copy()
#     sum_dir_resp_2 = sum_dir_resp_1.copy()
#     mean_dir_resp_2 = sum_dir_resp_1.copy()
#     sum_indir_resp_2 = sum_dir_resp_1.copy()
#     mean_indir_resp_2 = sum_dir_resp_1.copy()
    
#     for group_idx in range(n_groups):
#         group_mask_dir = dir_mask_weighted[:, group_idx] > 0
#         if group_mask_dir.shape[0] > 0: # No neuron catch
#             sum_dir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_dir, group_idx])
#             mean_dir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_dir, group_idx])
#             sum_dir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_dir, group_idx])
#             mean_dir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_dir, group_idx])
        
#         group_mask_indir = indir_mask_weighted[:, group_idx] > 0
#         if group_mask_indir.shape[0] > 0: # No neuron catch
#             sum_indir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_indir, group_idx])
#             mean_indir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_indir, group_idx])
#             sum_indir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_indir, group_idx])
#             mean_indir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_indir, group_idx])
        if correlation_mode == 'all':
            pairwsie_corrs_1 = data_dict['data']['trace_corr'][day_1_idx]
            pairwsie_corrs_2 = data_dict['data']['trace_corr'][day_2_idx]
            # Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
            pairwsie_corrs_1 = np.where(np.isnan(pairwsie_corrs_1), 0., pairwsie_corrs_1)
            pairwsie_corrs_2 = np.where(np.isnan(pairwsie_corrs_2), 0., pairwsie_corrs_2)
        elif correlation_mode == 'pre+post':
            pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'])
            pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'])
        elif correlation_mode == 'pre':
            pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(-2, 0))
            pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(-2, 0))
        elif correlation_mode == 'post':
            pairwsie_corrs_1 = compute_cross_corrs_special(data_1['trial_start_fs'], ts_trial=(0, 10))
            pairwsie_corrs_2 = compute_cross_corrs_special(data_2['trial_start_fs'], ts_trial=(0, 10))
        else:
            raise ValueError()
        
        if direct_response_mode == 'group_membership': # No response weighting
            group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted)
            group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted)
        elif direct_response_mode == 'ps_resp': # PS response weighting (sum of both to prevent spurious correlations)
#             group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * data_1['resp_ps'])
#             group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * data_2['resp_ps']) 
            group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * (data_1['resp_ps'] + data_2['resp_ps']))
            group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * (data_1['resp_ps'] + data_2['resp_ps'])) 
        elif direct_response_mode == 'pr_resp_thresh': # Threshold response weighting on average of responses
            threshold_mask = (1/2 * (data_1['resp_ps'] + data_2['resp_ps'])) > PS_RESPONSE_THRESH
            group_cors_1 = np.matmul(pairwsie_corrs_1, dir_mask_weighted * threshold_mask) 
            group_cors_2 = np.matmul(pairwsie_corrs_2, dir_mask_weighted * threshold_mask) 
        else:
            raise ValueError()
            
# 
#                 # More careful ways of combining over direct neurons
#                 group_cors_1 = np.zeros((n_neurons, n_groups,))
#                 group_cors_2 = np.zeros((n_neurons, n_groups,))
#                 for group_idx in range(n_groups):
#                     group_mask = dir_mask_weighted[:, group_idx] > 0 # Mask over neurons
# #                         resp_ps_weight_1 = data_1['resp_ps']
#                     resp_ps_weight_1 = data_1['resp_ps'] + data_2['resp_ps']
#                     resp_ps_weight_1 = resp_ps_weight_1 - np.mean(resp_ps_weight_1[group_mask, group_idx], keepdims=True)
#                     resp_ps_weight_1 = resp_ps_weight_1 / np.std(resp_ps_weight_1[group_mask, group_idx], keepdims=True)
# #                         resp_ps_weight_2 = data_2['resp_ps']
#                     resp_ps_weight_2 = data_1['resp_ps'] + data_2['resp_ps']
#                     resp_ps_weight_2 = resp_ps_weight_2 - np.mean(resp_ps_weight_2[group_mask, group_idx], keepdims=True)
#                     resp_ps_weight_2 = resp_ps_weight_2 / np.std(resp_ps_weight_2[group_mask, group_idx], keepdims=True)

#                     group_cors_1[:, group_idx] = np.matmul(
#                         pairwsie_corrs_1[:, group_mask], resp_ps_weight_1[group_mask, group_idx]
#                     )
#                     group_cors_2[:, group_idx] = np.matmul( 
#                         pairwsie_corrs_2[:, group_mask], resp_ps_weight_2[group_mask, group_idx]
#                     ) 

        # Control for what counts as photostimulation response
        resp_ps_plot_1 = data_1['resp_ps'] # Default is just the photostimulation response
#         resp_ps_plot_1 = resp_ps_1 / mean_dir_resp_1[np.newaxis, :]  # Normalize responses to a group by mean photostimulation size
        resp_ps_plot_2 = data_2['resp_ps'] # Default
#         resp_ps_plot_2 = resp_ps_2 / mean_dir_resp_2[np.newaxis, :]

        change_in_cors = group_cors_2.flatten() - group_cors_1.flatten()
        change_in_resp_ps = resp_ps_plot_2.flatten() - resp_ps_plot_1.flatten()

        ax_fit = None 
        if pair_idx == exemplar_pair_idx:
            ax1.scatter(change_in_cors[flat_indirect_filter],
                        change_in_resp_ps[flat_indirect_filter],
                        color=c_vals_l[fit_type_idx % 9], alpha=0.05, marker='.')
            
            ax_fit = ax1

        slope, intercept, rvalue, pvalue, se = add_regression_line(
            change_in_cors[flat_indirect_filter],
            change_in_resp_ps[flat_indirect_filter],      
            ax=ax_fit,
        )
        
        full_fit_data_x[fit_type_idx].append(change_in_cors[flat_indirect_filter])
        full_fit_data_y[fit_type_idx].append(change_in_resp_ps[flat_indirect_filter])
        
        print(' Pair idx {} r^2: {:.1e}'.format(pair_idx, rvalue**2))
        
        slopes[fit_type_idx, pair_idx] = slope
        log10_rsquares[fit_type_idx, pair_idx] = np.log10(rvalue**2)
        log10_pvalues[fit_type_idx, pair_idx] = np.log10(pvalue)
    
    # When done scanning over the various fit types

ax1.set_xlabel('Correlation to group')
ax1.set_ylabel('PS response')

ax1.legend()
        
# Reset globals               
D_DIRECT, D_NEAR, D_FAR = 30, 30, 100

# %% Cell 81: Code starts with: fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))
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

# %% Cell 83: ### Spurious Correlation Checks Looks for basic levels of correlation for a given group's causal con...
def causal_connectivity_corrs(ps_stats_params, session_idx_pairs, data_dict, 
                                    verbose=False,):
    """
    Create a compact function call to extract data from an intermediate processing step 
    to a recreation of the explainers of change of tuning.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'ps_pair_idx': [],
        'ps_CC': [],
    }
    
    n_pairs = len(session_idx_pairs)
    
    
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))
    
    all_sums_x = []
    all_means_x = []
    all_sum_diffs_x = []
    all_mean_diffs_x = []
    all_stds_x = []
    all_counts_x = []
    
    all_sums_y = []
    all_means_y = []
    all_sum_diffs_y = []
    all_mean_diffs_y = []
    all_stds_y = []
    all_counts_y = []
    
    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]
        
        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
            
        data_to_extract = ('d_ps', 'resp_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',)
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)

        indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
            ps_stats_params, data_1, data_2, verbose=verbose
        )

        n_neurons = data_1['resp_ps'].shape[0]
        n_groups = data_1['resp_ps'].shape[1]

        sum_dir_resp_1 = np.zeros((n_groups,))
        mean_dir_resp_1 = np.zeros((n_groups,))
        sum_indir_resp_1 = np.zeros((n_groups,))
        mean_indir_resp_1 = np.zeros((n_groups,))
        sum_dir_resp_2 = np.zeros((n_groups,))
        mean_dir_resp_2 = np.zeros((n_groups,))
        sum_indir_resp_2 = np.zeros((n_groups,))
        mean_indir_resp_2 = np.zeros((n_groups,))
        
        std_dir_resp_1 = np.zeros((n_groups,))
        std_dir_resp_2 = np.zeros((n_groups,))
        std_indir_resp_1 = np.zeros((n_groups,))
        std_indir_resp_2 = np.zeros((n_groups,))
        
        # These will be the same across days so long as you are using a constant mask
        group_counts_dir_1 = np.zeros((n_groups,))
        group_counts_dir_2 = np.zeros((n_groups,))
        group_counts_indir_1 = np.zeros((n_groups,))
        group_counts_indir_2 = np.zeros((n_groups,))
        
        for group_idx in range(n_groups):
            group_mask_dir = dir_mask_weighted[:, group_idx] > 0
            group_mask_indir = indir_mask_weighted[:, group_idx] > 0
            if group_mask_dir.shape[0] == 0: # No neuron catch
                sum_dir_resp_1[group_idx] = np.nan
                mean_dir_resp_1[group_idx] = np.nan
                sum_dir_resp_2[group_idx] = np.nan
                mean_dir_resp_2[group_idx] = np.nan
                std_dir_resp_1[group_idx] = np.nan
                std_dir_resp_2[group_idx] = np.nan
                group_counts_dir_1[group_idx] = np.nan
                group_counts_dir_2[group_idx] = np.nan
            else:
                sum_dir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_dir, group_idx])
                mean_dir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_dir, group_idx])
                sum_dir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_dir, group_idx])
                mean_dir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_dir, group_idx])
                std_dir_resp_1[group_idx] = np.std(data_1['resp_ps'][group_mask_dir, group_idx])
                std_dir_resp_2[group_idx] = np.std(data_2['resp_ps'][group_mask_dir, group_idx])
                group_counts_dir_1[group_idx] = data_1['resp_ps'][group_mask_dir, group_idx].shape[0]
                group_counts_dir_2[group_idx] = data_2['resp_ps'][group_mask_dir, group_idx].shape[0]
            if group_mask_indir.shape[0] == 0: # No neuron catch
                sum_indir_resp_1[group_idx] = np.nan
                mean_indir_resp_1[group_idx] = np.nan
                sum_indir_resp_2[group_idx] = np.nan
                mean_indir_resp_2[group_idx] = np.nan
                std_indir_resp_1[group_idx] = np.nan
                std_indir_resp_2[group_idx] = np.nan
                group_counts_indir_1[group_idx] = np.nan
                group_counts_indir_2[group_idx] = np.nan
            else:
                sum_indir_resp_1[group_idx] = np.sum(data_1['resp_ps'][group_mask_indir, group_idx])
                mean_indir_resp_1[group_idx] = np.mean(data_1['resp_ps'][group_mask_indir, group_idx])
                sum_indir_resp_2[group_idx] = np.sum(data_2['resp_ps'][group_mask_indir, group_idx])
                mean_indir_resp_2[group_idx] = np.mean(data_2['resp_ps'][group_mask_indir, group_idx])
                std_indir_resp_1[group_idx] = np.std(data_1['resp_ps'][group_mask_indir, group_idx])
                std_indir_resp_2[group_idx] = np.std(data_2['resp_ps'][group_mask_indir, group_idx])
                group_counts_indir_1[group_idx] = data_1['resp_ps'][group_mask_indir, group_idx].shape[0]
                group_counts_indir_2[group_idx] = data_2['resp_ps'][group_mask_indir, group_idx].shape[0]
        
        if pair_idx == 0: # Plot these only for the very first pair index
            fig2, ((ax11, ax22), (ax33, ax44), (ax55, ax66)) = plt.subplots(3, 2, figsize=(12, 12))
            
            ax11.scatter(sum_dir_resp_1, sum_indir_resp_1, marker='.', color='k')
            ax22.scatter(sum_dir_resp_2, sum_indir_resp_2, marker='.', color='k')
            ax33.scatter(mean_dir_resp_1, mean_indir_resp_1, marker='.', color='k')
            ax44.scatter(mean_dir_resp_2, mean_indir_resp_2, marker='.', color='k')
            ax55.scatter(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, marker='.', color='k')
            ax66.scatter(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_dir_resp_1, marker='.', color='k')

            _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax11, color=c_vals[0])
            _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax22, color=c_vals[0])
            _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax33, color=c_vals[0])
            _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax44, color=c_vals[0])
            _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax55, color=c_vals[0])
            _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_dir_resp_1, ax=ax66, color=c_vals[0])

        _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax1, color=c_vals[pair_idx % 9], label=None)
        _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax1, color=c_vals_l[pair_idx % 9], label=None)
        _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax3, color=c_vals[pair_idx % 9], label=None)
        _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax3, color=c_vals_l[pair_idx % 9], label=None)
        _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax5, color=c_vals[pair_idx % 9], label=None)
        _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_dir_resp_1, ax=ax6, color=c_vals[pair_idx % 9], label=None)
        
        all_sums_x.extend(sum_dir_resp_1)
        all_sums_x.extend(sum_dir_resp_2)
        all_means_x.extend(mean_dir_resp_1)
        all_means_x.extend(mean_dir_resp_2)
        all_sums_y.extend(sum_indir_resp_1)
        all_sums_y.extend(sum_indir_resp_2)
        all_means_y.extend(mean_indir_resp_1)
        all_means_y.extend(mean_indir_resp_2)
        all_sum_diffs_x.extend(sum_dir_resp_2 - sum_dir_resp_1)
        all_sum_diffs_y.extend(sum_indir_resp_2 - sum_indir_resp_1)
        all_mean_diffs_x.extend(mean_dir_resp_2 - mean_dir_resp_1)
        all_mean_diffs_y.extend(mean_indir_resp_2 - mean_dir_resp_1)
        
        _ = add_regression_line(std_dir_resp_1, std_indir_resp_1, ax=ax2, color=c_vals[pair_idx % 9], label=None)
        _ = add_regression_line(std_dir_resp_2, std_indir_resp_2, ax=ax2, color=c_vals_l[pair_idx % 9], label=None)
        all_stds_x.extend(std_dir_resp_1)
        all_stds_x.extend(std_dir_resp_1)
        all_stds_y.extend(std_indir_resp_1)
        all_stds_y.extend(std_indir_resp_2)
        
        all_counts_x.extend(group_counts_dir_1)
        all_counts_x.extend(group_counts_dir_2)
        all_counts_y.extend(group_counts_indir_1)
        all_counts_y.extend(group_counts_indir_2)
        ax4.scatter(group_counts_dir_1, group_counts_indir_1, color=c_vals[pair_idx % 9], marker='.', alpha=0.3)
        ax4.scatter(group_counts_dir_2, group_counts_indir_2, color=c_vals_l[pair_idx % 9], marker='.', alpha=0.3)

    _ = add_regression_line(all_sums_x, all_sums_y, ax=ax1, color='k')
    _ = add_regression_line(all_stds_x, all_stds_y, ax=ax2, color='k')
    _ = add_regression_line(all_means_x, all_means_y, ax=ax3, color='k')
    _ = add_regression_line(all_sum_diffs_x, all_sum_diffs_y, ax=ax5, color='k')
    _ = add_regression_line(all_mean_diffs_x, all_mean_diffs_y, ax=ax6, color='k')
        
    ax1.set_xlabel('Sum Dir. Responses')
    ax1.set_ylabel('Sum Indir. Responses')
    ax2.set_xlabel('Std Dir. Responses')
    ax2.set_ylabel('Std Indir. Responses')
    ax3.set_xlabel('Mean Dir. Responses')
    ax3.set_ylabel('Mean Indir. Responses')
    ax4.set_xlabel('Count Dir.')
    ax4.set_ylabel('Count Indir.')
    ax5.set_xlabel('$\Delta$ (Sum Dir. Responses)')
    ax5.set_ylabel('$\Delta$ (Sum Indir. Responses)')
    ax6.set_xlabel('$\Delta$ (Mean Dir. Responses)')
    ax6.set_ylabel('$\Delta$ (Mean Indir. Responses)')
    
    ax11.set_xlabel('Sum Dir. Responses (Day 1)')
    ax11.set_ylabel('Sum Indir. Responses (Day 1)')
    ax22.set_xlabel('Sum Dir. Responses (Day 2)')
    ax22.set_ylabel('Sum Indir. Responses (Day 2)')
    ax33.set_xlabel('Mean Dir. Responses (Day 1)')
    ax33.set_ylabel('Mean Indir. Responses (Day 1)')
    ax44.set_xlabel('Mean Dir. Responses (Day 2)')
    ax44.set_ylabel('Mean Indir. Responses (Day 2)')
    ax55.set_xlabel('$\Delta$ (Sum Dir. Responses)')
    ax55.set_ylabel('$\Delta$ (Sum Indir. Responses)')
    ax66.set_xlabel('$\Delta$ (Mean Dir. Responses)')
    ax66.set_ylabel('$\Delta$ (Mean Indir. Responses)')

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax11, ax22, ax33, ax44, ax55, ax66):
        ax.axhline(0.0, color='grey', linestyle='dashed')
        ax.axvline(0.0, color='grey', linestyle='dashed')
#     for ax in (ax11, ax22, ax33, ax44, ax55, ax66):
        ax.legend()
        
    return records
        
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

connectivity_metrics = None

ps_stats_params = {
    'trial_average_mode': 'trials_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': None,
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

records = causal_connectivity_corrs(
    ps_stats_params, session_idx_pairs, data_dict,
    verbose=False
)

# %% Cell 85: Checks for group correlations in neuron quantities in individual sessions and for between-sessions. ...
def check_for_group_corrs(
    neuron_metric, ps_stats_params, session_idx_pairs, data_dict, 
    verbose=False,
):
    """
    Create a compact function call to extract data from an intermediate processing step 
    to a recreation of the explainers of change of tuning.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'ps_pair_idx': [],
        'ps_CC': [],
    }
    
    n_pairs = len(session_idx_pairs)
    
#     fig3, (ax7, ax8, ax9) = plt.subplots(1, 3, figsize=(12, 4))
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))
    fig2, ((ax11, ax22), (ax33, ax44), (ax55, ax66)) = plt.subplots(3, 2, figsize=(12, 12))
    
    all_sums_x = []
    all_means_x = []
    all_sum_diffs_x = []
    all_mean_diffs_x = []
    all_stds_x = []
    
    all_sums_y = []
    all_means_y = []
    all_sum_diffs_y = []
    all_mean_diffs_y = []
    all_stds_y = []
    
    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]
        
        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
            
        data_to_extract = ('d_ps', 'resp_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',)
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params)

        indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
            ps_stats_params, data_1, data_2, verbose=verbose
        )

        n_neurons = data_1['resp_ps'].shape[0]
        n_groups = data_1['resp_ps'].shape[1]
        
        if neuron_metric == 'tuning':
            metric = 'tuning'
            neuron_metric_1 = data_1['tuning']
            neuron_metric_2 = data_2['tuning']
        elif neuron_metric == 'trial_resp':
            metric = 'trial_resp'
            neuron_metric_1 = data_1['trial_resp']
            neuron_metric_2 = data_2['trial_resp']
        elif neuron_metric == 'post':
            metric = 'post'
            neuron_metric_1 = data_1['post']
            neuron_metric_2 = data_2['post']
        elif neuron_metric == 'pre':
            metric = 'pre'
            neuron_metric_1 = data_1['pre']
            neuron_metric_2 = data_2['pre']
        else:
            raise NotImplementedError('Neuron metric {} not recognized.'.format(neuron_metric))
            
#         print('Max combined:', np.max(indir_mask_weighted + dir_mask_weighted))
#         dir_sim = np.matmul(dir_mask_weighted, dir_mask_weighted.T)
#         indir_sim = np.matmul(indir_mask_weighted, indir_mask_weighted.T)
#         far_sim = np.matmul(indir_mask_weighted + dir_mask_weighted, 
#                             (indir_mask_weighted + dir_mask_weighted).T)
        
#         dir_sim_flat = []
#         indir_sim_flat = []
#         far_sim_flat = []
#         tuning_diff_flat = []
#         for neuron_idx1 in range(dir_sim.shape[0]):
#             for neuron_idx2 in range(neuron_idx1): # Not inclusive of same index
#                 dir_sim_flat.append(dir_sim[neuron_idx1, neuron_idx2])
#                 indir_sim_flat.append(indir_sim[neuron_idx1, neuron_idx2])
#                 far_sim_flat.append(far_sim[neuron_idx1, neuron_idx2])
#                 tuning_diff_flat.append(np.abs(neuron_metric_1[neuron_idx1] - neuron_metric_1[neuron_idx2]))
        
#         for flat, ax in zip(
#             (dir_sim_flat, indir_sim_flat, far_sim_flat), (ax7, ax8, ax9)
#         ):
#             ax.scatter(flat, tuning_diff_flat, marker='.', color='k')
#             add_regression_line(flat, tuning_diff_flat, ax=ax, color=c_vals[pair_idx % 8])
        
#         for ax in (ax7, ax8, ax9):
#             ax.legend()
        
#         fig3.show()
#         print(sdfdsfsd)
        
        sum_dir_resp_1 = np.zeros((n_groups,))
        mean_dir_resp_1 = np.zeros((n_groups,))
        sum_indir_resp_1 = np.zeros((n_groups,))
        mean_indir_resp_1 = np.zeros((n_groups,))
        sum_dir_resp_2 = np.zeros((n_groups,))
        mean_dir_resp_2 = np.zeros((n_groups,))
        sum_indir_resp_2 = np.zeros((n_groups,))
        mean_indir_resp_2 = np.zeros((n_groups,))
        
        std_dir_resp_1 = np.zeros((n_groups,))
        std_dir_resp_2 = np.zeros((n_groups,))
        std_indir_resp_1 = np.zeros((n_groups,))
        std_indir_resp_2 = np.zeros((n_groups,))
        
        for group_idx in range(n_groups):
            group_mask_dir = dir_mask_weighted[:, group_idx] > 0
            group_mask_indir = indir_mask_weighted[:, group_idx] > 0
            
#             print('Group {} counts - dir: {} indir: {}'.format(
#                 group_idx, neuron_metric_1[group_mask_dir].shape[0],
#                 neuron_metric_1[group_mask_indir].shape[0]
#             ))
            if group_mask_dir.shape[0] == 0: # No neuron catch
                sum_dir_resp_1[group_idx] = np.nan
                mean_dir_resp_1[group_idx] = np.nan
                sum_dir_resp_2[group_idx] = np.nan
                mean_dir_resp_2[group_idx] = np.nan
                std_dir_resp_1 = np.nan
                std_dir_resp_2 = np.nan
            else:
                sum_dir_resp_1[group_idx] = np.sum(neuron_metric_1[group_mask_dir])
                mean_dir_resp_1[group_idx] = np.mean(neuron_metric_1[group_mask_dir])
                sum_dir_resp_2[group_idx] = np.sum(neuron_metric_2[group_mask_dir])
                mean_dir_resp_2[group_idx] = np.mean(neuron_metric_2[group_mask_dir])
                std_dir_resp_1[group_idx] = np.std(neuron_metric_1[group_mask_dir])
                std_dir_resp_2[group_idx] = np.std(neuron_metric_2[group_mask_dir])
            if group_mask_indir.shape[0] == 0: # No neuron catch
                sum_indir_resp_1[group_idx] = np.nan
                mean_indir_resp_1[group_idx] = np.nan
                sum_indir_resp_2[group_idx] = np.nan
                mean_indir_resp_2[group_idx] = np.nan
                std_indir_resp_1[group_idx] = np.nan
                std_indir_resp_2[group_idx] = np.nan
            else:
                sum_indir_resp_1[group_idx] = np.sum(neuron_metric_1[group_mask_indir])
                mean_indir_resp_1[group_idx] = np.mean(neuron_metric_1[group_mask_indir])
                sum_indir_resp_2[group_idx] = np.sum(neuron_metric_2[group_mask_indir])
                mean_indir_resp_2[group_idx] = np.mean(neuron_metric_2[group_mask_indir])
                std_indir_resp_1[group_idx] = np.std(neuron_metric_1[group_mask_indir])
                std_indir_resp_2[group_idx] = np.std(neuron_metric_2[group_mask_indir])
        
        if pair_idx == 0: # Plot these only for the very first pair index
            
            ax11.scatter(sum_dir_resp_1, sum_indir_resp_1, marker='.', color='k')
            ax22.scatter(sum_dir_resp_2, sum_indir_resp_2, marker='.', color='k')
            ax33.scatter(mean_dir_resp_1, mean_indir_resp_1, marker='.', color='k')
            ax44.scatter(mean_dir_resp_2, mean_indir_resp_2, marker='.', color='k')
            ax55.scatter(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, marker='.', color='k')
            ax66.scatter(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, marker='.', color='k')

            _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax11, color=c_vals[pair_idx])
            _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax22, color=c_vals[pair_idx])
            _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax33, color=c_vals[pair_idx])
            _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax44, color=c_vals[pair_idx])
            _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax55, color=c_vals[pair_idx])
            _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, ax=ax66, color=c_vals[pair_idx])
            
        _ = add_regression_line(sum_dir_resp_1, sum_indir_resp_1, ax=ax1, color=c_vals[pair_idx % 8], label=None)
        _ = add_regression_line(sum_dir_resp_2, sum_indir_resp_2, ax=ax1, color=c_vals_l[pair_idx % 8], label=None)
        _ = add_regression_line(mean_dir_resp_1, mean_indir_resp_1, ax=ax3, color=c_vals[pair_idx % 8], label=None)
        _ = add_regression_line(mean_dir_resp_2, mean_indir_resp_2, ax=ax3, color=c_vals_l[pair_idx % 8], label=None)
        _ = add_regression_line(sum_dir_resp_2 - sum_dir_resp_1, sum_indir_resp_2 - sum_indir_resp_1, ax=ax5, color=c_vals[pair_idx % 8], label=None)
        _ = add_regression_line(mean_dir_resp_2 - mean_dir_resp_1, mean_indir_resp_2 - mean_indir_resp_1, ax=ax6, color=c_vals[pair_idx % 8], label=None)
        
        all_sums_x.extend(sum_dir_resp_1)
        all_sums_x.extend(sum_dir_resp_2)
        all_means_x.extend(mean_dir_resp_1)
        all_means_x.extend(mean_dir_resp_2)
        all_sums_y.extend(sum_indir_resp_1)
        all_sums_y.extend(sum_indir_resp_2)
        all_means_y.extend(mean_indir_resp_1)
        all_means_y.extend(mean_indir_resp_2)
        all_sum_diffs_x.extend(sum_dir_resp_2 - sum_dir_resp_1)
        all_sum_diffs_y.extend(sum_indir_resp_2 - sum_indir_resp_1)
        all_mean_diffs_x.extend(mean_dir_resp_2 - mean_dir_resp_1)
        all_mean_diffs_y.extend(mean_indir_resp_2 - mean_dir_resp_1)
        
        _ = add_regression_line(std_dir_resp_1, std_indir_resp_1, ax=ax2, color=c_vals[pair_idx % 8], label=None)
        _ = add_regression_line(std_dir_resp_2, std_indir_resp_2, ax=ax2, color=c_vals_l[pair_idx % 8], label=None)
        all_stds_x.extend(std_dir_resp_1)
        all_stds_x.extend(std_dir_resp_1)
        all_stds_y.extend(std_indir_resp_1)
        all_stds_y.extend(std_indir_resp_2)

    _ = add_regression_line(all_sums_x, all_sums_y, ax=ax1, color='k')
    _ = add_regression_line(all_stds_x, all_stds_y, ax=ax2, color='k')
    _ = add_regression_line(all_means_x, all_means_y, ax=ax3, color='k')
    _ = add_regression_line(all_sum_diffs_x, all_sum_diffs_y, ax=ax5, color='k')
    _ = add_regression_line(all_mean_diffs_x, all_mean_diffs_y, ax=ax6, color='k')
        
    ax1.set_xlabel('Sum Dir. {}'.format(metric))
    ax1.set_ylabel('Sum Indir. {}'.format(metric))
    ax2.set_xlabel('Std Dir. {}'.format(metric))
    ax2.set_ylabel('Std Indir. {}'.format(metric))
    ax3.set_xlabel('Mean Dir. {}'.format(metric))
    ax3.set_ylabel('Mean Indir. {}'.format(metric))
    ax5.set_xlabel('$\Delta$ (Sum Dir. {})'.format(metric))
    ax5.set_ylabel('$\Delta$ (Sum Indir. {})'.format(metric))
    ax6.set_xlabel('$\Delta$ (Mean Dir. {})'.format(metric))
    ax6.set_ylabel('$\Delta$ (Mean Indir. {})'.format(metric))
    
    ax11.set_xlabel('Sum Dir. {} (Day 1)'.format(metric))
    ax11.set_ylabel('Sum Indir. {} (Day 1)'.format(metric))
    ax22.set_xlabel('Sum Dir. {} (Day 2)'.format(metric))
    ax22.set_ylabel('Sum Indir. {} (Day 2)'.format(metric))
    ax33.set_xlabel('Mean Dir. {} (Day 1)'.format(metric))
    ax33.set_ylabel('Mean Indir. {} (Day 1)'.format(metric))
    ax44.set_xlabel('Mean Dir. {} (Day 2)'.format(metric))
    ax44.set_ylabel('Mean Indir. {} (Day 2)'.format(metric))
    ax55.set_xlabel('$\Delta$ (Sum Dir. {})'.format(metric))
    ax55.set_ylabel('$\Delta$ (Sum Indir. {})'.format(metric))
    ax66.set_xlabel('$\Delta$ (Mean Dir. {})'.format(metric))
    ax66.set_ylabel('$\Delta$ (Mean Indir. {})'.format(metric))

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax11, ax22, ax33, ax44, ax55, ax66):
        ax.axhline(0.0, color='grey', linestyle='dashed')
        ax.axvline(0.0, color='grey', linestyle='dashed')
#     for ax in (ax11, ax22, ax33, ax44, ax55, ax66):
        ax.legend()
    
    fig2.show()
    
    return records
        
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

neuron_metric = 'pre' # tuning, trial_resp, pre, post

ps_stats_params = {
    'trial_average_mode': 'trials_first', 
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # Normalize things like tuning before computing metrics
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': None,
    'plot_pairs': None,
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

session_idx_pairs.remove((20, 21)) # Kayvon tosses out this session

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

records = check_for_group_corrs(
    neuron_metric, ps_stats_params, session_idx_pairs, data_dict,
    verbose=False
)

# %% Cell 88: Old version of this function that outputs in a slightly different format. New version updated to be ...
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

# %% Cell 89: Code starts with: def fit_each_xs_with_bases(
def fit_each_xs_with_bases(
    records, base_xs, ps_stats_params, fit_intercept=True, standardize_xs=False,
    standardize_ys=False, verbose=False
):
    """
    Fit each _y against each _x and the base _x's. 
    
    OUTPUTS:
    - full_fits: Dict with key for each connectivity_metrics_y, which contains a dict for each
      connectivity_metrics_x, which contains the full LS fit
        - e.g. full_fits[connectivity_metrics_y][connectivity_metrics_x]
    - session_fits: Same as above, with each dict key containing a list of sessions
        - e.g. full_fits[connectivity_metrics_y][connectivity_metrics_x][session_idx]
    """

    param_idx_offset = 1 if fit_intercept else 0

    # This can either be session pairs or individual sessions
    n_sessions = len(records[ps_stats_params['connectivity_metrics'][0]])

    full_fits = {} # Dict of fits across all sessions
    session_fits = {} # Dict of fits to individual sessions (optional)

    if standardize_xs and verbose:
        print('Standardizing _x\'s!')
    if standardize_ys and verbose:
        print('Standardizing _y\'s!')

    # Create lists of all _x's and _y's from connectivity_metrics
    if ps_stats_params['plot_pairs'] is not None:
        raise NotImplementedError('All possible pairs is hard coded for now.')

    # Get list of all possible 
    connectivity_metrics_xs, connectivity_metrics_ys = enumerate_plot_pairs(ps_stats_params)

    for base_x in base_xs: # Base_xs must exist
        assert base_x in connectivity_metrics_xs
    
    # First get base_metric idxs
    base_idxs = []
    for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
        if connectivity_metrics_x in base_xs:
            base_idxs.append(cm_x_idx)
        
    fit_metrics_xs = []
    fit_metrics_xs_idxs = []
    for cm_x_idx, connectivity_metrics_x in enumerate(connectivity_metrics_xs):
        if connectivity_metrics_x in base_xs:
            continue
        fit_metrics_xs.append(connectivity_metrics_x)
        
        fit_metrics_xs_idxs.append(base_idxs.copy()) # Base idxs
        fit_metrics_xs_idxs[-1].append(cm_x_idx) 
        fit_metrics_xs_idxs[-1] = np.array(fit_metrics_xs_idxs[-1])
    
    # Get all _x's used to fit the _y's below
    records_x_flat_all = get_all_xs(records, connectivity_metrics_xs, standardize_xs=standardize_xs)
    
    if 'raw_cc_y' in records.keys():
        exemplar_key = 'raw_cc_y'
    elif 'raw_delta_cc_y' in records.keys():
        exemplar_key = 'raw_delta_cc_y'
    else:
        raise ValueError()

    # Collect all _y's and then fit them
    for cm_y_idx, connectivity_metrics_y in enumerate(connectivity_metrics_ys):
        
        records_y_flat = []
        records_w_flat = []

        for session_idx, session_records_y in enumerate(records[connectivity_metrics_y]):
            exemplar_session_records_y = records[exemplar_key][session_idx]

            # Expand all ys so that they're the same shape 
            session_records_y_flat = []
            for group_idx in range(len(session_records_y)):
                if type(session_records_y) == np.ndarray: # Need to extend y for each x
                    session_records_y_flat.append(
                        session_records_y[group_idx] * np.ones((len(exemplar_session_records_y[group_idx]),))
                    )
                elif type(session_records_y) == list: # x and y already extended
                    session_records_y_flat.append(session_records_y[group_idx])

            # Concatenate across all groups, to get total number of indirect in session
            session_records_y_flat = np.concatenate(session_records_y_flat, axis=0)

            if standardize_ys:
                session_records_y_flat = (session_records_y_flat - np.nanmean(session_records_y_flat)) / np.nanstd(session_records_y_flat)

            if ps_stats_params['indirect_weight_type'] is not None: # Does the same for weights
                session_records_w_flat = []
                for group_idx in range(len(session_records_y)):
                    session_records_w_flat.append(records['indirect_weights_w'][session_idx][group_idx])
                session_records_w_flat = np.concatenate(session_records_w_flat, axis=0)

            # Append session records to across-session collection
            records_y_flat.append(session_records_y_flat)
            if ps_stats_params['indirect_weight_type'] is not None:
                records_w_flat.append(session_records_w_flat)
        
        # Now fit each y
        
        full_fits[connectivity_metrics_y] = {}
        session_fits[connectivity_metrics_y] = {}
        
        for fm_x_idx, fit_metrics_x in enumerate(fit_metrics_xs):
        
            session_fits[connectivity_metrics_y][fit_metrics_x] = [] # Only filled in if fitting individual sessions

            fm_idxs = fit_metrics_xs_idxs[fm_x_idx]
            
            for session_idx in range(len(records[connectivity_metrics_y])):
            
                if ps_stats_params['fit_individual_sessions']:

                    session_records_x = records_x_flat_all[session_idx][fm_idxs, :]
                    session_records_y_flat = records_y_flat[session_idx]

                    nonnan_mask = np.where(np.all(~np.isnan(session_records_x), axis=0) * ~np.isnan(session_records_y_flat))[0]
                    session_records_x_flat = session_records_x[:, nonnan_mask]
                    session_records_y_flat = session_records_y_flat[nonnan_mask]
                    if ps_stats_params['indirect_weight_type'] is not None:
                        session_records_w_flat = records_w_flat[session_idx]
                        session_weights = session_records_w_flat[nonnan_mask]
                    else:
                        session_weights = None

                    session_fit = {}

                    X = session_records_x_flat.T
                    Y = session_records_y_flat[:, np.newaxis]

                    if fit_intercept:
                        X = sm.add_constant(X)

                    if session_weights is None:
                        fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
                    else:
                        fit_model = sm.WLS(Y, X, weights=session_weights, missing='drop') # nans should be caught above
                    session_results = fit_model.fit()

                    session_fits[connectivity_metrics_y][fit_metrics_x].append(session_results)

            # Now that all sessions are collected, concatenate everything together
            records_x_flat_all_temp = np.concatenate(records_x_flat_all, axis=-1)[fm_idxs, :] # Temporary concat to fit everything
            records_y_flat_temp = np.concatenate(records_y_flat, axis=0) # Temporary concat to fit everything

            nonnan_mask = np.where(np.all(~np.isnan(records_x_flat_all_temp), axis=0) * ~np.isnan(records_y_flat_temp))[0]

            records_x_flat_all_temp = records_x_flat_all_temp[:, nonnan_mask]
            records_y_flat_temp = records_y_flat_temp[nonnan_mask]
            if ps_stats_params['indirect_weight_type'] is not None: 
                weights = np.concatenate(records_w_flat, axis=0)[nonnan_mask]
            elif ps_stats_params['group_weights_type'] == None and ps_stats_params['indirect_weight_type'] == None:
                weights = None
            else:
                raise NotImplementedError('Group weights type {} not recognized.'.format(ps_stats_params['group_weights_type']))

            X = records_x_flat_all_temp.T
            Y = records_y_flat_temp[:, np.newaxis]

            if fit_intercept:
                X = sm.add_constant(X)

            if weights is None:
                fit_model = sm.OLS(Y, X, missing='none') # nans should be caught above
            else:
                fit_model = sm.WLS(Y, X, weights=weights, missing='drop') # nans should be caught above
            results = fit_model.fit()

            full_fits[connectivity_metrics_y][fit_metrics_x] = results

    return full_fits, session_fits, (connectivity_metrics_xs, connectivity_metrics_ys)

base_xs = ('delta_laser_x', 'laser_resp_x',)

fit_intercept = True
standardize_xs = True
standardize_ys = True

full_fits, session_fits = fit_each_xs_with_bases(
    records, base_xs, ps_stats_params, fit_intercept=fit_intercept, standardize_xs=standardize_xs,
    standardize_ys=standardize_ys, verbose=True
)

print('Done!')

# %% Cell 91: MLR fits with base_xs only instead of all _x
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

# %% Cell 93: Get some metrics related to laser responses and # of directly stimulated neurons and see how many gr...
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(1, 1, figsize=(12, 8))

mice = []

# for ax, connectivity_metric, y_label in zip(
#     (ax1, ax2, ax3, ax4),
#     ('cc_x', 'tuning_x', 'trial_resp_x', 'post_resp_x'),
#     ('Sum direct neuron W_CC', 'Direct neuron tuning', )
# )

for idx, cc_x in enumerate(records['cc_x']):
    cc_x_below_zero = np.where(cc_x < 0)[0]
#     print(cc_x_below_zero.shape[0])
#     print('{} - num cc_x below zero: {}/{}'.format(
#         idx, cc_x_below_zero.shape[0], cc_x.shape[0]
#     ))
    print('Mouse {}, session {} - '.format(records['mice'][idx], records['session_idxs'][idx]), cc_x_below_zero)
    
    if records['mice'][idx] not in mice:
        mice.append(records['mice'][idx])
    
    ax1.scatter(records['mask_counts_x'][idx], cc_x, marker='.', color=c_vals[len(mice)-1], alpha=0.3)
    
ax1.axhline(0.0, color='grey', zorder=-5, linestyle='dashed')
ax1.set_xlabel('# of direct neurons')
ax1.set_ylabel('Sum direct neuron W_CC')    

# %% Cell 94: Code starts with: # For a single session, see the relative size of the various contributions to a gi...
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

# %% Cell 95: Code starts with: def fit_and_plot_up(x, y, weights=None, catch_nans=True, ax=None, axes_labels=None...
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

# %% Cell 97: This code was used to check that if we do the fit on the top magnitude direct neurons, if the predic...
exemplar_session_idx = 1 # 11

weight_type = 'rsquared' # None, rsquared

ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'top_mags', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 4,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

session_idx = 1

print('Session idx {}'.format(session_idx))

ps_events_group_idxs = data_dict['data']['seq'][session_idx] # This is matlab indexed so always need a -1 here
ps_fs = data_dict['data']['Fstim'][session_idx] # (ps_times, n_neurons, n_ps_events,)

n_ps_times = ps_fs.shape[0]
n_neurons = ps_fs.shape[1]
n_groups = int(np.max(ps_events_group_idxs))

d_ps_flat = data_dict['data']['x'][session_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps = unflatted_neurons_by_groups(d_ps_flat, n_neurons,)

resp_ps, resp_ps_extras = compute_resp_ps_mask_prevs(
    ps_fs, ps_events_group_idxs, d_ps, ps_stats_params, return_extras=False,
)

resp_ps_events = resp_ps_extras['resp_ps_events']

pairwsie_corrs = data_dict['data']['trace_corr'][session_idx]
# Some of these entries are np.nan, so just replace with zero so they don't contribute to matrix sums
pairwsie_corrs = np.where(np.isnan(pairwsie_corrs), 0., pairwsie_corrs)

direct_idxs_flat = np.where(d_ps.flatten() < D_DIRECT)[0]
indirect_idxs_flat = np.where(np.logical_and(d_ps.flatten() > D_NEAR, d_ps.flatten() < D_FAR))[0]

group_corrs = [] # Filled as we iterate across groups
indirect_resp_ps = []
indirect_predictions = []

r_squareds = []

direct_predictors_all = [] # will be (n_groups, n_direct_predictors, n_indirect)

if ps_stats_params['direct_predictor_mode'] in ('top_mags',):
    direct_to_indirect_params = [[[] for _ in range(n_neurons)] for _ in range(n_neurons)] 

for group_idx in range(n_groups):
    direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
    indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > D_NEAR, d_ps[:, group_idx] < D_FAR))[0]

    dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
    indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)

    direct_predictors, direct_shift, predictor_extras = find_photostim_variation_predictors(
        dir_resp_ps_events, ps_stats_params,
    )
    
    direct_predictors_all.append(direct_predictors)
    
    indirect_params, indirect_pvalues, fit_extras = fit_photostim_variation(
        dir_resp_ps_events, indir_resp_ps_events, direct_predictors, direct_shift,
        ps_stats_params, verbose=False, return_extras=True
    )
    
    r_squareds.append(fit_extras['r_squareds'])
    indirect_resp_ps.append(resp_ps[indirect_idxs, group_idx])
#         sum_direct_resp_ps = np.nansum(resp_ps[direct_idxs, group_idx]) # Avg. over events, then neurons
#         sum_direct_resp_ps =  np.nanmean(np.nansum(dir_resp_ps_events, axis=0)) # Over neurons, then avg events

    ### Gets average direct input ###
    # Note this way of doing determines average input for each event THEN averages over
    # events. This does not necessarily yield the same result as averaging over events
    # first then determining the input because of how we treat nans.

    # (n_direct_predictors, n_events) <- (n_direct_predictors, n_direct) x (n_direct, n_events)
    direct_predictors_events = nan_matmul(direct_predictors, dir_resp_ps_events)
    direct_input = np.nanmean(direct_predictors_events, axis=-1) # (n_direct_predictors,)

#         print('Sum:', sum_direct_resp_ps)
#         print('Sum events, mean first:', np.nansum(np.nanmean(dir_resp_ps_events, axis=-1)))
#         print('Sum events, sum first:', np.nanmean(np.nansum(dir_resp_ps_events, axis=0)))
#         print('Matmul:', direct_input)

#         if ps_stats_params['direct_predictor_intercept_fit']:
#             indirect_prediction = indirect_params[:, 0] + indirect_params[:, 1] * sum_direct_resp_ps
#         else:
#             indirect_prediction = indirect_params[:, 0] * sum_direct_resp_ps

#         print('Man pred', indirect_prediction[:3])

    ### Uses average direct input to predict photostimulation response ###
    indirect_predictions.append(photostim_predict(indirect_params, direct_input, ps_stats_params))

    group_corrs.append(np.matmul(pairwsie_corrs[:, direct_idxs], resp_ps[direct_idxs, group_idx])[indirect_idxs])
#         group_corrs_norm = np.matmul(pairwsie_corrs[:, direct_idxs], resp_ps[direct_idxs, group_idx] / np.sum(resp_ps[direct_idxs, group_idx]))[indirect_idxs]
    
    if ps_stats_params['direct_predictor_mode'] in ('top_mags',):
        top_mag_neurons = direct_idxs[np.where(direct_predictors > 0)[-1]]
        
        for direct_idx, top_mag_neuron_idx in enumerate(top_mag_neurons):
            for indirect_idx, indirect_neuron_idx in enumerate(indirect_idxs):
                direct_to_indirect_params[top_mag_neuron_idx][indirect_neuron_idx].append(
                    indirect_params[indirect_idx, direct_idx]
                )
    
    
group_corrs = np.concatenate(group_corrs, axis=0) # Filled as we iterate across groups
indirect_resp_ps = np.concatenate(indirect_resp_ps, axis=0)
indirect_predictions = np.concatenate(indirect_predictions, axis=0)
    
r_squareds = np.concatenate(r_squareds, axis=0)
    
if weight_type in (None,):
    weights = None
elif weight_type in ('rsquared',):
    weights = np.copy(r_squareds[session_idx_idx])
else:
    raise ValueError('Weight type {} not recognized'.format(weight_type))


fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

n_entries = np.zeros((n_neurons, n_neurons,))

MIN_COUNT = 5
count = 0

for direct_idx in range(n_neurons):
    for indirect_idx in range(n_neurons):
        entry = direct_to_indirect_params[direct_idx][indirect_idx]
        n_entries[direct_idx, indirect_idx] = len(entry)
        
        if len(entry) >= MIN_COUNT:
            ax2.scatter(count * np.ones((len(entry),)), entry, color=c_vals_l[0], marker='.', zorder=-4)
            ax2.errorbar(count, np.mean(entry), np.std(entry), color=c_vals[0], marker='_')
            
            
            count+=1
#             print(direct_to_indirect_params[direct_idx][indirect_idx])
ax2.axhline(0.0, color='lightgrey', zorder=-5, linestyle='dashed')

ax2.set_ylabel('Direct -> Indirect Parameter')
ax2.set_xlabel('Entries with >= {} fits.'.format(MIN_COUNT))

# %% Cell 99: Chasing down why certain neruons are nans...
ps_stats_params = {
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    
    'direct_predictor_mode': 'sum', # sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
}

exemplar_pair_idx = 0

n_pairs = len(session_idx_pairs)

differences = []
differences_vector = []
prs_direct_1 = []
prs_direct_2 = []
prs_direct_1_2 = []
n_direct_for_pr = []

n_direct = [[] for _ in range(n_pairs)] # Separate this out into pairs to plot by mouse
n_events_1 = [[] for _ in range(n_pairs)] 
n_events_2 = [[] for _ in range(n_pairs)]

percent_dir_nans_1 = [[] for _ in range(n_pairs)] # Gets data on how many entries are nans to justify how we're treating them
percent_dir_nans_2 = [[] for _ in range(n_pairs)] 

for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
    
    day_1_idx = session_idx_pair[0]
    day_2_idx = session_idx_pair[1]
    
    assert day_2_idx > day_1_idx
    assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
    
    print('Pair {} - Sessions {} and {} - Mouse {}.'.format(
        pair_idx, day_1_idx, day_2_idx, data_dict['data']['mouse'][day_2_idx]
    )) 

    ps_events_group_idxs_1 = data_dict['data']['seq'][day_1_idx] # This is matlab indexed so always need a -1 here
    ps_fs_1 = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_1 = data_dict['data']['x'][day_1_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, ps_fs_1.shape[1],)
    resp_ps_1, resp_ps_extras_1 = compute_resp_ps_mask_prevs(
        ps_fs_1, ps_events_group_idxs_1, d_ps_1, ps_stats_params,
    )
    resp_ps_events_1 = resp_ps_extras_1['resp_ps_events']

    ps_events_group_idxs_2 = data_dict['data']['seq'][day_2_idx] # This is matlab indexed so always need a -1 here
    ps_fs_2 = data_dict['data']['Fstim'][day_2_idx] # (ps_times, n_neurons, n_ps_events,)
    d_ps_flat_2 = data_dict['data']['x'][day_2_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
    d_ps_2 = unflatted_neurons_by_groups(d_ps_flat_2, ps_fs_2.shape[1],)
    resp_ps_2, resp_ps_extras_2 = compute_resp_ps_mask_prevs(
        ps_fs_2, ps_events_group_idxs_2, d_ps_2, ps_stats_params,
    )
    resp_ps_events_2 = resp_ps_extras_2['resp_ps_events']

    n_groups = int(np.max(ps_events_group_idxs_1)) # +1 already accounted for because MatLab indexing
    
    sum_dir_resp_ps_1 = np.zeros((n_groups,))
    sum_dir_resp_ps_2 = np.zeros((n_groups,))
    
    for group_idx in range(n_groups):
        direct_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] < D_DIRECT, d_ps_2[:, group_idx] < D_DIRECT))[0]
        indirect_idxs = np.where(np.logical_and(
            np.logical_and(d_ps_1[:, group_idx] > D_NEAR, d_ps_1[:, group_idx] < D_FAR),
            np.logical_and(d_ps_2[:, group_idx] > D_NEAR, d_ps_2[:, group_idx] < D_FAR)
        ))[0]
        
        # Mean response-based metrics
        dir_resp_ps_1 = resp_ps_1[direct_idxs, group_idx] # (n_direct,)
        indir_resp_ps_1 = resp_ps_1[indirect_idxs, group_idx] # (n_indirect,)

        dir_resp_ps_2 = resp_ps_2[direct_idxs, group_idx] # (n_direct,)
        indir_rresp_ps_2 = resp_ps_2[indirect_idxs, group_idx] # (n_indirect,)
        
        sum_dir_resp_ps_1[group_idx] = np.nansum(dir_resp_ps_1)
        sum_dir_resp_ps_2[group_idx] = np.nansum(dir_resp_ps_2)
        
        differences_vector.append(
            np.linalg.norm(dir_resp_ps_1 - dir_resp_ps_2) /
            (1/2 * (np.linalg.norm(dir_resp_ps_1) + np.linalg.norm(dir_resp_ps_2)))
        )
        
        # Event-based metrics
        dir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_1 = np.array(resp_ps_events_1[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        percent_dir_nans_1[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_1), 1., 0.), axis=0) / dir_resp_ps_events_1.shape[0]
        )
#         dir_resp_ps_events_1 = np.where(np.isnan(dir_resp_ps_events_1), 0., dir_resp_ps_events_1) # Fill nans with 0s
#         keep_event_idxs_1 = np.where(~np.any(np.isnan(dir_resp_ps_events_1), axis=0))[0]
#         dir_resp_ps_events_1 = dir_resp_ps_events_1[:, keep_event_idxs_1]
        
        dir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[direct_idxs, :] # (n_direct, n_events,)
        indir_resp_ps_events_2 = np.array(resp_ps_events_2[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)
        percent_dir_nans_2[pair_idx].append(
            np.sum(np.where(np.isnan(dir_resp_ps_events_2), 1., 0.), axis=0) / dir_resp_ps_events_2.shape[0]
        )
#         dir_resp_ps_events_2 = np.where(np.isnan(dir_resp_ps_events_2), 0., dir_resp_ps_events_2) # Fill nans with 0s
#         keep_event_idxs_2 = np.where(~np.any(np.isnan(dir_resp_ps_events_2), axis=0))[0]
#         dir_resp_ps_events_2 = dir_resp_ps_events_2[:, keep_event_idxs_2]
        
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        
        ax1.matshow(np.isnan(dir_resp_ps_events_1))
        ax1.set_xlabel('Event idx')
        ax1.set_ylabel('Direct neuron idx')
        
        print(percent_dir_nans_1[pair_idx])
        
        print(dsfsdfsd)

# %% Cell 100: Code starts with: day_1_idx = 1
day_1_idx = 1
group_idx = 0
seq = data_dict['data']['seq'][day_1_idx]
raw_resp_ps = data_dict['data']['Fstim'][day_1_idx] # (ps_times, n_neurons, n_ps_events,)
sq = (seq - 1).astype(np.int32) # -1 to account for Matlab indexing

d_ps_flat_1 = data_dict['data']['x'][day_1_idx] # distance between a neuron and its nearest photostim point for each group (n_groups x n_neurons)
d_ps_1 = unflatted_neurons_by_groups(d_ps_flat_1, ps_fs_1.shape[1],)

group_trial_idxs = np.where(sq == group_idx)[0] # All ps event indexes that match group index
assert group_trial_idxs.shape[0] > 0
        
direct_idxs = np.where(np.logical_and(d_ps_1[:, group_idx] < D_DIRECT, d_ps_2[:, group_idx] < D_DIRECT))[0]
    
    
# Mean over time first, then PS events
raw_resp_ps_by_group = raw_resp_ps[:, :, group_trial_idxs] # (ps_times, n_neurons, n_group_events,)    

pre_resp_ps_events = np.nanmean(raw_resp_ps_by_group[IDXS_PRE_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,) 
post_resp_ps_events = np.nanmean(raw_resp_ps_by_group[IDXS_POST_PS, :, :], axis=0) # (ps_times, n_neurons, n_group_events,) -> (n_neurons, n_group_events,)
baseline_resp_ps_events = np.nanmean(pre_resp_ps_events, axis=-1) # (n_neurons, n_group_events,) -> (n_neurons,) 

resp_ps_events = (post_resp_ps_events - pre_resp_ps_events) / baseline_resp_ps_events[:, np.newaxis] # (n_neurons, n_group_events,) 

fig1, (ax1, ax1p) = plt.subplots(1, 2, figsize=(9, 4))
        
ax1.matshow(np.isnan(resp_ps_events[direct_idxs, :]))
ax1.set_xlabel('Event idx')
ax1.set_ylabel('Direct neuron idx')

event_idx = 1
direct_neuron_idx1 = 5
direct_neuron_idx2 = 6

ax1.scatter(event_idx, direct_neuron_idx1, color=c_vals[1], marker='o')
ax1.scatter(event_idx, direct_neuron_idx2, color=c_vals[0], marker='o')

print('Direct idx: {}'.format(direct_neuron_idx1), raw_resp_ps[:, direct_idxs[direct_neuron_idx1], group_trial_idxs[event_idx]])
print('Direct idx: {}'.format(direct_neuron_idx2), raw_resp_ps[:, direct_idxs[direct_neuron_idx2], group_trial_idxs[event_idx]])

ax1p.plot(raw_resp_ps[:, direct_idxs[direct_neuron_idx1], group_trial_idxs[event_idx]], color=c_vals[1])
ax1p.plot(raw_resp_ps[:, direct_idxs[direct_neuron_idx2], group_trial_idxs[event_idx]], color=c_vals[0])

print('Direct idx: {} - Neuron idx {}'.format(direct_neuron_idx1, direct_idxs[direct_neuron_idx1]))
print('Direct idx: {} - Neuron idx {}'.format(direct_neuron_idx2, direct_idxs[direct_neuron_idx2]))

# %% Cell 101: Code starts with: # OLDER VERSION OF THESE FUNCTIONS
# OLDER VERSION OF THESE FUNCTIONS

def eval_over_neurons(
    neuron_metric, neuron_by_group, ps_stats_params, weights=None,
    over_neuron_mode='matrix_multiply_sanity', debug=False, second_weights=None
):
    """
    Various ways of computing "A" or "W" for a given group OR group/indirect neuron
    combination (e.g. in computing correlation). 
    
    
    An example usage would be the computation of 5c, where the y-axis's W, for a 
    given group, is given by the matrix multiplication of delta tuning and 
    indirect change in photostimulation.

    This function gives added functionality to decompose various contributions 
    to said matrix multiplication including things like doing mean subtraction
    or compuating the pearson correlation instead of matrix multiplication. All
    of these still function as taking something that is shape (n_neurons,) and 
    another quantity that is (n_neurons, n_groups,) to something that is 
    (n_groups,)

    Note that many of these measures require being more careful than simply doing
    a matrix multiplication, so the masks for each group are done by hand
    rather than relying on zeros in the neuron_by_group to properly mask things.
    
    Note this function can return NaNs for certain elements and they will be omitted
    by a nan mask later.
    
    Finally note that over_neuron_mode can be a list of modes, in which case it 
    iterates over the various modes
    
    INPUTS:
    - neuron_metric: string OR np.array of shape: (n_neurons,) OR (n_neurons2, n_neurons)
    - neuron_by_group: (n_neurons, n_groups,)
    - over_neuron_mode: string or list of strings
        - matrix_multiply: usual way of doing matrix multiplication that relies
            on zeros to mask groups
        - matrix_multiply_sanity: sanity check to make sure everything is done 
            correctly, should yield same results as matrix-multiply, but 
            eliminates group members by hand
        - means: isolate mean contribution to matrix_multiply_sanity
        - matrix_multiply_centered: isolate mean subtraction contribution to matrix_multiply_sanity
    - second_weights: optional additional weights to mask n_neurons2 if neuron_metric is 2d
    """
    
    group_vals_all = []
    if type(over_neuron_mode) == str: # Default case, creates single element tuple to iterate over
        on_modes = (over_neuron_mode,)
    else:
        on_modes = over_neuron_mode
        
    neuron_metric, neuron_by_group, weights = validation_tests(
        neuron_metric, neuron_by_group, ps_stats_params, weights=weights, second_weights=second_weights,
        on_modes=on_modes
    )
    
    if type(neuron_metric) != str:
        if ps_stats_params['neuron_metrics_adjust'] in ('normalize',):
            neuron_metric /= np.linalg.norm(neuron_metric, axis=-1, keepdims=True)
        elif ps_stats_params['neuron_metrics_adjust'] in ('standardize',):
            neuron_metric = (
                (neuron_metric - np.nanmean(neuron_metric, axis=-1, keepdims=True)) / 
                np.nanstd(neuron_metric, axis=-1, keepdims=True)
            )
        elif ps_stats_params['neuron_metrics_adjust'] != None:
            raise NotImplementedError('neuron_metrics_adjust {} not recognized.'.format(ps_stats_params['neuron_metrics_adjust']))
        
    for on_mode in on_modes: 
        # This is the slow way of doing things but treats things more carefully
        if on_mode in ('pearson', 'matrix_multiply_sanity', 'means', 'matrix_multiply_centered', 
                       'matrix_multiply_standardized', 'product_nc',
                       'neuron_centered_conn_standardized') or type(neuron_metric) == str:
            assert weights is not None

            n_groups = neuron_by_group.shape[-1]
            weights_mask = weights > 0 # (n_neurons, n_groups,)
            if type(neuron_metric) == str:
                if neuron_metric in ('raw_nc',): # Do not collapse over indirect neurons
                    group_vals = [] # (n_groups, n_neurons)
                else:
                    group_vals = np.zeros((n_groups,))
            elif on_mode in ('product_nc',): # Metrics that do not collapse over indirect neurons
                group_vals = [] # (n_groups, n_neurons)
            else:
                if len(neuron_metric.shape) == 1: # (n_neurons,)
                    group_vals = np.zeros((n_groups,))
                elif len(neuron_metric.shape) == 2: # (n_neurons2, n_neurons,)
                    group_vals = []
                    assert second_weights is not None
                    second_weights_mask = second_weights > 0 # (n_neurons, n_groups)

            for group_idx in range(n_groups):
                masked_neuron_by_group = neuron_by_group[weights_mask[:, group_idx], group_idx]

                if masked_neuron_by_group.shape[0]== 0: # Need to have at least one neuron make it past the mask
    #                     print('Mouse {}, group {}: no neurons in mask!'.format(
    #                         data_dict['data']['mouse'][day_1_idx], group_idx,
    #                     ))
                    if type(group_vals) == list: 
                        if len(neuron_metric.shape) == 1: # (n_neurons = 0)
                            group_vals.append(np.array(()))
                        elif len(neuron_metric.shape) == 2: # (n_neurons2,)
                            masked_neuron_metric = neuron_metric[second_weights_mask[:, group_idx], :]
                            group_vals.append(np.nan * np.ones(masked_neuron_metric.shape[0]))
                    else:
                        group_vals[group_idx] = np.nan
                    continue

                if type(neuron_metric) == str: # Special sum, all are independent of over_neuron_mode
                    if neuron_metric in ('raw_nc',):
                        assert on_mode in ('product_nc',)
                        group_vals.append(masked_neuron_by_group) # n_indirect
                    elif neuron_metric in ('sum',):
                        group_vals[group_idx] = np.sum(masked_neuron_by_group)
                    elif neuron_metric in ('abs_sum',):
                        group_vals[group_idx] = np.sum(np.abs(masked_neuron_by_group))
                    elif neuron_metric in ('mask_sum',): # Zeros eliminated in sum
                        group_vals[group_idx] = np.sum(weights[:, group_idx])
                else:
                    # Handle masking for the various possible shapes of the neuron metric
                    if len(neuron_metric.shape) == 1: # (n_neurons,)
                        masked_neuron_metric = neuron_metric[weights_mask[:, group_idx]]
                    elif len(neuron_metric.shape) == 2: # (n_neurons2, n_neurons,)
                        masked_neuron_metric = neuron_metric[:, weights_mask[:, group_idx]]
                        masked_neuron_metric = masked_neuron_metric[second_weights_mask[:, group_idx], :]
                        if on_mode not in ('matrix_multiply_sanity',):
                            NotImplementedError('neuron_metric shape not implemented for on_mode:', on_mode)
                    else:
                        raise NotImplementedError('neuron_metric shape not recognized:', neuron_metric.shape)                
                        
                    if on_mode in ('pearson',):
                        if masked_neuron_metric.shape[0] == 1: # Pearson makes no sense for a single neuron
                            group_vals[group_idx] = np.nan
                            continue
                        group_vals[group_idx] = np.corrcoef(masked_neuron_metric, masked_neuron_by_group)[0, 1]
                    elif on_mode in ('matrix_multiply_sanity',): # Sanity check that this yields same as mm above
                        if len(masked_neuron_metric.shape) == 1: # yields scale
                            group_vals[group_idx] = np.dot(masked_neuron_metric, masked_neuron_by_group)
                        elif len(masked_neuron_metric.shape) == 2: # yields (n_neurons2,)
                            group_vals.append(np.matmul(masked_neuron_metric, masked_neuron_by_group))
                    elif on_mode in ('product_nc',): # Sanity check that this yields same as mm above
                        group_vals.append(masked_neuron_metric * masked_neuron_by_group) # n_indirect
                    elif on_mode in ('means',):
                        group_vals[group_idx] = np.dot(
                            np.mean(masked_neuron_metric) * np.ones_like(masked_neuron_metric),
                            np.mean(masked_neuron_by_group) * np.ones_like(masked_neuron_by_group) 
                        )
                        # Uncomment below to decompose this quantity further
#                         group_vals[group_idx] = np.dot(
#                             np.mean(masked_neuron_metric) * np.ones_like(masked_neuron_metric),
#                             masked_neuron_by_group 
#                         )
#                         group_vals[group_idx] = np.mean(masked_neuron_by_group) # Just mean of masked_neuron_by_group (e.g. delta causal conn)
#                         group_vals[group_idx] = np.mean(masked_neuron_metric) # Just mean of neuron metric (e.g. tuning)
#                         group_vals[group_idx] = np.sum(masked_neuron_metric) # Sum of neuron metric instead of mean (proportional up to constant) 
                    elif on_mode in ('matrix_multiply_centered',):
                        group_vals[group_idx] = np.dot(
                            masked_neuron_metric - np.mean(masked_neuron_metric), 
                            masked_neuron_by_group - np.mean(masked_neuron_by_group)
                        )
                    elif on_mode in ('neuron_centered_conn_standardized',):
                        group_vals[group_idx] = np.dot(
                            masked_neuron_metric - np.mean(masked_neuron_metric), 
                            (masked_neuron_by_group - np.mean(masked_neuron_by_group)) / np.std(masked_neuron_by_group)
                        )
                    elif on_mode in ('matrix_multiply_standardized',):
                        group_vals[group_idx] = np.dot(
                            (masked_neuron_metric - np.mean(masked_neuron_metric)) / np.std(masked_neuron_metric), 
                            (masked_neuron_by_group - np.mean(masked_neuron_by_group)) / np.std(masked_neuron_by_group)
                        )

            group_vals_all.append(group_vals)
        elif on_mode == 'matrix_multiply':
            # Mask sum is already incorporated into neuron_by_group by setting 
            # corresponding elements to zero
            if weights is not None:
                raise ValueError('Weights dont make sense here.')

            group_vals_all.append(np.matmul(neuron_metric.T, neuron_by_group))
        else:
            raise ValueError('Over_neuron_mode {} not recognized.'.format(on_mode))
            
    if type(over_neuron_mode) == str: # Default case, returns first element in list
        return group_vals_all[0]
    else: # Retruns entire list instead
        return group_vals_all
    
# D_NEAR = 30
# D_FAR = 100

def get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict, 
    verbose=False,
):
    """
    Looks over several possible neuron-specific metrics to understand
    what explains causal connectivity for a SINGLE session.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'mice': [],
        'session_idxs': [],
        'ps_CC': [],
    }
    for key in ps_stats_params['connectivity_metrics']:
        records[key] = []
    
#     n_pairs = len(session_idx_pairs)
    n_sessions = len(session_idxs)
    
    for session_idx_idx, session_idx in enumerate(session_idxs):
        day_1_idx = session_idx
            
        records['session_idxs'].append(day_1_idx)
        records['mice'].append(data_dict['data']['mouse'][day_1_idx])
        
        data_to_extract = ['d_ps', 'trial_start_metrics', 'trial_start_fs', 'd_masks',]
        if 'pairwise_corr_x' in ps_stats_params['connectivity_metrics']: 
            data_to_extract.append('pairwise_corr')
        
        if ps_stats_params['direct_predictor_mode'] is not None:
            data_to_extract.append('resp_ps_pred')
            resp_ps_type = 'resp_ps_pred'
            if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
                data_to_extract.append('resp_ps')
                resp_ps_type = 'resp_ps'
        else:
            data_to_extract.append('resp_ps')
            resp_ps_type = 'resp_ps'
        
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params)
        
        # May need to save fit data for weights later on, saves in same format as records
        if ps_stats_params['indirect_weight_type'] is not None:
            assert ps_stats_params['direct_predictor_mode'] is not None # Doesn't make sense if we're not fitting
            
            if 'indirect_weights_w' not in records: # Initialize first time through
                records['indirect_weights_w'] = []
            group_vals_weights = [] # (n_groups, n_indirect)
            
            indir_mask = data_1['indir_mask'] > 0 # (n_neurons, n_groups,)
            for group_idx in range(indir_mask.shape[-1]):
                r_squareds_1 = data_1['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]
                if ps_stats_params['indirect_weight_type'] in ('rsquared',):
                    group_vals_weights.append(r_squareds_1)
                else:
                    raise NotImplementedError('indirect_weight_type {} not recognized'.format(ps_stats_params['indirect_weight_type']))
            records['indirect_weights_w'].append(group_vals_weights)
        
        for key in ps_stats_params['connectivity_metrics']: # All of these should take (n_neurons) x (n_neurons, n_groups) -> (n_groups)
            second_weights = None # Only needed when _x differs for each direct/indirect pair (like pairwise corr)
            if key[-2:] == '_x': # Metrics that capture laser-stimulated properties
                if key in ('pairwise_corr_x',):
                    neuron_weight_1 = data_1['pairwise_corr']
                    second_weights = data_1['indir_mask']
                elif key in ('tuning_x',):
                    neuron_weight_1 = data_1['tuning']
                elif key in ('trial_resp_x',):
                    neuron_weight_1 = data_1['trial_resp']
                elif key in ('post_resp_x',):
                    neuron_weight_1 = data_1['post']
                elif key in ('pre_resp_x',):
                    neuron_weight_1 = data_1['pre']
                elif key in ('cc_x',):
                    neuron_weight_1 = 'sum'
                elif key in ('cc_mag_x',):
                    neuron_weight_1 = 'abs_sum'
                elif key in ('mask_counts_x',):
                    neuron_weight_1 = 'mask_sum'
                group_vals_1 = eval_over_neurons(
                    neuron_weight_1, data_1[resp_ps_type], ps_stats_params, data_1['dir_mask'],
                    over_neuron_mode=ps_stats_params['x_over_neuron_mode'],
                    second_weights=second_weights
                )

            elif key[-2:] == '_y': # Metrics that capture causal connectivity properties
                if key in ('tuning_y',):
                    neuron_weight_1 = data_1['tuning']
                elif key in ('trial_resp_y',):
                    neuron_weight_1 = data_1['trial_resp']
                elif key in ('post_resp_y',):
                    neuron_weight_1 = data_1['post']
                elif key in ('pre_resp_y',):
                    neuron_weight_1 = data_1['pre']
                elif key in ('raw_cc_y',):
                    neuron_weight_1 = 'raw_nc'
                elif key in ('cc_y',):
                    neuron_weight_1 = 'sum'
                elif key in ('cc_mag_y',):
                    neuron_weight_1 = 'abs_sum'
                elif key in ('mask_counts_y',):
                    neuron_weight_1 = 'mask_sum'
                group_vals_1 = eval_over_neurons(
                    neuron_weight_1, data_1[resp_ps_type], ps_stats_params, data_1['indir_mask'],
                    over_neuron_mode=ps_stats_params['y_over_neuron_mode'],
                    second_weights=second_weights
                )
                
            else:
                raise NotImplementedError('Connectivity_metric {} not recognized.'.format(key))

            records[key].append(group_vals_1)
    
    return records
            
which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)

connectivity_metrics = (
    'cc_x',
    'pairwise_corr_x',
#     'tuning_x', 
    'trial_resp_x', 
#     'post_resp_x',
#     'pre_resp_x',
#     'cc_mag_x',
#     'mask_counts_x',
    
    'raw_cc_y',
#     'tuning_y',
    'trial_resp_y', 
#     'post_resp_y',
#     'pre_resp_y',
#     'cc_y',
#     'cc_mag_y',
#     'mask_counts_y',
)

plot_pairs = None
# plot_pairs = (
#     ('tuning_x', 'tuning_y'),
#     ('trial_resp_x', 'trial_resp_y'),
# #     ('mask_counts_x', 'mask_counts_y'),
# #     ('mean_tuning_x', 'delta_tuning_y'),
# #     ('mean_tuning_x', 'mean_tuning_y'),
# #     ('mean_tuning_x', 'cc_y'),
# #     ('mean_tuning_x', 'cc_mag_y'),
# #     ('delta_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'mean_trial_resp_y'),
# #     ('delta_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'mean_post_resp_y'),
# #     ('delta_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'mean_pre_resp_y'),
# )

ps_stats_params = {
    'ps_analysis_type': 'single_session',
    
    'trial_average_mode': 'time_first', # trials_first, time_first
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # None, normalize, standardize
    'pairwise_corr_type': 'trial', # trace, trial, pre, post
    
    ### Various ways of computing the over_neuron modes (this can be a tuple to do multiple)
    # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, 
    # matrix_multiply_standardized, neuron_centered_conn_standardized, product_nc
    'x_over_neuron_mode': 'matrix_multiply_sanity', #('matrix_multiply_sanity', 'matrix_multiply_centered', 'means',),
    'y_over_neuron_mode': 'product_nc', #('matrix_multiply_sanity', 'matrix_multiply_centered', 'means',),
    
    ### Fitting photostim parameters
    'direct_predictor_mode': None, # CHANGE 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average', # average, minimum
    'use_only_predictor_weights': False, 
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': plot_pairs,
    'group_weights_type': None, # None, direct_resp, direct_resp_norm_sessions
    'indirect_weight_type': None, # CHANGE 'rsquared', # None, rsquared, minimum_rsquared
    # shuffle_X_within_group, shuffle_X_within_session, shuffle_A_within_group, mean_X_within_session, fake_ps, shuffle_indirect_events
    'validation_types': (), # (), 
    
    'plot_up_mode': None, # all, significant, None; what to actually create plots for 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)

# if (20, 21) in session_idx_pairs:
#     print('Removing session pair to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

session_idxs = get_unique_sessions(session_idx_pairs, verbose=True)

print('Evaluating {} session idxs...'.format(len(session_idxs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])
    
records = get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict,
    verbose=False
)

print(STOP_HERE_FOR_NOW)
exemplar_session_idx = None

# Only do full fit if these are single modes, otherwise code not ready to handle
if type(ps_stats_params['x_over_neuron_mode']) == str and type(ps_stats_params['y_over_neuron_mode']) == str:
    full_fits, session_fits, ps_stats_params = scan_over_connectivity_pairs(
        ps_stats_params, records, exemplar_session_idx=exemplar_session_idx, verbose=True
)

def get_causal_connectivity_metrics_pair(
    ps_stats_params, session_idx_pairs, data_dict, verbose=False,
):
    """
    Looks over several possible neuron-specific metrics to understand
    what explains causal connectivity for a PAIRED sessions.
    
    INPUTS:
    session_idx_pairs: list of session pairs
    data_dict: loaded data file
    return_dict: compact dictionary for easy saving an futher plotting.
    
    """
    records = { # Fill with defaults used no matter what metrics
        'mice': [],
        'ps_pair_idx': [],
        'ps_CC': [],
        'cc_x': [], 
    }
    for key in ps_stats_params['connectivity_metrics']:
        records[key] = []
    
    n_pairs = len(session_idx_pairs)
    
    for pair_idx, session_idx_pair in enumerate(session_idx_pairs):
        day_2_idx = session_idx_pair[1]
        day_1_idx = session_idx_pair[0]
        
        assert day_2_idx > day_1_idx
        assert data_dict['data']['mouse'][day_2_idx] == data_dict['data']['mouse'][day_1_idx]
            
#         print('{} - Mouse {}, session_idxs {} and {} pass'.format(
#             pair_idx, data_dict['data']['mouse'][day_1_idx], day_1_idx, day_2_idx
#         ))
        records['ps_pair_idx'].append(pair_idx)
        records['mice'].append(data_dict['data']['mouse'][day_1_idx])
        
        data_to_extract = ['d_ps', 'trial_start_metrics', 'd_masks', 'pairwise_corr']
        
        if ps_stats_params['direct_predictor_mode'] is not None:
            if 'fake_ps' in ps_stats_params['validation_types']:
                raise NotImplementedError()
            
            data_to_extract.append('resp_ps_pred')
            resp_ps_type = 'resp_ps_pred'
            if ps_stats_params['use_only_predictor_weights']: # Special control case to see how much fit weights help
                data_to_extract.append('resp_ps')
                resp_ps_type = 'resp_ps'
        else:
            data_to_extract.append('resp_ps')
            resp_ps_type = 'resp_ps'
        data_1 = extract_session_data(day_1_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_2_idx)
        data_2 = extract_session_data(day_2_idx, data_dict, data_to_extract, ps_stats_params, paired_session_idx=day_1_idx)
        
        # Similarity of photostimulation via distances
        records['ps_CC'].append(np.corrcoef(data_1['d_ps'].flatten(), data_2['d_ps'].flatten())[0, 1])

        delta_tuning = data_2['tuning'] - data_1['tuning']        # change in tuning between days (n_neurons,)
        mean_tuning = 1/2 * (data_2['tuning'] + data_1['tuning']) # mean tuning across days (n_neurons,)

        delta_trial_resp = data_2['trial_resp'] - data_1['trial_resp']
        mean_trial_resp = 1/2 * (data_2['trial_resp'] + data_1['trial_resp'])

        delta_post_resp = data_2['post'] - data_1['post']
        mean_post_resp = 1/2 * (data_2['post'] + data_1['post'])

        delta_pre_resp = data_2['pre'] - data_1['pre']
        mean_pre_resp = 1/2 * (data_2['pre'] + data_1['pre'])
                                            
        indir_mask_weighted, dir_mask_weighted = get_dir_indir_masks(
            ps_stats_params, data_1, data_2, verbose=verbose
        )
        
        if ps_stats_params['mask_mode'] in ('constant', 'kayvon_match',): # Same mask for both sessions considered, * functions as AND

            delta_cc = indir_mask_weighted * (data_2[resp_ps_type] - data_1[resp_ps_type])

            delta_laser_resp = dir_mask_weighted * (data_2[resp_ps_type] - data_1[resp_ps_type])
            mean_laser_resp = dir_mask_weighted * 1/2 * (data_2[resp_ps_type] + data_1[resp_ps_type])
#             mean_laser_resp = dir_mask_weighted * (data_2[resp_ps_type] + data_1[resp_ps_type]) # No 1/2 for Kayvon's
        elif ps_stats_params['mask_mode'] in ('each_day',): # Individual distance masks for each session
            
            delta_cc = (data_2[resp_ps_type] * data_2['indir_mask'] - data_1[resp_ps_type] * data_1['indir_mask'])        # Change in causal connectivity, (n_neurons, n_groups)

            delta_laser_resp = (data_2[resp_ps_type] * data_2['dir_mask'] - data_1[resp_ps_type] * data_1['dir_mask'])        # Change in the laser response, (n_neurons, n_groups)
            mean_laser_resp = 1/2 * (data_2[resp_ps_type] * data_2['dir_mask'] + data_1[resp_ps_type] * data_1['dir_mask'])  # Mean laser response, (n_neurons, n_groups)
        
        # Always gather raw laser responses for both sessions because this is sometimes used for masking/weighting
        # Note stores the two pairs sessions as a tuple to stay in sync with other paired metrics
        records['cc_x'].append((
            eval_over_neurons('sum', data_1[resp_ps_type], ps_stats_params, dir_mask_weighted),
            eval_over_neurons('sum', data_2[resp_ps_type], ps_stats_params, dir_mask_weighted),
        ))
        
        # May need to save fit data for weights later on, saves in same format as other records
        if ps_stats_params['indirect_weight_type'] is not None:
            assert ps_stats_params['direct_predictor_mode'] is not None # Doesn't make sense if we're not fitting
            
            if 'indirect_weights_w' not in records: # Initialize first time through
                records['indirect_weights_w'] = []
            group_vals_weights = [] # (n_groups, n_indirect)
            
            indir_mask = indir_mask_weighted > 0 # (n_neurons, n_groups,)
            for group_idx in range(indir_mask.shape[-1]):
                r_squareds_1 = data_1['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]
                r_squareds_2 = data_2['resp_ps_pred_extras']['r_squareds'][indir_mask[:, group_idx], group_idx]            
                if ps_stats_params['indirect_weight_type'] in ('minimum_rsquared',):
                    group_vals_weights.append(np.minimum(r_squareds_1, r_squareds_2))
                else:
                    raise NotImplementedError('indirect_weight_type {} not recognized'.format(ps_stats_params['indirect_weight_type']))
            records['indirect_weights_w'].append(group_vals_weights)
        
        for key in ps_stats_params['connectivity_metrics']: # All of these should take (n_neurons) x (n_neurons, n_groups) -> (n_groups)
            second_weights = None # Only needed when _x differs for each direct/indirect pair (like pairwise corr)
            if key[-2:] == '_x': # Metrics that capture laser-stimulated properties
                resp_weight = mean_laser_resp
#                         resp_weight = delta_laser_resp
                if key in ('delta_laser_x',):
                    neuron_weight = 'sum'
                    resp_weight = delta_laser_resp
                elif key in ('laser_resp_x',):
                    neuron_weight = 'sum'
                elif key in ('delta_pairwise_corr_x',):
                    neuron_weight = data_2['pairwise_corr'] - data_1['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('pairwise_corr_1_x',): # Just corr from Day 1
                    neuron_weight = data_1['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('pairwise_corr_2_x',): # Just corr from Day 2
                    neuron_weight = data_2['pairwise_corr']
                    second_weights = indir_mask_weighted
                elif key in ('delta_tuning_x',):
                    neuron_weight = delta_tuning
                elif key in ('mean_tuning_x',):
                    neuron_weight = mean_tuning
                elif key in ('delta_trial_resp_x',):
                    neuron_weight = delta_trial_resp
                elif key in ('mean_trial_resp_x',):
                    neuron_weight =  mean_trial_resp
                elif key in ('delta_post_resp_x',):
                    neuron_weight = delta_post_resp
                elif key in ('mean_post_resp_x',):
                    neuron_weight = mean_post_resp
                elif key in ('delta_pre_resp_x',):
                    neuron_weight = delta_pre_resp
                elif key in ('mean_pre_resp_x',):
                    neuron_weight = mean_pre_resp
                elif key in ('laser_resp_mag_x',):
                    neuron_weight = 'abs_sum'
                elif key in ('mask_counts_x',):
                    neuron_weight = 'mask_sum'

                group_vals = eval_over_neurons(
                    neuron_weight, resp_weight, ps_stats_params, dir_mask_weighted,
                    over_neuron_mode=ps_stats_params['x_over_neuron_mode'],
                    second_weights=second_weights,
                )

            elif key[-2:] == '_y': # Metrics that capture causal connectivity properties
                resp_weight = delta_cc

                if key in ('delta_tuning_y',):
                    neuron_weight = delta_tuning # 5c's W
                elif key in ('mean_tuning_y',):
                    neuron_weight = mean_tuning
                elif key in ('delta_trial_resp_y',):
                    neuron_weight = delta_trial_resp
                elif key in ('mean_trial_resp_y',):
                    neuron_weight = mean_trial_resp
                elif key in ('delta_post_resp_y',):
                    neuron_weight = delta_post_resp
                elif key in ('mean_post_resp_y',):
                    neuron_weight = mean_post_resp 
                elif key in ('delta_pre_resp_y',):
                    neuron_weight = delta_pre_resp
                elif key in ('mean_pre_resp_y',):
                    neuron_weight = mean_pre_resp 
                elif key in ('raw_delta_cc_y',):
                    neuron_weight = 'raw_nc'
                elif key in ('delta_cc_y',):
                    neuron_weight = 'sum'
                elif key in ('delta_cc_mag_y',):
                    neuron_weight = 'abs_sum'
                elif key in ('mask_counts_y',):
                    neuron_weight = 'mask_sum'

                group_vals = eval_over_neurons(
                    neuron_weight, resp_weight, ps_stats_params, indir_mask_weighted, 
                    over_neuron_mode=ps_stats_params['y_over_neuron_mode'],
                    second_weights=second_weights,
                )
            else:
                raise NotImplementedError('Connectivity_metric {} not recognized.'.format(key))

            # Some of these can be nans because a group might not have valid direct/indirect mask
            # Currently case for BCI35 group 52, BCI37, group 11
            records[key].append(group_vals)
        
    return records
                
connectivity_metrics = (
    'delta_laser_x', 
    'laser_resp_x',
    'delta_pairwise_corr_x',
#     'pairwise_corr_1_x',
#     'pairwise_corr_2_x',
#     'delta_tuning_x',
#     'mean_tuning_x',
#     'delta_trial_resp_x',
#     'mean_trial_resp_x',
    'delta_post_resp_x',
    'mean_post_resp_x',
#     'delta_pre_resp_x',
#     'mean_pre_resp_x',
#     'laser_resp_x',
#     'laser_resp_mag_x',
#     'mask_counts_x',
    
    'raw_delta_cc_y',
#     'delta_tuning_y',
#     'mean_tuning_y',
#     'delta_trial_resp_y',
#     'mean_trial_resp_y',
    'delta_post_resp_y',
    'mean_post_resp_y',
#     'delta_pre_resp_y',
#     'mean_pre_resp_y',
#     'delta_cc_y',
#     'delta_cc_mag_y',
#     'mask_counts_y',
)

plot_pairs = None
# plot_pairs = (
#     ('delta_tuning_x', 'delta_tuning_y'),
# #     ('mean_trial_resp_x', 'delta_tuning_y'),
# #     ('mask_counts_x', 'mask_counts_y'),
# #     ('mean_tuning_x', 'delta_tuning_y'),
# #     ('mean_tuning_x', 'mean_tuning_y'),
# #     ('mean_tuning_x', 'cc_y'),
# #     ('mean_tuning_x', 'cc_mag_y'),
# #     ('delta_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'delta_trial_resp_y'),
# #     ('mean_trial_resp_x', 'mean_trial_resp_y'),
# #     ('delta_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'delta_post_resp_y'),
# #     ('mean_post_resp_x', 'mean_post_resp_y'),
# #     ('delta_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'delta_pre_resp_y'),
# #     ('mean_pre_resp_x', 'mean_pre_resp_y'),
# )

ps_stats_params = {
    'ps_analysis_type': 'paired',
    
    'trial_average_mode': 'time_first', # trials_first, time_first
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'constant', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': None, # None, normalize, standardize
    'pairwise_corr_type': 'trace', # trace, trial, pre, post
    
    ### Various ways of computing the over_neuron modes for x and y ###
    # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, 
    # matrix_multiply_standardized, neuron_centered_conn_standardized, product_nc
    'x_over_neuron_mode': 'matrix_multiply_sanity',
    'y_over_neuron_mode': 'product_nc',
    
    ### Fitting photostim parameters
    'direct_predictor_mode': 'sum', # None, sum, top_mags, top_devs, top_devs_center
    'n_direct_predictors': 1,
    'direct_predictor_intercept_fit': False,
    'direct_predictor_nan_mode': 'ignore_nans', # ignore_nans, eliminate_events, 
    'direct_input_mode': 'average_equal_sessions', # average, average_equal_sessions, minimum
    'modify_direct_weights': True,
    'use_only_predictor_weights': False,
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': plot_pairs,
    'group_weights_type': None, # None, direct_resp, direct_resp_norm_sessions
    'indirect_weight_type': 'minimum_rsquared', # None, rsquared, minimum_rsquared
    # shuffle_X_within_group, shuffle_X_within_session, shuffle_A_within_group, mean_X_within_session, fake_ps, shuffle_indirect_events
    'validation_types': (), # (), 

    'plot_up_mode': 'all', # all, significant, None; what to actually create plots for 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
# if (20, 21) in session_idx_pairs:
#     print('Removing session to match Kayvons sessions')
#     session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))
if ps_stats_params['validation_types'] != ():
    print('Using validations:', ps_stats_params['validation_types'])

records = get_causal_connectivity_metrics_pair(
    ps_stats_params, session_idx_pairs, data_dict, verbose=False,
)

print(STOP_HERE_FOR_NOW)
exemplar_session_idx = None

full_fits, session_fits, ps_stats_params = scan_over_connectivity_pairs(
    ps_stats_params, records, exemplar_session_idx=exemplar_session_idx, verbose=True
)

# %% Cell 104: Testing weighted linear regression equivalence so that getting p-values is easy.
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

np.random.seed(111111)

x = np.arange(5, 30, dtype=np.float32)
y = 0.3 * x + np.random.normal(size=x.shape)

# weights = np.abs(np.random.normal(size=x.shape))
weights = None

reg = LinearRegression()
reg.fit(x[:, np.newaxis], y[:, np.newaxis], sample_weight=weights)

print('LinearRegression slope:', reg.coef_[0, 0])

if weights is not None:
#     x_copy = np.sqrt(weights) * x
#     y_copy = np.sqrt(weights) * y
    x_copy = np.matmul(np.diag(np.sqrt(weights)), x)
    y_copy = np.matmul(np.diag(np.sqrt(weights)), y)    
else:
    x_copy = x.copy()
    y_copy = y.copy()

X = x_copy[:, np.newaxis]
Y = y_copy[:, np.newaxis]
print('Inverse slope:', np.linalg.inv(X.T @ X) @ X.T @ Y)
    
slope, intercept, rvalue, pvalue, se = linregress(x_copy, y_copy)
print('linregress slope:', slope)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax1.scatter(x, y, color='k', marker='.')

fig.show()

# %% Cell 106: Minimum reproduction of Kayvon's ps for 5c
# Some globals may have changed, so restore to Kayvon's old values

D_NEAR = 30                   # Distance from photostim (microns) to include as minumum distance for indirect
D_FAR = 100                   # Distance from photostim (microns) to include as maximum distance for indirect
D_DIRECT = 30                 # Distance from photostim (microns) if less than considered targeted by the photostim

SAMPLE_RATE = 20 # Hz
T_START = -2 # Seconds, time relative to trial start where trial_start_fs begin
TS_POST = (0, 10) # Seconds, time points to include for post-trial start average
TS_PRE = (-2, -1) # Seconds, time points to include for pre-trial start average

IDXS_PRE_PS = np.arange(0, 4)   # Indexes in FStim to consider pre-photostim response (~200 ms)
IDXS_PS = (4, 5, 6, 7, 8,)      # Indexes in FStim to consider photostim response (~250 ms of PS)
IDXS_POST_PS = np.arange(9, 16) # Indexes in FStim to consider post-photostim response (~350 ms)

connectivity_metrics = (
    'delta_tuning_x',
    'delta_tuning_y',
)

plot_pairs = (
    ('delta_tuning_x', 'delta_tuning_y'),
)

ps_stats_params = {
    'trial_average_mode': 'trials_first', # trials_first, time_first
    'resp_ps_average_mode': 'trials_first',
    'resp_ps_n_trials_back_mask': 1, # Skip PS response when neuron has been stimulated recently
    'mask_mode': 'kayvon_match', # constant, each_day, kayvon_match
    'normalize_masks': False, # Normalize distance masks by the number of neurons that make it through
    'neuron_metrics_adjust': 'normalize', # Normalize things like tuning before computing metrics
    
    # Various ways of computing the over_neuron modes
    'x_over_neuron_mode': 'matrix_multiply_sanity', # matrix_multiply, pearson, matrix_multiply_sanity, means, matrix_multiply_centered, matrix_multiply_standardized
    'y_over_neuron_mode': 'matrix_multiply_sanity',
    
    ### Plotting/fitting parameters
    'fit_individual_sessions': True, # Fit each individual session too
    'connectivity_metrics': connectivity_metrics,
    'plot_pairs': plot_pairs,
    
    
    'ps_analysis_type': 'paired',
    'plot_up_mode': 'all', # all, significant, None; what to actually create plots for 
}
ps_stats_params = default_ps_stats_params(ps_stats_params)

which_sessions_to_include = np.arange(len(data_dict['data']['F'])-1)
session_idx_pairs = find_valid_ps_pairs(
    ps_stats_params, which_sessions_to_include, data_dict,
    verbose=False
)
if (20, 21) in session_idx_pairs:
    print('Removing session to match Kayvons sessions')
    session_idx_pairs.remove((20, 21))

print('Evaluating {} session pairs...'.format(len(session_idx_pairs)))

records = get_causal_connectivity_metrics_pair(
    ps_stats_params, session_idx_pairs, data_dict, verbose=False,
)

full_fits, _, _ = scan_over_connectivity_pairs(ps_stats_params, records, verbose=False)

print('P value result:  \t', full_fits[('delta_tuning_x', 'delta_tuning_y')]['pvalue'])
print('P value to match:\t 0.014509242863871622')

# %% Cell 108: A check on a single-session metric also using Kayvon's setup
ps_stats_params['connectivity_metrics'] = (
    'trial_resp_x',
    'trial_resp_y',
)

ps_stats_params['plot_pairs'] = (
    ('trial_resp_x', 'trial_resp_y'),
)


session_idxs = get_unique_sessions(session_idx_pairs, verbose=False)

print('Evaluating {} session idxs...'.format(len(session_idxs)))

records = get_causal_connectivity_metrics_single(
    ps_stats_params, session_idxs, data_dict, verbose=False
)

full_fits = scan_over_connectivity_pairs(ps_stats_params, records, verbose=False)

print('P value result:  \t', full_fits[('trial_resp_x', 'trial_resp_y')]['pvalue'])

# %% Cell 111: If you want to access the corresponding behavior files and data which has a slightly different motif...
# mypath = '/data/bci_oct24_upload/'
BEHAV_DATA_PATH = '/data/bci_data/'

behav, data, maps = get_behav_and_data_maps(BEHAV_DATA_PATH, verbose=False)
session_idx_to_behav_idx = maps['session_idx_to_behav_idx']
session_idx_to_data_idx = maps['session_idx_to_data_idx']
print('Done!')

# %% Cell 112: Code starts with: session_idx = 20 # 9, 15
session_idx = 20 # 9, 15
load_behav = True
load_data = False

# Load the corresponding behavioral for each paired session
if load_behav:
    behav_idx = session_idx_to_behav_idx[session_idx]
    print('Loading behav from: {}'.format(mypath + behav[behav_idx]))
    data_dict_behav = scipy.io.loadmat(mypath + behav[behav_idx])

# Load the corresponding data
if load_data: 
    data_idx = session_idx_to_data_idx[session_idx]
    print('Loading data from: {}'.format(mypath + data[data_idx]))
    data_photo = scipy.io.loadmat(mypath + data[data_idx])

# %% Cell 113: Code starts with: # Plot some exemplar profiles
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

# %% Cell 114: Code starts with: SAMPLE_RATE = 20 # Hz
SAMPLE_RATE = 20 # Hz
T_START = -2 # Seconds, time relative to trial start where trial_start_fs begin
TS_POST = (0, 10) # Seconds, time points to include for post-trial start average
TS_PRE 

idxs_post = np.arange(
int((TS_POST[0] - T_START) * SAMPLE_RATE), int((TS_POST[1] - T_START) * SAMPLE_RATE),
)
idxs_pre = np.arange(
    int((TS_PRE[0] - T_START) * SAMPLE_RATE), int((TS_PRE[1] - T_START) * SAMPLE_RATE),
)

# %% Cell 115: Code starts with: ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_idxs = np.arange(
    int((ts_range[0] - T_START) * SAMPLE_RATE), int((ts_range[1] - T_START) * SAMPLE_RATE),
)

n_ts_idxs = ts_idxs.shape[0]

print(ts_idxs.shape[0])

# %% Cell 116: Code starts with: dff = data_dict_behav['df_closedLoop']
dff = data_dict_behav['df_closedLoop']

trial_start_idxs = np.where(data_dict_behav['trial_start'][0, :] == 1)[0]
rew_idxs = np.where(data_dict_behav['rew'][0, :] == 1)[0]
n_trial_start = trial_start_idxs.shape[0]

ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_idxs = np.arange(
    int(ts_range[0] * SAMPLE_RATE), int(ts_range[1] * SAMPLE_RATE),
) - 1 # - 1 keeps these in sync
n_ts_idxs = ts_idxs.shape[0]

n_neurons = dff.shape[1]

trial_start_dff = np.empty((n_ts_idxs, n_neurons, n_trial_start))
trial_start_dff[:] = np.nan

# Pads dff with nans so we don't need to trim start and end
nan_pad = np.empty((n_ts_idxs, n_neurons,))
nan_pad[:] = np.nan

print(dff.shape)

dff = np.concatenate((nan_pad, dff, nan_pad), axis=0)

print(dff.shape)

for trial_start_idx_idx, trial_start_idx in enumerate(trial_start_idxs):
    
    rel_trial_start_idx = trial_start_idx + n_ts_idxs # Accounts for nan padding
    
#     print(rel_trial_start_idx)
#     print(next_rel_trial_start_idx)
#     print(ts_idxs[:10])
    
    trial_start_dff[:, :, trial_start_idx_idx] = dff[rel_trial_start_idx + ts_idxs, :]
#     if trial_start_idx_idx < n_trial_start - 1:
#         next_rel_trial_start_idx = trial_start_idxs[trial_start_idx_idx + 1] + n_ts_idxs 
#         trial_start_dff[next_rel_trial_start_idx - rel_trial_start_idx:, :, trial_start_idx_idx] = np.nan
    
# This is a hack that just copies tha nan statistics from one to the other
trial_start_dff = np.where(np.isnan(data_dict['data']['F'][session_idx_idx]), np.nan, trial_start_dff)


# %% Cell 117: Code starts with: idxs_post = np.arange(
idxs_post = np.arange(
int((TS_POST[0] - T_START) * SAMPLE_RATE), int((TS_POST[1] - T_START) * SAMPLE_RATE),
)
idxs_pre = np.arange(
    int((TS_PRE[0] - T_START) * SAMPLE_RATE), int((TS_PRE[1] - T_START) * SAMPLE_RATE),
)

fit_changes = True

tuning, trial_resp, pre, post, ts_extras  = compute_trial_start_metrics(
    data_dict['data']['F'][session_idx_idx], (idxs_pre, idxs_post), mean_mode='time_first',
    fit_changes=fit_changes
)

tuning_n, trial_resp_n, pre_n, post_n, ts_extras_n  = compute_trial_start_metrics(
    trial_start_dff, (idxs_pre, idxs_post), mean_mode='time_first',
    fit_changes=fit_changes
)

# %% Cell 118: Code starts with: fig, ((ax1, ax2,), (ax3, ax4,)) = plt.subplots(2, 2, figsize=(8, 8))
fig, ((ax1, ax2,), (ax3, ax4,)) = plt.subplots(2, 2, figsize=(8, 8))

ax1.scatter(tuning_n, tuning, marker='.', color=c_vals[0])
ax1.set_xlabel('Tuning from df_closedLoop')
ax1.set_ylabel('Tuning from data_dict[F]')

ax2.scatter(trial_resp_n, trial_resp, marker='.', color=c_vals[0])
ax2.set_xlabel('Trial resp. from df_closedLoop')
ax2.set_ylabel('Trial resp. from data_dict[F]')

ax3.scatter(post_n, post, marker='.', color=c_vals[0])
ax3.set_xlabel('Post from df_closedLoop')
ax3.set_ylabel('Post from data_dict[F]')

ax4.scatter(pre_n, pre, marker='.', color=c_vals[0])
ax4.set_xlabel('Pre from df_closedLoop')
ax4.set_ylabel('Pre from data_dict[F]')

for ax in (ax1, ax2, ax3, ax4):
    ax.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
    ax.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

# %% Cell 119: Code starts with: fig, ax = plt.subplots(1, 1, figsize=(6, 4))
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

trial_idx = 40
neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][:10, neuron_idx, trial_idx], color='r', label='data_dict F')
ax.plot(trial_start_dff[:10, neuron_idx, trial_idx], color='b', label='my trial_start aligned')
ax.legend()

# %% Cell 120: Code starts with: fig, ax = plt.subplots(1, 1, figsize=(6, 4))
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][40:, neuron_idx, 0], color='r')
ax.plot(data_dict_behav['df_closedLoop'][:300, neuron_idx], color='b')

scale = data_dict['data']['F'][session_idx_idx][40, neuron_idx, 0] / data_dict_behav['df_closedLoop'][0, neuron_idx]

# ax.plot(scale * data_dict_behav['df_closedLoop'][:300, neuron_idx], color='g')

# %% Cell 121: Code starts with: session_idx = 18
session_idx = 18

ps_stats_params = {
    'pairwise_corr_type': 'behav_full', # trace, trial, pre, post, behav_full, behav_start, behav_end
}

neuron_corrs = get_correlation_from_behav(session_idx, ps_stats_params)
trace_corr = data_dict['data']['trace_corr'][session_idx]

print(neuron_corrs.shape)
print(trace_corr.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.scatter(neuron_corrs.flatten(), trace_corr.flatten(), marker='.', color='k', alpha=0.005)

ax1.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
ax1.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

ax1.set_xlabel('Pairwise corr from corrcoef(df_closedLoop)')
ax1.set_ylabel('Pairwise corr from trace_corr')

ps_stats_params['pairwise_corr_type'] = 'behav_start'
start_corrs = get_correlation_from_behav(session_idx, ps_stats_params)
ps_stats_params['pairwise_corr_type'] = 'behav_end'
end_corrs = get_correlation_from_behav(session_idx, ps_stats_params)

ax2.scatter(start_corrs.flatten(), end_corrs.flatten(), marker='.', color='k', alpha=0.005)

ax2.axhline(0.0, color='lightgrey', zorder=5, linestyle='dashed')
ax2.axvline(0.0, color='lightgrey', zorder=5, linestyle='dashed')

ax2.set_xlabel('Pairwise corr start')
ax2.set_ylabel('Pairwise corr end')

# %% Cell 122: Code starts with: # Some Moore-Penrose inversion sanity checks to ensure shifting to neuron -> neuro...
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
