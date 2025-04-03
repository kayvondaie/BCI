from utils import preprocessing, plotting, metrics
import config

# Cell 1
# BCI/Photostim analysis

This analysis primarily draws from a combined data_dict file and sometimes from a corresponding behav file too.  

data_dict (combined data) keys:
* "F" : A list across sessions where each entry is the (time x neuron x trial) response during behavior triggered by trial start (CRITICAL)
* "Fraw": The same as "F" but without df/f normalization
* "Fstim": The response of each neuron to each photostimulation event in (time x neuron x event) form (CRITICAL)
* "Fstim_sort" : The response of every neuron to each presentation of each group sorted by which group was active
* "GRP": A flattened index matrix, 1s for first n_neurons, 2s for next n_neurons, ... (Neuron x Photostim Group)
* "Gx" : distance to center of the SLM fov (targeted neurons only)
* "Gx_all" : distance to center of the SLM fov (all neurons)distance to center of the SLM fov (all neurons)
* "Gy" : Response amplitude (targeted neurons only)
* "condition_coordinates": where in the field of view the CN is positioned
* "condition_neuron": The id for the CN in matlab index (CRITICAL)
* "dat_file": The file loaction for the raw data_file (see the files in data/Jan18/bci_session_data/)
* "dist" : The distance between the CN and the neuron with that ID  (CRITICAL)
* "dt_si" : The sampling frequency is 20 Hz
* "e": The Standard Devision of the response of each neuron to each photostim group (Neuron x PSgroup) which has been flattened by mat73's import.
* "id": Needs clarification
* "mouse": The mouse's name to be used to compare across mice (CRITICAL)
* "pval" : The pval of the response of each neuron to each group.
* "seq": The matlab indexed id for each group of 10 neurons at each time (CRITICAL)
* "session": The MMDDYYYY datetime of the session
* "stim": The relative magnitude of the laser power
* "stimAmp": The mean amplitude of the stim response ; identical to reshaped "y"
* "stimDist": The distance betweeen each neuron and the closest point illuminated for each photostim group ; identical to reshaped "x"
* "trace_corr": The cross-correlation (zero lag) between every pair of neurons during behavior. (CRITICAL)
* "tsta": The time samples relative to the event in our time x neuron x events tensors
* "x": The distance between a neuron and its nearest photostim point for each group which has been flattened by mat73's import. (CRITICAL)
* "y": The mean response for a neuron to a given photostimulation which has been flattened by mat73's import. 

Behav data file keys:
- F_rew: Reward aligned dff. (rel_time, neuron, trial) 
- F_pre: Pre-trial aligned dff. (rel_time, neuron, trial) 
- vel: Velocity of lickspout. Note if lickspout is already at reward position, still on but lickspout not actually moving. Binary variable. (1, abs_time)
- rew: Reward achieved requires lickspout to be in position and the mouse to be licking. Zero if a miss trial. Binary variable. (1, abs_time)
- thr: When the lickspout reaches threshold position at -1 mm (does not trigger reward until mouse licks). Binary variable. (1, abs_time)
- trial_start: Trial start. Binary variable. (1, abs_time)
- dt_si: dt between steps. Float.
- df_closedLoop: Raw dff (abs_time, neuron)
- dist:
- conditioned_neuron:
- conditioned_coordinates:
- mouse:
- session:
- dat_file:

# Cell 2
Loads the combined data that is used across all analyses below (see above for details). Data loading may take a few minutes.

# Cell 3
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.stats as stats
import io
try:
    import mat73
except:
    !pip install mat73
    import mat73
%matplotlib inline

mypath2 = '/data/bci/combined_new_old_060524.mat'
BEHAV_DATA_PATH = '/data/bci_data/'

print('Loading data_dict...')
data_dict = mat73.loadmat(mypath2)
print('Done!')

# Cell 4
Various helper functions and nice colors.
- add_regression_line: fit OLS or WLS and optionally plot
- add_bin_plot: add binned data to plot (equal spaced bins, uneven numbers in bins)
- participation_ratio_vector: compute PR
- shuffle_along_axis
- add_identity
- visualize_dict

# Cell 6
### Set global parameters
Extracts maps that allow one to connect combined data to raw data files

# Cell 7
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

# Cell 8
### Helper functions
Helper functions for evaluating pairs of sessionsHelper functions for evaluating pairs of sessions
- unflatted_neurons_by_groups: Used for unflattening loaded MatLab data (n_neurons * n_groups,) -> (n_neurons, n_groups,)
- compute_resp_ps_mask_prevs: Computes the PS responses and statitistics from FStim with nan masking of previously directly stimulated neurons
- find_valid_ps_pairs: From combined data, evaluates potential pairs of sessions for analysis
- get_unique_sessions: Takes pairs of sessions and extracts unique sessions
- extract_session_data: Extracts useful data from a session, including many pre-processing steps
    - compute_trial_start_metrics: Computes metrics related to trial_starts (e.g. tuning, trial response, etc.)
    - get_dir_indir_masks: Combines two sessions distances to get single direct/indirect mask for both sessions.
    - compute_cross_corrs_special: Computes pairwise correlations from trial-start-aligned fluorescence ('F')
    - get_correlation_from_behav: Computes pairwise correlations from behav files
    - get_fake_ps_data: Make fake photostim data for calibration    
Optional aditional weighting features such as normalizing by number of neurons that get past mask

# Cell 10
### Fit PS functions
Functions for fitting the variation in photostimulation response. 
- get_resp_ps_pred: Wrapper function to efficiently get predictions
    - find_photostim_variation_predictors: determines directions in direct neuron space to use as independent predictors of photostimulation response
    - fit_photostim_variation: given the direct_predictors, fits the photostim variation
    - photostim_predict: given parameters and direct input, yield photostim predictions

# Cell 12
# Fit PS Responses

# Cell 13
### Exemplar plots
This generates some exemplar plots showing various ways of doing interpolation on direct photostimulation data in order to quantify 

# Cell 14
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


# Cell 15
### Many session fitting

Now use the fitting metrics across several sessions and session pairs to compare to old method of computing metrics.

# Cell 16
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

# Cell 17
For pairs of sessions, evaluate dissimilarity of photostimulation to justify needing to control for it. Also evaluate some statistics of each group's photostimulation such as its dimensionality and number of direct/events

# Cell 18
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

# Cell 19
Separate out this plot because code ocean is having problems outputting both plots above.

# Cell 20
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

# Cell 21
Some additional nan statistics.

# Cell 22
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

# Cell 23
#### Single session
Use the fit parameters in single-sessions measures.

# Cell 24
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

# Cell 25
#### Paired sessions
Same as above but for paired sessions. First find valid session pairs

# Cell 26
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

# Cell 27
Consistency of fits across pairs

# Cell 29
# Global photostim results
("A" vs "W" plots)

# Cell 30
Some helper functions for the below code

# Cell 32
### Fit functions
Functions for taking various raw "A" and "W" metrics and fitting results.Functions for taking various raw "A" and "W" metrics and fitting results. Here "connectivity metrics" are various types of A or W's and "plot_pairs" are possible combinations of A and W to plot/fit against one another. Functions to compute raw "A" and "W" metrics defined below.

Single linear regression functions:
- scan_over_connectivity_pairs: gets all possible combinations of A vs. W, fits them against one another, AND generates corresponding plots.

Multiple linear regression functions:
- enumerate_plot_pairs: enumerate all possible A and W
- get_all_xs: gathers all A together to be plotted at once, does some reshaping if needed
- fit_all_xs_at_once: fit all A at once using MLR, note this does NOT generate plots, need to run additional code below
- get_hier_bootstrap_shuffle: optional hier boostrapping for MLR fits

# Cell 34
### Single session
Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Gathers A and W metrics single sessions (not session pairs). Change "ps_fit_type" to move between SLR and MLR fits.

# Cell 35
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

# Cell 36
### Paired sessions
Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Functions and runs for garther raw "A" and "W" metrics (and fitting for SLR). Gathers A and W metrics session pairs (not single sessions). Change "ps_fit_type" to move between SLR and MLR fits.

# Cell 37
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

# Cell 38
### MLR fit/plot
Further MLR fitting/plotting code for both single and paired sessions
Fit raw A and W using MLR. Can run with optional boostrapping. Just gathers fit results, code below for plotting up.

# Cell 39
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

# Cell 40
Run this for plotting up results from fit_all_xs_at_once without bootstrapping

# Cell 41
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
        

# Cell 42
Run this for an analysis on bootstrapped metrics (not the best plots, work in progress)

# Cell 44
Sometimes want to fit against some validation data, run this to save full_fits as validation

# Cell 45
import copy
valid_full_fits = copy.deepcopy(full_fits)
print('Fit assigned as validation.')

# Cell 46
# CN specific changes

Initialize some parameters that are used in all tests below. Also function for drawing CN-like neurons.

# Cell 48
### Single session

# Cell 49
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

# Cell 50
### Paired (no CC)
Just plot some paired statistics to check if distributions looks different for certain pairs. This can does not look at 

# Cell 51
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

# Cell 52
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

# Cell 53
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

# Cell 54
Distance to CN versus trial-start (within session) change plots.

# Cell 55
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

# Cell 56
Distance to CN vs. generic neuron metric plot

# Cell 57
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

# Cell 58
### Paired (w CC)
Similar to scan above, but actually gets the predicted responses too so takes a lot longer to collect.

# Cell 59
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

# Cell 60
This is not a CN specific test.

# Cell 61
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

# Cell 62
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

# Cell 63
# Photostimulation Sanity Checks
For a single session, look at some PS statistics. Generates some basic plots that serve as sanity checks on how the PS response behave in an exemplar session and neuron within that session.

Valid session idxs right now: [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]

# Cell 64
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


# Cell 65
Similar to above cell but runs over all session_idxs rather than a single session. Still not looking at anything pair-dependent, just doing aggregate photostim analyses

# Cell 66
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


# Cell 67
Some basic analysis on the effects of photostimulation on responses for paired sessions

# Cell 68
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

# Cell 69
## Look for simple predictors of photostim, like correlations

Tries to see how well correlations predict changes in causal connectivity

# Cell 70
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

# Cell 71
For single sessions, looks at what is a good predictor of causal connectivity.

# Cell 72
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

# Cell 73
Same as above but now tries to fit change in correlation between two session to the resulting change in causal connectivity

# Cell 74
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

# Cell 75
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

# Cell 76
### Spurious Correlation Checks
Looks for basic levels of correlation for a given group's causal connectivity to eliminate possible underlying sources of correlations

# Cell 77
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

# Cell 78
Checks for group correlations in neuron quantities in individual sessions and for between-sessions. Completmentary to above analysis that looks at causal connectivity correlations. Useful for justifying if we need to control for mean subtraction or standarization

# Cell 79
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

# Cell 80
# Old Code

# Cell 81
Old version of this function that outputs in a slightly different format. New version updated to be more compatible with bootstrapping.

# Cell 83
MLR fits with base_xs only instead of all _x

# Cell 84
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

# Cell 85
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

# Cell 86
Get some metrics related to laser responses and # of directly stimulated neurons and see how many groups are being omitted

# Cell 87
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

# Cell 88
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

# Cell 90
This code was used to check that if we do the fit on the top magnitude direct neurons, if the predicted responses of the indirect neurons were consistent or not.

# Cell 91
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

# Cell 92
Chasing down why certain neruons are nans...

# Cell 94
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

# Cell 95
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

# Cell 96
# Code appendix

# Cell 97
Testing weighted linear regression equivalence so that getting p-values is easy.

# Cell 98
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

# Cell 99
Minimum reproduction of Kayvon's ps for 5c

# Cell 100
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

# Cell 101
A check on a single-session metric also using Kayvon's setup

# Cell 102
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

# Cell 103
# Load behav and data
Some code to further analyze raw behav and data files.

# Cell 104
If you want to access the corresponding behavior files and data which has a slightly different motif structure. 

# Cell 105
# mypath = '/data/bci_oct24_upload/'
BEHAV_DATA_PATH = '/data/bci_data/'

behav, data, maps = get_behav_and_data_maps(BEHAV_DATA_PATH, verbose=False)
session_idx_to_behav_idx = maps['session_idx_to_behav_idx']
session_idx_to_data_idx = maps['session_idx_to_data_idx']
print('Done!')

# Cell 107
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

# Cell 109
ts_range = (np.min(TS_PRE), np.max(TS_POST))
ts_idxs = np.arange(
    int((ts_range[0] - T_START) * SAMPLE_RATE), int((ts_range[1] - T_START) * SAMPLE_RATE),
)

n_ts_idxs = ts_idxs.shape[0]

print(ts_idxs.shape[0])

# Cell 110
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


# Cell 112
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

# Cell 113
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

trial_idx = 40
neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][:10, neuron_idx, trial_idx], color='r', label='data_dict F')
ax.plot(trial_start_dff[:10, neuron_idx, trial_idx], color='b', label='my trial_start aligned')
ax.legend()

# Cell 114
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

neuron_idx = 5

ax.plot(data_dict['data']['F'][session_idx_idx][40:, neuron_idx, 0], color='r')
ax.plot(data_dict_behav['df_closedLoop'][:300, neuron_idx], color='b')

scale = data_dict['data']['F'][session_idx_idx][40, neuron_idx, 0] / data_dict_behav['df_closedLoop'][0, neuron_idx]

# ax.plot(scale * data_dict_behav['df_closedLoop'][:300, neuron_idx], color='g')

# Cell 115
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

# Cell 116
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

