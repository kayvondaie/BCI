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
try:
    import mat73
except:
    !pip install mat73
    import mat73

mypath = 'H:/My Drive/Learning rules/BCI_data/combined_new_old_060524.mat'
data_dict = mat73.loadmat(mypath)

#%%

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from helper_functions1 import *
from helper_functions2 import *



# visualize_dict(data_dict)

#%%

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
mypath = 'H:/My Drive/Learning rules/BCI_data/'

BEHAV_FILES, DATA_FILES, DATA_MAPS = get_behav_and_data_maps(mypath, verbose=False)
SESSION_IDX_TO_BEHAV_IDX = DATA_MAPS['session_idx_to_behav_idx']
SESSION_IDX_TO_DATA_IDX = DATA_MAPS['session_idx_to_data_idx']
print('Maps to behav and data files loaded!')
        
#%% fits1
from sklearn.decomposition import PCA
import copy
import numpy as np
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

#%%
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
exemplar_group_idx = 17 # 0, 5

group_event_slope = np.zeros((n_groups,))
group_event_rsquared = np.zeros((n_groups,))

#for group_idx in range(exemplar_group_idx + 1):
group_idx = exemplar_group_idx
direct_idxs = np.where(d_ps[:, group_idx] < D_DIRECT)[0]
indirect_idxs = np.where(np.logical_and(d_ps[:, group_idx] > -10, d_ps[:, group_idx] < 2000))[0]

dir_resp_ps_events = np.array(resp_ps_events[group_idx])[direct_idxs, :] # (n_direct, n_events,)
indir_resp_ps_events = np.array(resp_ps_events[group_idx])[indirect_idxs, :] # (n_indirect, n_events,)


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


# Find predictors for this group
ps_stats_params_copy = copy.deepcopy(ps_stats_params)
ps_stats_params_copy['direct_predictor_mode'] = 'sum'
ps_stats_params_copy['n_direct_predictors'] = 1

direct_predictors, direct_shift, _ = find_photostim_variation_predictors(
    dir_resp_ps_events, ps_stats_params_copy, return_extras=True
)

# Run the fit
indirect_params, pvals, _ = fit_photostim_variation(
    dir_resp_ps_events,
    indir_resp_ps_events,
    direct_predictors,
    direct_shift,
    ps_stats_params_copy
)

# Get per-neuron slope and mean
slope_idx = 1 if ps_stats_params_copy['direct_predictor_intercept_fit'] else 0
slopes = indirect_params[:, slope_idx]
pvals = pvals[:,slope_idx]
mean_indir_response = np.nanmean(indir_resp_ps_events, axis=1)

# Mask invalid values
valid_mask = ~np.isnan(slopes) & ~np.isnan(mean_indir_response)
slopes = slopes[valid_mask]
mean_indir_response = mean_indir_response[valid_mask]


b = np.argsort(slopes)       
cl = b[-5]
bb = np.argsort(sum_dir_resp_ps_events[cl,:]) 
ind=np.where(ps_events_group_idxs-1==exemplar_group_idx)[0]
f_all = ps_fs[:,cl,ind].copy()

# Set some rows to NaN as an example
f_all[5:8, :] = np.nan

# Only interpolate the first 10 rows
f_all[:10, :] = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
        m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
    ),
    axis=0,
    arr=f_all[:10, :]
)
bl = np.nanmean(f_all[0:4,:]);
f_all = (f_all - bl)/bl



num_top = 9
f_big = f_all[:,bb[-num_top :]]
bl = np.nanmean(f_big[0:4,:]);
f_big= (f_big- bl)
dt_si = data_dict['data']['dt_si'][-1]
t_sta = np.arange(0,dt_si*f_all.shape[0],dt_si)
t_sta = t_sta - t_sta[8]

# f_big = Fstim[:,cl,ind[bb[-10:]]]
# f_all = Fstim[:,cl,:]
plt.subplot(121)
plt.plot(t_sta[0:18],np.nanmean(f_all[0:18],axis=1),'b')       
plt.plot(t_sta[0:18],np.nanmean(f_big[0:18],axis=1),'k')    
plt.xlabel('Time (s)')
plt.ylabel('$\Delta$F/F')   
plt.subplot(122)
plt.plot(sum_dir_resp_ps_events[cl,:],indir_resp_ps_events[cl,:],'b.')
plt.plot(sum_dir_resp_ps_events[cl,bb[-num_top :]],indir_resp_ps_events[cl,bb[-num_top :]],'k.')
plt.ylabel('$\Delta$F/F indir.')   
plt.xlabel('$\Delta$F/F dir.')   
plt.tight_layout()