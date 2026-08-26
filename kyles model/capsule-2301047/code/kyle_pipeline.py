"""AUTO-GENERATED from bci_toy_setup.ipynb by build_kyle_pipeline.py.
Contains only definition/import statements from the notebook (no execution).
Do NOT edit by hand -- re-run build_kyle_pipeline.py to regenerate."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA

import sys

import warnings

from importlib import reload

import networks

import net_helpers

import bci_analysis

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]

c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]

c_vals_p = {
    'pre_start': '#33b983',
    'post_start_nopre': '#1077f3',
    'pre_reward': '#0050ae',
    'post_reward': '#bf8cfc',
    'cn': '#f98517',
    'direct': '#e83328',
    'indirect': '#f98517',
}

c_vals_pl = { 
    'pre_start': '#A7D9C1',
    'post_start_nopre': '#92B9F3',
    'pre_reward': '#87A6D2',
    'post_reward': '#DAC6F9',
    'cn': '#F2C392',
    'direct': '#E69C96',
    'indirect': '#F2C392',
}

c_vals_pd = { # 35% darkened
    'pre_start': '#217855',
    'post_start_nopre': '#084da0',
    'pre_reward': '#003471',
    'post_reward': '#7506f8',
    'cn': '#ac5604',
    'direct': '#a01911',
    'indirect': '#ac5604',
}

def participation_ratio_vector(C, axis=None):
    """Computes the participation ratio of a vector of variances."""
    return np.sum(C, axis=axis) ** 2 / np.sum(C*C, axis=axis)

def add_regression_line(x, y, ax=None, color='red', zorder=0, linestyle='solid', catch_nans=True):

    if catch_nans:
        nonnan_mask = ~np.isnan(x) * ~np.isnan(y)
        x = np.array(x)[nonnan_mask]
        y = np.array(y)[nonnan_mask]

    slope, intercept, rvalue, pvalue, _ = linregress(x, y)
    x_plot = np.linspace(np.min(x), np.max(x), 10)
    y_plot = slope * x_plot + intercept
    label = 'p {:.2e}, $r^2$: {:.2e}'.format(pvalue, rvalue**2)
    if ax is not None:
        ax.plot(x_plot, y_plot, color=color, zorder=zorder, linestyle=linestyle, label=label)

    return slope, intercept, rvalue

def add_bin_plot(x, y, bins=None, n_bins=10, mode='equal_spaced', ax=None, perc_range=(0, 100), yerr_mode='sem', verbose=False, 
                 nan_mask=False, error_draw_mode='errorbar', error_color=None, **scatter_kwargs):
    """
    Bin plot function. 
    
    INPUTS:
    - bins: uses these bins to make data, otherwise generates from data directly
    - n_bins: number of bins to make if generating
    - mode: equal_spaced OR equal_sized
      -  equal_spaced: equal spaced bins, unequal number in each bin
      -  equal_sized: equal sized bins, uneven size of bins
    - perc_range: percentile range of bins, can eliminate outliers
      - (0, 100): will include all points
    - yerr_mode: sem, std
    - error_draw_mode: errorbar, fill_between
    
    """
    
    assert x.shape == y.shape
    size = x.shape[0] # Number of points
    
    if nan_mask:
        nonnan_mask = ~np.isnan(x) * ~np.isnan(y)
        x = np.array(x)[nonnan_mask]
        y = np.array(y)[nonnan_mask]
    
    x_data_sort_idxs = np.argsort(x)
    x_data = x[x_data_sort_idxs]
    y_data = y[x_data_sort_idxs]
    
    if bins is None: # If bins were not passed, generate them
        if mode == 'equal_spaced':
            top_perc = (1 + 1e-5) * np.percentile(x, perc_range[1]) # +1e-5 ensures this is inclusive of top percentile
            bins = np.linspace(np.percentile(x, perc_range[0]), top_perc, n_bins+1) # +1 so includes end point        
        elif mode == 'equal_sized':
            bin_percentiles = np.linspace(perc_range[0], perc_range[1], n_bins+1) # +1 so includes end point
            bins = np.array([np.percentile(x, bin_percentile) for bin_percentile in bin_percentiles])
            bins[-1] = (1 + 1e-5) * bins[-1] # +1e-5 ensures this is inclusive of top percentile
        else:
            raise NotImplementedError('Mode {} not recognized.'.format(mode))
    else:
        n_bins = bins.shape[0] - 1
        
#     bin_x = [[] for _ in range(n_bins)]
#     bin_y = [[] for _ in range(n_bins)]

#     for idx, (x, y) in enumerate(zip(x_data, y_data)):
        

#         bin_x[bin_idxs[idx]].append(x)
#         bin_y[bin_idxs[idx]].append(y)

    bin_x_means = np.zeros((n_bins,))
    bin_x_stds = np.zeros((n_bins,))
    bin_y_means = np.zeros((n_bins,))
    bin_y_stds = np.zeros((n_bins,))
    bin_counts = np.zeros((n_bins,), dtype=np.int32)

    # for bin_idx in range(n_bins):
    #     bin_x_means[bin_idx] = np.mean(bin_x[bin_idx])
    #     bin_x_stds[bin_idx] = np.std(bin_x[bin_idx]) / np.sqrt(len(bin_x[bin_idx]))
    #     bin_y_means[bin_idx] = np.mean(bin_y[bin_idx])
    #     bin_y_stds[bin_idx] = np.std(bin_y[bin_idx]) / np.sqrt(len(bin_y[bin_idx]))
        
    for bin_idx in range(n_bins):
        bin_x_means[bin_idx] = np.mean(x[np.logical_and(x>=bins[bin_idx], x<bins[bin_idx+1])])
        bin_x_stds[bin_idx] = np.std(x[np.logical_and(x>=bins[bin_idx], x<bins[bin_idx+1])])
        bin_y_means[bin_idx] = np.mean(y[np.logical_and(x>=bins[bin_idx], x<bins[bin_idx+1])])
        bin_y_stds[bin_idx] = np.std(y[np.logical_and(x>=bins[bin_idx], x<bins[bin_idx+1])])
        bin_counts[bin_idx] = np.sum(np.logical_and(x>=bins[bin_idx], x<bins[bin_idx+1]))
    
    if verbose:
        print('Bin seps:', bins)
        print('Number per bin: (total: {})'.format(np.sum(bin_counts)))
        for bin_idx in range(n_bins):
            print('  Bin {}: {}'.format(bin_idx, bin_counts[bin_idx]))
    
    if yerr_mode is None:
        yerr = None
    elif yerr_mode == 'sem':
        yerr = bin_y_stds / np.sqrt(bin_counts)
    elif yerr_mode == 'std':
        yerr = bin_y_stds
    else:
        print('yerr_mode {} not recognized'.format(yerr_mode))
    
    if error_draw_mode == 'errorbar':
        ax.errorbar(bin_x_means, bin_y_means, yerr=yerr, **scatter_kwargs)
    elif error_draw_mode == 'fill_between':
        ax.plot(bin_x_means, bin_y_means, **scatter_kwargs)
        if error_color == None:
            error_color = 'k'
        ax.fill_between(bin_x_means, bin_y_means+yerr, bin_y_means-yerr, linewidth=0., color=error_color, alpha=0.3)

def build_scan_val_string(scan_val):
    """
    Creates unique idetnifier for each scan_val, note this assumes the name of the 
    particular scan that has been run already distingiushes them.
    """
    scan_val_str = ''
    if scan_val is None:
        scan_val_str += 'no_scan_val'
    elif type(scan_val) in (tuple, np.ndarray,): # 2d scans
        if len(scan_val) == 2:
            if type(scan_val[0]) in (float, np.float64,):
                scan_val_str += '{:.1e}_{:.1e}'.format(scan_val[0], scan_val[1])
            elif type(scan_val[0]) in (str, np.str_, np.int32,): # This can sometimes lead to long strings when a float has been convereted into a string
                scan_val_str += '{}_{}'.format(scan_val[0], scan_val[1])
            else:
                NotImplementedError('scan_val type {} not recognized'.format(type(scan_val[0])))
        else:
            raise NotImplementedError('Scan vals beyond length 2 not yet implemented')
    elif type(scan_val) in (float, np.float64,):
        scan_val_str += '{:.1e}'.format(scan_val)
    elif type(scan_val) in (str, int, np.int32,):
        scan_val_str += '{}'.format(scan_val)
    else:
        raise NotImplementedError('scan_val type {} not recognized'.format(type(scan_val)))
        
    return scan_val_str

from scipy.stats import linregress

from sklearn.linear_model import LinearRegression

from scipy.signal import fftconvolve

from net_helpers import get_stimulus

from net_helpers import accumulate_decay

def get_reward(activity, task_params, bci_masks=None, task=None, seq_idx=None):
    """
    Generic reward function that either uses built in task reward function for
    tasks that have a corresponding class OR various types of reward functions
    below for tasks that do not have a corresponding class.
    
    Takes in relevant activity of network, computes BCI relevant activity, then
    returns reward function.

    INPUT:
    - activity.shape: (n_neurons)
    - bci_masks.shape: (n_bci_mask, n_neurons)
    
    OUTPUT:
    
    """

    if task is None:
        if bci_masks is None:
            return get_reward_masked(activity, task_params)
            # return 0.1 * get_reward_masked(activity, task_params)  + 0.9 * np.random.normal(0., task_params['task_scale'])
        else:
            raise NotImplementedError('Reward from explicit cn idxs is depricated.')
    else:
        return task.get_reward(activity, task_params, seq_idx)

def get_reward_masked(activity, task_params):
    """
    Wrapper for various reward function for tasks that do not have a corresponding
    class. Passes in reward-relevant neural activity, applies BCI masks, and 
    returns reward

    INPUTS:
    - activity.shape: (n_neurons)
    - task_params:
        - bci_masks.shape: (n_bci_masks, n_neurons)
        - activity_subtract.shape: (n_neurons)

    How this scales with various mask normalizations:

    - Two-hot (1, 1) means individual activities need only be half as big to
    yield same bci_activity. Correlations will be smaller, but will need to
    move a comparitively smaller distance for same activity, so roughly the
    same length of training.
    - Two-hot (1/2, 1/2) means individual activities need to be same size as they
    would be for a one-hot to yield same bci_activity. Correlation will be smaller,
    and will need to move the same distance, so longer training.

    """
    bci_activity = get_bci_activity(activity, task_params)

    threshold = task_params['threshold']
    width = task_params['width']

    if task_params['reward_structure'] in ('trapezoid',):

        if bci_activity.shape[0] != 1:
            raise NotImplementedError('Only implemented for a single BCI mask.')
        bci_mask_idx = 0
        bci_activity = bci_activity[bci_mask_idx]

        if task_params['normalize_reward']: # Normalizes size of rewards to size of BCI activity during stabilization
            if task_params['task_type'] not in ('simple_bci',):
                raise NotImplementedError('Check that normalization makes sense for this task!')

            task_scale = task_params['task_scale']
            reward_scale = task_params['reward_scale']

            return trapezoid_reward(bci_activity, threshold, width, reward_scale, task_scale)
        else: # Normal unscaled reward, redundant but just ensures unscaled is used
            return trapezoid_reward(bci_activity, threshold, width)
    elif task_params['reward_structure'] in ('quadratic', 'abs',):

        if bci_activity.shape[0] != 1:
            raise NotImplementedError('Only implemented for a single BCI mask.')
        bci_mask_idx = 0
        bci_activity = bci_activity[bci_mask_idx]

        if task_params['normalize_reward']: # Normalizes size of rewards to size of BCI activity during stabilization
            if task_params['task_type'] not in ('simple_bci',):
                raise NotImplementedError('Check that normalization makes sense for this task!')

            task_scale = task_params['task_scale']
            reward_scale = task_params['reward_scale']
            
            if task_params['reward_structure'] in ('quadratic',):
                return quadratic_reward(bci_activity, threshold, width, reward_scale, task_scale)
            elif task_params['reward_structure'] in ('abs',):
                return abs_reward(bci_activity, threshold, width, reward_scale, task_scale)
        else: # Normal unscaled reward, redundant but just ensures unscaled is used
            if task_params['reward_structure'] in ('quadratic',):
                return quadratic_reward(bci_activity, threshold)
            elif task_params['reward_structure'] in ('abs',):
                return abs_reward(bci_activity, threshold)
    else:
        raise NotImplementedError('Reward stucture {} not implemented.'.format(task_params['reward_structure']))

def trapezoid_reward(activity, target, width, reward_scale=None, task_scale=None):
    """
    Trapezoidal shaped reward function, that is at maximum between target and target+width
    and then linearly interpolates outside of this range.
    
    INPUTS:
    - activity: scalar, activity to compare to target
    - target: scalar, minimum activity needed to get maximum reward
    - width: scalar, width of maximum reward band
    - reward_scale: Important for normalizing the size of the reward relative to baseline BCI activities.
        Generally this is set to the median BCI activity during stabilization. Effectively normalizes
        the neural distance the BCI task needs to move by how big the BCI activity is to begin with.
    - task_scale: when reward_scale is used to noramlize the size of reward, this is used to normalize 
        the size of rewards such that they are comparable to when reward_scale isn't used. Since task_scale
        controls the size of activity fluctuations, this renormalizes how big rewards are to activity
        fluctuation sizes
    """

    assert np.prod(activity.shape) == 1 # Activity must be a scalar quantity

    if reward_scale is not None:
        if activity < target: # activity at reward_scale = -1/2 * task_params['task_scale'] reward
            return  -1/2 * np.abs(activity - target) / (target - reward_scale) * task_scale
        elif activity > target + width: # activity one reward_scale away from boundary = -1/2 * task_params['task_scale'] reward
            return  -1/2 * np.abs(activity - (target + width)) / (target - reward_scale) * task_scale
        else:
            return 0.
    else: # Normal unscaled reward
        if activity < target:
            return  -1 * np.abs(activity - target)
        elif activity > target + width:
            return  -1 * np.abs(activity - (target + width))
        else:
            return 0.

def quadratic_reward(activity, target, width=None, reward_scale=None, task_scale=None):
    """
    Quadratic shaped reward function, penalizing activity quadratically away from
    target value.

    Activity/target/reward_scale must be the same shape.
    Doesn't use width, just include for same function call.
    """

    assert activity.shape == target.shape

    if reward_scale is not None:
        return  -1/2 * np.sum((target - activity)**2) / np.sum(reward_scale**2) * task_scale
    else:
        return -1/2 * np.sum((target - activity)**2)

def abs_reward(activity, target, width=None, reward_scale=None, task_scale=None):
    """
    Absolute value shaped reward function. Similar to trapezoidal one above
    but with no width.

    Activity/target/reward_scale must be the same shape.
    Doesn't use width, just include for same function call.
    """

    assert activity.shape == target.shape
    if reward_scale is not None:
        assert activity.shape == reward_scale.shape

    if reward_scale is not None:
        return -1 * np.abs(target - activity) / reward_scale
    else:
        return -1 * np.abs(target - activity)

def center_out_reward(activity, target, width=None, reward_scale=None, task_scale=None):
    """
    Special reward for the center-out task. Very similar to quadratic reward
    but doesn't penalize BCI activity that is LARGER than the target (equiv.
    to moving to the target faster). Essentially implements quadratic reward
    for any activity whose dot product with the target is smaller than 1.
    (Concretely, this measures the shortest euclidean distance to ray starting
    at the target and stretching to infinity along the center-out direction.)

    Activity/target/reward_scale must be the same shape.
    Doesn't use width, just include for same function call.
    """

    assert activity.shape == target.shape
    assert activity.shape == reward_scale.shape

    if reward_scale is not None:
        # Targets are assumed to be unit length, so this will only be >1 if beyond target
        activity_target_dot = np.dot(activity, target)
        if activity_target_dot <= 1: # Just quadratic reward
            return  -1/2 * np.sum((target - activity)**2) / np.sum(reward_scale**2) * task_scale
        else: # Subtracts out portion of activity parallel to target
            perp_activity = activity - activity_target_dot * target
            return  -1/2 * np.sum(perp_activity**2) / np.sum(reward_scale**2) * task_scale
    else:
        raise NotImplementedError()
        return -1 * np.sum((target - activity)**2)

def center_out_distance(activity, target, width=None, reward_scale=None, task_scale=None):
    """
    Special reward for the center-out task. Very similar to quadratic reward
    but doesn't penalize BCI activity that is LARGER than the target (equiv.
    to moving to the target faster). Essentially implements quadratic reward
    for any activity whose dot product with the target is smaller than 1.
    (Concretely, this measures the shortest euclidean distance to ray starting
    at the target and stretching to infinity along the center-out direction.)

    Activity/target/reward_scale must be the same shape.
    Doesn't use width, just include for same function call.
    """

    assert activity.shape == target.shape
    assert activity.shape == reward_scale.shape

    if reward_scale is not None:
        # Targets are assumed to be unit length, so this will only be >1 if beyond target
        activity_target_dot = np.dot(activity, target)
        if activity_target_dot <= 1: # Just quadratic reward
            return  np.linalg.norm((target - activity))
        else: # Subtracts out portion of activity parallel to target
            perp_activity = activity - activity_target_dot * target
            return  np.linalg.norm((perp_activity))
    else:
        raise NotImplementedError()
        return -1 * np.sum((target - activity)**2)

def custom_corrcoef(x, y):
    """
    Custom version of np.corrcoef that doesn't calculate x's and y's correlation
    with themselves, just cross correlation.
    """
    # x shape (n_neurons, n_time_steps)
    # y shape (n_neurons, n_time_steps)

    x = x - np.mean(x, axis=-1, keepdims=True)
    y = y - np.mean(y, axis=-1, keepdims=True)

    outer = (
        np.linalg.norm(x, axis=-1)[:, np.newaxis] *
        np.linalg.norm(y, axis=-1)[np.newaxis, :]
    )

    return np.where(outer > 0., np.matmul(x, y.T) / outer, np.nan)

def fluorescence_init(task_params):
    """ Initialize fluorescence kernel into task_params """

    fl_timescale = task_params.get('fl_timescale', 1000.0) # in ms
    fl_kernel_times = np.arange(0., 3 * fl_timescale, task_params['t_step']) # Goes out 3 time constants, min is 0.05 roughly.
    n_fl_kernel = fl_kernel_times.shape[0]
    task_params['fl_kernel'] = np.exp(-1 * fl_kernel_times / fl_timescale)

    # Normalizes fl_kernel so sum is = 1, this ensures that fl activity is roughly
    # the same scale as raw activity, so BCI activity and RPEs are also same scale
    task_params['fl_kernel'] = task_params['fl_kernel'] / np.sum(task_params['fl_kernel'])

    return task_params

def fluorescence_convolution(activity, task_params, last_only=False):
    """
    Returns fluorescence-convolved activity from 'fl_kernel' in task_params. Can
    return either full convolution or only minimal convolution to update next
    step of activity.

    last_only: True/Flase If true, only returns the very last idx of the fluorescence
    convolution with activity. This is useful when we only want the fluorescence
    present time step
    """
    if last_only:
        if activity.shape[0] < task_params['fl_kernel'].shape[0]: # Need to do full convolution
            return fftconvolve(activity, task_params['fl_kernel'][:, np.newaxis], mode='full', axes=(0,))[activity.shape[0] - 1]
        elif activity.shape[0] == task_params['fl_kernel'].shape[0]: # Same size, so only valid is needed now
            # print(fftconvolve(activity, task_params['fl_kernel'][:, np.newaxis], mode='full', axes=(0,))[activity.shape[0] - 1])
            # print(fftconvolve(activity, task_params['fl_kernel'][:, np.newaxis], mode='valid', axes=(0,))[0])
            return fftconvolve(activity, task_params['fl_kernel'][:, np.newaxis], mode='valid', axes=(0,))[0]
        else:
            raise ValueError('Activity should not be longer than fl kernel for last_only=True')
    else:
        return fftconvolve(activity, task_params['fl_kernel'][:, np.newaxis], mode='full', axes=(0,))[:activity.shape[0]]

def set_to_toy_bci_defaults(task_params, train_params, net_params):
    """
    Default parameters for the toy BCI task to better understand basic learning results
    """

    local_task_params = {

        ### Task choice
        'task_type': 'simple_bci', # (should already be set to this, just to be explicit)

        'z_score_activities': False,

        ### BCI mask choice (threshold/width used to set spout movement threshold/max speed)
        'bci_choice': 'random',
        ## Parameters for various bci_choice settings
            'n_bci_masks': 1,
            'activity_percentile': 0.7, # For percentile and high_activity, determines tuning percentile.
            'n_cns': 1, # Used for settings where number of CNs is varaible (n_hot), automatically set for other settings
        ### What to set the threshold based on: percentile_bci_activity, bci_activity_mean_and_std, fixed_neural_distance
        'dyn_threshold_type': 'percentile_bci_activity', # mean_all_neurons, percentile_bci_activity (what to set the threshold based on)
        ## Parameters for various threshold choices
            'dyn_threshold_perc': 0.7, # 0.7, # In our_bci, used to set spout move threshold
            'dyn_width_perc': 1.0, # In our_bci, used to set max movement speed

        ### Session length parameters
        't_step': 50, # ms
        'n_steps': 10000,
        'n_steps_stabilize': 2400,

        'normalize_reward': False,
        'n_reward_delay': 0, # Number of time steps to delay the current reward, 10 = 500 ms

        'task_scale': 0.1, # Determines the scale of the noise input and also the constant input when used

        'n_sessions': 1,
        'use_max_activity': False, # Clip activity at maximum, helps stabilize training with recurrent adjustment

        # Noise params
        'noise_type': 'iid', # iid, tc_weight, None

        ### Reward-relevant parameters
        'reward_structure': None,
        'reward_mode': 'water_and_spout', # water_only, water_and_spout, thirst, spout_and_thirst
        'state_mode': 'mix_spout_loc_1d', # mix_spout_loc, mix_spout_loc_1d, mix_spout_movement, mix_spout_movement_abs
        'start_avg_reward': 'dynamic', # how to compute avg_reward at start: None, float, dynamic
        'start_avg_reward_mult': 1.0, # For Hebbian-idx tests, >1.0 means higher reward expected and leads to initial negative RPE

        'stim_to_noise_ratio': 0.25, # > 1 more stim than noise
        'spout_movement_reward_scale': 1.0, # Relative amount of reward from stim movement and actual reward
        'thirst_reward_timescale': 500,

        # Fluorescence parameters
        'add_fl': False,
    }

    local_train_params = {
        # W_rec train: 2e0 for 0.25 stim/noise, 5e-1 for 1.0 stim/noise
        # W_inp train: 1e2 for 0.25 stim noise, 5e0 for 1.0 stim/noise
        # W_rec train, mix_spout_loc_1d stim, 1.0 stim/noise:
        'eta': 1e0,
        'n_window_reward': 300 * 20, #60 * 20, # Size of the average window for reward baseline
        'n_window_baseline': 10 * 20, # Size of the average window for activity
        'n_steps_per_loss': 5, # How often to total RPE and adjust weights
        'eligibility_acc_type': 'running_average', # acc_and_wipe, running_average
        'n_window_elig': 40, # Only used for 'running_average' option, how long to keep eligibility

        'rpe_clip': 0.05,
    }

    local_net_params = {
        'direct_input': False,

        'weight_mask_modes': None, # None, cn_freeze, cn_only
    }

    for local_key in local_task_params.keys():
        task_params[local_key] = local_task_params[local_key]
    for local_key in local_train_params.keys():
        train_params[local_key] = local_train_params[local_key]
    for local_key in local_net_params.keys():
        net_params[local_key] = local_net_params[local_key]

    return task_params, train_params, net_params

from sklearn.decomposition import FactorAnalysis

from net_helpers import get_stimulus

def determine_bci_mapping(activities_stabilize, params, net=None, task=None,
                          train_outputs_all=None, prev_seq_vars_tuning=None, 
                          output_vars=[], verbose=False):
    """
    Determine the BCI mask(s) from activity during some stabilization period.
    Note this can be very simple (e.g. just choosing the mask randomly not based
    on activity at all) to very complicated (e.g. determining some low-dim
    manifold of activity and selecting the mask based on that).

    For multiple BCI masks, can iterate through to compute one at a time but
    sometimes they are dependent upon one another, in which case all need to
    be computed at once.

    This can include optional z-scoring and activity subtraction of the activity
    (like in monkey center-out experiments).
    
    This code can also potentially gather additional information needed to determine 
    the BCI mapping, such as running a test session of the task to determine the tuning
    of neurons to certain task stimuli. Sometimes this test session data is relevant
    for further analysis so can be output from the training.

    Possible choices from task_params['bci_choice'] with required additional
    task_param keys in (...):
    - random: random choice (n_cns, n_bci_masks)
    - activity_percentile: choice closest to desired activity percentile (n_cns, n_bci_masks, activity_percentile)
    - activity_high: random choice within percentile (n_cns, n_bci_masks, activity_percentile)
    - manifold: BCI mask is one of the manifold dims (manifold_mode, manifold_idx, n_bci_masks)
    - intuitive: BCI masks chosen to best solve task
    - intuitive_maifold: Same as above, but based on low-dim manifold activity
    
    INPUTS:
    - params:
        - net_params and train_params: only needed for computing tuning, for running test network
    - net: only needed for computing tuning, for running test network
    - train_outputs_all: only for computing tuning, when tunings determined from previous session
    - prev_seq_vars_tuning: only needed for computing tuning, for running test network
    - output_vars: sometimes output relevant variables from test section used to compute tuning
    
    OUTPUT:
    - task_params:
        - task_params['bci_masks']: (n_bci_masks, n_neurons)
        - task params is also filled with a bunch of extra `extra_bci_quantities'
          that have to do with BCI mask choice and may be used later for
          analysis
        - task_params['activity_subtract']: Used later to reproduce the activity
          transformation before passing through BCI mask
        - task_params['activity_stds']: Used later to reproduce the activity
          transformation before passing through BCI mask
    - determine_bci_extras: currently only used to output additional test_tuning quantities

    """

    determine_bci_extras = {}
    task_params, train_params, net_params = params
    
    if task_params['bci_masks'] is not None: # BCI mask is already passed, skip everything below
        if verbose: print('No need to determine BCI masks, using passed masks.')
        return task_params
    
    if task_params['bci_choice'] not in ('random', 'activity_percentile', 'activity_high', 'manifold', 'intuitive', 'intuitive_manifold',):
        raise ValueError('BCI mask choice {} not recoginized'.format(task_params['bci_choice']))
    
    
    assert len(activities_stabilize.shape) == 2
    assert activities_stabilize.shape[1] == task_params['n_neurons']

    # Compute tunings for certain tasks
    if task_params['task_type'] in ('our_bci',): 
        # Compute tuning either from test task or from previous session
        estimated_tunings, tuning_extras = compute_estimated_tunings_our_bci(
            (task_params, train_params, net_params), net, task, train_outputs_all, prev_seq_vars_tuning, output_vars=output_vars
        )
        if 'test_tunings' in output_vars:
            determine_bci_extras['test_tunings'] = tuning_extras['test_tunings']
            determine_bci_extras['test_tuning_stds'] = tuning_extras['test_tuning_stds']
    else:
        estimated_tunings, tuning_extras = None, None
    
    bci_masks = np.zeros((task_params['n_bci_masks'], task_params['n_neurons'],))
    activity_subtract = np.zeros((task_params['n_neurons'],))
    cn_idxs = [] # Nested lists for each
    cn_idxs_activity_percentile = []
    cn_idxs_tuning_percentile = []

    if task_params['z_score_activities']: # Determines z-scores
        activity_subtract = np.mean(activities_stabilize, axis=0) # Mean of each neuron activity across sequence
        activities_stabilize_center = activities_stabilize - activity_subtract

        activity_stds = np.std(activities_stabilize_center, axis=0) # Std of each neuron
        # Anything that has std deviation below some threshold has it set to
        # this threshold to avoid things blowing up
        activity_stds = np.where(activity_stds < 1e-2 * task_params['task_scale'],
                                 1e-2 * task_params['task_scale'], activity_stds)

        task_params['activity_subtract'] = activity_subtract
        task_params['activity_stds'] = activity_stds

    # In these cases its useful to do a stablilization period activity
    # quantification using PCA
    if task_params['bci_choice'] in ('manifold', 'intuitive', 'intuitive_manifold',):

        # Compute the low-dimensional projection of the stabilize activity
        activities_stabilize_center = activities_stabilize - np.mean(activities_stabilize, axis=0)

        if task_params['z_score_activities']: # Fit z-scored activity
            activity_fit = activities_stabilize_center / task_params['activity_stds']
        else:
            activity_fit = activities_stabilize_center

        if task_params['manifold_mode'] in ('pc',):
            activity_pca = PCA()
            activity_pca.fit(activity_fit)
            stabilize_pca_var_exp = activity_pca.explained_variance_ratio_
            stabilize_pca_pr = participation_ratio_vector(activity_pca.explained_variance_ratio_)

            # Remove ambiguity in PC directions by definining the mean dot product with the raw activity to be positive
            manifold_axes = np.sign(np.mean(np.matmul(activities_stabilize, activity_pca.components_.T), axis=0))[:, np.newaxis] * activity_pca.components_

            # IF WE WANT TO CALCULATE THESE SHOULD PUT THROUGH get_bci_activity now
            # # Goes through pre-session  PCA projections
            # activities_stabilize_pca = np.matmul(activities_stabilize, manifold_axes.T)

            # # Thresholds along various PC directions
            # pc_thresholds = np.percentile(activities_stabilize_pca, 100, axis=0)

            # # Sanity check: maximum possible activity along each PC
            # max_activity_manifold_axes = np.zeros((task_params['n_neurons'],))
            # for pc_idx in range(task_params['n_neurons']):
            #     max_activity_manifold_axes[pc_idx] = np.sum(manifold_axes[pc_idx, manifold_axes[pc_idx] > 0], axis=-1)
        elif task_params['manifold_mode'] in ('fa',):
            n_components = 10
            print('Hard coding FA dim = 10 for now.')
            activity_fa = FactorAnalysis(n_components=n_components, random_state=task_params['seed'])
            activity_fa.fit(activity_fit)

            f_proj = np.matmul(activity_fa.components_, np.linalg.inv(
                np.diag(activity_fa.noise_variance_) + np.matmul(activity_fa.components_.T, activity_fa.components_)
            ))

            raise NotImplementedError()

    # BCI masks are independent of one another, iterate and determine one at a time
    # (for manifold masks, they are dependent on one another but their dependence
    #  has already been determined, so can iterate)
    if task_params['bci_choice'] in ('manifold', 'random', 'activity_percentile', 'activity_high',):

        for bci_mask_idx in range(task_params['n_bci_masks']):

            bci_mask = np.zeros((task_params['n_neurons'],))
            if task_params['bci_choice'] in ('manifold',): # Manifold-based masks

                # BCI mask is just one of the manifold dimensions calculated earlier
                bci_mask = manifold_axes[task_params['manifold_idx'] + bci_mask_idx]
                bci_var_exp = stabilize_pca_var_exp[task_params['manifold_idx'] + bci_mask_idx]
                task_params['n_cns'] = task_params['n_neurons']

                if verbose:
                    print_str = ' Approx activity dim: {:.1f} (ratio: {:.2f})'.format(
                        stabilize_pca_pr, stabilize_pca_pr / task_params['n_neurons']
                    )
                    print_str += '\n BCI mask set to manifold idx: {} Var exp: {:.3f}'.format(
                        task_params['manifold_idx'], bci_var_exp,
                    )
                    print(print_str)

                extra_bci_quantities = {
                    'cn_idxs': None,
                    'cn_idxs_activity_percentile': None,
                    'cn_idxs_tuning_percentile': None,
                    'bci_var_exp': bci_var_exp,
                    'manifold_axes': manifold_axes,
                    'manifold_axes_var_exps': stabilize_pca_var_exp,
                    'mean_act_stims': None,
                    'bci_masks_manifold': None,
                    'manifold_project': None,
                }

            # CN-based BCI-masks (note that a mask can have multiple CNs)
            elif task_params['bci_choice'] in ('random', 'activity_percentile', 'activity_high',):

                if task_params['z_score_activities']:
                    raise NotImplementedError('Z-scoring with activity percentile not yet implemented.')

                estimated_activities = np.mean(activities_stabilize, axis=0)

                for n_cn_idx in range(task_params['n_cns']):
                    bci_mask, new_cn_idx, new_cn_idx_activity_percentile, new_cn_idx_tuning_percentile = find_another_cn(
                        task_params, bci_mask, estimated_activities,
                        estimated_tunings=estimated_tunings, verbose=verbose
                    )
                    cn_idxs.append(new_cn_idx)
                    cn_idxs_activity_percentile.append(new_cn_idx_activity_percentile)
                    cn_idxs_tuning_percentile.append(new_cn_idx_tuning_percentile)

                extra_bci_quantities = {
                    'cn_idxs': cn_idxs,
                    'cn_idxs_activity_percentile': cn_idxs_activity_percentile,
                    'cn_idxs_tuning_percentile': new_cn_idx_tuning_percentile,
                    'bci_var_exp': np.nan,
                    'manifold_axes': None,
                    'manifold_axes_var_exps': None,
                    'mean_act_stims': None,
                    'bci_masks_manifold': None,
                    'manifold_project': None,
                }

            bci_masks[bci_mask_idx] = bci_mask

    # BCI masks are dependent on desired target direction and are not independent
    # of one another, calcluate all at once
    elif task_params['bci_choice'] in ('intuitive', 'intuitive_manifold',):

        assert task is not None # This needs to use task

        # Mean of each neuron activity across sequence
        activity_subtract = np.mean(activities_stabilize, axis=0)
        activities_stabilize_center = activities_stabilize - activity_subtract

        if task_params['z_score_activities']: # Fit z-scored activity
            activity_intuitive_fit = activities_stabilize_center / task_params['activity_stds']
        else:
            activity_intuitive_fit = activities_stabilize_center

        # Note seq idxs of separate stimuli are irrelevant here.
        activities_intuitive_stims, _ = task.separate_activity_into_stims(activity_intuitive_fit, task_params)

        mean_act_stims = np.zeros((
            len(activities_intuitive_stims), activities_intuitive_stims[0].shape[-1]
        ))

        for stim_idx in range(len(activities_intuitive_stims)):
            print(activities_intuitive_stims[stim_idx].shape)
            mean_act_stims[stim_idx] = np.mean(
                activities_intuitive_stims[stim_idx], axis=0
            )

        if task_params['bci_choice'] in ('intuitive_manifold',): # Now project activity into manifold space and z-score again
            if task_params['manifold_mode'] in ('pc',):
                # Used to determine size of manifold for PC space
                pr_val = participation_ratio_vector(stabilize_pca_var_exp)
                pr_val = int(np.ceil(pr_val))

                if pr_val < 20:
                    print('Manifold low dim: {:.1f}, raising size to 20'.format(pr_val))
                    pr_val = 20

                # Project into PC space ((not yet z-scored)
                proj_matrix = manifold_axes[:pr_val, :]
            elif task_params['manifold_mode'] in ('fa',):
                # Project into factor space (not yet z-scored)
                proj_matrix = f_proj

            # Given the projection matrix, z-score it now based on activity
            activity_intuitive_fit_manifold = np.matmul(activity_intuitive_fit, proj_matrix.T)
            std_activity_intuitive_fit_manifold = np.std(activity_intuitive_fit_manifold, axis=0)

            manifold_project = np.matmul(
                np.diag(1 / std_activity_intuitive_fit_manifold), # (n_dims_activity_space, n_dims_activity_space)
                proj_matrix # (n_dims_activity_space, n_neurons)
            )

            # Fit to the projected mean activity
            mean_act_stims_fit = np.matmul(mean_act_stims, manifold_project.T)
        else:
            mean_act_stims_manifold = None # This isnt computed in this case
            mean_act_stims_fit = mean_act_stims

        # Note at this point activity_intuitive_fit could have an arbitrary
        # size in the last dimension, because it could have been projected to a lower dimension
        # task.stim_targets is (n_stim, n_bci_mask)
        # mean_act_stims_fit is (n_stim, n_dims_activity_space)
        reg = LinearRegression().fit(mean_act_stims_fit, task.stim_targets)

        # bci_mask = np.matmul(np.matmul(
        #     np.linalg.inv(np.matmul(mean_act_periods.T, mean_act_periods)),
        #     mean_act_periods
        # ), task.stim_targets).T

        if task_params['bci_choice'] in ('intuitive_manifold',): # Needs to include projection and normalization
            bci_masks_manifold = reg.coef_ # (n_bci_mask, n_dims_activity_space)
            bci_masks = np.matmul(
                bci_masks_manifold, # (n_bci_mask, n_dims_activity_space)
                manifold_project # (n_dims_activity_space, n_neurons)
            )
        elif task_params['bci_choice'] in ('intuitive',): # Just the reg
            bci_masks = reg.coef_ # (n_bci_mask, n_neurons)
            manifold_project = None
            bci_masks_manifold = None

        for stim_idx in range(len(activities_intuitive_stims)):
            print('Stim idx {} mean location:'.format(stim_idx), np.matmul(
                bci_masks, mean_act_stims[stim_idx]
            ))

        task_params['n_cns'] = task_params['n_neurons']

        extra_bci_quantities = {
            'cn_idxs': None,
            'cn_idxs_activity_percentile': None,
            'cn_idxs_tuning_percentile': None,
            'bci_var_exp': None,
            'manifold_axes': manifold_axes,
            'manifold_axes_var_exps': stabilize_pca_var_exp,
            'mean_act_stims': mean_act_stims, # (n_stim, n_neurons) Note these include the z-scoring and manifold projection if on
            'bci_masks_manifold': bci_masks_manifold,
            'manifold_project': manifold_project,
        }
    else:
        raise NotImplementedError('BCI choice {} not recognized.'.format(task_params['bci_choice']))

    # Normalize all BCI masks in these cases
    if task_params['bci_choice'] not in ('intuitive', 'intuitive_manifold',):
        bci_masks = bci_masks / np.linalg.norm(bci_masks, axis=-1, keepdims=True)

    task_params['bci_masks'] = bci_masks
    if task_params['activity_subtract'] is None: # May have already been set above
        task_params['activity_subtract'] = activity_subtract

    for key in extra_bci_quantities.keys():
        task_params[key] = extra_bci_quantities[key]

    return task_params, determine_bci_extras

def find_another_cn(task_params, bci_mask, estimated_activities,
                    estimated_tunings=None, verbose=False):
    """
    Code used to draw new CNs and modify a single BCI mask based on CN mask.
    Note that this assumes a SINGLE BCI mask is passed and not the full
    set of BCI masks. Draws candidate indexes based on their activity percentile
    and, for our bci task, also their tuning percentile.

    bci_mask (n_neurons,)
    estimated_activities (n_neurons,): estimate of mean activity of each neuron,
        used to determine which cn choices are valid
    estimated_tunings (n_neurons,): only passed for our BCI task where tunings
        are meaningful, used along with estimated_activities to determine which
        cn_idx choices are valid

    Various ways of choosing based on task_params['bci_choice']:
    - activity_high: Choose a random neuron_idx above a certain percentile
    - activity_percentile: Choose closest neuron_idx to a given percentile
    - random: Choose a random neuron_idx from all possible neurons

    """

    def candidate_idx(offset=0):
        """ Draw another potential CN index """
        if task_params['bci_choice'] in ('activity_high',):
            activity_percentile_idx = np.random.choice(activity_percentile_idx_bound) # Choose a random idx within the bounds
            return activity_sort_idxs[activity_percentile_idx], activity_percentile_idx
        elif task_params['bci_choice'] in ('activity_percentile',):
            activity_percentile_idx = activity_percentile_idx_bound + offset # Just choose given percentile, or closest to it
            return activity_sort_idxs[activity_percentile_idx], activity_percentile_idx
        elif task_params['bci_choice'] in ('random',):
            cn_idx = np.random.randint(task_params['n_neurons'])
            return cn_idx, None

    assert len(bci_mask.shape) == 1

    # Sort estimated activity from largest to smallest
    activity_sort_idxs = np.argsort(estimated_activities)[::-1]

    if estimated_tunings is not None: # Additional tuning considerations for our task
        # Sort estimated tunings from largest to smallest
        tuning_sort_idxs = np.argsort(estimated_tunings)[::-1]
        if 'tuning_percentile' not in task_params:
            tuning_percentile = 0.25 # Matched to what Kayvon/Marton do in experiment, note we want to be BELOW this percentile
        else:
            tuning_percentile = task_params['tuning_percentile']
    else: # No tuning considerations
        tuning_sort_idxs = None
        cn_tuning_percentile = None

    if task_params['bci_choice'] in ('activity_high', 'activity_percentile',):  # Bounds on possible choices
        activity_percentile_idx_bound = int((1. - task_params['activity_percentile']) * (task_params['n_neurons'] - 1e-5))
        offset = 0 # Only used for "activity_percentile"
        # Makes sure it is possible to choose correct number of CNs
        if task_params['n_cns'] >  activity_percentile_idx_bound + 1: # +1 because if =0 this still yields 1 CN choice
            raise ValueError('Not enough neurons {} for n_cns {}'.format(
                activity_percentile_idx_bound, task_params['n_cns']
            ))

    cn_idx, activity_percentile_idx = candidate_idx(offset) # Initial draw

    if tuning_sort_idxs is None: # Only take into account repeat choices
        while bci_mask[cn_idx] > 0.: # Check if BCI mask already has given element filled
            offset += 1
            cn_idx, activity_percentile_idx = candidate_idx(offset) # Draws a new CN
    else: # Also checks to see if cn_idx is within bottom tuning_percentile of tunings (only for our BCI task)
        tuning_percentile_idx_bound = int((1. - tuning_percentile) * (task_params['n_neurons'] - 1e-5))
        tuning_percentile_idx = np.where(tuning_sort_idxs == cn_idx)[0][0]
        # print(' -- CN {} at idx {} of tuning_sort (Bound is >={})'.format(
        #     cn_idx, tuning_percentile_idx, tuning_percentile_idx_bound
        # ))
        # Largest to smallest, so keep looking for idxs that are SMALLER than bound
        while bci_mask[cn_idx] > 0. or tuning_percentile_idx < tuning_percentile_idx_bound:
            offset += 1
            cn_idx, activity_percentile_idx = candidate_idx(offset) # Draws a new CN
            tuning_percentile_idx = np.where(tuning_sort_idxs == cn_idx)[0][0]
            # print(' -- CN {} at idx {} of tuning_sort (Bound is >={})'.format(
            #     cn_idx, tuning_percentile_idx, tuning_percentile_idx_bound
            # ))
            if offset > 100: break # Breaks infinite loops where not possible to find appropriate CN idx

        cn_tuning_percentile = 1. - (tuning_percentile_idx / (task_params['n_neurons'] - 1))

    bci_mask[cn_idx] = 1.0
    cn_activity_percentile = 1. - (activity_percentile_idx / (task_params['n_neurons'] - 1))

    # if verbose:
    print_str = ' New CN idx {}'.format(cn_idx)
    if task_params['bci_choice'] in ('activity_high', 'activity_percentile',):
        print_str += ' (chosen from {:.2f} percentile, actual activity percentile: {:.2f})'.format(
            task_params['activity_percentile'], cn_activity_percentile
        )
    if tuning_sort_idxs is not None: # Tuning-related info
        print_str += '\n (tuning: chosen from {:.2f} percentile, actual {:.2f}. Estimated tuning: {:.1e})'.format(
            tuning_percentile, cn_tuning_percentile, estimated_tunings[cn_idx]
        )
    print(print_str)

    return bci_mask, cn_idx, cn_activity_percentile, cn_tuning_percentile

def get_bci_activity(activity, task_params, bci_masks=None):
    """
    Converts all raw neuron activity (maybe after fluorescence convolution)
    into activity relevant for BCI tasks. Can do this for either single
    time steps of activity or sequences of activity.

    Can also do various conversions of the raw activity that are necessary
    before passing through the BCI masks, like z-scoring the activity (e.g. to
    mimic monkey BCI setups).

    Optially, can override with custom BCI mask(s) if desired (useful if
    task_params doesn't contain desired BCI mask).
    """
    if bci_masks is None:
        bci_masks = task_params['bci_masks']

    if len(activity.shape) == 1: # activity.shape (n_neurons,) -> (n_bci_masks,)
        if task_params['activity_subtract'] is not None:
            shifted_activity = activity - task_params['activity_subtract']
        else:
            shifted_activity = activity

        if task_params['z_score_activities']:
            return np.matmul(
                bci_masks, shifted_activity / task_params['activity_stds']
            )
        else:
            return np.matmul(
                bci_masks, shifted_activity
            )
    elif len(activity.shape) == 2: # activity.shape (seq_len, n_neurons,) -> (seq_len, n_bci_masks,)
        if task_params['activity_subtract'] is not None:
            shifted_activity = activity - task_params['activity_subtract'][np.newaxis, :]
        else:
            shifted_activity = activity

        if task_params['z_score_activities']:
            return np.matmul(
                shifted_activity / task_params['activity_stds'][np.newaxis, :], bci_masks.T
            )
        else:
            return np.matmul(
                shifted_activity, bci_masks.T
            )
    else:
        raise NotImplementedError('Activity shape', activity.shape, 'not compatible with BCI mask conversion.')

def set_task_difficulty(activities_stabilize, task_params, task=None, verbose=False):
    """
    Sets the difficulty of the task based on the stabilization period activity.
    Will either modify task_params or the corresponding task instantiation to
    set the difficulty.

    - For most tasks, simply sets the corresponding parameters of the reward
      function (i.e. threshold, width, reward_scale) via task_params
    - 'our_bci': Computes threshold and width in the same way, but then
      directly modifies the task instantiation parameters. Uses threshold to
      set spout movement threshold, and threshold + width to set spout movement
      speed.
    - 'trial_structure_task': Calls the corresponding set_difficulty function
       of the class, which interally calls set_thresholds_and_scales for each
      stimulus.
    """

    print_str = ''

    if task_params['task_type'] in ('simple_bci', 'our_bci',):
        threshold, width, reward_scale, print_str, extras = set_thresholds_and_scales(
            activities_stabilize, task_params, task=task, print_str=print_str,
        )
        task_params['threshold'] = threshold
        task_params['width'] = width
        task_params['reward_scale'] = reward_scale
        
        task_params['sol_scale'] = extras['sol_scale'] if 'sol_scale' in extras else None
        if task_params['sol_scale'] is not None: print('Sol scale set to {:.1e}'.format(extras['sol_scale']))
        
        if task_params['use_max_activity']: # Max activity set based on threshold
            if task_params['dyn_threshold_type'] in ('percentile_bci_activity',):
                raise NotImplementedError('Make sure this setting is compatible with decrease activity thresholds and no width')
                task_params['max_activity'] = 2 * task_params['threshold'] + task_params['width']
            print_str += ' max activity {:.2f}'.format(task_params['max_activity'])

        if task_params['task_type'] in ('our_bci',): # For these tasks, also need to set attributes
            task.set_threshold(threshold, threshold + width)
            # task.threshold = threshold
            # task.threshold_upper = threshold + width

#     elif task_params['task_type'] in ('bci_activity_decrease',):
#         task_params['threshold'] = 0.0
#         task_params['width'] = 0.0
#         task_params['reward_scale'] = None
#         print_str += ' Threshold set to {:.2f} width {:.2f}'.format(task_params['threshold'], task_params['width'])

#         if task_params['use_max_activity']:
#             raise NotImplementedError('Need to define max activity for silence.')

    elif task_params['task_type'] in ('trial_structure_task',):
        task.set_task_difficulty_fn(activities_stabilize, task_params)

        if task_params['use_max_activity']:
            task_params['max_activity'] = task.max_activity
            print_str += ' max activity {:.2f}'.format(task_params['max_activity'])
    else:
        raise NotImplementedError('Presession threshold not implemented for task type {}'.format(task_params['task_type']))
    # task_params['threshold'] = np.mean(activities_stabilize, axis=0)[cn_idx] + 0.025 # CN-dependent threshold

    if verbose:
        print(print_str)

    return task_params, task

def set_thresholds_and_scales(activities_stabilize, task_params, task=None, print_str=''):
    """
    Based off of some activity, returns threshold and reward_scale values. Used
    to set the difficulty of various tasks.

    So far, this is only used in setting the difficulty of the task within the
    "set_task_difficulty" function and also used to set difficulty for the
    "TrialStructureTask" class, where it is called for each stimulus separately.

    Important quantities (that are reward-function dependent are)
    - threshold: the target activity
    - width: used in some rewards (e.g. trapezoidal) to further specify target
    - reward_scale: determines how large of rewards to be given for a change in
      activity, effectively the slope of the reward function. Optional.

    """

    assert len(activities_stabilize.shape) == 2
    
    SOL_SCALE_FRAC = 0.2 # Fraction of distance to set solution bounds within (for quadratic and abs rewards)

    if task is None: # Copies existing task_params values, which may be None
        task_scale = task_params['task_scale']
        current_threshold = task_params['threshold']
        current_width = task_params['width']
    else:
        task_scale = task.task_scale
        current_threshold = None
        current_width = None

    extras = {} # For outputting some extra values, only used for abs reward solution for now

    extras['sol_scale'] = None # Default value, overriden below
    
    if current_threshold is not None: # Threshold is already specified in task_params
        assert task_params['threshold'] is not None
        if task_params['reward_structure'] in ('trapezoid',):
            assert task_params['width'] is not None
        print_str += ' Fixed threshold and width passed.\n'
        threshold = task_params['threshold']
        reward_scale = task_params['reward_scale'] # Its okay if this is None
        extras['sol_scale'] = task_params['sol_scale']
    elif task_params['dyn_threshold_type'] in ('bci_activity_mean_and_std', 'fixed_neural_distance',):
         # Threshold is a certain number of std deviations away from mean bci activity,
         # or literally just a fixed value away from mean bci activity

        if task_params['bci_masks'].shape[0] > 1:
            raise NotImplementedError('Currently only implemented for a single BCI mask.')
        bci_mask_idx = 0

        bci_activity_stablize = get_bci_activity(activities_stabilize, task_params)[:, bci_mask_idx]

        # Solution scale is used to compute when a solution is reached for
        # the 'abs' reward_structure. Determines distance from threshold that
        # must be held to count as a solution.

        if task_params['dyn_threshold_type'] in ('fixed_neural_distance',):
            assert task_params['threshold_distance'] is not None
            bci_activity_stablize_mean = bci_activity_stablize.mean()
            threshold = bci_activity_stablize_mean + task_params['threshold_distance']
            extras['sol_scale'] = np.abs(SOL_SCALE_FRAC * task_params['threshold_distance']) # abs for negative distances
            
            print(' Threshold {:.1e} (fixed distance, BCI activity mean: {:.1e}, dist: {:.1e})'.format(
                threshold, bci_activity_stablize_mean, task_params['threshold_distance']
            ))
            
            reward_scale = np.array(1.0) # Since solution is a fixed neural distance away from mean, equal reward across all setups already taken care of

        elif task_params['dyn_threshold_type'] in ('bci_activity_mean_and_std',):
            assert task_params['n_stds'] is not None
            bci_activity_stablize_std = bci_activity_stablize.std()
            bci_activity_stablize_mean = bci_activity_stablize.mean()
            
            threshold = bci_activity_stablize_mean + task_params['n_stds'] * bci_activity_stablize_std
            print(' Threshold {:.1e} (BCI activity mean: {:.1e}, std: {:.1e})'.format(
                threshold, bci_activity_stablize_mean, bci_activity_stablize_std
            ))
                
            # Important that this scales with neural distance to target, which in this setting is proportional to bci_activity_stablize.std()
            extras['sol_scale'] = np.abs(SOL_SCALE_FRAC * task_params['n_stds'] * bci_activity_stablize_std) # abs for negative distances
            
            # # This does not scale with neural distance to target
            # extras['sol_scale'] = 0.5 * bci_activity_stablize_std
            
            # Having a reward scale in this case ensures that for n_stds,
            # the total reward from mean to mean + n_std * std is the same
            # note: normalization is set so n_stds = 2 doesnt change
            reward_scale = np.abs(np.array(task_params['n_stds'] / 2))

        # These methods should use abs/quadratic reward
        assert task_params['reward_structure'] in ('abs', 'quadratic',)
        
        # # no reward scale, the scale is accounted for in standard deviation weighting of the distance
        # assert task_params['normalize_reward'] == False

    elif task_params['dyn_threshold_type'] in ('percentile_bci_activity',): # Threshold is percentile of bci-mask dotted activity

        if task_params['bci_masks'].shape[0] > 1:
            raise NotImplementedError('Currently only implemented for a single BCI mask.')
        bci_mask_idx = 0

        bci_activity_stablize = get_bci_activity(activities_stabilize, task_params)[:, bci_mask_idx]
        
        threshold = np.percentile(bci_activity_stablize, 100 * task_params['dyn_threshold_perc'])
        # This is used to correctly scale reward when needed, just hard coded to be median for now
        reward_scale = np.percentile(
            get_bci_activity(activities_stabilize, task_params)[:, bci_mask_idx], 50
        )
        print('Threshold {:.1e}, mean BCI activity {:.1e}, dist {:.1e}'.format(
            threshold, bci_activity_stablize.mean(), threshold - bci_activity_stablize.mean()
        ))
        if threshold == 0.:
            raise ValueError('Threshold set to 0: neuron isnt active enough need new neuron or higher threshold criterion!')
            
        if task_params['reward_structure'] in ('abs', 'quadratic',): 
            extras['sol_scale'] = SOL_SCALE_FRAC * (threshold - bci_activity_stablize.mean())
    else:
        raise NotImplementedError('Dynamic threshold type {} not recoginized.'.format(task_params['dyn_threshold_type']))

    if task_params['reward_structure'] in ('abs', 'quadratic',): # No need to set the width for these reward structures
            width = None
    else:
        if current_width is not None:
            width = current_width
        elif task_params['dyn_width_perc'] is not None: # Override to manually set width. Used to reproduce our experiments
            width = (
                np.percentile(
                    get_bci_activity(activities_stabilize, task_params)[:, 0],
                    100 * task_params['dyn_width_perc']
                ) - threshold
            )
        elif task_params['normalize_reward']:
            width = 0.25 * np.sign(threshold) * np.abs(threshold - reward_scale)
        else:
            width = 0.2 * threshold # 20% of threshold

    if width is not None:
        print_str += ' Threshold set to {:.3f} width {:.3f} reward scale {:.3f} (from presession act.)'.format(threshold, width, reward_scale)
    else:
        print_str += ' Threshold set to {:.3f} width: None, reward scale {:.3f} (from presession act.)'.format(threshold, reward_scale)
        
    return threshold, width, reward_scale, print_str, extras

def initialize_sequence_variables(task_params, net=None, task=None, prev_seq_vars=None, n_steps=1, verbose=False):
    """
    These are all variables that need to be initialized for any sequential task
    setup, regardless if training is being done or not. This is separated out for
    potential test runs, where all these quantities might be copied from another
    sequence to compute characteristics of a network/task at a frozen point in
    training.

    This includes the initialization/resetting of the task too.
    
    INPUTS:
    - n_steps: used to check if sequence variables need to be extended or not

    """

    if prev_seq_vars is None: # Need to initialize these quantities from scratch

        # Generate both the net input and the corresponding task instantiation
        # (if task is not None, transitions to new session)
        net_input_temp, task = setup_task_and_input(task_params, net=net, task=task)

        if task_params['task_type'] in ('our_bci',): # Only generated very first input (inputs are action dependent)
            net_input = np.zeros((n_steps, task_params['n_inp'],))
            net_input[0] = net_input_temp
            assert task_params['n_bci_masks'] == 1 # Assumes this below by setting bci_mask_idx = 0
        else: # All the input was generated at once
            assert len(net_input_temp.shape) == 2
            net_input = net_input_temp

        output = np.zeros((n_steps, task_params['n_neurons'],))
        output_pre_act = np.zeros((n_steps, task_params['n_neurons'],))
        reward = np.zeros((n_steps,))
        avg_reward = np.zeros((n_steps,))
        avg_activity = np.zeros((n_steps, task_params['n_neurons'],))
        avg_activity_pre_act = np.zeros((n_steps, task_params['n_neurons'],))
        mean_stabilize_actvities = None # Overriden below, init here just in case no training occurs
        if task_params['add_fl']: # Keep fl signal separate from normal output since one is for RPE and other is for eligibility
            output_fl = np.zeros((n_steps, task_params['n_neurons'],))
        else:
            output_fl = None
        # BCI activity is based on fluorescence if its enabled, otherwise based on raw outputs.
        # Note BCI activity incorporates no effects from delay in it, delays are
        # handled directly in reward calculations/environment interactions
        bci_activity = np.zeros((n_steps, task_params['n_bci_masks'],))
        avg_bci_activity = np.zeros((n_steps, task_params['n_bci_masks'],))

        start_step_idx = 0

        if task_params['separate_stim_reward_baselines']:
            if task_params['trial_type'] not in ('center_out_1d', 'center_out_1d_1', 'center_out_2d', 'center_out_2d_8',):
                raise NotImplementedError()
            avg_reward_stims = np.zeros((n_steps, task.n_stim))
            print('Using different baseline rewards for each stimulus!')
        else:
            avg_reward_stims = None
            
        if train_params['adjust_type'] in ('3factor_node_pert', '3factor_node_pert_pre',): # Generate perturbations
            if train_params['act_perturbation_scale'] is None:
                perturbation_scale = task_params['task_scale']
            elif type(train_params['act_perturbation_scale']) in (float, np.float64,):
                perturbation_scale = train_params['act_perturbation_scale']
            elif type(train_params['act_perturbation_scale']) == np.ndarray:
                if train_params['act_perturbation_scale'].shape[0] == 1:
                    perturbation_scale = train_params['act_perturbation_scale']
                elif train_params['act_perturbation_scale'].shape[0] == task_params['n_neurons']:
                    perturbation_scale = train_params['act_perturbation_scale']
                    print('Vector scale used!')
                else:
                    raise ValueError('Numpy array shape not recognized.')
            else:
                raise ValueError('type(act_perturbation_scale) {} not recognized.'.format(type(train_params['act_perturbation_scale'])))
                    
            # ### No perturbations during stabilization ###
            # act_perturbations = np.zeros((task_params['n_steps'], task_params['n_neurons'],)) # Default pertrubations are all zeros
            # # Pertrubations during training are non-zero. If perturbations were applied
            # # during the stabilization period this would change the spectrum of neural
            # # activity. Different from how node perturbation is applied in practice.
            # act_perturbations[task_params['n_steps_stabilize']:, :] = np.random.normal(
            #     scale=perturbation_scale, size=(task_params['n_steps'] - task_params['n_steps_stabilize'], task_params['n_neurons'],)
            # )
            
            ### Perturbations at all times ###
            # Noise should be turned off in this case
            perturbations = np.random.normal(
                scale=perturbation_scale, size=(n_steps, task_params['n_neurons'],)
            )
            
            if train_params['adjust_type'] in ('3factor_node_pert',):
                act_perturbations = perturbations
                preact_perturbations = None
            elif train_params['adjust_type'] in ('3factor_node_pert_pre',):
                act_perturbations = None
                preact_perturbations = perturbations
        else:
            act_perturbations = None
            preact_perturbations = None

        return task, start_step_idx, (
            net_input, output, output_pre_act, reward, avg_reward,
            avg_activity, avg_activity_pre_act, mean_stabilize_actvities,
            output_fl, bci_activity, avg_bci_activity, avg_reward_stims,
            act_perturbations, preact_perturbations,
        )
    else: # In this case the previous sequence variables will just be used (assuming this is continuation of session, so no task reset either)
        if task_params['task_type'] in ('our_bci',):
            # For this to make any sense, the previously used task needs to have
            # been passed too
            assert task is not None
            # No need to setup net_input to be special here, the test function
            # is assummed to have been run after the net_input[step_idx+1]
            # has been assigned, so the next input to network has already been
            # computed.
            start_step_idx = np.copy(task.seq_idx)
            if verbose: print('Using passed task, starting at seq_idx={}.'.format(start_step_idx))
            
            N_RETURN_SEQ_VARS = 14
            
            # The convention of prev_seq_vars was updated but some old saves still use the 
            # previous convention, so make them compatible by just appending None to them.
            if len(prev_seq_vars) < N_RETURN_SEQ_VARS: # 14 distinct seq vars returned above
                # print('Appending None to seq vars to match new convention')
                prev_seq_vars = (*prev_seq_vars, None, None)
                
            if n_steps > prev_seq_vars[0].shape[0]:
                # Note this implicitly assumes the passed prev_seq_vars are long enough
                # to accomodate the test sequence length. This code pads
                # the prev_seq_vars with the desired test sequence length...
                print('n_steps {} start_step_idx {} prev_seq_vars shape {}'.format(n_steps, start_step_idx, prev_seq_vars[0].shape[0]))
                n_steps_new = n_steps - prev_seq_vars[0].shape[0]
                print('Passing prev_seq_vars {} more steps ({} to {})'.format(n_steps_new, prev_seq_vars[0].shape[0], prev_seq_vars[0].shape[0]+n_steps_new))
                (
                    net_input, output, output_pre_act, reward, avg_reward,
                    avg_activity, avg_activity_pre_act, mean_stabilize_actvities,
                    output_fl, bci_activity, avg_bci_activity, avg_reward_stims,
                    act_perturbations, preact_perturbations,
                ) = prev_seq_vars
                
                net_input = np.concatenate((net_input, np.zeros((n_steps_new, net_input.shape[-1],))), axis=0)
                output = np.concatenate((output, np.zeros((n_steps_new, output.shape[-1],))), axis=0)
                output_pre_act = np.concatenate((output_pre_act, np.zeros((n_steps_new, output.shape[-1],))), axis=0)
                reward = np.concatenate((reward, np.zeros((n_steps_new,))), axis=0)
                avg_reward = np.concatenate((avg_reward, np.zeros((n_steps_new,))), axis=0)
                avg_activity = np.concatenate((avg_activity, np.zeros((n_steps_new, avg_activity.shape[-1],))), axis=0)
                avg_activity_pre_act = np.concatenate((avg_activity_pre_act, np.zeros((n_steps_new, avg_activity_pre_act.shape[-1],))), axis=0)
                if mean_stabilize_actvities is not None:
                    raise ValueError('This shouldnt happen, thought this was depricated.')
                if output_fl is not None:
                     output_fl = np.concatenate((output_fl, np.zeros((n_steps_new, output_fl.shape[-1],))), axis=0)
                bci_activity = np.concatenate((bci_activity, np.zeros((n_steps_new, bci_activity.shape[-1],))), axis=0)
                avg_bci_activity = np.concatenate((avg_bci_activity, np.zeros((n_steps_new, avg_bci_activity.shape[-1],))), axis=0)
                if avg_reward_stims is not None:
                    avg_reward_stims = np.concatenate((avg_reward_stims, np.zeros((n_steps_new, avg_reward_stims.shape[-1],))), axis=0)
                if act_perturbations is not None:
                    raise NotImplementedError('Need to figure out how to do this in test session')
                if preact_perturbations is not None:
                    raise NotImplementedError('Need to figure out how to do this in test session')
                prev_seq_vars = (
                    net_input, output, output_pre_act, reward, avg_reward,
                    avg_activity, avg_activity_pre_act, mean_stabilize_actvities,
                    output_fl, bci_activity, avg_bci_activity, avg_reward_stims,
                    act_perturbations, preact_perturbations,
                )
        else:
            raise NotImplementedError('Continued sequence not yet implemented.')

        return task, start_step_idx, prev_seq_vars

def setup_task_and_input(task_params, net=None, task=None):
    """
    Sets up the task, including instantiation of the corresponding class and
    the corresponding inputs for the task (including noise).

    Note that when a task is already instatiated, can be used to create a new
    session of the task.

    INPUTS:

    OUTPUTS:
    - net_input: could either be inputs for all times or just input for the
        very first time step
    - task
    """

    if task_params['task_type'] in ('simple_bci',):
        if task_params['direct_input']:
            n_input = task_params['n_neurons']
        else:
            n_input = task_params['n_inp']
                
        # Generate noise to drive the network , which is set by task_scale
        iid_noise = np.random.normal(scale=task_params['task_scale'], size=(task_params['n_steps'], n_input,))

        if train_params['adjust_type'] in ('3factor_node_pert', '3factor_node_pert_pre',): # Special cases with no noise
            print('Node perturbation, zeroing noise!')
            iid_noise = np.zeros_like(iid_noise)
        
        if task_params['task_type'] in ('simple_bci',) and task_params['constant_stim_input']: # Generate a stimulus to be added to all noise
            # I messed around with the scale of this input so that mean activity didn't look too huge compared to baseline, 0.1 looks good
            stim = get_stimulus(task_params, stim_type='normal', rel_stimulus_scale=0.1)
            print('Task scale: {} Stim:'.format(task_params['task_scale']), stim[0, :5])
        elif task_params['task_type'] in ('simple_bci',) and net.act_fn_type in ('linear', 'Tanh'):
            raise ValueError('For odd activation functions, there needs to be something to give neurons a nonzero baseline! Turn on stim input.')
        else: # Stimulus is just zeros
            stim = np.zeros((1, n_input,))

        if task_params['direct_input']:
            raise NotImplementedError('This setting not yet implemented for non-task setup.')
        
        # We view the input layer as similar types of neurons to those in the hidden layer, so also pass through same activation
        if net.act_fn_type in ('linear', 'Tanh'):
            net_input = net.act_fn(iid_noise + stim)
            ### Positive noise for odd functions is depricated, a better method is to provide a constant stimulus input.
            # net_input = np.maximum(iid_noise + stim, np.zeros_like(iid_noise)) # Make noise positive
        else:
            net_input = net.act_fn(iid_noise + stim)

    elif task_params['task_type'] in ('trial_structure_task',):
        if task is None:
            net_params_temp = {'act_fn_type': net.act_fn_type}
            task = TrialStructureTask(task_params, net_params_temp)
        else:
            # Reset task here
            raise NotImplementedError()
        net_input = task.generate_input_sequence(task_params['n_steps'])

    elif task_params['task_type'] in ('our_bci',):
        if task is None:
            task = BCIGym.BCI_Env(task_params)
        else: # Transitions to new session (reset below)
            task.set_new_session()
        net_input, _ = task.reset() # Only the net input of the very first time step now
    else:
        raise ValueError('Task type {} not recognized'.format(task_params['task_type']))

    return net_input, task

def initialize_sequence_variables_training(task_params, net, output_vars, train=True, verbose=False):
    """
    These are variables that are only tracked if trianing
    """
    W_inp_elg_vals = None
    W_rec_elg_vals = None
    act_fn_p_pre_act_vals = None
    if train:
        # These don't need to be updated every step, but every time there is a gradient update if training
        loss_steps = [0,] # 0 to keep in sync with initial values below
        total_rpes = [0.,]
        W_inp_vals = [net.W_inp,]
        W_rec_vals = [net.W_rec,]
        # Only track these if needed
        if 'W_inp_elg_vals' in output_vars:
            W_inp_elg_vals = [np.zeros_like(net.W_inp),] # init to keep in sync with initial values below
        if 'W_rec_elg_vals' in output_vars:
            W_rec_elg_vals = [np.zeros_like(net.W_rec),] # 0 to keep in sync with initial values below
        if 'act_fn_p_pre_act_vals' in output_vars: # For analyzing eligibility trace in more detail
            act_fn_p_pre_act_vals = np.zeros((task_params['n_steps'], task_params['n_neurons'],))
    else:
        if verbose: print('No training, skipping initialization of training-quantities.')
        loss_steps = None
        total_rpes = None
        W_inp_vals = None
        W_rec_vals = None

    return (
        loss_steps, total_rpes, W_inp_vals, W_rec_vals, W_inp_elg_vals,
        W_rec_elg_vals, act_fn_p_pre_act_vals
    )

def apply_bci_mask_perturbations(step_idx, task_params, verbose=False):
    """
    The net effect of this function is to modify task_params['bci_masks'] to
    a new set of BCI masks. Currently called within the session loop of the train 
    function, immediately before the forward pass.

    task_params['perturbations'] are of form (step_idx, perturbation_type, perturbation_params)
    """

    terminate_training = False

    # Step idxs for each perurbation
    pert_step_idxs = [pert[0] for pert in task_params['perturbations']]

    if step_idx not in pert_step_idxs:
        return task_params, terminate_training

    ### Perturbation being applied ###
    if 'prev_bci_masks' not in task_params:
        task_params['prev_bci_masks'] = []
    # Saves mask for post-training analysis
    task_params['prev_bci_masks'].append(np.copy(task_params['bci_masks']))

    pert_idx = pert_step_idxs.index(step_idx)

    pert_type = task_params['perturbations'][pert_idx][1]
    pert_params = task_params['perturbations'][pert_idx][2]

    if verbose:
        print('Step idx: {} - applying {} perturbation.'.format(step_idx, pert_type))

    if pert_type in ('bci_masks_new_pc',): # Change BCI mask, but leave everything else unchanged
        assert task_params['manifold_axes'] is not None

        if task_params['n_bci_masks'] > 1:
            raise NotImplementedError('This code needs to be changed to account for more than one BCI mask')

        new_bci_mask = task_params['manifold_axes'][pert_params['pc_idx']]
        task_params['bci_masks'][0, :] =  new_bci_mask
    elif pert_type in ('center_out',):
        assert task_params['task_type'] == 'trial_structure_task'
        assert task_params['trial_type'] in ('center_out_1d', 'center_out_1d_1', 'center_out_2d', 'center_out_2d_8',)
        if 'intuitive_bci_masks' not in task_params: # Only save the very first time, used to restore it later
            # Repetitive with 'prev_bci_masks' but convenient
            task_params['intuitive_bci_masks'] = np.copy(task_params['bci_masks']) # Save for restoration later

        perm_mode = pert_params['perm_mode']
        if perm_mode in ('intuitive',): # Just restore old mask
            print('Restoring inuitive masks.')
            task_params['bci_masks'] = np.copy(task_params['intuitive_bci_masks'])
        elif perm_mode in ('wm', 'wm_n_top', 'om_raw', 'om_top_stds', 'om_top_stim_stds',):
            bci_masks_pert, pert_extras = get_perturbation_center_out(
                task_params, perm_mode, n_perm_dims=pert_params['n_perm_dims'], verbose=verbose
            )
            if bci_masks_pert is None:
                # raise ValueError('No valid perturbation found!')
                typed_input = input('Proceed? (y/n)')
                if typed_input in ('y',):
                    task_params['bci_masks'] = pert_extras['best_pert_so_far']
                else:
                    terminate_training = True
            else:
                task_params['bci_masks'] = bci_masks_pert
        else:
            raise NotImplementedError('Perm mode {} not recoginized.'.format(perm_mode))
    else:
        raise ValueError('Perturbation of type {} not recognized!'.format(pert_type))

    return task_params, terminate_training

def set_special_test(task_params, train_params, net_params, net, verbose=False):
    """
    A bunch of special tests packaged together in a function to better understand learning.
    Currently called immediately after initialization of network, the task,
    and various sequence variables tracked over training.
    
    zero_cn_input: Randomly determine CN prior to training, zero the corresponding
        W_inp contribution to the CN (assumes W_inp is not adjusted, so will
        remain zero)
    zero_cn_input_and_sparsify_rec: Same as above, but also sparsify the W_rec
        inputs to the CN. Also set rec_adjust_mask so that all these inputs
        remain zero as well. Note currently sets all inputs into the CN to be
        the same: 1 / sqrt(n_nonzero_inputs)
    zero_cn_input_and_sparsify_rec_copy: Same as above, but makes the inputs
        to the nonzeros identical so they are on completely equal footing and
        we can investigate if stimulation modifies them
    """

    if task_params['special_test_type'] in (
        'zero_cn_input', 'zero_cn_input_and_sparsify_rec',
        'zero_cn_input_and_sparsify_rec_copy', 'zero_cn_input_and_sparsify_rec_copy_hold',
        ):

        assert task_params['n_bci_masks'] == 1

        # Set BCI mask and change network so that CN input has no input
        cn_idx = np.random.choice(task_params['n_neurons'])
        print('Fixing BCI mask to have CN {}'.format(cn_idx))
        task_params['bci_masks'] = np.zeros((task_params['n_bci_masks'], task_params['n_neurons'],))
        task_params['bci_masks'][0, cn_idx] = 1.0
        task_params['cn_idxs'] = [cn_idx,]

        assert net_params['W_inp_adjust'] == False

        print('Setting CN to have zero input.')
        net.W_inp[cn_idx, :] = 0.0

        if task_params['special_test_type'] in (
            'zero_cn_input_and_sparsify_rec', 'zero_cn_input_and_sparsify_rec_copy',
            'zero_cn_input_and_sparsify_rec_copy_hold',
        ):
            N_NONZERO_INPUTS = 2

            neuron_idxs = np.delete(np.arange(task_params['n_neurons']), cn_idx)
            nonzero_rec_idxs = np.random.choice(neuron_idxs, N_NONZERO_INPUTS, replace=False)
            print('Sparsifying the CN input too, nonzero rec inputs are:', nonzero_rec_idxs)

            rec_adjust_mask = np.ones_like(np.ones_like(net.W_rec))

            net.W_rec[cn_idx, :] = 0.0 # Zero all inputs
            rec_adjust_mask[cn_idx, :] = 0.0 # Default is no adjusting rec into CN
            for nonzero_rec_idx in nonzero_rec_idxs: # Turn on only a few weights going into the CN and corresponding adjustment
                net.W_rec[cn_idx, nonzero_rec_idx] = 1 / np.sqrt(N_NONZERO_INPUTS)
                rec_adjust_mask[cn_idx, nonzero_rec_idx] = 1.0

            # Also freeze inputs into some of the important neurons
            # print('Also freezing neuron idx 22')
            # rec_adjust_mask[22, :] = 0.0
            # print('Also freezing neuron idx 15')
            # rec_adjust_mask[15, :] = 0.0
            # print('Freezing everything except neuron idx 22')
            # for neuron_idx in range(200):
            #     if neuron_idx != 22:
            #         rec_adjust_mask[neuron_idx, :] = 0.0

            if net.W_rec_adjust_mask is None:
                net.W_rec_adjust_mask = np.ones_like(net.W_rec)
            net.W_rec_adjust_mask = net.W_rec_adjust_mask * rec_adjust_mask # Combines multiple masks

            # Make inputs into the nonzero recs identical too, so they train on even footing
            if task_params['special_test_type'] in (
                'zero_cn_input_and_sparsify_rec_copy', 'zero_cn_input_and_sparsify_rec_copy_hold',
            ):
                first_rec_idx = nonzero_rec_idxs[0]
                print('Making inputs to recs identical too...')
                for nonzero_rec_idx in nonzero_rec_idxs[1:]:
                    net.W_inp[nonzero_rec_idx, :] = net.W_inp[first_rec_idx, :]
                    net.W_rec[nonzero_rec_idx, :] = net.W_rec[first_rec_idx, :]

            task_params['nonzero_rec_idxs'] = nonzero_rec_idxs

    elif task_params['special_test_type'] is not None:
        print('Special test type {} modified nothing at initialization'.format(
            task_params['special_test_type']
        ))

    return task_params, train_params, net_params, net

def set_special_test_post_bci_mask(activities_stabilize, task_params, train_params, net_params, net, verbose=False):
    """
    Special test applied after the BCI mask is set after the stabilization period. Note like set_special_test,
    this is only called once.
    
    zero_cn_input_and_sparsify_rec_high_activity: same as 'zero_cn_input_and_sparsify_rec'
        done at initialization, but instead of choosing random neurons to connect
        to the CN, chooses only high activity neurons to connect to it. This
        should make training a bit more stable since these neurons are already
        pre-disposed to change quite a bit.
    """

    assert len(activities_stabilize.shape) == 2
    assert activities_stabilize.shape[1] == task_params['n_neurons']

    if task_params['special_test_type'] in ('zero_cn_input_and_sparsify_rec_high_activity',):

        N_NONZERO_INPUTS = 2
        assert task_params['n_bci_masks'] == 1
        assert net_params['W_inp_adjust'] == False

        estimated_activities = np.mean(activities_stabilize, axis=0)

        # Looks for high activity neurons in the same way it would look for CNs
        nonzero_rec_idxs = []
        nonzero_rec_idxs_percentile = []
        bci_mask = np.zeros((task_params['n_neurons'],))
        for n_cn_idx in range(N_NONZERO_INPUTS):
            bci_mask, new_cn_idx, new_cn_idx_activity_percentile, _ = find_another_cn(
                task_params, bci_mask, estimated_activities,
                estimated_tunings=None, verbose=verbose
            )
            nonzero_rec_idxs.append(new_cn_idx)
            nonzero_rec_idxs_percentile.append(new_cn_idx_activity_percentile)

        # Finds neurons with large activity
        print('Nonzero rec idxs:', nonzero_rec_idxs)
        print('Percentiles:', nonzero_rec_idxs_percentile)

        # Now set BCI mask and change network so that CN input has no input
        cn_idx = np.random.choice(task_params['n_neurons'])
        while cn_idx in nonzero_rec_idxs: # Make sure its not one of the nonzero_rec_idxs
            cn_idx = np.random.choice(task_params['n_neurons'])
        print('Fixing BCI mask to have CN {}'.format(cn_idx))
        task_params['bci_masks'] = np.zeros((task_params['n_bci_masks'], task_params['n_neurons'],))
        task_params['bci_masks'][0, cn_idx] = 1.0
        task_params['cn_idxs'] = [cn_idx,]

        print('Setting CN to have zero input.')
        net.W_inp[cn_idx, :] = 0.0

        rec_adjust_mask = np.ones_like(np.ones_like(net.W_rec))

        net.W_rec[cn_idx, :] = 0.0 # Zero all inputs
        rec_adjust_mask[cn_idx, :] = 0.0 # Default is no adjusting rec into CN
        for nonzero_rec_idx in nonzero_rec_idxs: # Turn on only a few weights going into the CN and corresponding adjustment
            net.W_rec[cn_idx, nonzero_rec_idx] = 1 / np.sqrt(N_NONZERO_INPUTS)
            rec_adjust_mask[cn_idx, nonzero_rec_idx] = 1.0

        if net.W_rec_adjust_mask is None:
            net.W_rec_adjust_mask = np.ones_like(net.W_rec)
        net.W_rec_adjust_mask = net.W_rec_adjust_mask * rec_adjust_mask # Combines multiple masks

        task_params['nonzero_rec_idxs'] = nonzero_rec_idxs

        raise NotImplementedError('Need some way of adjusting task difficulty now that a new BCI mask is set')

    elif task_params['correlation_type'] is not None: # Optional additional correlation with CN
        print('Correlating CN and ON idx {}, weight: {:.2f}'.format(task_params['on_idx'], task_params['on_weight']))
        raise NotImplementedError('Moved this to an external function, so need to update from that too')
        raise NotImplementedError('Need to update to more general BCI mask setup')
        raise NotImplementedError('Need to update to new network class')
        if task_params['correlation_type'] in ('weights',): # Correlate via weights
            net.W_inp = correlate_weight(net.W_inp, cn_idx, task_params['on_idx'], task_params['on_weight'])
            net.W_rec = correlate_weight(net.W_rec, cn_idx, task_params['on_idx'], task_params['on_weight'])
            W_inp_vals[0] = net.W_inp # Reset these because they're been updated
            W_rec_vals[0] = net.W_rec
        elif task_params['correlation_type'] in ('activity',): # Correlate via direct activity change
            raise NotImplementedError('Make sure this work with new step_idx assignments')
            # Override the mean activity of ON at this time step, and at each time step henceforth will override activity (mean activity will be update)
            avg_activity[step_idx, task_params['on_idx']] = (
                (1 - task_params['on_weight']) * avg_activity[step_idx, task_params['on_idx']] + task_params['on_weight'] * avg_activity[step_idx, cn_idx]
            )
            output[step_idx, task_params['on_idx']] = (
                (1 - task_params['on_weight']) * output[step_idx, task_params['on_idx']] + task_params['on_weight'] * output[step_idx, cn_idx]
            )

    elif task_params['special_test_type'] is not None:
        print('Special test type {} modified nothing after stabilization'.format(
            task_params['special_test_type']
        ))

    return task_params, train_params, net_params, net

def correlate_neurons(net):
    """
    Add additional correlations into the network between a cn and other neurons
    to test effects of correlation on
    """

    raise NotImplementedError('This function is incomplete')

    def correlate_weight(W, cn_idx, on_idx, on_weight):
        """
        Makes one row of a weight matrix more similar to another to induce additional correlation
        """
        W[on_idx, :] = (1. - on_weight) * W[on_idx, :] + on_weight * W[cn_idx, :]
        return W

    net.W_inp = correlate_weight(net.W_inp, cn_idx, task_params['on_idx'], task_params['on_weight'])
    net.W_rec = correlate_weight(net.W_rec, cn_idx, task_params['on_idx'], task_params['on_weight'])
    W_inp_vals[0] = net.W_inp # Reset these because they're been updated
    W_rec_vals[0] = net.W_rec

def get_weight_adjust_mask(net, task_params, net_params, W_inp, W_rec):
    """
    Freeze certain weight adjustments within the network. Used to test the
    effects of knockout certain types of plasticity experimentally.
    
    This is called after the BCI mask is set after the stabilization period. 
    Note like set_special_test, this is only called once.
    """

    assert task_params['bci_masks'] is not None

    if net_params['W_inp_adjust']:
        raise NotImplementedError('This code assumes that only the recurrent layer is trained.')
    if len(task_params['cn_idxs']) > 1:
        raise NotImplementedError('This code assumes that there is only one cn_idx')

    # Default is everything can be adjusted, this then gets updated below
    W_rec_adjust_mask_full = np.ones_like(W_rec)
        
    if 'cn_freeze' in net_params['weight_mask_modes']: # Freeze all weights entering CN index
        print('Freezing all recurrent weights that go into CN.')
        cn_idx = task_params['cn_idxs'][0]
        W_rec_adjust_mask = np.ones_like(W_rec)
        W_rec_adjust_mask[cn_idx, :] = 0.0
        W_rec_adjust_mask_full *= W_rec_adjust_mask
    
    if 'cn_only' in net_params['weight_mask_modes']: # Freeze all weights other than those entering CN index
        print('Freezing all recurrent weights that go into non-CNs')
        cn_idx = task_params['cn_idxs'][0]
        W_rec_adjust_mask = np.zeros_like(W_rec)
        W_rec_adjust_mask[cn_idx, :] = 1.0
        W_rec_adjust_mask_full *= W_rec_adjust_mask
       
    if 'cn_no_output' in net_params['weight_mask_modes']: # Zero all weights coming out of CN and freeze them so they stay that way
        print('Freezing and zeroing all recurrent weights leave the CN')
        cn_idx = task_params['cn_idxs'][0]
        net.W_rec[:, cn_idx] = 0.
        W_rec_adjust_mask = np.ones_like(W_rec)
        W_rec_adjust_mask[:, cn_idx] = 0.0
        W_rec_adjust_mask_full *= W_rec_adjust_mask
    
    for weight_mask_mode in net_params['weight_mask_modes']:
        if weight_mask_mode not in ('cn_freeze', 'cn_only', 'cn_no_output',):
            raise NotImplementedError('Weight mask adjust mode {} not recognized'.format(weight_mask_mode))

    net_params['W_rec_adjust_mask'] = W_rec_adjust_mask_full

    if net.W_rec_adjust_mask is None:
        net.W_rec_adjust_mask = np.ones_like(net.W_rec)
    net.W_rec_adjust_mask = net.W_rec_adjust_mask * net_params['W_rec_adjust_mask'] # Combines multiple masks

    return net, net_params

from scipy import signal

import BCIGym_numpy_setup as BCIGym

import copy

import timeit

def get_session_summary_our_bci_task(
        reward, total_rpes, output, output_fl, avg_activity, task, task_params,
        cn_idxs, cn_idxs_activity_percentile, output_vars, verbose=False
    ):
    """
    Collects useful summary statistics of a given session. Useful for comparing
    many sessions to one another (e.g. for parameter scans).

    Messy parameter pass because don't want to have to save these to
    train_outputs if not necessary
    """
    N_AVG_TRIALS = 10

    activity_type = 'raw_fl' # raw_activity, raw_fl, dff
    tuning_mode = 'omit_misses' # default of experiment, so use it here too
    trial_avg_window = 1 / N_AVG_TRIALS * np.ones((N_AVG_TRIALS,))

    session_idx = task.session_idx
    
    assert len(cn_idxs) == 1
    cn_idxs_idx = 0
    cn_idx = cn_idxs[cn_idxs_idx]

    session_summary = {}
    session_summary['cn_idx'] = cn_idx
    session_summary['cn_idx_activity_percentile'] = cn_idxs_activity_percentile[cn_idxs_idx]
    session_summary['total_reward'] = np.nansum(reward)
    if total_rpes is None:
        session_summary['total_rpe'] = np.nan
    else:
        session_summary['total_rpe'] = np.nansum(total_rpes)
    session_summary['n_trials'] = len(task.hist['trial_ends'])

    session_summary['hit_rate'] = np.sum(task.hist['trial_outcomes']) / session_summary['n_trials']
    average_trial_outcomes = fftconvolve(task.hist['trial_outcomes'], trial_avg_window, mode='valid')

    session_summary['hit_rate_init'] = average_trial_outcomes[0]
    session_summary['hit_rate_final'] = average_trial_outcomes[-1]
    session_summary['hit_rate_max'] = np.max(average_trial_outcomes)

    if 'max_activity' in task_params:
        if task_params['max_activity'] is not None:
            saturation_thresh = task_params['max_activity']
        else:
            saturation_thresh = np.nan
    else:
        saturation_thresh = act_fn(10 * task_params['task_scale'])
    session_summary['perc_saturated_neurons'] = np.sum(avg_activity[-1] > 0.95 * saturation_thresh) / task_params['n_neurons']

    ### Task period analysis ###
    n_steps_total = output.shape[0]
    if activity_type in ('raw_activity',):
        activity_analyze = output[task_params['n_steps_stabilize']:n_steps_total-task_params['n_steps_evaluate']]
    elif activity_type in ('raw_fl',):
        activity_analyze = output_fl[task_params['n_steps_stabilize']:n_steps_total-task_params['n_steps_evaluate']]
    elif activity_type in ('dff',):
        raise NotImplementedError()
        activity_analyze = output_fl[task_params['n_steps_stabilize']:n_steps_total-task_params['n_steps_evaluate']]
        
    task_metrics, ts_extras = bci_analysis.compute_task_metric_aligned_values(
        activity_analyze, task.hists[session_idx], task_params, tuning_mode=tuning_mode, fit_changes=True,
    )
    
    # This greatly increases the size of session summary, but these are useful metrics to track
    session_summary['task_metrics'] = task_metrics
    session_summary['ts_extras'] = ts_extras
    
    # Tuning can sometimes be unspecified because there are no pre_trials
    if session_summary['n_trials'] > N_AVG_TRIALS and not np.isnan(cn_idx) and task_metrics['tuning'] is not None:
        tuning_slopes = ts_extras['tuning']['slope']
        tuning_slopes_norm = tuning_slopes / np.max(np.abs(tuning_slopes))
        tuning_slopes_sort = np.argsort(tuning_slopes) # Smallest to largest

        cn_sort_idx = np.where(tuning_slopes_sort == cn_idx)[0][0]
        # idx = 0 (idx = n-1) -> perc = 0.0 (1.0)
        session_summary['cn_tuning_slope'] = tuning_slopes[cn_idx]
        session_summary['cn_tuning_percentile'] = cn_sort_idx / (tuning_slopes_sort.shape[0] - 1)
    else:
        session_summary['cn_tuning_slope'] = np.nan
        session_summary['cn_tuning_percentile'] = np.nan
    
    # This tuning analysis is depricated, use same pipeline we use in experiment to compute various session period tunings
#     ### Activity change calculations ###
#     start_time = timeit.timeit()
#     if activity_type in ('raw_activity',):
#         activity_analyze = output
#     elif activity_type in ('raw_fl',):
#         activity_analyze = output_fl
#     elif activity_type in ('dff',):
#         raise NotImplementedError()
#         activity_analyze = output_fl
#     # Use current session_idx
#     post_activities, pre_activities, tuning_lengths = bci_analysis.get_our_bci_trial_aligned_activities(
#         activity_analyze, task, task.session_idx, tuning_type='trial'
#     )
#     tuning = bci_analysis.get_our_bci_tuning(post_activities, pre_activities)

#     if session_summary['n_trials'] > N_AVG_TRIALS and not np.isnan(cn_idx): # Need at least the average number of trials for this to be valid

#         tuning_slopes = np.zeros((task_params['n_neurons'],))
#         for neuron_idx in range(task_params['n_neurons']): # Compute each
#             mean_tuning = fftconvolve(tuning[:, neuron_idx], trial_avg_window, mode='valid')
#             tuning_slopes[neuron_idx], intercept, rvalue = add_regression_line(np.arange(mean_tuning.shape[0]), mean_tuning)

#         # Normalizes
#         tuning_slopes_norm = tuning_slopes / np.max(np.abs(tuning_slopes))

#         tuning_slopes_sort = np.argsort(tuning_slopes) # Smallest to largest

#         cn_sort_idx = np.where(tuning_slopes_sort == cn_idx)[0][0]
#         # idx = 0 (idx = n-1) -> perc = 0.0 (1.0)
#         session_summary['cn_tuning_slope'] = tuning_slopes[cn_idx]
#         session_summary['cn_tuning_percentile'] = cn_sort_idx / (tuning_slopes_sort.shape[0] - 1)
#     else:
#         session_summary['cn_tuning_slope'] = np.nan
#         session_summary['cn_tuning_percentile'] = np.nan

    if verbose:
        print('Total reward: {:.2f}, total rpe: {:.2f}\nTrials: {}, perc. sat. neurons: {:.2f}'.format(
            session_summary['total_reward'], session_summary['total_rpe'],
            session_summary['n_trials'], session_summary['perc_saturated_neurons'],
        ))
        print('Hit rates - init: {:.2f}, final: {:.2f}, all: {:.2f}'.format(
            session_summary['hit_rate_init'], session_summary['hit_rate_final'],
            session_summary['hit_rate'],
        ))
        print('CN idx {} (activity perc: {:.2f}) // Tuning - perc: {:.2f} (slope: {:.2e})'.format(
            session_summary['cn_idx'], session_summary['cn_idx_activity_percentile'],
            session_summary['cn_tuning_percentile'], session_summary['cn_tuning_slope'],
        ))

    return session_summary

def plot_more_our_bci_task(axs, x_axis, task_params, task, session_idx):
    """ Code specifically for analyzing our BCI task """

    assert len(axs) == 2
    ax1, ax2 = axs

    t_window = 30 # sec
    n_mean_window = int(t_window * 1000 / task_params['t_step'])
    mean_window = 1 / n_mean_window * np.ones((n_mean_window,))
    count_window = np.ones((n_mean_window,))

    seq_idxs = np.arange(task_params['n_steps'])
    dt = task_params['t_step']
    task_hist = task.hists[session_idx]

    n_trials = len(task_hist['trial_outcomes'])
    trial_start_times = dt * np.array(task_hist['trial_starts']) / 1000
    trial_end_times = dt * np.array(task_hist['trial_ends']) / 1000
    reward_times = dt * np.array(task_hist['rewards']) / 1000

    ## Get all the start times of successful trials
    # Might have one more start than outcome because last trial incomplete
    trial_start_successes = np.array(task_hist['trial_starts'][:n_trials]) * np.array(task_hist['trial_outcomes'])
    trial_start_successes = trial_start_successes[trial_start_successes > 0] # Filters only successful trials

    # Mask over all seq idxs for when trial starts/successful trials occur
    trial_starts_mask = np.array([seq_idx in np.array(task_hist['trial_starts']) for seq_idx in seq_idxs]).astype(np.int32)
    trial_starts_hit_mask = np.array([seq_idx in trial_start_successes for seq_idx in seq_idxs]).astype(np.int32)

    rolling_count_trials = signal.fftconvolve(trial_starts_mask, count_window, mode='full')[:-n_mean_window+1]
    rolling_count_hits = signal.fftconvolve(trial_starts_hit_mask, count_window, mode='full')[:-n_mean_window+1]

    ax1.plot(x_axis, rolling_count_trials, label='Avg # Trials',
            color=c_vals[0], zorder=5)
    ax1.plot(x_axis, rolling_count_hits, label='Avg # Reward Trials',
            color=c_vals[1], zorder=6)
    ax1.set_ylabel('Avg. trials/rewards\n(window {} sec)'.format(t_window))

    # Code below calculates rolling averages in a different way than above because number of events in a window can change
    # and this matters for all averaging

    t_window = 30 # sec
    n_mean_window = int(t_window * 1000 / task_params['t_step'])
    count_window = np.ones((n_mean_window,))

    trial_lens = np.array(task_hist['trial_ends']) - np.array(task_hist['trial_starts'][:n_trials])
    pretrial_lens = np.array(task_hist['pretrial_ends']) - np.array(task_hist['pretrial_starts'][:len(task_hist['pretrial_ends'])])

    trial_starts_idxs = task_hist['trial_starts'][:n_trials]
    pretrial_starts_idxs = task_hist['pretrial_starts'][:len(task_hist['pretrial_ends'])]

    # Mask over all seq idxs for when trial starts/successful trials occur
    trial_idx = 0
    trial_starts_by_seq_idx = np.zeros((task_params['n_steps'],))
    trial_lens_by_seq_idx = np.zeros((task_params['n_steps'],))
    trial_hits_by_seq_idx = np.zeros((task_params['n_steps'],))
    pretrial_idx = 0
    pretrial_starts_by_seq_idx = np.zeros((task_params['n_steps'],))
    pretrial_lens_by_seq_idx = np.zeros((task_params['n_steps'],))
    for seq_idx in seq_idxs:
        if seq_idx in trial_starts_idxs:
            trial_starts_by_seq_idx[seq_idx] = 1.0
            trial_lens_by_seq_idx[seq_idx] = trial_lens[trial_idx] * task_params['t_step'] / 1000
            trial_hits_by_seq_idx[seq_idx] = task_hist['trial_outcomes'][trial_idx]
            trial_idx += 1
        if seq_idx in pretrial_starts_idxs:
            pretrial_starts_by_seq_idx[seq_idx] = 1.0
            pretrial_lens_by_seq_idx[seq_idx] = pretrial_lens[pretrial_idx] * task_params['t_step'] / 1000
            pretrial_idx += 1

    # trial_starts_mask = np.array([seq_idx in np.array(task_hist['trial_starts']) for seq_idx in seq_idxs]).astype(np.int32)

    rolling_avg_trial_lens = signal.fftconvolve(trial_lens_by_seq_idx, count_window, mode='full')[:-n_mean_window+1]
    rolling_avg_trial_hits = signal.fftconvolve(trial_hits_by_seq_idx, count_window, mode='full')[:-n_mean_window+1]
    rolling_avg_trial_starts = signal.fftconvolve(trial_starts_by_seq_idx, count_window, mode='full')[:-n_mean_window+1] + 1e-7 # For numerical stability
    rolling_avg_trial_lens = np.where(rolling_avg_trial_lens / rolling_avg_trial_starts > 10000,
                                    0.0, rolling_avg_trial_lens / rolling_avg_trial_starts)
    rolling_avg_trial_hits = np.where(rolling_avg_trial_hits / rolling_avg_trial_starts > 10000,
                                    0.0, rolling_avg_trial_hits / rolling_avg_trial_starts)
    rolling_avg_pretrial_lens = signal.fftconvolve(pretrial_lens_by_seq_idx, count_window, mode='full')[:-n_mean_window+1]
    rolling_avg_pretrial_starts = signal.fftconvolve(pretrial_starts_by_seq_idx, count_window, mode='full')[:-n_mean_window+1] + 1e-7 # For numerical stability
    rolling_avg_pretrial_lens = np.where(rolling_avg_pretrial_lens / rolling_avg_pretrial_starts > 10000,
                                    0.0, rolling_avg_pretrial_lens / rolling_avg_pretrial_starts)

    ax2.plot(x_axis, rolling_avg_trial_hits, color='k', label='Hit rate')
    ax2.set_ylabel('Hit rate\n(window: {} sec)'.format(t_window))
    ax2.set_ylim((-0.1, 1.1))
    ax2.legend()

    ax2p = ax2.twinx()
    ax2p.plot(x_axis, rolling_avg_trial_lens, color=c_vals[0],
            label='Trial time')
    ax2p.plot(x_axis, rolling_avg_pretrial_lens, color=c_vals[1],
            label='Pre_trials time')
    ax2p.set_ylabel('Time within state (s)\n(window: {} sec)'.format(t_window))
    ax2p.set_ylim((-1, 11.))
    ax2p.legend()

def plot_dynamic_threshold(task_hist, ax, task_params):
    """ Extracts from task_hist what is needed to plot a dynamic threshold on the chosen plot """
    n_threshold_changes = len(task_hist['threshold_changes'])
    for threshold_idx, (threshold_change_idx, threshold_vals) in enumerate(
        zip(task_hist['threshold_changes'], task_hist['threshold_vals'])
    ):
        if threshold_idx == n_threshold_changes-1:
            start_idx, end_idx = threshold_change_idx, task_params['n_steps']
        else:
            start_idx, end_idx = threshold_change_idx, task_hist['threshold_changes'][threshold_idx+1]
        start_idx = task_params['t_step'] / 1000 * start_idx
        end_idx = task_params['t_step'] / 1000 * end_idx
        ax.plot( # Lower
            (start_idx, end_idx), (threshold_vals[0], threshold_vals[0]), color=c_vals_d[2], linestyle='dashed', zorder=5
        ) 
        ax.plot( # Upper
            (start_idx, end_idx), (threshold_vals[1], threshold_vals[1]), color=c_vals_d[2], linestyle='dashed', zorder=5
        )

def run_test_session_our_bci(params, net, task, prev_seq_vars, output_vars, n_steps_test=2400, run_ps=False,
                             clip_train_outputs=True):
    """
    Runs a test session of our BCI task. 
    
    Makes use of 'prev_seq_vars' and internally tracked state in 'task', which allows the 
    test session to be initialized exactly where the current task left off. Copies the network,
    task, and parameters so that test session can be run without affecting the current task.
    
    Handles all the copying of internally tracked variables.
    
    INPUTS:
    - n_steps_test: number of steps to run test session for
    - clip_train_outputs: bool, clips train_outputs so that they only contain the test_session indexes
        rather than repeated information in the prev_seq_vars.
    """
    # Saves random state so the effect of running this on randomly generated
    # quantities is minimal
    start_rng_state = np.random.get_state()

    task_test = copy.deepcopy(task)
    net_test = copy.deepcopy(net)
    current_step_idx = np.copy(task_test.seq_idx)

    task_params, train_params, net_params = params
    task_params_test = copy.deepcopy(task_params)
    task_params_test['n_sessions'] = 1
    if run_ps:
        task_params_test['photostim'] = 'every_session' 
    else:
        task_params_test['photostim'] = None 
    
    init_step_idx = np.copy(task_test.seq_idx - 1) # -1 since one step ahead
    
    # Run the task again, with prev_seq_vars passed so it starts from previous point, and no training 
    params_test, train_outputs_test, net_test, task_test, _ = train_task(
        (task_params_test, train_params, net_params), output_vars=output_vars,
        net=net_test, task=task_test, prev_seq_vars=prev_seq_vars,
        n_steps = current_step_idx+n_steps_test, train=False, verbose=False
    )
    
    if clip_train_outputs: # Clip each output_var to be only the test session indexes 
        assert task_params['n_sessions'] == 1 # Only implemented for single session right now
        session_idx = 0 
        final_step_idx = np.copy(task_test.seq_idx - 1) # -1 since one step ahead
        for output_var in output_vars:
            # print('{} shape:'.format(output_var), train_outputs_test[session_idx][output_var].shape)
            train_outputs_test[session_idx][output_var] = train_outputs_test[session_idx][output_var][init_step_idx:final_step_idx]
            # print('{} clip shape:'.format(output_var), train_outputs_test[session_idx][output_var].shape)
    
    # # Verify the test network is actually performing as expected
    # print('Outcomes:', task_test.hists[0]['trial_outcomes'])
    # print('Trial starts:', task_test.hists[0]['trial_starts'])
    # print('Rewards:', task_test.hists[0]['rewards'])
    
    np.random.set_state(start_rng_state) # Reset random state

    return params_test, train_outputs_test, net_test, task_test

def compute_avg_reward_our_bci(params, net, task, prev_seq_vars, n_steps_test=2400):
    """
    Computes the average reward at the start of the session for our BCI task.
    This requires a test session be run because the network's activity influences
    the environment. Thus a test session must be run to see what reward would be
    at this point in the task.

    Unlike the toy task, cannot simply be estimated from presession activity
    because presession does not progress through the task environment.
    """
    print('Running test network/task to compute starting reward...')
    
    task_params, train_params, net_params = params
    
    if task_params['start_avg_reward'] not in ('dynamic',):
        raise NotImplementedError('Default for this mode is dynamic.')
    if task_params['env_drive_mode'] in ('saved_activity',):
        raise NotImplementedError('Check to see if saved activity drive makes sense for running test network, env_drive_activity set to actual activity here.')
    
    # We need to temporarily compute what the next net_input[step_idx+1] will be, since 
    # this code is called before it is computed, so unpack prev_seq_vars, compute, then repack
    ( 
        net_input, output, output_pre_act, reward, avg_reward, avg_activity,
        avg_activity_pre_act, mean_stabilize_actvities, output_fl,
        bci_activity, avg_bci_activity, avg_reward_stims
    ) = prev_seq_vars
    
    bci_mask_idx = 0
    step_idx = task.seq_idx # These are in sync because next task step hasn't been taken yet
    env_drive_activity = bci_activity[step_idx - task_params['n_reward_delay']][bci_mask_idx] # Delay incorporated into offset of bci_activity (defined to have no delay)
    task_copy = copy.deepcopy(task) # Use copy of task, since we dont want to iterate the actual task forward
    net_input[step_idx+1], _, _, _, _ = ( # BCI activity includes fl activity if used
        task_copy.step(env_drive_activity)
    )
    prev_seq_vars = ( # Repackage updates seq_vars
        net_input, output, output_pre_act, reward, avg_reward, avg_activity,
        avg_activity_pre_act, mean_stabilize_actvities, output_fl,
        bci_activity, avg_bci_activity, avg_reward_stims
    )
    
    ### Run test session ###
    # Note its important we pass task_copy here to stay in sync, since its on step_idx+1 and task is not. 
    output_vars = ('reward',)
    _, train_outputs_test, _, _ = run_test_session_our_bci(
        params, net, task_copy, prev_seq_vars, output_vars, n_steps_test=n_steps_test, clip_train_outputs=True
    )
    del task_copy
    
    session_idx = 0
    current_step_idx = np.copy(task.seq_idx)
    return np.mean(train_outputs_test[session_idx]['reward'])

def compute_estimated_tunings_our_bci(params, net, task, train_outputs_all, prev_seq_vars, n_steps_test=2400, output_vars=[]):
    """
    Computes estimated tuning of neurons from either (1) a previous session or
    (2) a separate breakout test session. In the latter case, this requires
    the network run test_session. Currently, this is done using an external
    drive for the task, so that no neuron is on special footing to control 
    the BCI task.
    
    Note when a test session is run, the tuning is computed using the OLD
    definition of tuning that Kayvon originally used, i.e. termination only
    at the start of the next trial, and keeping all pres regardless of whether
    it followed a hit or not. 
    
    Our newer analysis definitions are sometimes also computed for certain
    analyses here, i.e. seeing if certain epoch activities are predictive 
    of performance differences.

    INPUTS:
    - train_outputs_all: Only used to compute tunings from previous session
    
    OUTPUTS:
    -
    - tuning_extras: dictionary of extra quantities, used to output test_tunings if relevant 

    """

    tuning_extras = {}
    
    # Saves random state so the effect of running this on randomly generated
    # quantities is minimal
    start_rng_state = np.random.get_state()

    task_params, train_params, net_params = params

    
    if task.session_idx > 0: # Compute tuing from previous seession's activity
        print('Computing tuning from previous session...')
        # Use ENTIRE previous session's activity to compute tuning
        tuning_session_idx = task.session_idx - 1
        if task_params['add_fl']:
            if 'output_fl' in train_outputs_all[tuning_session_idx]:
                tuning_activity = train_outputs_all[tuning_session_idx]['output_fl']
            else: # Might need to reproduce fl output
                tuning_activity = bci_analysis.reproduce_output_fl(
                    train_outputs_all[tuning_session_idx]['output'], task_params['fl_kernel']
                )
        else:
            tuning_activity = train_outputs_all[tuning_session_idx]['output']
        tuning_task = task
    elif task_params['test_tuning']: # Specical case where we need to compute tunings by running a test session
        print('Running test session to compute tuning...')

        task_test = copy.deepcopy(task)
        net_test = copy.deepcopy(net)
        current_step_idx = np.copy(task_test.seq_idx)

        task_params_test = copy.deepcopy(task_params)
        # This will skip any BCI mask assignment, since it is not needed in this setup.
        # Creates a fake BCI mask, the activity is not relevant here either
        task_params_test['n_steps_stabilize'] = 0
        task_params_test['n_sessions'] = 1
        task_params_test['bci_masks'] = np.zeros((task_params['n_bci_masks'], task_params['n_neurons'],))
        task_params_test['cn_idxs'] = [np.nan,]
        task_params_test['cn_idxs_activity_percentile'] = [np.nan,]

        task_params_test['photostim'] = None # Turn off photostim

        task_params_test['env_drive_mode'] = 'saved_activity' # Saved BCI activty drives environemnt

        # Load states from saved file
        # load_path = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/test_bci_activity'
        load_path = '/scratch/test_bci_activity'
        task_params_load, bci_activity_load, task_load = net_helpers.load_bci_activity(load_path, 22222)

        task_test.copy_attributes(task_load) # Copies threshold and other bci-relevant values

        # Tuple of keys to check are the same between loaded task and current task
        # reward_mode not needed since it just drives activity, doesn't actually train
        task_syncs = ('task_scale', 't_step', 'simple_states', 'noise_type', 'noise_timescale',
                      'state_mode', 'add_fl', 'stim_to_noise_ratio',)
        # 'stim_to_noise_ratio',
        for task_sync in task_syncs:
            if task_sync in ('state_mode', 'stim_to_noise_ratio',):
                print(' TUNING COMPUTATION: Skipping task_sync on {} for now.'.format(task_sync))
                continue
            assert task_params_test[task_sync] == task_params_load[task_sync]

        task_params_test['env_drive_activity'] = bci_activity_load
        
        output_vars_test = ('reward', 'output', 'output_fl',)
        
        _, train_outputs_test, _, _, _ = train_task(
            (task_params_test, train_params, net_params), output_vars=output_vars_test,
            net=net_test, task=task_test, prev_seq_vars=prev_seq_vars,
            n_steps = current_step_idx + n_steps_test, train=False, verbose=False
        )

        # Pass everything up to run so far so indexing in trial and test_activity are aligned.
        tuning_session_idx = 0
        if task_params_test['add_fl']:
            tuning_activity = train_outputs_test[tuning_session_idx]['output_fl'][:current_step_idx+n_steps_test, :]
        else:
            tuning_activity = train_outputs_test[tuning_session_idx]['output'][:current_step_idx+n_steps_test, :]
        tuning_task = task_test
    else:
        raise NotImplementedError('This shouldnt happen.')

    assert not np.isnan(tuning_activity).any() # Shouldn't have any nans
    
    # Kept for backwards compatibility purposes, this is Kayvon's old definition of tuning which does not 
    # match our analysis definition, but is what he used to use to determine CN choice
    post_activities, pre_activities, tuning_lengths = bci_analysis.get_our_bci_trial_aligned_activities(
        tuning_activity, tuning_task, tuning_session_idx, tuning_type='trial'
    )
    tuning = bci_analysis.get_our_bci_tuning(post_activities, pre_activities)

    if 'test_tunings' in output_vars:
        # Semi-repetitive as above, used to compare to data analysis techniques
        tuning_mode = 'omit_misses' # default of experiment, so use it here too
        task_metrics, ts_extras = bci_analysis.compute_task_metric_aligned_values(
            tuning_activity[task_params['n_steps_stabilize']:], tuning_task.hists[tuning_session_idx], task_params, tuning_mode=tuning_mode, 
            fit_changes=False, return_trial_ts_metrics=False, return_task_metric_stds=True,
        )
        tuning_extras['test_tunings'] = task_metrics
        tuning_extras['test_tuning_stds'] = ts_extras['ts_metric_stds']
    
    np.random.set_state(start_rng_state) # Reset random state
    
    estimated_tunings = np.mean(tuning, axis=0)  # Mean over trials
    
    return estimated_tunings, tuning_extras

def set_to_our_bci_defaults(task_params, train_params, net_params):
    """
    Default parameters for our BCI task.
    """

    local_task_params = {

        ### Task choice
        'task_type': 'our_bci', # (should already be set to this, just to be explicit)

        'z_score_activities': False,
        
        ### BCI mask choice (threshold/width used to set spout movement threshold/max speed)
        'bci_choice': 'activity_high', # random, activity_percentile, activity_high, manifold, intuitive, intuitive_manifold
        ## Parameters for various bci_choice settings
            'n_bci_masks': 1,
            'activity_percentile': 0.7, # For percentile and high_activity, determines tuning percentile.
            'n_cns': 1, # Used for settings where number of CNs is varaible (n_hot), automatically set for other settings
            'constant_stim_input': False,        
        ### What to set the threshold based on: percentile_bci_activity, bci_activity_mean_and_std, fixed_neural_distance
        'dyn_threshold_type': 'percentile_bci_activity', # mean_all_neurons, percentile_bci_activity (what to set the threshold based on)
        ## Parameters for various threshold choices
            'dyn_threshold_perc': 0.7, # 0.7, # In our_bci, used to set spout move threshold
            'dyn_width_perc': 1.0, # In our_bci, used to set max movement speed
        
        ### Session length parameters
        't_step': 50, # ms
        'n_steps': 20000,
        'n_steps_stabilize': 2400,
        'n_steps_evaluate': 0, # Stops training to observe solution        

        'task_scale': 0.1, # Determines the scale of the noise input and also the constant input when used

        'n_sessions': 1,
        'use_max_activity': False, # Clip activity at maximum, helps stabilize training with recurrent adjustment

        'run_test_idxs': None, #(2400, 19999,), # Indices to run test_network, None or tuple of indices 
        'n_steps_test': 4800,
        
        'track_raw_states': False, # Keep track of spout position, trial start tone, reward
        'solution_type': None, # Not yet implemented
        'perturbations': None,

        ### Parameters specific for our BCI task model ###
        'tuning_percentile': 0.25, # Tuning percentile, note want to be BELOW this threshold
        'test_tuning': True, # True, # whether or not to run separate test session to estimate tunings prior to training
        'env_drive_mode': 'default', # default, saved_activity; used to compute pre-session tuning
        'simple_states': False, # If true, removes pretrial period

        # Noise params
        'noise_type': 'iid', # iid, tc_weight, None
        'noise_timescale': 500, # units of ms, only for tc noise
        'steps_stored': 1000,

        ### Reward-relevant parameters
        'reward_structure': 'trapezoid',
        'normalize_reward': False,
        'n_reward_delay': 10, # Number of time steps to delay the current reward, 10 = 500 ms
        'reward_mode': 'water_and_spout', # water_only, water_and_spout, thirst, spout_and_thirst
        'state_mode': 'mix_spout_loc_1d', # 'mix_spout_movement', # mix_spout_loc, mix_spout_loc_1d, mix_spout_movement, mix_spout_movement_abs
        'start_avg_reward': 'dynamic', # how to compute avg_reward at start: None, float, dynamic
        'start_avg_reward_mult': 1.0, # For Hebbian-idx tests, >1.0 means higher reward expected and leads to initial negative RPE

        'stim_to_noise_ratio': 0.25, # > 1 more stim than noise
        'spout_movement_reward_scale': 1.0, # Relative amount of reward from stim movement and actual reward
        'thirst_reward_timescale': 500,

        # Fluorescence parameters
        'add_fl': True,

        ### Photostim parameters
        'photostim': None, #'every_session', # None, every_session
        'neuron_fidelity_mode': 'ones', # ones, uniform_random, copy
        'n_groups': 100,
        'max_fidelty': 0.5, #1.5, #0.5, # Since neuron is leaky, this needs to be quite large to create large change in activity
        'n_repeats_per_group': 20,
        # 'n_steps_photostim': 19200, # 16 minutes
        
        ### Dynamic threshold parameters
        'threshold_change_type': 'fixed', # fixed, seq_idxs
        'threshold_change_idxs': (8000, 14000,),
        'threshold_change_intertrial': True, # Change thresholds only during inter-trial
        'threshold_change_mag_type': 'fixed', # fixed, uniform 
        'threshold_change_mag_params': (1.5, 0.0) # (mean, std), in terms of gain_new / gain_old distribution
         
    }

    local_train_params = {
        # W_rec train: 2e0 for 0.25 stim/noise, 5e-1 for 1.0 stim/noise
        # W_inp train: 1e2 for 0.25 stim noise, 5e0 for 1.0 stim/noise
        # W_rec train, mix_spout_loc_1d stim, 1.0 stim/noise:
        'eta': 1e0,
        'n_window_reward': 300 * 20, #60 * 20, # Size of the average window for reward baseline
        'n_window_baseline': 10 * 20, # Size of the average window for activity
        'n_steps_per_loss': 5, # How often to total RPE and adjust weights
        'eligibility_acc_type': 'running_average', # acc_and_wipe, running_average
        'n_window_elig': 40, # Only used for 'running_average' option, how long to keep eligibility

        'rpe_clip': 0.05,
    }

    local_net_params = {
        'direct_input': False,

        'weight_mask_modes': None, # None or tuple with: cn_freeze, cn_only, cn_no_output
        
        # alpha, 0.0 means no leak. 0.8 corresponds to 250 ms timescale, 0.5 to 100 ms
        'leak_term': 0.5, 
    }

    for local_key in local_task_params.keys():
        task_params[local_key] = local_task_params[local_key]
    for local_key in local_train_params.keys():
        train_params[local_key] = local_train_params[local_key]
    for local_key in local_net_params.keys():
        net_params[local_key] = local_net_params[local_key]

    return task_params, train_params, net_params

def run_photostim(params, output_vars=[], task=None, task_ps=None, net=None, verbose=True):
    """
    Run a photostim session. This is a very similar loop to training, but
    instead interacts with the photostim task/environment.

    task_ps: If None, initializes. If not None, starts new session and resets.
    """

    # Saves random state so the effect of running this on randomly generated
    # quantities is minimal
    start_rng_state = np.random.get_state()

    train_outputs_ps = {}

    task_params, train_params, net_params = params
    assert task is not None
    assert net is not None

    if net_params['direct_input']:
        raise NotImplementedError('Setup only allows for PS to be direc')

    if task_ps is None: # Initialize photostim
        task_ps = BCIGym.Photostim_Env(task_params, task)
    else: # Iterate photostim to next session, reset below
        task_ps.set_new_session()

    n_steps_photostim = task_ps.n_steps_photostim

    ### Quantities to track throughout photostim run ###
    output_ps = np.zeros((n_steps_photostim, task_params['n_neurons'],))
    output_pre_act_ps = np.zeros((n_steps_photostim, task_params['n_neurons'],))

    # # Turn this on just for consistency checks, if we want identical noise in the two setups
    # print('Seed reset!')
    # np.random.seed(task_params.get('seed', 0))

    net_input_ps = np.zeros((n_steps_photostim, task_params['n_inp'],))
    net_photostim_input_ps = np.zeros((n_steps_photostim, task_params['n_neurons'],))
    net_input_init, _ = task_ps.reset()
    net_input_ps[0] = net_input_init[0]
    net_photostim_input_ps[0] = net_input_init[1]

    print('Running {:.1f} minutes photostim ({} steps)...'.format(
        n_steps_photostim * task_ps.dt / 60, n_steps_photostim
    ))

    for step_idx in range(n_steps_photostim):

        if step_idx == 0: # Initialization values for each session (these are not carried between sessions)
            # Up to a minute between photostim and BCI task, so don't carry activity between the two
            prev_activity = np.zeros((task_params['n_neurons'],))
            prev_activity_pre_act = np.zeros((task_params['n_neurons'],))
        else:
            prev_activity = output_ps[step_idx-1]
            prev_activity_pre_act = output_pre_act_ps[step_idx-1]

        # Network forward pass #
        current_input = net_input_ps[step_idx]
        output_ps[step_idx], output_pre_act_ps[step_idx] = net.forward(
            current_input, prev_activity, prev_activity_pre_act, net_params,
            perturbation_preact=net_photostim_input_ps[step_idx]
        )

        # Clip total activity for numerical stability
        if 'max_activity' in task_params:
            if task_params['max_activity'] is not None:
                output_ps[step_idx] = np.clip(output_ps[step_idx], None, task_params['max_activity'])

        # Fl activity is computed externally, since not needed for task
        net_input_temp, _, _, _, _ =  task_ps.step(None)
        if step_idx < n_steps_photostim - 1: # No assignment on final step
            net_input_ps[step_idx+1] = net_input_temp[0]
            net_photostim_input_ps[step_idx+1] = net_input_temp[1]
            
        # # Print out summary
        # if step_idx > 0 and step_idx % train_params['print_every'] == 0:
        #     if verbose: print('PS Step {}'.format(step_idx))

    if 'output_fl_ps' in output_vars:
        output_fl_ps = bci_analysis.reproduce_output_fl(
            output_ps, task_params['fl_kernel']
        )

    if 'output_dff_ps' in output_vars: # Needs fl to compute
        output_dff_ps, train_outputs_ps['output_dff_perc_cutoff_ps']  = bci_analysis.compute_dff(
            output_fl_ps, task_params, #f0_CUTOFF= 0.1 * task_params['task_scale'],
            verbose=verbose
        )

    assert 'task_ps' not in output_vars # Output separately, shouldnt be in train_outputs (depricated)
    for output_var in output_vars:
        exec('train_outputs_ps[\'{}\'] = {}'.format(output_var, output_var))

    np.random.set_state(start_rng_state) # Reset random state

    params = task_params, train_params, net_params

    return params, train_outputs_ps, net, task_ps

def simple_null(task_params, mode='task'):
    """
    A simple Poisson rate null model for generating a fluorecence-like signal

    Note that fl convolution has been implemented directly into the environment,
    so option to turn if off here.

    """

    if mode in ('task',):
        timesteps = task_params['n_steps']
    elif mode in ('photostim',):
        timesteps = task_ps.n_steps_photostim
    else:
        raise ValueError('mode not recognized')
    n_neurons = task_params.get('n_neurons', 1)
    firingrate = task_params.get('FR', 0.1)

    return np.random.poisson(lam=firingrate, size=(timesteps, n_neurons,))

def print_null_results(task):
    time_elapsed = len(task.hist['world_trajectory']) * task_params['t_step'] / 1000 # Seconds
    print('--- Session {}, CN idx ---'.format(task.session_idx))

    if task_params['task_type'] in ('our_bci',): # These do not make sense for toy task

        print(' Success rate: {:.2f}'.format(
            len(task.hist['rewards']) / len(task.hist['trial_starts'])
        ))
        print('  Trials: {}, rate {:.2f} (per second)'.format(
            len(task.hist['trial_starts']), len(task.hist['trial_starts']) / time_elapsed
        ))

        print('  Rewards: {}, rate {:.2f} (per second)'.format(
            len(task.hist['rewards']), len(task.hist['rewards']) / time_elapsed
        ))

n_neurons = 50

seed = 2222

train_params = {}

net_params = {}

default_fills = {'bci_masks': None, 'activity_subtract': None, 'activity_stds': None,
                 'threshold': None, 'width': None, 'perturbations': None,
                 'dyn_width_perc': None, 'max_activity': None,
                 'z_score_activities': False,
                 'use_max_activity': False, 'add_fl': False,}

n_noise = 3

seq_start = 2520

seq_end = 2540

verbose = False

session_idx = 0

output_vars_ps = ['output_ps', 'output_fl_ps', 'output_dff_ps', 'net_input_ps',
                  'net_photostim_input_ps',]

session_idx = 0

n_neurons_plot = 3

omit_ps_times = True

ps_activity_type = 'dff'

session_idx = 0

omit_ps_times = True

ps_activity_type = 'dff'

causal_connectivity_mode = 2

import copy

class TrialStructureTask():
    """
    Generates a basic trial-structure task.
    """

    def __init__(self, task_params, net_params):
        """
        Set defaults based on task_params (only needs net_params to get
        act_fn_type to generate appropriate inputs).
        """

        self.n_steps_trial = task_params.get('n_steps_trial', 100)
        self.trial_type = task_params.get('trial_type', 'high_stim') # high_stim, high_low_stim, center_out_1d, center_out_2d,

        self.n_neurons = task_params.get('n_neurons')
        self.n_bci_masks = task_params.get('n_bci_masks', 1) # Overriden for certain tasks
        self.task_scale = task_params.get('task_scale', 0.1)
        # Relative size of stim and noise (similar to signal to noise).
        # Note noise is always passed through W_inp, so can change with direct input
        # >1: Larger stim compared to noise
        # <1: Larger noise comapred to stim
        self.stim_to_noise_ratio = task_params.get('stim_to_noise_ratio', 1.0)

        self.reward_scale = task_params.get('reward scale', 0.025) # Overriden below for certain tasks

        # Way to more directly control stimulus signal where it doesn't need to pass through random matrix anymore
        self.net_direct_input = task_params.get('direct_input', False)
        if self.net_direct_input:
            self.net_W_inp = task_params.get('W_inp')

        self.act_fn_type = net_params.get('act_fn_type', 'ReTanh')

        if self.trial_type in ('high_stim', 'high_low_stim',):
            assert self.n_bci_masks == 1
            self.n_stim = 2
            # All these are filled after difficulty is set
            self.stim_targets = np.zeros((self.n_stim, self.n_bci_masks,))
            self.stim_widths = np.zeros((self.n_stim, self.n_bci_masks,))
            self.stim_reward_scales = np.zeros((self.n_stim, self.n_bci_masks,))

            self.n_trial_periods = self.n_stim # Trial periods defined by number of distinct stimuli
            self.get_trial_fn = self.activity_stim_trial
            self.set_task_difficulty_fn = self.activity_stim_set_task_difficulty
            self.reward_fn = trapezoid_reward
            self.stim_input = np.zeros((self.n_stim, self.n_neurons,))
            self.stim_input[0:1, :] = self.stim_to_noise_ratio * get_stimulus(task_params)
            self.stim_input[1:2, :] = self.stim_to_noise_ratio * get_stimulus(task_params)
            if self.trial_type in ('high_stim',):
                self.trial_period_percs = 1.0 * np.ones((self.n_trial_periods,)) # Highest in both
            elif self.trial_type in ('high_low_stim',):
                assert self.n_trial_periods == 2 # Doesn't make sense for more than two currently
                self.trial_period_percs = np.array((1.0, 0.)) # Highest in first, lwoest in second
                self.reward_smoothing = False # Can sometimes lead to runaway rewards because crosses zero
        elif self.trial_type in ('center_out_1d', 'center_out_1d_1', 'center_out_2d', 'center_out_2d_8',):

            self.n_trial_periods = 1 # Each trial is just a different stim
            self.stim_sample_mode = 'pseudo-random' # pseudo-random, random
            self.possible_stim_idxs = [] # Only used for pseudo-random option
            self.get_trial_fn = self.variable_stim_trial
            self.set_task_difficulty_fn = self.stim_target_set_task_difficulty
            # self.reward_fn = quadratic_reward
            self.reward_fn = center_out_reward
            self.reward_scale = np.sqrt(2) # Sets this to something on order of typical distances in BCI space

            if self.trial_type in ('center_out_1d',): # 1d BCI space, 2 targets
                self.n_bci_masks = 1
                self.n_stim = 2
                self.stim_targets = np.zeros((self.n_stim, self.n_bci_masks,)) # Used to set intrinsic mapping
                self.stim_targets[0, 0] = 1.
                self.stim_targets[1, 0] = -1.
            elif self.trial_type in ('center_out_1d_1',): # 1d BCI space, 1 target
                self.n_bci_masks = 1
                self.n_stim = 1
                self.stim_targets = np.zeros((self.n_stim, self.n_bci_masks,))
                self.stim_targets[0, 0] = 1.
            elif self.trial_type in ('center_out_2d',): # 2d BCI space, 4 targets
                self.n_bci_masks = 2
                self.n_stim = 4
                self.stim_targets = np.zeros((self.n_stim, self.n_bci_masks,))
                self.stim_targets[0, :] = np.array((1., 0.))
                self.stim_targets[1, :] = np.array((0., 1.))
                self.stim_targets[2, :] = np.array((-1., 0.))
                self.stim_targets[3, :] = np.array((0., -1.))
            elif self.trial_type in ('center_out_2d_8',): # 2d BCI space, 8 targets
                self.n_bci_masks = 2
                self.n_stim = 8
                self.stim_targets = np.zeros((self.n_stim, self.n_bci_masks,))
                self.stim_targets[0, :] = np.array((1., 0.))
                self.stim_targets[1, :] = 1/np.sqrt(2) * np.array((1., 1.))
                self.stim_targets[2, :] = np.array((0., 1.))
                self.stim_targets[3, :] = 1/np.sqrt(2) * np.array((-1., 1.))
                self.stim_targets[4, :] = np.array((-1., 0.))
                self.stim_targets[5, :] = 1/np.sqrt(2) * np.array((-1., -1.))
                self.stim_targets[6, :] = np.array((0., -1.))
                self.stim_targets[7, :] = 1/np.sqrt(2) * np.array((1., -1.))

            self.stim_widths = None # This doesn't use a reward function with width
            self.stim_reward_scales = np.zeros((self.n_stim, self.n_bci_masks,))

            if not np.all(np.abs(np.linalg.norm(self.stim_targets, axis=-1) - 1) < 1e-3):
                raise NotImplementedError(
                    'Center-out reward function assumes targets have magnitude 1!!'
                )

            self.stim_input = np.zeros((self.n_stim, self.n_neurons,))
            for stim_idx in range(self.n_stim): # Generate stimuli corresponding to distinct inputs
                # The size of this determines how much stimuli dominate activity
                if self.stim_to_noise_ratio < 1.0: # Ensures maximum scale of stim and noise is roughly task_scale
                    stim_mult = self.stim_to_noise_ratio # Reduces stim relative to noise
                else:
                    stim_mult = 1.0 # Reduces noise relative to stim
                # print('Setting stim_mult to:', stim_mult)
                # self.stim_input[stim_idx:stim_idx+1, :] = stim_mult * get_stimulus(task_params)
                self.stim_input[stim_idx:stim_idx+1, :] =  get_stimulus_special(task_params, stim_idx, self.n_stim, 'sparse_binary')
        else:
            raise ValueError('Trial type {} not recognized.'.format(self.trial_type))

        ### Masking conditions for context switching ###
        # Determines number of steps at beginning of a new stim to not count
        # in presession activity for scaling task difficulty
        self.n_steps_tol = task_params.get('n_steps_tol', 20)

        ## NaN reward conditions (equivalent to zero RPE) ##
        # Determines number of steps at beginning of a new trial to return nans
        self.n_steps_trial_start_nans = task_params.get('n_steps_trial_start_nans', 0)
        # Determines number of steps at beginning of a new stim to return nans
        # (since for some task types, stims can change mid-trial, distinct from above)
        self.n_steps_stim_change_nans = task_params.get('n_steps_stim_change_nans', 0)

        # Used to smooth rewards to compensate for maximum speed a network activity could change
        self.reward_smoothing = task_params.get('reward_smoothing', True)
        self.net_leak_term = task_params.get('net_leak_term', 0.8)

        ### Tracked parameters ###
        self.trial_onsets = []
        self.trial_period_onsets = [[] for _ in range(self.n_trial_periods)]

    def activity_stim_trial(self, trial_override=False):
        """
        Generate a single activity_stim trial. For distinct stimuli, requires the
        activity meet distinct levels under distinct stimuli, but all levels
        are high relative to what was seen before.

        Each trial consists of two periods, which go through periods in the same
        order each time. Thus each trial is identical other than the noise.
        """

        if trial_override:
            raise NotImplementedError()

        # Seq idx is always relative to start of trial
        half_trial_idx = int(np.round(self.n_steps_trial / 2))

        stims = np.zeros((self.n_steps_trial, self.n_neurons))
        stims[:half_trial_idx, :] = stims[:half_trial_idx, :] + self.stim_input[0:1, :]
        stims[half_trial_idx:, :] = stims[half_trial_idx:, :] + self.stim_input[1:2, :]
        input = self.get_noise_input(self.n_steps_trial, stims=stims)

        # bci_targets = 0.5 * self.task_scale * np.ones((self.n_steps_trial,))
        # bci_targets[half_trial_idx:, :] = 0.0
        # bci_targets[half_trial_idx:] = 0.25 * self.task_scale

        trial_stims = np.zeros((self.n_steps_trial,), dtype=np.int32)
        trial_stims[half_trial_idx:] = 1
        trial_period_onsets = (0, half_trial_idx)

        return input, trial_stims, trial_period_onsets

    def variable_stim_trial(self, trial_override=False):
        """
        Generate a single variable_stim_trial. Each trial randomly determines
        one distinct n_stim. Each trial only has one period, but trials can
        be distinct (from stim_idx and noise).
        """

        if trial_override: # Same stim every single trial
            stim_idx = 0
        elif self.stim_sample_mode in ('random',):
            stim_idx = np.random.choice(self.n_stim)
        elif self.stim_sample_mode in ('pseudo-random',): # Choose randomly from existing list or generate new list
            if self.possible_stim_idxs == []: # Empty condition
                self.possible_stim_idxs = [idx for idx in range(self.n_stim)]

            list_idx = np.random.randint(len(self.possible_stim_idxs))
            stim_idx = self.possible_stim_idxs[list_idx]
            self.possible_stim_idxs.pop(list_idx) # Delete the stim idx

        stims = np.ones((self.n_steps_trial, 1,)) * self.stim_input[stim_idx:stim_idx+1, :]
        input = self.get_noise_input(self.n_steps_trial, stims=stims)

        trial_stims = stim_idx * np.ones((self.n_steps_trial,), dtype=np.int32)

        trial_period_onsets = (0,)

        return input, trial_stims, trial_period_onsets

    def separate_activity_into_stims(self, activities_stabilize, task_params,
                                     seq_start_idx=0, reject_criterion='nan_mask'):
        """
        Separates activity into activity observed during distinct stims
        Used for pre-session activity to set difficulty and/or BCI mask,
        i.e. the inutitive mapping. Also used for analysis of how responses
        of each stimulus change during training.

        Returns both activity and corresponding sequence indexes. Latter useful
        for post-train analysis to further filter/separate activity.
        """

        seq_len = activities_stabilize.shape[0]

        activities_stabilize_stims = [[] for _ in range(self.n_stim)]
        seq_idxs_stabilize_stims = [[] for _ in range(self.n_stim)]

        current_stim_idx = None
        n_steps_in_stim = None

        # Goes through each seq_idx time, rejects times immediately following a transition.
        # Note this can be optionally offset by passing a nonzero seq_start_idx,
        # but this offset needs to be removed when referencing the passed activity
        for seq_idx in range(seq_start_idx, seq_len + seq_start_idx):
            trial_stim_idx = self.trial_stims[seq_idx]

            if reject_criterion in ('n_steps_tol',): # Old way of filtering
                if current_stim_idx is None: # Initialization
                    current_stim_idx = np.copy(trial_stim_idx)
                    n_steps_in_stim = 1
                else:
                    if current_stim_idx == trial_stim_idx: # Same state
                        n_steps_in_stim += 1
                    else: # New state
                        current_stim_idx = np.copy(trial_stim_idx)
                        n_steps_in_stim = 1

                if n_steps_in_stim <= self.n_steps_tol:
                    continue
            elif reject_criterion in ('nan_mask',): # Just use nan mask to filter, similar to experiment
                if self.nan_mask[seq_idx]:
                    continue
            else:
                raise ValueError()

            activities_stabilize_stims[trial_stim_idx].append(activities_stabilize[seq_idx - seq_start_idx])
            seq_idxs_stabilize_stims[trial_stim_idx].append(seq_idx)

        for trial_stim_idx in range(self.n_stim): # Converts each stim into numpy arrays
            activities_stabilize_stims[trial_stim_idx] = np.array(
                activities_stabilize_stims[trial_stim_idx]
            )

        return activities_stabilize_stims, seq_idxs_stabilize_stims

    def set_sequence_reward_parameters(self):
        """
        Set BCI targets based on stimulus of each step. Requires that the
        targets, reward scales, and (optionally) widths already be defined for
        the different stimuli.

        Option to smoothly interpolate based on max speed of the network's
        activity change
        """

        n_steps = self.input.shape[0]
        # self.nan_mask is initialized in generate_input_sequence since its easier to set trial-dependent parameters there
        self.bci_targets = np.zeros((n_steps, self.n_bci_masks,))
        self.bci_reward_scales = np.zeros((n_steps, self.n_bci_masks,))
        if self.reward_fn == trapezoid_reward:
            self.bci_widths = np.zeros((n_steps, self.n_bci_masks,))
        else:
            self.bci_widths = None

        if self.reward_smoothing:
            leak_term = self.net_leak_term
        else:
            leak_term = 0.

        current_stim_idx = None
        n_steps_in_stim = None

        for seq_idx in range(n_steps):
            stim_idx = self.trial_stims[seq_idx]
            if seq_idx == 0:
                prev_threshold = 0.
                prev_reward_scale = 1e-3 # For stability
            else:
                prev_threshold = self.bci_targets[seq_idx-1]
                prev_reward_scale = self.bci_reward_scales[seq_idx-1]

            self.bci_targets[seq_idx] = (
                leak_term * prev_threshold + (1 - leak_term) * self.stim_targets[stim_idx]
            )
            self.bci_reward_scales[seq_idx] = (
                leak_term * prev_reward_scale + (1 - leak_term) *  self.stim_reward_scales[stim_idx]
            )
            if self.bci_widths is not None:
                if seq_idx == 0:
                    prev_width = 0.
                else:
                    prev_width = self.bci_widths[seq_idx-1]
                self.bci_widths[seq_idx] = (
                    leak_term * prev_width + (1 - leak_term) * self.stim_widths[stim_idx]
                )

            if np.sum(np.abs(self.bci_reward_scales[seq_idx])) < 1e-3: # Stability condition
                self.bci_reward_scales[seq_idx, :] = 1e-3

            if current_stim_idx is None: # Initialization
                current_stim_idx = np.copy(stim_idx)
                n_steps_in_stim = 1
            else:
                if current_stim_idx == stim_idx: # Same state
                    n_steps_in_stim += 1
                else: # New state
                    current_stim_idx = np.copy(stim_idx)
                    n_steps_in_stim = 1

            if n_steps_in_stim < self.n_steps_stim_change_nans:
                self.nan_mask[seq_idx] = 1

    def activity_stim_set_task_difficulty(self, activities_stabilize, task_params):
        """
        Sets the task difficulty in a task that requires meeting a certain BCI
        activity level under distinct stimuli. The difficulty level is set
        dynamically based on stabilization activity.

        First, goes through stabilization stim activities and separates into
        distinct stims. Then for each distinct stim, sets the difficulty
        scale independent of the other stims, based on the desired percentiles.
        Then optionally smoothes the desired targets (thresholds).
        """

        seq_len = activities_stabilize.shape[0]

        ### Separates pre-session activity into distinct stims (seq_idxs dont matter)
        activities_stabilize_stims, _ = self.separate_activity_into_stims(
            activities_stabilize, task_params
        )

        ### For each distinct stim, computes activity scale to determine task difficulty
        for stim_idx in range(self.n_trial_stims):

            print('--- Stim idx {}, activity shape:'.format(stim_idx), activities_stabilize_stims[stim_idx].shape, '---')

            print('Mean: {:.3f}'.format(np.mean(activities_stabilize_stims[stim_idx])))

            # Creates a local version of task_params to modify
            task_params_copy = copy.deepcopy(task_params)
            task_params_copy['dyn_threshold_perc'] = self.trial_stim_percs[stim_idx]

            threshold, width, reward_scale, print_str, extras = set_thresholds_and_scales(
                activities_stabilize_stims[stim_idx], task_params_copy, task=self,
            )

            self.stim_targets[stim_idx, 0] = threshold
            self.stim_widths[stim_idx, 0] = width
            self.stim_reward_scales[stim_idx, 0] = reward_scale

        self.max_activity = 2 * max(self.stim_targets) + max(self.stim_widths) # This may not be used

        # Now set BCI targets based on trial stim of each step
        self.set_sequence_reward_parameters()

    def stim_target_set_task_difficulty(self, activities_stabilize, task_params):
        """
        Sets the task difficulty in a task that requires meeting a certain BCI
        activity level under distinct stimuli. The difficulty level is fixed by
        the targets.
        """

        seq_len = activities_stabilize.shape[0]

        ### Separates pre-session activity into distinct stims (seq_idxs dont matter)
        activities_stabilize_stims, _ = self.separate_activity_into_stims(
            activities_stabilize, task_params
        )

        ### For each distinct stim, sets reward scale to determine task difficulty
        for stim_idx in range(self.n_stim):
            # For now just set scale to initialized value
            self.stim_reward_scales[stim_idx, :] = self.reward_scale * np.ones((self.n_bci_masks,))

        self.max_activity = 0.3

        # Now set BCI targets based on trial stim of each step
        self.set_sequence_reward_parameters()

    def get_noise_input(self, n_steps, stims=None):
        """ Creates a noisy input with optional stimulus. """

        if stims is None:
            stims = np.zeros((n_steps, self.n_neurons,))

        if self.stim_to_noise_ratio < 1.0: # Ensures maximum scale of stim and noise is roughly task_scale
            noise_mult = 1.0 # Reduces stim relative to noise
        else:
            noise_mult = 1.0 / self.stim_to_noise_ratio # Reduces noise relative to stim
        # print('Setting noise_mult to:', noise_mult)
        raw_input = noise_mult * np.random.normal(scale=self.task_scale, size=(n_steps, self.n_neurons,))

        if self.net_direct_input: # Passes noise through excluded input layer for comparable noise distribution
            # Make noise positive (can still have inhibitory effect via W_input components)
            if self.act_fn_type in ('Tanh'):
                raw_input = np.maximum(raw_input, np.zeros_like(raw_input))
            elif self.act_fn_type in ('ReTanh'):
                raw_input = np.maximum(np.tanh(raw_input), np.zeros_like(raw_input))
            else:
                raise NotImplementedError('act_fn_type {} not yet implemented'.format(self.act_fn_type))

            # Note this is NOT passed through non-linearity, since it functions as
            # an input to into the hidden layer which has a non-linearity
            raw_input = np.matmul(raw_input, self.net_W_inp.T)

            return raw_input + stims
        else: # Stim and noise added, passed through non-linearity, then passed through W_input externally
            if self.act_fn_type in ('Tanh'): # Make noise positive
                return np.maximum(raw_input + stims, np.zeros_like(raw_input))
            elif self.act_fn_type in ('ReTanh'):
                return np.maximum(np.tanh(raw_input), np.zeros_like(raw_input))
            else:
                raise NotImplementedError('act_fn_type {} not yet implemented'.format(self.act_fn_type))

    def generate_input_sequence(self, n_steps):
        """
        Generate the full train sequences and targets. Does this by generating
        one trial at a time using appropriate target function then appends
        everything together.

        Generates self.input and self.trial_stims and save internally. Note this
        does not genreate self.bci_targets yet because these are set after task
        difficulty is set, which often requires running pre-session period to
        observe intial activity.
        """

        n_trials = int(np.round(n_steps / self.n_steps_trial))

        # Hardcoded override to produce some test sequences for debugging
        # Currently makes it so each trial is the same, just to see if learning
        # is possible without stimulus variation
        db_mode = False
        db_start_idx = 6000

        if n_steps % self.n_steps_trial != 0:
            raise NotImplementedError('Last trial wil be clipped, change n_steps to multiple of n_steps_trial')

        self.input = np.zeros((n_steps, self.n_neurons))
        self.bci_targets = None # These are set after task difficulty is set
        self.trial_stims = np.zeros((n_steps,), dtype=np.int32)

        self.nan_mask = np.zeros((n_steps,), dtype=np.int32) # When reward should just be NaN

        current_seq_idx = 0

        for trial_idx in range(n_trials):

            if db_mode and current_seq_idx >= db_start_idx:
                trial_override = True
            else:
                trial_override = False

            trial_input, trial_stims, trial_period_onsets = self.get_trial_fn(trial_override)

            self.input[current_seq_idx:current_seq_idx + self.n_steps_trial, :] = trial_input
            # self.bci_targets[current_seq_idx:current_seq_idx + self.n_steps_trial] = trial_activity_targets
            self.trial_stims[current_seq_idx:current_seq_idx + self.n_steps_trial] = trial_stims

            self.trial_onsets.append(current_seq_idx)
            for period_idx in range(self.n_trial_periods):
                self.trial_period_onsets[period_idx].append(current_seq_idx + trial_period_onsets[period_idx])

            if self.n_steps_trial_start_nans > 0:
                self.nan_mask[current_seq_idx:current_seq_idx + self.n_steps_trial_start_nans] = 1

            current_seq_idx += self.n_steps_trial

        return self.input

    def get_reward(self, activity, task_params, seq_idx):
        """
        activity.shape = (n_neurons,)
        bci_masks.shape = (n_bci_masks, n_neurons,)

        """

        if self.nan_mask[seq_idx]:
            return np.nan

        if self.bci_widths is None: # Some reward functions don't use width
            bci_width = None
        else:
            bci_width = self.bci_widths[seq_idx]

        bci_activity = get_bci_activity(activity, task_params) # (n_bci_masks)

        total_reward = self.reward_fn(
            bci_activity, self.bci_targets[seq_idx], bci_width,
            self.bci_reward_scales[seq_idx], self.task_scale
        )

        return total_reward

def get_perturbation_center_out(
        task_params, perm_mode, task=None, max_perts=2e6, n_perts_at_once=1e5,
        pass_criterion={
                'max_angle': 50., # 44.4 # In degrees
                'min_angle': 12., # 19.7
                'max_speed': 3, # AU, target speeds have speed 1
                'min_speed': 0.3,
        },
        tolerance = 0.0, n_perm_dims=60,
        return_all_perts=False, compute_rewards=False, verbose=False
    ):
    """
    Retrieves a BCI masks perturbation that meets the required criterion.
    Evaluates perturbations based on task_params['mean_act_stims'], which is
    computed at the end of the stabilization period.

    perm_mode:
    - wm: shuffles top axes within manifold space (determined by PR)
    - wm_n_top: shuffles top axes, determined by n_perm_dims
    - om_raw: shuffles neuron indices
    - om_top_stds: only shuffles neurons with top standard deviations during
      stabilization period
    - om_top_stim_stds: only shuffles neurons with top standard deviations
      across the various stimulus types
    task: only needed if compute_rewards = True
    max_perts: maximum number of perturbations to consider
    n_perts_at_once: how many perts to evaluate at once
    n_perm_dims: number of dimensions to permute, used for wm_n_top,
      om_top_stds, and om_top_stim_stds
    """

    def compute_bci_masks_diff_center_out(bci_masks_perm, task_params, task):
        """
        Evaluates stimulus velocity and relative angles of many permutation
        simultaneously.

        INPUTS:
        bci_masks_perm.shape = (n_perm, n_bci_masks, n_perm_space)

        OUTPUTS:
        speeds.shape (n_perm, n_stims)
        """

        # if task_params['manifold_project'] is not None:
        #     raise NotImplementedError('Mean act stims are in manifold space.')

        # (n_perm, n_bci_masks, n_neurons) x (n_neurons, n_stims) =  (n_perm, n_bci_masks, n_stims)
        # Recall mean_act_stims already include the z-scoring and manifold projection if on
        if task_params['bci_choice'] in ('intuitive_manifold',) and perm_mode in ('wm', 'wm_n_top',):
            # Well-defined manifold, mean_act_stims need to be projected to manifold space first
            mean_bci_act_stims = np.matmul(bci_masks_perm, np.matmul(
                task_params['manifold_project'], task_params['mean_act_stims'].T
            ))
        else:
            mean_bci_act_stims = np.matmul(bci_masks_perm, task_params['mean_act_stims'].T)

        # (n_perm, n_bci_masks, n_stims) -> (n_perm, n_stims)
        speeds = np.linalg.norm(mean_bci_act_stims, axis=1)

        # if task_params['bci_choice'] in ('intuitive_manifold',) and perm_mode in ('wm', 'wm_n_top',):
        #     # Well-defined manifold, mean_act_stims fit to manifold space not full space
        #     intuitive_velocities = np.matmul(task_params['bci_masks_manifold'], task_params['mean_act_stims'].T)
        # else:
        # (n_bci_masks, n_neurons) x (n_neurons, n_stims) =  (n_bci_masks, n_stims) -> (n_bci_masks, n_stims)
        intuitive_velocities = np.matmul(task_params['bci_masks'], task_params['mean_act_stims'].T)

        dots = np.einsum('piI, iI -> pI', mean_bci_act_stims, intuitive_velocities) # (n_perm, n_stims)
        cosine_angles = dots / (
            np.linalg.norm(intuitive_velocities, axis=0)[np.newaxis, :] * # (1, n_stims)
            np.linalg.norm(mean_bci_act_stims, axis=1) # (n_perm, n_stims)
        )

        if compute_rewards: # This is slow because in for loop, at some point might want to matrix-ify it
            rewards = np.zeros_like(speeds) # (n_perm, n_stims)
            dists = np.zeros_like(speeds)
            for perm_idx in range(rewards.shape[0]):
                for stim_idx in range(rewards.shape[-1]):
                    target = task.stim_targets[stim_idx]
                    rewards[perm_idx, stim_idx] = task.reward_fn(
                        mean_bci_act_stims[perm_idx, :, stim_idx], target, None,
                        task.stim_reward_scales[stim_idx], task.task_scale
                    )
                    dists[perm_idx, stim_idx] = center_out_distance(
                        mean_bci_act_stims[perm_idx, :, stim_idx], target, None,
                        task.stim_reward_scales[stim_idx], task.task_scale
                    )
                    # if perm_idx == 119 and stim_idx == 0:
                    #     print('Target:', target)
                    #     print('Mean act stim', mean_bci_act_stims[perm_idx, :, stim_idx])
                    #     print('Reward scale:', task.stim_reward_scales[stim_idx])
                    #     print('task scale:', task.task_scale)
        else:
            rewards = None
            dists = None

        return speeds, cosine_angles, rewards, dists

    bci_masks = task_params['bci_masks']
    n_neurons = bci_masks.shape[-1]
    mean_act_stims = task_params['mean_act_stims']

    # Used for some types of perturbations to set size of perturbation space
    pr_val = participation_ratio_vector(task_params['manifold_axes_var_exps'])
    pr_val = int(np.ceil(pr_val))

    bci_masks_pert_return = None
    total_perts_eval = 0
    current_min_dist = 1e5 # just a large number
    # Used to keep track of best perturbation so far
    best_pert_so_far = None
    best_speeds_so_far = None
    best_cas_so_far = None

    start_time = time.time()

    while total_perts_eval < max_perts: # Runs n_perts_at_once perturbation evals

        start_time_local = time.time()

        print_str = 'Perturbation type {} search'.format(perm_mode)

        ### Permute the (n_bci_masks, n_perm_space) elements, the same for each mask
        # n_perm_space is determined by the type of shuffle
        if perm_mode in ('om_raw',): # Shuffle over neuron indexes
            bci_masks_perm_space = bci_masks # Just the raw BCI masks (shuffle neurons)
        elif perm_mode in ('om_top_stds', 'om_top_stim_stds',): # Projects to subspace containing neurons with highest standard deviation
            # Note this projection just determines which neurons to include,
            # so only has has 1.0 elements in projection matrix
            if perm_mode in ('om_top_stds',):
                stds_stort = np.argsort(task_params['activity_stds'])[::-1] # Largest to smallest
            elif perm_mode in ('om_top_stim_stds',):
                stds_stort = np.argsort(np.std(task_params['mean_act_stims'], axis=0))[::-1] # Largest to smallest
            print_str += f' (n_permuted neurons: {n_perm_dims})'
            top_stds_proj = np.zeros((n_perm_dims, n_neurons,))
            for std_idx in range(n_perm_dims):
                top_stds_proj[std_idx, stds_stort[std_idx]] = 1.0
            top_stds_mask = np.sum(top_stds_proj, axis=0)
            bci_masks_perm_space = np.matmul(bci_masks, top_stds_proj.T)
        elif perm_mode in ('wm', 'wm_n_top',): # Project BCI masks to top few PC space
            if task_params['bci_choice'] in ('intuitive_manifold',): # Manifold projection well defined
                if perm_mode in ('wm_n_top',):
                    raise NotImplementedError('Manifold has been defined earlier, cant override.')
                print_str += ' (n_permuted set by manifold projection: {})'.format(task_params['manifold_project'].shape[0])
                bci_masks_perm_space = task_params['bci_masks_manifold']
            else: # No well-defined manifold
                if perm_mode in ('wm',):
                    n_dims_manifold = pr_val
                elif perm_mode in ('wm_n_top',):
                    n_dims_manifold = n_perm_dims
                print_str += f' (n_permuted dims manifold: {n_dims_manifold})'
                bci_masks_perm_space = np.matmul(bci_masks, task_params['manifold_axes'][:n_dims_manifold].T)
        else:
            raise NotImplementedError('Perm mode {} not recognized.'.format(perm_mode))
        print(print_str)

        shuffle_idxs = np.arange(bci_masks_perm_space.shape[-1])

        bci_masks_perm = np.repeat(bci_masks_perm_space[np.newaxis, :, :], n_perts_at_once, axis=0)
        bci_masks_perm_idxs = np.repeat(shuffle_idxs[np.newaxis, :], n_perts_at_once, axis=0)

        # rng = np.random.default_rng(seed = task_params['seed'] + total_perts_eval) # Change seed each runthrough
        rng = np.random.default_rng(seed = int(task_params['seed'] + total_perts_eval)) # Change seed each runthrough
        bci_masks_perm_idxs = rng.permuted(bci_masks_perm_idxs, axis=-1)

        for bci_mask_idx in range(bci_masks.shape[0]):
            bci_masks_perm[:, bci_mask_idx, :] = np.take_along_axis(
                bci_masks_perm[:, bci_mask_idx, :], bci_masks_perm_idxs, axis=-1
            )

        if perm_mode in ('om_raw',): # No need to project back
            bci_masks_pert = bci_masks_perm
        elif perm_mode in ('om_top_stds', 'om_top_stim_stds',): # Projects out of subspace comtaining neurons with highest standard deviation, copies rest
            bci_masks_pert_top_stds = np.matmul(bci_masks_perm, top_stds_proj) # BCI from just the shuffled space
            bci_masks_repeat = np.repeat(bci_masks[np.newaxis, :, :], n_perts_at_once, axis=0) # All other BCI

            bci_masks_pert = bci_masks_pert_top_stds + (1 - top_stds_mask) * bci_masks_repeat
        elif perm_mode in ('wm', 'wm_n_top',):  # Project back into full space
            if task_params['bci_choice'] in ('intuitive_manifold',): # Well-defined manifold
                bci_masks_pert = bci_masks_perm # Stimulus fits done directly in the manifold space
            else: # No well-defined manifold
                bci_masks_pert = np.matmul(bci_masks_perm, task_params['manifold_axes'][:n_dims_manifold])

        speeds, cosine_angles, rewards, dists = compute_bci_masks_diff_center_out(
            bci_masks_pert, task_params, task
        )

        # # Now evaluate the four conditions across all the stimuli
        # min_speed_pass = np.all(np.where(speeds > pass_criterion['min_speed'], True, False), axis=-1)
        # max_speed_pass = np.all(np.where(speeds < pass_criterion['max_speed'], True, False), axis=-1)

        # max_cosine_angle = np.cos(pass_criterion['min_angle'] * (np.pi / 180)) # Smaller angles -> larger cosine angles
        # min_cosine_angle = np.cos(pass_criterion['max_angle'] * (np.pi / 180)) # Larger angles -> smaller cosine angles

        # min_cosine_angle_pass = np.all(np.where(cosine_angles > min_cosine_angle, True, False), axis=-1)
        # max_cosine_angle_pass = np.all(np.where(cosine_angles < max_cosine_angle, True, False), axis=-1)

        # # Combine all conditions into one giant array
        # full_pass = np.concatenate((
        #     min_speed_pass[np.newaxis, :],
        #     max_speed_pass[np.newaxis, :],
        #     min_cosine_angle_pass[np.newaxis, :],
        #     max_cosine_angle_pass[np.newaxis, :],
        # ), axis=0)

        # n_passes = np.sum(np.all(full_pass, axis=0))

        # Now evaluate distance to all four conditions across all the stimuli

        to_min_speed = pass_criterion['min_speed'] - speeds # Negative passes
        min_speed_dist = np.where(to_min_speed > 0, to_min_speed, 0.)
        to_max_speed = speeds - pass_criterion['max_speed'] # Negative passes
        max_speed_dist = np.where(to_max_speed > 0, to_max_speed, 0.)
        total_speed_dist = np.sum(min_speed_dist + max_speed_dist, axis=-1) / (pass_criterion['max_speed'] - pass_criterion['min_speed'])

        max_cosine_angle = np.cos(pass_criterion['min_angle'] * (np.pi / 180)) # Smaller angles -> larger cosine angles
        min_cosine_angle = np.cos(pass_criterion['max_angle'] * (np.pi / 180)) # Larger angles -> smaller cosine angles

        to_min_ca = min_cosine_angle - cosine_angles # Negative passes
        min_ca_dist = np.where(to_min_ca > 0, to_min_ca, 0.)
        to_max_ca = cosine_angles - max_cosine_angle # Negative passes
        max_ca_dist = np.where(to_max_ca > 0, to_max_ca, 0.)
        total_ca_dist = np.sum(min_ca_dist + max_ca_dist, axis=-1) / (max_cosine_angle - min_cosine_angle)

        # Combine both conditions into a single number
        full_pass = total_speed_dist + total_ca_dist

        # n_passes = np.sum(full_pass == 0.)
        n_passes = np.sum(full_pass <= tolerance)

        if n_passes > 0:
            if verbose:
                print(' {} valid perturbations found! (time {:.2f})'.format(
                    n_passes, time.time() - start_time_local
                ))
            # pert_idx = np.where(np.all(full_pass, axis=0))[0][0] # Just get the first valid one, everything is random anyway
            pert_idx = np.where(full_pass <= tolerance)[0][0] # Just get the first valid one, everything is random anyway
            print('Pert idx: {}'.format(pert_idx))
            print(' speeds:', ['{:.2f}'.format(speed) for speed in speeds[pert_idx]])
            print(' angles:', ['{:.1f}'.format(angle) for angle in 180 / np.pi * np.arccos(cosine_angles[pert_idx])])
            if task_params['bci_choice'] in ('intuitive_manifold',) and perm_mode in ('wm', 'wm_n_top',):
                # Project perturbed mask to full space
                bci_masks_pert_return = np.matmul(bci_masks_pert[pert_idx], task_params['manifold_project'])
            else:
                bci_masks_pert_return = bci_masks_pert[pert_idx]
            break
        else: # While it fails to find a perturbation, keep track of the one that is closest to passing
            local_min_dist = min(full_pass)
            if local_min_dist < current_min_dist:
                current_min_dist = np.copy(local_min_dist)
                pert_idx = np.where(full_pass == current_min_dist)[0][0]
                if task_params['bci_choice'] in ('intuitive_manifold',) and perm_mode in ('wm', 'wm_n_top',):
                    # Project perturbed mask to full space
                    best_pert_so_far = np.matmul(bci_masks_pert[pert_idx], task_params['manifold_project'])
                else:
                    best_pert_so_far = bci_masks_pert[pert_idx]
                best_speeds_so_far = speeds[pert_idx]
                best_cas_so_far = cosine_angles[pert_idx]

            total_perts_eval += n_perts_at_once
            if verbose:
                print(' No passes found, min dist: {:.3f}, running again (time {:.2f}).'.format(
                    min(full_pass), time.time() - start_time_local
                ))

    if bci_masks_pert_return is not None:
        if verbose:
            print('Valid perturbation found, total time: {:.2f} seconds.'.format(
                time.time() - start_time
            ))
    else:
        print('No valid perturbation found in maximum number of iterations!')
        print('Best found over search:')
        print(' speeds:', ['{:.2f}'.format(speed) for speed in best_speeds_so_far])
        print(' angles:', ['{:.1f}'.format(angle) for angle in 180 / np.pi * np.arccos(best_cas_so_far)])

    extras = {
        'speeds': speeds, # Note these will only be for the most recent set of perturbations
        'cosine_angles': cosine_angles,
        'rewards': rewards,
        'dists': dists, # Special center-out distance to target
        'best_pert_so_far': best_pert_so_far,
        'best_speeds_so_far': best_speeds_so_far,
        'best_cas_so_far': best_cas_so_far,
        'min_speed': pass_criterion['min_speed'],
        'max_speed': pass_criterion['max_speed'],
        'min_ca': min_cosine_angle,
        'max_ca': max_cosine_angle,
    }

    if return_all_perts:
        if task_params['bci_choice'] in ('intuitive_manifold',) and perm_mode in ('wm', 'wm_n_top',):
            # Project perturbed mask to full space
            extras['bci_masks_pert'] = np.matmul(bci_masks_pert, task_params['manifold_project'])
        else:
            extras['bci_masks_pert'] = bci_masks_pert
        extras['pass_criterion'] = full_pass

    return bci_masks_pert_return, extras

def get_stimulus_special(task_params, stim_idx, n_stim, stim_type='uniform'):
    """
    Generates stimulus inputs for tasks that have many stimuli.

    Stimulus inputs are various random matrices.
    """
    if 'n_inp' in task_params: # Defaults to number of inputs
        n_neurons = task_params['n_inp']
    elif 'n_neurons' in task_params:
        n_neurons = task_params['n_neurons']
    else:
        raise NotImplementedError('Unknown stimulus size')

    assert n_neurons % n_stim == 0 # Needs to be nicely divisible
    n_per_stim = int(np.round(n_neurons / n_stim)) # Number of neurons per stimulus

    stim = np.zeros((1, n_neurons,))

    if stim_type in ('uniform',):
        stim[:, stim_idx*n_per_stim:(stim_idx+1)*n_per_stim] = np.random.uniform(
            -1 * task_params['task_scale'], 1 * task_params['task_scale'], size=(1, n_per_stim,)
        )
        return stim
    elif stim_type in ('sparse_binary',):
        p = 0.2
        stim = task_params['task_scale'] * (np.random.rand(1, n_neurons) < p).astype(int)
        return stim
    elif stim_type in ('normal',):
        raise NotImplementedError()

seed = 2002

task_params = {}

net_params = {'act_fn_type': 'ReTanh',}

from sklearn.decomposition import FactorAnalysis

n_components = 10

session_idx = 0

def set_pert_defaults(task_params, train_params, net_params):
    """
    Default parameters for the perturbation task
    """

    local_task_params = {

        ### Task choice
        'task_type': 'trial_structure_task', # (should already be set to this, just to be explicit)
        ## Trial_structure task parameters
            'trial_type': 'center_out_2d', # high_stim, high_low_stim, center_out_1d, center_out_1d_1, center_out_2d, 'center_out_2d_8'
            'stim_to_noise_ratio': 1.0/0.3, # 0.25
            'n_steps_trial': 200, # 200 is about 10 seconds; 500,
            'n_steps_period_change_nans': 40, # 100
            'n_steps_trial_start_nans': 40, #  grace

        'z_score_activities': True, # z-scores activities based on pre-session, does this before BCI map and for manifold calculation

        ### BCI mask choice (threshold/width used to set spout movement threshold/max speed)
        'bci_choice': 'intuitive_manifold',
        ## Parameters for various bci_choice settings
            'n_bci_masks': 2,
            'manifold_mode': 'pc', # pc, fa; whether to use PCA or FA to determine the low-dimensional manifold

        ### Session length parameters
        'n_steps': 50000, #5200,
        'n_steps_stabilize': 5000, #1000, #500,

        ### Reward-relevant parameters
        'normalize_reward': True, # Makes value of rewards depend on median and threshold during presession
        'n_reward_delay': 0, # Number of time steps to delay the current reward
        'separate_stim_reward_baselines': True, # Different reward baseline for each distinct stimulus

        'task_scale': 0.1, # Determines the scale of the noise input and also the constant input when used

        'n_sessions': 1,
        'use_max_activity': False, # Clip activity at maximum, helps stabilize training with recurrent adjustment

        'solution_type': None, # Not yet implemented
        # 'perturbations': None,
        'perturbations': ( # To test BCI mask perturbations
            (8000, 'center_out', {'perm_mode': 'om_raw', # wm, wm_n_top, om_raw, om_top_stds, om_top_stim_stds
                                  'n_perm_dims': 100,}),
            (42000, 'center_out', {'perm_mode': 'intuitive',
                                  'n_perm_dims': 60,}),
        )


# task_params['z_score_activities'] = False
# task_params['dyn_threshold_type'] = 'percentile_bci_activity'
# task_params['dyn_threshold_perc'] = 1.0
# task_params['normalize_reward'] = False
# task_params['net_leak_term'] = 0.8
# task_params['direct_input'] = True
# task_params['stim_to_noise_ratio'] = 10.0 #1.0/0.3
    }

    local_train_params = {
        'eta': 2e0,
        'n_window_reward': 10 * 20, #60 * 20, # Size of the average window for reward baseline
        'n_window_baseline': 1 * 20, # Size of the average window for activity
        'n_steps_per_loss': 5, # How often to total RPE and adjust weights
        'eligibility_acc_type': 'running_average', # acc_and_wipe, running_average
        'n_window_elig': 40, # Only used for 'running_average' option, how long to keep eligibility

        'rpe_clip': 0.05,
    }

    local_net_params = {
        'direct_input': True, # More directly control over stimulus signal where it doesn't need to pass through random matrix anymore
        'weight_mask_modes': None, # None or tuple with: cn_freeze, cn_only, cn_no_output
    }

    local_task_params['direct_input'] = local_net_params['direct_input']

    for local_key in local_task_params.keys():
        task_params[local_key] = local_task_params[local_key]
    for local_key in local_train_params.keys():
        train_params[local_key] = local_train_params[local_key]
    for local_key in local_net_params.keys():
        net_params[local_key] = local_net_params[local_key]

    return task_params, train_params, net_params

import time

run = True

minimal = False

train = True

save_run = False

save_sessions = False

save_path = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/test_plots_'

seed = 2013

def default_toy_params(seed=0, verbose=False):

    task_params = {
        ### Task choice
        # - simple_bci: just change bci activity, no trial structure
        # - our_bci: Special task setup to reproduce our BCI task
        # - trial_structure_task: various trial-structure based tasks (including WMP/OMP setup)
        'task_type': 'our_bci',

        'z_score_activities': False, # z-scores activities based on pre-session, does this before BCI map and for manifold calculation

        ### BCI mask choice (parameters that need to be specified)
        # - random: random CN choices (n_cns, n_bci_masks)
        # - activity_percentile: choice closest to desired activity percentile (n_cns, n_bci_masks, activity_percentile)
        # - activity_high: random choice within percentile (n_cns, n_bci_masks, activity_percentile)
        # - manifold: BCI mask is one of the manifold dims (manifold_mode, manifold_idx, n_bci_masks)
        # - intuitive: BCI masks chosen to best solve task
        # - intuitive_maifold: Same as above, but based on low-dim manifold activity
        'bci_choice': 'activity_high',
        ## Parameters for various bci_choice settings
            'n_bci_masks': 1,
            'activity_percentile': 0.7, # For activity_percentile and activity_high, determines CN percentile.
            'n_cns': 1, # Used for settings where number of CNs for a single mask is variable
            'manifold_mode': 'pc', # pc, fa; whether to use PCA or FA to determine the low-dimensional manifold
            'manifold_idx': 25, # For manfiold settings, determines what idx is the mask (lower = higher var exp)
            'constant_stim_input': False, # Add constant stim to this task (otherwise just zeros, needed for odd activations)
        ### What to set the threshold based on: percentile_bci_activity, bci_activity_mean_and_std, fixed_neural_distance
        'dyn_threshold_type': 'bci_activity_mean_and_std',
        ## Parameters for various threshold choices
            'dyn_threshold_perc': 0.9, #0.9, # Sets threshold to be at certain percentile of activity (only used for perc_ above)
            'dyn_width_perc': None,
            'n_stds': 2.0, # for 'bci_activity_mean_and_std' option
            'threshold_distance': 0.05, # for 'fixed_neural_distance' option

        ### Session length parameters
        't_step':  50, # ms, only needed to define fluorescence in some tasks
        'n_steps': 6000, #10000, # 2500,
        'n_steps_stabilize': 2000, #1000, #500,
        'n_steps_evaluate': 0, # Stops training to observe solution

        ### Reward-relevant parameters
        'reward_structure': 'abs', # trapezoid, quadratic, abs
        'normalize_reward': True, # False, # Makes value of rewards depend on median and threshold during presession
        'n_reward_delay': 5, # Number of time steps to delay the current reward

        'task_scale': 0.1, # Determines the scale of the noise input and also the constant input when used
        'start_avg_reward': 'dynamic', # dynamic
        
        'n_sessions': 1,
        'use_max_activity': False, # Clip activity at maximum, helps stabilize training with recurrent adjustment

        'add_fl': False,

        'seed': seed,

        # Solution criteria
        'solution_type': 'avg_activity', # None, avg_activity
        'sol_n_steps': 20,
        'terminate_on_sol': False,

        ### Parameters used only for special tests ###
        
        'constant_rpe_val': None, # 0.025, # If anything other than None, RPE is held constant

        # Used to introduce additional correlation into activity
        'correlation_type': None, # None, activity, weights
        'on_idx': 20,
        'on_weight': 1.0,

        # Used to test how non-BCI components affect training
        'non_bci_deviation_zero': False,

        ## Perturbations to training
        # Step_idx to apply perturbation, preturbation type, perturbation params
        # 'perturbations': ( # To test BCI mask perturbations
        #     (2000, 'bci_masks_new_pc', {'pc_idx': 80}), # 10, 80
        #     (3000, 'bci_masks_new_pc', {'pc_idx': 2}),
        # )
        # 'perturbations': ( # To test BCI mask perturbations
        #     (6000, 'center_out', {'perm_mode': 'om_top_stim_stds', # wm, wm_n_top, om_raw, om_top_stds, om_top_stim_stds
        #                           'n_perm_dims': 100,}),
        #     (26000, 'center_out', {'perm_mode': 'intuitive',
        #                           'n_perm_dims': 60,}),
        # )

        ## Special training criterion
        # None, zero_cn_input, zero_cn_input_and_sparsify_rec, zero_cn_input_and_sparsify_rec_copy
        # zero_cn_input_and_sparsify_rec_copy_hold
        'special_test_type': None,
    }

    # Some defaults
    default_fills = {'bci_masks': None, 'activity_subtract': None, 'activity_stds': None,
                    'threshold': None, 'width': None, 'perturbations': None,
                    'dyn_width_perc': None, 'max_activity': None, 'photostim': None,
                    'special_test_type': None,
                    'start_avg_reward': None, 'z_score_activities': False,
                    'use_max_activity': False, 'add_fl': False, 'terminate_on_sol': False,
                    'separate_stim_reward_baselines': False, 'run_test_idxs': None,}
    for default_fill in default_fills.keys():
        if default_fill not in task_params: task_params[default_fill] = default_fills[default_fill]

    assert task_params['n_reward_delay'] >= 0. # Negative values don't make sense here

    train_params = {
        # 5e-1 works well for 3-factor toy task with abs reward
        # 50.0 works well for 3-factor toy task with quadratic
        'eta': 1e0, #5e0,
        'n_window_reward': 200, # Size of the average window for reward baseline
        'n_window_baseline': 50, #20, # Size of the average window for activity
        'n_steps_per_loss': 5, # How often to total RPE and adjust weights
        'eligibility_acc_type': 'running_average', # acc_and_wipe, running_average
        'n_window_elig': 5, # Only used for 'running_average' option, how long to keep eligibility

        'adjust_type': '3factor',
        # 3factor: our 3-factor, post dev * phi^prime * pre
        # backprop: same as our 3-factor, but full dh_dW including leak
        # backprop_two_step: same as our 3-factor, just one more step back for dh_dW
        # 3factor_leak: our 3-factor, but correctly calculates leak term contribution
        # 3factor_nophiprime: our 3-factor without phi^prime term (EH learning with postactivation)
        # 3factor_nodev: our 3-factor without post dev, just post * phi^prime * pre
        # 3factor_vanilla_hebbian: 3-factor with just the usual Hebbian, essentially equivalent to having both no phi prime and no deviation
        # 3factor_eh: True EH learning, use preactivations for postsynaptic activity
        # 3factor_miconi: Same rule used in Miconi
        # 3factor_miconi_insp: Same rule used in Miconi, but actually uses postsynaptic deviations
        # 3factor_node_pert: Node perturbation to postactivations (applies perturbations but still uses same RPE)
        # 3factor_node_pert_pre: Node perturbation to preactivations
        # 3factor_predevs: Deviation only in presyn
        # 3factor_two_devs: Deviation in post and presyn deviations
        # 3factor_sl: 3-factor supervised approximation, with knowledge of BCI mask
        # 3factor_sl_uniform: 3-factor supervised approximation, with no knowledge of BCI mask
        # backprop_sl: full dh_dW with knowlege of BCI mask
        
        # scalar or vector: scale of activity perturbations for 3factor_nope_pert training (if None, uses task_scale)
        'act_perturbation_scale': None, #0.05, 
        
        'rpe_clip': 0.05, # None, 0.1; maximum RPE (per step) magnitude, stabilizes training a bit

        'print_every': 100,
        'weight_reg': 'L2',
        'reg_lambda': 0., #1e-5, #1e-4,
    }

    net_params = {
        'n_inp': 200, # input layer
        'n_neurons': 200, # hidden layer

        'W_inp_type': 'gaussian', # gaussian, sparse_gaussian, log_normal
        'W_rec_type': 'gaussian',
        'W_inp_adjust': False,
        'W_rec_adjust': True,
        # 'W_rec_weight_norm': 0.0,

        'act_fn_type': 'ReTanh', # ReTanh, Tanh, ReLU, linear

        'weight_mask_modes': None, # None or tuple with: cn_freeze, cn_only, cn_no_output
        'W_inp_adjust_mask': None,
        'W_rec_adjust_mask': None,

        'direct_input': False, # For more control over input into hidden activity

        'leak_term': 0.8, # alpha, 0.0 means no leak
        'leak_type': 'membrane', # activity, membrane
    }

    task_params['n_stim'] = None
    task_params['net_leak_term'] = net_params['leak_term']
    task_params['direct_input'] = net_params['direct_input']

    if 'n_inp' not in net_params:
        net_params['n_inp'] = net_params['n_neurons']
    # Tasks need to know about these to create correct input sizes
    task_params['n_inp'] = net_params['n_inp']
    task_params['n_neurons'] = net_params['n_neurons']

    if task_params['task_type'] in ('our_bci',):
        if verbose: print('Overriding task_params with our BCI defaults.')
        task_params, train_params, net_params = set_to_our_bci_defaults(task_params, train_params, net_params)
        task_params['track_raw_states'] = True # Track some extra things for nice plots (like spout position)
    if task_params['task_type'] in ('trial_structure_task',):
        if verbose: print('Overriding task_params with our pertrubation defaults.')
        task_params, train_params, net_params = set_pert_defaults(task_params, train_params, net_params)

    if train_params['adjust_type'] in ('3factor_sl', '3factor_sl_uniform', 'backprop_sl',):
        print('INCREASING RPE CLIP FOR SL-LIKE LEARNING')
        train_params['rpe_clip'] = 1.0
    
    if task_params['add_fl'] and ('fl_kernel' not in task_params): # Initialization of fl kernel
        # print('Initializing fluorescence!')
        task_params = fluorescence_init(task_params)
        if task_params['task_type'] not in ('our_bci',):
            train_params['eta'] = 0.4 * train_params['eta'] # Compensate for lower standard deviation
            print(' FL: lowering learning rate by factor of 2/5: {:.1e} -> {:.1e}'.format(
                train_params['eta'] / 0.4, train_params['eta']
            ))

    return (task_params, train_params, net_params)

def train_task(params, output_vars=[], net=None, task=None, prev_seq_vars=None,
               train=True, n_steps=None, save_sessions=False, save_path='',
               verbose=True):
    """
    Run a network through a training sequence. This is setup to be run under
    several conditions including those where the network/task are generated
    internally but also ones where a network/task generated elsewhere is run
    through a sequence.

    INPUTS:
    - params = task_params, train_params, net_params
    - output_vars: list controlling what internal parameters are returned from
        training (some are returned regardless of contents of list)
    - net: if None, initializes new networks, otherwise uses passed network
    - task: the task environment, for some tasks there is no environment. For
        tasks with an environment, generates internally if None, otherwise uses
        passed task
    - prev_seq_vars: a bunch of internally tracked parameters over training. If None,
        generates internally, otherwise uses passed value
    - n_steps: Number of steps to train, if None just uses task_params value. Used for
        training test sessions that may be distinct from full task_params value.
        
    - save_sessions: save individual sessions within a run
    - save_path: only used here to save individual sessions, otherwise saved externally

    OUTPUTS:
    - train_outputs_all: giant dictionary with default ouputs along with "output_vars" input
    
    """

    task_params, train_params, net_params = params

    task_scale = task_params['task_scale'] # Determines the scale of the noise input and also the constant input when used

    # Set seed/state here, determines input noise (see below for additional seed sets)
    if net is not None:
        if net.rng_state is not None:
            np.random.set_state(net.rng_state)
    else:
        np.random.seed(task_params['seed'])

    train_outputs_all = []
    task_ps = None
    if n_steps is None: n_steps = task_params['n_steps']

    # Repeats everything for each session, just doesn't reset network and carries over certain baselines
    for session_idx in range(task_params['n_sessions']):

        # Dict will be populated throughout each individual session, then all
        # are collected in a list for output
        train_outputs = {}

        prev_activity = None # Overriden below if specifying the network's state (only do this if generating more activity)
        prev_activity_pre_act = None

        if net is not None: # Using a passed network
            if verbose: print('--- Session idx {}: Using passed network ---'.format(session_idx))
            if hasattr(net, 'prev_activity') and hasattr(net, 'prev_activity_pre_act'): # Initialize these if we want to set the state too
                prev_activity = net.prev_activity
                prev_activity_pre_act = net.prev_activity_pre_act
            if net_params['weight_mask_modes'] is not None:
                raise NotImplementedError()
        elif session_idx == 0: # Things set only in first session, unless an existing network is passed
            if verbose: print('--- Session idx {}: Initializing network ---'.format(session_idx))
            net = networks.SimpleRNN(net_params, task_params, train_params)
        else:
            raise ValueError('Net needs to be passed if this isnt the first session')

        # Initialize BCI-mask quantities if not yet determined
        if task_params['bci_masks'] is None:
            task_params['bci_var_exp'] = None
            task_params['cn_idxs'] = None
            
        task_params['W_inp'] = np.copy(net.W_inp) if net_params['direct_input'] else None  # task needs to know aobut W_inp in case it is being circumnavigated

        ### Initialize task and quantities to track throughout sequence run (whether training or not) ###
        # Also handles resetting for a new session, start step_idx can be nonzero if we are splitting training at a certain point
        task, start_step_idx, (
            net_input, output, output_pre_act, reward, avg_reward, avg_activity,
            avg_activity_pre_act, mean_stabilize_actvities, output_fl,
            bci_activity, avg_bci_activity, avg_reward_stims, 
            act_perturbations, preact_perturbations,
        ) = initialize_sequence_variables(task_params, net=net, task=task, prev_seq_vars=prev_seq_vars, n_steps=n_steps, verbose=verbose)
        
        # Quantities to determine if solution has been found, if used
        sol_n_steps = 0
        sol_step = None # Step solution was achieved

        ### Initialize training-relevant quantities, if relevant ###
        (loss_steps, total_rpes, W_inp_vals, W_rec_vals, W_inp_elg_vals,
         W_rec_elg_vals, act_fn_p_pre_act_vals) = initialize_sequence_variables_training(task_params, net, output_vars, train=train, verbose=verbose)

        task_params, train_params, net_params, net = set_special_test(
            task_params, train_params, net_params, net
        )

        # Initialize and run initial photostim (only do this when doing a single session)
        if task_params['n_sessions'] == 1 and task_params['photostim'] in ('every_session',):
            _, train_outputs_ps_init, _, task_ps = run_photostim(
                (task_params, train_params, net_params), output_vars=output_vars_ps,
                task=task, task_ps=task_ps, net=net, verbose=verbose
            )
            train_outputs['train_outputs_ps_init'] = train_outputs_ps_init

        # Now iterate through sequence times
        for step_idx in range(start_step_idx, n_steps):

            if step_idx == 0: # Initialization values for each session (these are not carried between sessions)
                if prev_activity is None and prev_activity_pre_act is None:
                    prev_activity = np.zeros((task_params['n_neurons'],))
                    prev_activity_pre_act = np.zeros((task_params['n_neurons'],))
                prev_reward = 0.
                prev_avg_activity = np.zeros((task_params['n_neurons'],))
                prev_avg_activity_pre_act = np.zeros((task_params['n_neurons'],))
                prev_avg_bci_activity = np.zeros((task_params['n_neurons'],))
                if avg_reward_stims is not None:
                    prev_reward_stims = np.zeros((task.n_stim,))
            else:
                prev_activity = output[step_idx-1]
                prev_activity_pre_act = output_pre_act[step_idx-1]
                prev_reward = avg_reward[step_idx-1] # Delay not needed here
                prev_avg_activity = avg_activity[step_idx-1]
                prev_avg_activity_pre_act = avg_activity_pre_act[step_idx-1]
                prev_avg_bci_activity = avg_bci_activity[step_idx-1]
                if avg_reward_stims is not None:
                    prev_reward_stims = avg_reward_stims[step_idx-1]

            if task_params['perturbations'] is not None:
                task_params, terminate_training = apply_bci_mask_perturbations(step_idx, task_params, verbose=verbose)
                if terminate_training: # If a valid perturbation is not found, sometimes breaks from training
                    break                

            # Network forward pass
            current_input = net_input[step_idx]
            current_act_perturbation = None if act_perturbations is None else act_perturbations[step_idx]
            current_preact_perturbation = None if preact_perturbations is None else preact_perturbations[step_idx]
            output[step_idx], output_pre_act[step_idx] = net.forward(
                current_input, prev_activity, prev_activity_pre_act, net_params, 
                perturbation=current_act_perturbation, perturbation_preact=current_preact_perturbation,
            )

            # Some ways that activity could be adjusted (special tests or clips)
            if 'max_activity' in task_params: # Clip total activity for numerical stability
                if task_params['max_activity'] is not None:
                    output[step_idx] = np.clip(output[step_idx], None, task_params['max_activity'])
                # if train_params['adjust_type'] in ('3factor_miconi',):
                #     print('Output activity being clipped, what about preact?')
            
            if task_params['correlation_type'] in ('activity',): # Correlate via direct activity change
                raise NotImplementedError('This was making code messier, since it requires on_idx it was further down. Update so it works here.')
                # output[step_idx, task_params['on_idx']] = (
                #     (1 - task_params['on_weight']) * output[step_idx, task_params['on_idx']] + task_params['on_weight'] * output[step_idx, cn_idx]
                # )
            if task_params['special_test_type'] in ('zero_cn_input_and_sparsify_rec_copy_hold',):
                raise NotImplementedError('This was making code messier, since it requires nonzero_rec_idxs it was further down. Update so it works here.')
                # # Holds one of the special neurons at its average activity
                # output[step_idx, task_params['nonzero_rec_idxs'][0]] = avg_activity[step_idx, task_params['nonzero_rec_idxs'][0]]

            if task_params['add_fl']: # Computed after activity adjustments
                n_steps_fl = np.minimum(step_idx, task_params['fl_kernel'].shape[0] - 1) # Only pass whats needed, -1 because inclusive of step_idx
                output_fl[step_idx] = fluorescence_convolution(output[step_idx - n_steps_fl:step_idx+1], task_params, last_only=True)

            avg_activity[step_idx] = accumulate_decay(prev_avg_activity, output[step_idx], n_window=train_params['n_window_baseline'])
            avg_activity_pre_act[step_idx] = accumulate_decay(prev_avg_activity_pre_act, output_pre_act[step_idx], n_window=train_params['n_window_baseline'])

            output_deviation = output[step_idx] - avg_activity[step_idx] # n_neurons
            output_preact_deviation = output_pre_act[step_idx] - avg_activity_pre_act[step_idx]
            act_fn_p_pre_act = net.act_fn_p(output_pre_act[step_idx])
            if 'act_fn_p_pre_act_vals' in output_vars and train: # For analyzing eligibility trace in more detail
                act_fn_p_pre_act_vals[step_idx] = np.copy(act_fn_p_pre_act)

            ### Now that forward is complete, compute reward/state adjusts and (for some tasks) state computation based on stage of training ###
            if step_idx < task_params['n_steps_stabilize']:
                if task_params['task_type'] in ('our_bci',): # Generate a new input at every time step
                    temp_bci_activity = None # Prior to stabilization, bci_activity is not well defined
                    if step_idx < n_steps - 1: # Doesn't run on last step
                        net_input[step_idx+1], _, _, _, env_extras = task.step(temp_bci_activity) # currently, done and term not used, reward ignored too

            elif step_idx >= task_params['n_steps_stabilize']:
                # Stablilization over, determines bci_mask(s), set difficulty, retroactively computes reward, sets baseline reward.
                # This could have in theory been done at the end of the previous step, but clearner to do here.
                if step_idx == task_params['n_steps_stabilize']:
                    # This ignores effects from delays, but should have an overall very small effect
                    if task_params['add_fl']:
                        reward_relevant_stabilize = output_fl[:task_params['n_steps_stabilize']]
                    else:
                        reward_relevant_stabilize = output[:task_params['n_steps_stabilize']]
                    
                    ### Determine BCI mask ###
                    prev_seq_vars_tuning = (
                        net_input, output, output_pre_act, reward, avg_reward, avg_activity,
                        avg_activity_pre_act, mean_stabilize_actvities, output_fl,
                        bci_activity, avg_bci_activity, avg_reward_stims
                    )
                    task_params, determine_bci_extras = determine_bci_mapping(
                        reward_relevant_stabilize, (task_params, train_params, net_params), net=net, task=task,
                        train_outputs_all=train_outputs_all, prev_seq_vars_tuning=prev_seq_vars_tuning, output_vars=output_vars,
                        verbose=verbose
                    )
                    del prev_seq_vars_tuning

                    train_outputs['bci_mean_activity'] = np.mean(get_bci_activity(reward_relevant_stabilize, task_params), axis=0)

                    # Special tests that make adjustments after BCI mask is set
                    task_params, train_params, net_params, net = set_special_test_post_bci_mask(
                        reward_relevant_stabilize, task_params, train_params, net_params, net
                    )

                    ### Set the difficulty of the task based on stabilization activity (changes task_params) ###
                    task_params, task = set_task_difficulty(reward_relevant_stabilize, task_params, task=task, verbose=verbose)

                    # Retroactively calculate what BCI activity and (optionally) rewards would have been during stabilization period
                    # Starts at n_delay_steps since needs previous activity to compute reward, so first few bci_activity/rewards will just be zero
                    for step_idx_retro in range(task_params['n_reward_delay'], step_idx):
                        # Incorporates delay by using past activity for current reward computation
                        if  task_params['add_fl']: # If fl is added, passes this to compute reward instead of raw activity
                            bci_relevant_activity = output_fl[step_idx_retro] # No delay here, bci activity is activity without delay
                            reward_relevant_activity = output_fl[step_idx_retro - task_params['n_reward_delay']]
                        else:
                            bci_relevant_activity = output[step_idx_retro]
                            reward_relevant_activity = output[step_idx_retro - task_params['n_reward_delay']]

                        bci_activity[step_idx_retro] = get_bci_activity(bci_relevant_activity, task_params)
                        avg_bci_activity[step_idx_retro] = np.nan # Since avg activity isn't calculated during presession, no need
                        if task_params['task_type'] not in ('our_bci',): # In this case, calculation of prev reward done in special fn below
                            # Delay was accounted for in offset of index assignment above
                            reward[step_idx_retro] = get_reward(
                                reward_relevant_activity, task_params,
                                task=task, seq_idx=step_idx_retro
                            )

                    if net_params['weight_mask_modes'] is not None: # Optional freezing of certain weights
                        net, net_params = get_weight_adjust_mask(net, task_params, net_params, net.W_inp, net.W_rec)

                    ### Computes previous reward, which is needed to compute RPE ###
                    if net.last_reward is not None: # Just use the network's last reward (from previous session)
                        print('Setting avg_reward to last_reward')
                        avg_reward[step_idx-1] = np.copy(net.last_reward)
                    elif task_params['task_type'] in ('our_bci',) and session_idx == 0:
                        
                        prev_seq_vars_reward = (
                            net_input, output, output_pre_act, reward, avg_reward, avg_activity,
                            avg_activity_pre_act, mean_stabilize_actvities, output_fl,
                            bci_activity, avg_bci_activity, avg_reward_stims
                        )
                        avg_test_reward = compute_avg_reward_our_bci((task_params, train_params, net_params), net, task, prev_seq_vars_reward)
                        del prev_seq_vars_reward
                        if 'start_avg_reward_mult' in task_params:
                            avg_test_reward *= task_params['start_avg_reward_mult']
                            print(' Setting avg_reward to test average: {:.4f} (mult: {:.2f})'.format(avg_test_reward, task_params['start_avg_reward_mult']))
                        else:
                            print(' Setting avg_reward to test average: {:.4f}'.format(avg_test_reward))
                        avg_reward[step_idx-1] = avg_test_reward
                    elif task_params['start_avg_reward'] in ('dynamic',):
                        avg_reward[step_idx-1] = np.nanmean(
                            reward[task_params['n_reward_delay']:task_params['n_steps_stabilize']], axis=0
                        ) # Delay already accounted for in assignment
                    elif type(task_params['start_avg_reward']) == float:
                        # print('Setting avg_reward to start_avg_reward: {:.4f}'.format(task_params['start_avg_reward']))
                        avg_reward[step_idx-1] = np.copy(task_params['start_avg_reward'])
                    else:
                        raise NotImplementedError('task_params[start_avg_reward]: {} not recognized.'.format(task_params['start_avg_reward']))

                    prev_reward = avg_reward[step_idx-1] # Update this with new assignment, since used below to calculate running reward
                    prev_avg_bci_activity = np.mean(bci_activity[:step_idx]) # Update, used below to calculate running BCI activity

                ### Things to do on every step after stabilization (including step_idx == n_steps_stabilize) ###
                if  task_params['add_fl']: # If fl is added, passes this to compute reward instead of raw activity
                    bci_relevant_activity = output_fl[step_idx] # No delay here, bci activity is activity without delay
                    reward_relevant_activity = output_fl[step_idx - task_params['n_reward_delay']]
                else:
                    bci_relevant_activity = output[step_idx]
                    reward_relevant_activity = output[step_idx - task_params['n_reward_delay']]

                bci_activity[step_idx] = get_bci_activity(bci_relevant_activity, task_params)
                avg_bci_activity[step_idx] = accumulate_decay(prev_avg_bci_activity, bci_activity[step_idx], n_window=train_params['n_window_baseline'])
                # avg_bci_activity[step_idx] = get_bci_activity(avg_activity[step_idx], task_params) # THIS DOES NOT USE THE FL ACTIVITY SO WILL BE INCORRECT

                if task_params['task_type'] in ('our_bci',): # Calculate reward and next input via environment
                    bci_mask_idx = 0
                    if task_params['env_drive_mode'] in ('default',): # Current net's activty drives environemnt
                        # Delay incorporated into offset of bci_activity (defined to have no delay)
                        env_drive_activity = bci_activity[step_idx - task_params['n_reward_delay']][bci_mask_idx]
                    elif task_params['env_drive_mode'] in ('saved_activity',): # Saved BCI activty drives environemnt
                        # Delay doesn't matter here, since its external activity driving network anyway
                        env_drive_activity = task_params['env_drive_activity'][step_idx, bci_mask_idx]
                    net_input_temp, reward[step_idx], _, _, env_extras = ( # BCI activity includes fl activity if used
                        task.step(env_drive_activity)
                    )
                    if step_idx < n_steps - 1: # No assignment on final step
                        net_input[step_idx+1] = np.copy(net_input_temp)
                else: # Reward calculation via get_reward function now that threshold and BCI masks are set
                    reward[step_idx] = get_reward( # Note: delay incorporated into reward_relevant_activity above already
                        reward_relevant_activity, task_params,
                        task=task, seq_idx=step_idx
                    )

                # # Create a copy of the network and task to see what reward would be at this point in test task
                # # Note this uses net_input[step_idx+1] so must be run after first step above (hence awkard placement)
                # # The net effect of this is to set avg_reward and prev_reward, which is used below for this step
                # if step_idx == task_params['n_steps_stabilize'] and task_params['task_type'] in ('our_bci',) and session_idx == 0:
                #     if task_params['start_avg_reward'] == 'dynamic':
                #         prev_seq_vars_reward = (
                #             net_input, output, output_pre_act, reward, avg_reward, avg_activity,
                #             avg_activity_pre_act, mean_stabilize_actvities, output_fl,
                #             bci_activity, avg_bci_activity, avg_reward_stims
                #         )
                #         avg_test_reward = compute_avg_reward_our_bci((task_params, train_params, net_params), net, task, prev_seq_vars_reward)
                #         del prev_seq_vars_reward
                #         if 'start_avg_reward_mult' in task_params:
                #             avg_test_reward *= task_params['start_avg_reward_mult']
                #             print('Setting avg_reward to test average: {:.4f} (mult: {:.2f})'.format(avg_test_reward, task_params['start_avg_reward_mult']))
                #         else:
                #             print('Setting avg_reward to test average: {:.4f}'.format(avg_test_reward))
                #         avg_reward[step_idx-1] = avg_test_reward
                #         prev_reward = avg_reward[step_idx-1]
                
                # Option to create a test version of the network to evaluate statistics in a training-frozen session
                # This is only really needed for tasks that interact with the enviroment. We do this after next input is caluclated for starting point
                # Needs to be training, prevents test sessions from triggering themselves
                if task_params['task_type'] in ('our_bci',) and train and task_params['run_test_idxs'] is not None:
                    if step_idx in task_params['run_test_idxs']:
                        print('Running test session at step {} (duration {} steps)...'.format(step_idx, task_params['n_steps_test']))
                        prev_seq_vars = (
                            net_input, output, output_pre_act, reward, avg_reward, avg_activity,
                            avg_activity_pre_act, mean_stabilize_actvities, output_fl,
                            bci_activity, avg_bci_activity, avg_reward_stims
                        )
                        test_output_vars = ('reward', 'output',)
                        _, train_outputs_test, _, _ = run_test_session_our_bci(
                            params, net, task, prev_seq_vars, test_output_vars, n_steps_test=task_params['n_steps_test']
                        )
                        # Now save the outputs from the test session to unique key
                        test_session_idx = 0
                        while 'train_outputs_test_{}'.format(test_session_idx) in train_outputs.keys():
                            test_session_idx += 1
                        session_idx = 0 # test sessions are always run for only one session
                        train_outputs[f'train_outputs_test_{test_session_idx}'] = train_outputs_test[session_idx]
                        del prev_seq_vars
                        del train_outputs_test
                
                # Average reward is calculated without potential reward delay, assumed to based on current rewards
                avg_reward[step_idx] = accumulate_decay(prev_reward, reward[step_idx], n_window=train_params['n_window_reward']) # nans handled within

                # Special version of RPE that is stimulus-dependent (used for WMP/OMP model)
                if avg_reward_stims is not None:
                    current_stim_idx = task.trial_stims[step_idx]
                    for stim_idx in range(task.n_stim):
                        if stim_idx == current_stim_idx: # Actually decay reward based on current step
                            avg_reward_stims[step_idx, stim_idx] = accumulate_decay(prev_reward_stims[stim_idx], reward[step_idx], n_window=train_params['n_window_reward']) # nans handled within
                            avg_reward[step_idx] = avg_reward_stims[step_idx, stim_idx] # Override avg reward
                        else: # Just carry average reward over
                            avg_reward_stims[step_idx, stim_idx] = prev_reward_stims[stim_idx]

                ### Determine if solution has been found ###
                if task_params['solution_type'] is not None:
                    if task_params['solution_type'] in ('avg_activity',):
                        if task_params['add_fl']: raise NotImplementedError('Need to incorporate FL in this setup.')
                        # Uses the reward criterion to see if avg activity is within threshold
                        avg_act_reward = get_reward(
                            avg_activity[step_idx], task_params,
                            task=task, seq_idx=step_idx
                        )
                        # Threshold average reward needs to be under (0.0 for trapezoidal most of the time)
                        sol_scale = task_params['sol_scale'] if task_params['sol_scale'] is not None else 0.0
                        if ~np.isnan(avg_act_reward):
                            if np.abs(avg_act_reward) <= sol_scale:
                                sol_n_steps += 1
                            else:
                                sol_n_steps = 0
                        # print('Sol at step idx {}, sol steps: {}'.format(step_idx, sol_n_steps))
                        if sol_n_steps == task_params['sol_n_steps'] and sol_step is None:
                            sol_step = np.copy(step_idx - task_params['n_steps_stabilize'])
                            if verbose: print('Solution criteria met at step: {}'.format(sol_step))
                            # Break training if relevant
                            if task_params['terminate_on_sol']: break
            # Continue run below regardless of training step (reward is just zero without BCI mask)

            if task_params['constant_rpe_val'] is not None: # Overrides actual rewards with custom signal
                reward[step_idx] = task_params['constant_rpe_val']
                avg_reward[step_idx] = task_params['constant_rpe_val']
            # avg_reward[step_idx] = 0.0025 # Override avg reward for some constant baseline checks

            # Possible delay in reward already accounted for in offset of step_idx assignment above
            rpe = reward[step_idx] - avg_reward[step_idx] # scalar, can be nan

            # Print out summary during training
            if step_idx > 0 and step_idx % train_params['print_every'] == 0:
                if verbose:
                    print('step {} - avg reward: {:.2e}'.format(step_idx, avg_reward[step_idx]))

            # Some early exit conditions that will skip weight adjustment calculations
            if not train: # No adjustment run
                continue
            if step_idx < task_params['n_steps_stabilize']: # No adjustment during stabilization period
                continue
            if step_idx >= n_steps - task_params['n_steps_evaluate']: # No adjustment during evaluation period
                if step_idx == n_steps - task_params['n_steps_evaluate']:
                    if verbose: print('Evaluation period: pausing training')
                continue

            ### Eligibility calculation and accumulation, RPE accumulation ###
            if train_params['adjust_type'] in ('3factor_sl', '3factor_sl_uniform', 'backprop_sl',): # Override RPE with supervised signal
                if task_params['task_type'] in ('simple_bci',):
                    assert task_params['reward_structure'] == 'abs'
                    # assert task_params['normalize_reward'] == False
                    assert task_params['add_fl'] == False
                    # assert task_params['n_reward_delay'] == 0
                    assert task_params['n_bci_masks'] == 1
                    # BCI activity below threshold is equivalent to positive RPE
                    bci_mask_idx = 0
                    rpe = (task_params['threshold'] - bci_activity[step_idx, bci_mask_idx])
                elif task_params['task_type'] in ('our_bci',):
                    assert task_params['n_bci_masks'] == 1
                    bci_mask_idx = 0
                    env_drive_activity = bci_activity[step_idx - task_params['n_reward_delay']][bci_mask_idx]
                    rpe = (env_extras['oracle_signal'] - env_drive_activity)
                else:
                    raise NotImplementedError()
                # # Effective credit assignment of BCI mask
                # act_fn_p_pre_act = task_params['bci_masks'][bci_mask_idx] * act_fn_p_pre_act

            # Update eligibility traces of network (tracked internally)
            # (messy call because lots of options for eligibility adjustment)
            net.backward(
                task_params, rpe, output_deviation, current_input, prev_activity,
                act_fn_p_pre_act, prev_avg_activity, output_pre_act[step_idx],
                output[step_idx], output_preact_deviation, current_act_perturbation,
                current_preact_perturbation, task_params['bci_masks'],
            )

            ### Weight adjustment, only if criterion are met ###
            if (step_idx + 1) % train_params['n_steps_per_loss'] == 0: # +1 prevent adjustment on step_idx == 0

                net.loss_step(task_params, train_params)

                # Save some stuff for post-training analysis
                loss_steps.append(step_idx)
                total_rpes.append(net.total_rpe)
                if net.W_inp_adjust:
                    W_inp_vals.append(np.copy(net.W_inp))
                    if 'W_inp_elg_vals' in output_vars:
                        W_inp_elg_vals.append(np.copy(net.total_W_inp_elg))
                if net.W_rec_adjust:
                    W_rec_vals.append(np.copy(net.W_rec))
                    if 'W_rec_elg_vals' in output_vars:
                        W_rec_elg_vals.append(np.copy(net.total_W_rec_elg))

                # Should be called after saving of values above, since it wipes many things
                net.set_grads_for_next_step()

        ### At the end of the session ###
        if verbose and task_params['task_type'] not in ('our_bci',):
            print('Total reward: {:.2f}, total rpe: {:.2f} (std: {:.2f})'.format(np.nansum(reward), np.nansum(total_rpes), np.nanstd(total_rpes)))
        session_summary = None
        if task_params['task_type'] in ('our_bci',) and 'session_summary' in output_vars: # Special summary printout for our BCI task
            session_summary = get_session_summary_our_bci_task(
                reward, total_rpes, output, output_fl, avg_activity, task, task_params,
                task_params['cn_idxs'], task_params['cn_idxs_activity_percentile'], output_vars, verbose=verbose
            )

        # Run end of session photostim if being used
        # (internally initializes/resets new session)
        if task_params['photostim'] in ('every_session',):
            _, train_outputs_ps, _, task_ps = run_photostim(
                (task_params, train_params, net_params), output_vars=output_vars_ps,
                task=task, task_ps=task_ps, net=net,
                verbose=verbose
            )
            train_outputs['train_outputs_ps'] = train_outputs_ps

        if train:
            W_inp_vals = np.array(W_inp_vals)
            W_rec_vals = np.array(W_rec_vals)
            delta_W_inp = W_inp_vals[-1, :, :] - W_inp_vals[0, :, :]
            delta_W_rec = W_rec_vals[-1, :, :] - W_rec_vals[0, :, :]
            if 'W_inp_elg_vals' in output_vars: W_inp_elg_vals = np.array(W_inp_elg_vals)
            if 'W_rec_elg_vals' in output_vars: W_rec_elg_vals = np.array(W_rec_elg_vals)
        else:
            delta_W_inp = None
            delta_W_rec = None

        train_outputs['bci_masks'] = np.copy(task_params['bci_masks'])
        if task_params['cn_idxs'] is not None:
            train_outputs['cn_idxs'] = np.copy(task_params['cn_idxs'])
        else: # Prevents this from being turned into an array with None in it
            train_outputs['cn_idxs'] = None
        train_outputs['threshold'] = np.copy(task_params['threshold'])
        train_outputs['start_avg_reward'] = avg_reward[task_params['n_steps_stabilize']-1] # Used to estimate reward with no training
        if 'test_tunings' in output_vars: test_tunings = determine_bci_extras['test_tunings'] 
        if 'test_tuning_stds' in output_vars: test_tuning_stds = determine_bci_extras['test_tuning_stds']
        assert 'task' not in output_vars # Output separately, shouldnt be in train_outputs (depricated)
        assert 'task_ps' not in output_vars # Output separately, shouldnt be in train_outputs (depricated) 
        for output_var in output_vars:
            if output_var in ('FILL',): # Omitted output_vars that are saved elesewhere
                continue
            else:
                exec('train_outputs[\'{}\'] = {}'.format(output_var, output_var))

        if save_sessions: # train_outputs saved, so wipe to save memory
            net.rng_state = np.random.get_state()

            net_helpers.save_session(
                session_idx, (task_params, train_params, net_params),
                train_outputs, net, task, task_ps, save_path,
                path_mode='raw_path', overwrite=False, verbose=verbose
            )

            # Delete information that will be TWO sessions back (after next iteration),
            # need to keep previous session info because it is sometimes used to
            # compute things like tuning
            if session_idx > 0:
                if verbose: print('Wiping session {} data'.format(session_idx - 1))
                train_outputs_all[session_idx - 1] = {}
                task.hists[session_idx - 1] = {}
                if task_ps is not None:
                    task_ps.hists[session_idx - 1] = {}

        # Accumulate train_outputs (might be wiped later, see above)
        train_outputs_all.append(train_outputs)

        if len(avg_reward) > 0:
            net.last_reward = np.copy(avg_reward[-1]) # Carries over average reward to next session
        # else: # No training case
        #     net.last_reward = 0.

        # Reset mask to trigger reset at start of next session (mask saved to train_outputs above)
        task_params['bci_masks'] = None
        task_params['cn_idxs'] = None

    ### At the end of all sessions ###
    # Some params are updated here, so return them
    params = task_params, train_params, net_params

    net.rng_state = np.random.get_state()

    return params, train_outputs_all, net, task, task_ps

output_vars = ['output', 'reward', 'avg_reward', 'avg_activity', 'loss_steps',
               'W_inp_vals', 'W_rec_vals', 'bci_activity', 'avg_bci_activity',
               'total_rpes', 'mean_stabilize_actvities', 'sol_step',
               'delta_W_inp', 'delta_W_rec', 'output_fl', 'session_summary',]

output_vars_ps = ['output_ps',]

session_idx = 0

n_cells = 5

needs_thresold_plot = True

label = 'not CN examples'

show_trial_starts = False

import statsmodels.api as sm

n_offset = 0

session_idx = 0

predictor_names = (
    # 'RPE-activity product',
    # 'RPE-dev product',
    'CN act-act product',
    'CN dev-act product',
    'CN dev-dev product',
    'W_rec input to CN',
    'W_rec output from CN',
    'W_inp sim rel. to CN',
)

reg_names = []

delta_W_regressors = []

step_init = 10000

step_final = 20000

n_steps_corr = 5000

session_idx = 0

def get_corrs_and_weights(step):
    output_steps = output[step-n_steps_corr:step]
    corrs = np.cov(output_steps.T)

    # Find closest loss_step index to desired final time step
    loss_step_idx = np.argmin(np.abs(step - np.array(loss_steps)))
    weights = W_vals[loss_step_idx]

    corrs_flat = []
    weights_flat = []

    for neuron_idx1 in range(corrs.shape[0]):
        for neuron_idx2 in range(corrs.shape[0]):
            if neuron_idx1 == neuron_idx2: continue # Skip diagonals
            corrs_flat.append(corrs[neuron_idx1, neuron_idx2])
            weights_flat.append(weights[neuron_idx1, neuron_idx2])

    return np.array(corrs_flat), np.array(weights_flat)

import copy

def get_more_acitivty(params, net, W_rec_new=None, W_inp_new=None, state=None,
                      state_pre_act=None):

    task_params, train_params, net_params = params

    task_params_copy = copy.deepcopy(task_params)
    train_params_copy = copy.deepcopy(train_params)
    net_params_copy = copy.deepcopy(net_params)
    net_copy = copy.deepcopy(net)

    train_params_copy['eta'] = 0.0 # Zeros out learning rate, so no training
    task_params_copy['n_steps_stabilize'] = task_params_copy['n_steps'] + 10 # Never leave stabilization (repetitive with above)

    if W_rec_new is not None:
        net_copy['W_rec'] = W_rec_new

    if W_inp_new is not None:
        net_copy['W_inp'] = W_inp_new

    if state is not None:
        raise NotImplementedError('Update to new network')
        assert state_pre_act is not None # Both these need to be nonzero for this to make sense
        net_copy['prev_activity'] = state
        net_copy['prev_activity_pre_act'] = state_pre_act

    _, train_outputs_activity, _, _, _ = train_task((task_params_copy, train_params_copy, net_params_copy),
                                                     output_vars=output_vars, net=net_copy, verbose=False)

    session_idx = 0

    return train_outputs_activity[session_idx]['output'], train_outputs_activity

session_idx = 0

percentile = 90

max_activity = 1.

correlation_type = 'deviation_raw_rpes'

session_idx = 0

plot_extras = True

training_only = False

cmap_type = 'cool'

var_exps = []

slopes = []

dev_self_corrs = []

rpe_self_corrs = []

rpe_dev_prods = []

comparison_type = 'dev-cn_dev'

max_offset = 9

w_val_idx = 0

session_idx = 0

cmap_type = 'viridis'

session_idx = 0

session_idx = 0

session_idx = 0

TOLERANCE = 1e-2

session_idx = 0

W_elg_values_reconstruct = []

W_elg_values_no_phi_p = []

neuron_idx = 407

session_idx = 19

session_idx = 0

session_idx = 1

step_idxs = None

output_corrs_flat = []

delta_W_rec_flat = []

session_idx = 0

correlations = []

n_divisions = 10

div_step_idxs = []

session_idx = 0

def get_reward_mask(activity, mask, threshold=1.5, width=0.1):
    raise NotImplementedError('This is old.')
    bci_activity = np.dot(mask, activity)

    if bci_activity < threshold:
        return  -1 * np.abs(bci_activity - threshold)
    elif bci_activity > threshold + width:
        return  -1 * np.abs(bci_activity - (threshold + width))
    else:
        return 0.

bci_masks = []

n_divisions = 20

session_idx = 0

div_step_idxs = []

def get_div_idxs(task_hist, loss_steps, hi_params):
    if hi_params['div_mode'] == 'idxs':
        n_divisions = hi_params['n_divisions']
        # Divide indexes of loss/weight values into equal sizes, start at 1 since loss_steps[0] is 0 by default
        loss_step_divs = np.linspace(1, len(loss_steps)-1, n_divisions+1).astype(np.int32)
    elif hi_params['div_mode'] == 'trials':
        n_trials_per_div = hi_params['n_trials_per_div']
        n_trials = len(task_hist['trial_starts']) - 1 # Last trial is always omitted so we have end point
        n_divisions = int(np.floor(n_trials / n_trials_per_div))
        print('{} trials into {} divisions ({} per trial, last {} omitted)'.format(
            n_trials, n_divisions, n_trials_per_div, n_trials - n_trials_per_div * n_divisions
        ))

        # +1 here ensures that we get trial start idx of trial after last division, there will always be one because of above omission
        div_step_idxs = [task_hist['trial_starts'][div_idx * n_trials_per_div] for div_idx in range(n_divisions+1)]

        loss_step_divs = []
        for div_step_idx in div_step_idxs:
            loss_step_divs.append(np.argmin(np.abs(div_step_idx - np.array(loss_steps))))

        loss_step_divs = np.array(loss_step_divs)
    
    return loss_step_divs

def compute_local_hebbian_indexes(
    train_outputs, task_hist, params, hi_params, return_elig_divs=False, run_mlr=False):
    """
    Compute the local Hebbian index across potentially many eligibility estimations, including the true eligibility.
    
    Note this code computes the full session Hebbian index still, which is an addition computation
    but just kept it in for convenience
    
    INPUTS: 
    - train_outputs
    - n_divisions: number of divisions to divide session up into. Does this from the loss time steps, 
        so looks only at training times
    - return_elig_divs: return internally computed variable elig_divs, which is quite large but sometimes used for 
        additional analysis
    - run_mlr: MLR fit over divisions, this is the original way Kayvon was doing, kept for backwards compatibility
        
    OUTPUTS:
    - rpes_divs: shape (n_divisions,); RPE in each division
    - div_slopes: shape (n_elig_types, n_divisions,) the true local HI for each division (fit to change in weight in division)
    - div_slopes_full_delta: shape (n_elig_types, n_divisions,) the approximate local HI for each division (fit to change in weight across entire session)
    - full_slopes: shape (n_elig_types,) the HI across the entire session
    - elig_divs: usually None, otherwise internal variable
    - div_slopes_mlr: (n_elig_types, n_divisions,) MLR HI
    """
    
    task_params, train_params, net_params = params
    
    output = train_outputs['output']
    reward = train_outputs['reward']
    total_rpes = train_outputs['total_rpes']
    
    loss_step_divs = get_div_idxs(task_hist, train_outputs['loss_steps'], hi_params)
    n_divisions = loss_step_divs.shape[0] - 1 # -1 since this has both bounds of each division
    
    # if task_params['add_fl']:
    #     print('SHOULD COMPUTE ELIGIBILITY FROM FLUORESCENCE!')

    if net_params['W_inp_adjust']:
        W_vals = train_outputs['W_inp_vals']
        W_elg = train_outputs['W_inp_elg_vals']
        n_presyn = task_params['n_inp']
    elif net_params['W_rec_adjust']:
        W_vals = train_outputs['W_rec_vals']
        W_elg = train_outputs['W_rec_elg_vals']
        n_presyn = task_params['n_neurons']
        
    delta_W_divs = np.zeros((n_divisions, task_params['n_neurons'], n_presyn)) # Change in W over each division
    rpes_divs = {}
    if 'true' in hi_params['rpe_types']:
        rpes_divs['true'] = np.zeros((n_divisions,)) # Total RPE in each division

    # Different eligibility types
    n_elig_types = len(hi_params['elig_types'])
    elig_divs = np.zeros((n_elig_types, n_divisions, task_params['n_neurons'], n_presyn)) # Sum of each elig in each division
    elig_full = np.zeros((n_elig_types, task_params['n_neurons'], n_presyn)) # Across full session, so no divisions

    ### Full session values for entire session Hebbian index ###
    idx_start = 0
    idx_end = loss_step_divs[-1]
    delta_W = W_vals[idx_end] - W_vals[idx_start]

    # Translate the loss steps (start at idx 1 because first loss_step is always [0]
    step_idx_start = train_outputs['loss_steps'][1] - train_params['n_steps_per_loss'] + 1
    step_idx_end = train_outputs['loss_steps'][idx_end] + 1

    # Hebbian index fits, eligibility to change in weights
    div_slopes = np.zeros((n_elig_types, n_divisions,)) # Elig in division to weight change in division
    div_slopes_full_delta = np.zeros((n_elig_types, n_divisions,)) # Elig in division to total weight change
    full_slopes = np.zeros((n_elig_types,)) # Over entire session

    for elig_type_idx, elig_type in enumerate(hi_params['elig_types']):
        if elig_type == 'true':
            elig_full[elig_type_idx] = np.sum(W_elg[idx_start+1:idx_end+1, :, :], axis=0) # +1 because first entry is all zeros
        elif elig_type == 'hebb':
            elig_full[elig_type_idx] = np.matmul(output[step_idx_start:step_idx_end].T, output[step_idx_start:step_idx_end])
            # elig_full[elig_type_idx] = np.corrcoef(output[step_idx_start:step_idx_end].T)
        else:
            continue
        # Hebbian index fit over entire session
        full_slopes[elig_type_idx], _, _, _, _ = linregress(elig_full[elig_type_idx].flatten(), delta_W.flatten())

    ### Scan over eligibility types then divisions ###
    # This order means we only need to compute the full eligibility once for each division
    for elig_type_idx, elig_type in enumerate(hi_params['elig_types']):

        # Compute eligibility for the relevant type
        if elig_type in ('hebb', 'dpost_pre', 'post_dpre', 'dpost_dpre', 'dpost_pre_acc',):
            eligibility = bci_analysis.compute_eligibility_approx(output, output, train_params, elig_type=elig_type)
        elif elig_type not in ('true',): # Already saved in this place, so no need to compute.
            raise ValueError('Eligibility type {} not recognized!'.format(elig_type))

        for division_idx in range(n_divisions):
            # Start and end of division in loss steps
            idx_start = loss_step_divs[division_idx]
            idx_end = loss_step_divs[division_idx+1]

            # Translate loss steps intro raw steps
            step_idx_start = train_outputs['loss_steps'][idx_start] - train_params['n_steps_per_loss'] + 1
            step_idx_end = train_outputs['loss_steps'][idx_end] + 1

            # print('Loss idxs: {} to {}\tStep idxs: {} to {}'.format(idx_start, idx_end, step_idx_start, step_idx_end))

            if elig_type_idx == 0: # These things are not eligibility dependent, so do only on first pass through
                delta_W_divs[division_idx] =  W_vals[idx_end] - W_vals[idx_start]
                rpes_divs['true'][division_idx] = np.sum(np.array(total_rpes[idx_start:idx_end]))
                
            if elig_type == 'true': 
                # Division relevant eligibility and RPE
                W_elg_trunc = W_elg[idx_start:idx_end, :, :]
                elig_divs[elig_type_idx, division_idx] = np.sum(W_elg_trunc, axis=0) # Sum over all time steps within division
            elif elig_type in ('hebb', 'dpost_pre', 'post_dpre', 'dpost_dpre', 'dpost_pre_acc',):
                elig_divs[elig_type_idx, division_idx] = np.sum(eligibility[step_idx_start:step_idx_end], axis=0)

            # Now that eligibility in division is computed, fit to weight changes (both local or full weight change)
            div_slopes[elig_type_idx, division_idx], _, _, _, _ = linregress(
                elig_divs[elig_type_idx, division_idx].flatten(), delta_W_divs[division_idx].flatten()
            )
            div_slopes_full_delta[elig_type_idx, division_idx], _, _, _, _ = linregress(
                elig_divs[elig_type_idx, division_idx].flatten(), delta_W.flatten()
            )

    div_slopes_mlr = None
    if run_mlr:
        div_slopes_mlr = np.zeros((n_elig_types, n_divisions,))

        for elig_type_idx, elig_type in enumerate(hi_params['elig_types']):
            # Kayvon's MLR version of Hebbian index
            X = elig_divs[elig_type_idx, :].reshape((n_divisions, -1)).T # (n_divs, n_pairs) -> (n_pairs, n_divs)
            X = sm.add_constant(X)
            Y = delta_W.flatten()[:, np.newaxis]
            # Y = delta_W_divs[div_idx, :, :].flatten()[:, np.newaxis]

            fit_model = sm.OLS(Y, X, missing='drop')
            results = fit_model.fit()

            div_slopes_mlr[elig_type_idx] = results.params[1:]     
    if not return_elig_divs: # Wipe this before returning it
        elig_divs = None
        
    ### Gather various performance metrics over each division ###
    rpe_estimates = get_rpe_estimates(task_hist)
    for rpe_type in hi_params['rpe_types']:
        if rpe_type == 'true':
            continue
        if hi_params['div_mode'] == 'idxs':
            rpes_divs[rpe_type] = None 
        elif hi_params['div_mode'] == 'trials':
            rpes_divs[rpe_type] = np.zeros((n_divisions,)) # Total RPE in each division
            if rpe_type in ('hits', 'hits_rpe', 'speed', 'speed_rpe',): # Divide by trials
                for div_idx in range(n_divisions):
                    rpes_divs[rpe_type][div_idx] = np.mean(
                        rpe_estimates[rpe_type][div_idx*hi_params['n_trials_per_div']:(div_idx+1)*hi_params['n_trials_per_div']]
                    )
            elif rpe_type in ('hits_steps', 'hits_rpe_steps', 'spout_steps', 'spout_rpe_steps',):
                for div_idx in range(n_divisions):
                    # Get loss step divisions
                    idx_start = loss_step_divs[div_idx]
                    idx_end = loss_step_divs[div_idx+1]
                    # Translate loss steps intro raw steps
                    step_idx_start = train_outputs['loss_steps'][idx_start] - train_params['n_steps_per_loss'] + 1
                    step_idx_end = train_outputs['loss_steps'][idx_end] + 1
                    # if rpe_type == 'hits_steps':
                    #     print('{} - steps {} to {}'.format(div_idx, step_idx_start, step_idx_end))
                    rpes_divs[rpe_type][div_idx] = np.mean(
                        rpe_estimates[rpe_type][step_idx_start:step_idx_end]
                    )
                
    return rpes_divs, div_slopes, div_slopes_full_delta, full_slopes, elig_divs, div_slopes_mlr

MAX_L = 3.

MAX_T = 10.

T_STEP = 0.05

def get_rpe_estimates(task_hist, n_trial_avg=10, n_steps_miss=200, n_reward_window=6000,):
    """
    INPUTS:
    - n_trial_avg: # of trials to average over
    - n_steps_miss: how many steps to count misses as
    - n_reward_window: # of steps to average over
    """

    ### Trial metrics ###
    start_idxs = np.array(task_hist['trial_starts'])
    rew_idxs = bci_analysis.pad_rew_idxs(task_hist) # Adds nan pads to rew_idxs, should be same length as trial start idxs now

    n_trials = start_idxs.shape[0]

    hits = np.astype(~np.isnan(rew_idxs), np.int32)
    trial_lengths = rew_idxs - start_idxs
    trial_lengths = np.where(np.isnan(trial_lengths), n_steps_miss, trial_lengths)

    avg_hits = np.nan * np.ones_like(hits)
    avg_trial_lengths = np.nan * np.ones_like(hits)
    for trial_idx in range(n_trials):
        if trial_idx < n_trial_avg: # Early session clipping
            avg_hits[trial_idx] = np.mean(hits[:trial_idx+1])
            avg_trial_lengths[trial_idx] = np.mean(trial_lengths[:trial_idx+1])
        else:
            avg_hits[trial_idx] = np.mean(hits[trial_idx-n_trial_avg+1:trial_idx+1])
            avg_trial_lengths[trial_idx] = np.mean(trial_lengths[trial_idx-n_trial_avg+1:trial_idx+1])
    
    ### Step metrics ###
    gamma = 1. - 1./n_reward_window
    
    raw_states = np.array(task_hist['raw_states']) # (trial tone, water available, spout location)

    water_reward = raw_states[:, 1] # Times water is being delivered
    forward_spout_move = np.zeros((raw_states.shape[0])) 
    forward_spout_move[1:] = -1 * (raw_states[1:, 2] - raw_states[:-1, 2]) # Spout movement = difference in spout location
    forward_spout_move = np.where(forward_spout_move > 0, forward_spout_move, 0.) # Remove backward movements

    avg_water_reward = np.zeros_like(water_reward)
    avg_forward_spout_move = np.zeros_like(forward_spout_move)

    for step_idx in range(raw_states.shape[0]):
        if step_idx == 0:
            prev_avg_water_reward = 0.
            prev_avg_forward_spout_move = 0.
        else:
            prev_avg_water_reward = avg_water_reward[step_idx-1]
            prev_avg_forward_spout_move = avg_forward_spout_move[step_idx-1]

        avg_water_reward[step_idx] = gamma * prev_avg_water_reward + (1 - gamma) * water_reward[step_idx]
        avg_forward_spout_move[step_idx] = gamma * prev_avg_forward_spout_move + (1 - gamma) * forward_spout_move[step_idx]
    
    
    return {
        ### Trial metrics ###
        'hits': hits,
        'hits_rpe': np.where(np.isnan(hits - avg_hits), 0., hits - avg_hits), # If first trial is miss, prevents nan
        'speed': MAX_L / (trial_lengths * T_STEP),
        'speed_rpe': MAX_L / (trial_lengths * T_STEP) - MAX_L / (avg_trial_lengths * T_STEP),
        
        ### Step metrics ###
        'hits_steps': water_reward,
        'hits_rpe_steps': water_reward - avg_water_reward,
        'spout_steps': forward_spout_move,
        'spout_rpe_steps': forward_spout_move - avg_forward_spout_move,
    }

import statsmodels.api as sm

hi_params = {
    'div_mode': 'trials', # idxs, trials
        'n_trials_per_div': 5, # Number of trials per division bin, used when div_mode = trials
        'n_divisions': 20, # Number of equal time bins the session should be divided into, used when div_mode = idxs
    'elig_types': ('true',), # 'hebb', 'dpost_pre', 'post_dpre', 'dpost_dpre', 'dpost_pre_acc',),
    'rpe_types': ('true', 'hits', 'hits_rpe', 'speed', 'speed_rpe', 'hits_steps', 'hits_rpe_steps', 'spout_steps', 'spout_rpe_steps',),
}

session_idx = 0

div_idx = 0

elig_type_idx = 0

elig_type_idx = 0

rpe_type = 'true'

rpe_type_idx = 0

from sklearn.model_selection import train_test_split

test_ratio = 0.1

import statsmodels.api as sm

n_divisions = 50

session_idx = 0

div_step_idxs = []

div_idx = 5

eligibility_type = 'true'

def compute_and_plot_tuning_toy(session_idx, task_params, train_outputs_all, ax1=None, ax2=None):
    """
    This computes the "tuning" of the toy task (i.e. the mean activity.)
    Note the use of "tuning" to describe this is depricated, now just call it
    mean_activity, so should change vocabulary used in this function at some
    point.
    """

    assert session_idx < task_params['n_sessions'] - 1

    pretrain_tunings = train_outputs_all[session_idx]['mean_stabilize_actvities']
    posttrain_tunings = train_outputs_all[session_idx + 1]['mean_stabilize_actvities']

    current_cn_idx = train_outputs_all[session_idx]['cn_idx']

    delta_tunings = posttrain_tunings - pretrain_tunings
    delta_tuning_sort_idxs = np.argsort(delta_tunings)
    sort_delta_tunings = delta_tunings[delta_tuning_sort_idxs]

    current_cn_perc = (np.where(delta_tuning_sort_idxs == current_cn_idx)[0][0] + 1) / sort_delta_tunings.shape[0]

    if session_idx > 0: # Prev CN idx
        prev_cn_idx = train_outputs_all[session_idx - 1]['cn_idx']
        prev_cn_perc = (np.where(delta_tuning_sort_idxs == prev_cn_idx)[0][0] + 1) / sort_delta_tunings.shape[0]
    else:
        prev_cn_perc = None

    if ax1 is not None:
        ax1.scatter(pretrain_tunings, posttrain_tunings, color='k', marker='.')
        ax1.scatter(pretrain_tunings[current_cn_idx], posttrain_tunings[current_cn_idx], color=c_vals[0], marker='o', zorder=5,
                    label='Day X CN')
        if session_idx > 0: # Prev CN idx
            ax1.scatter(pretrain_tunings[prev_cn_idx], posttrain_tunings[prev_cn_idx], color=c_vals[2], marker='o', zorder=5,
                        label='Day X-1 CN')

        max_tuning = np.max((np.max(pretrain_tunings), np.max(posttrain_tunings)))
        ax1.plot((0, max_tuning), (0, max_tuning), color='lightgrey', linestyle='dashed', zorder=-1)

        ax1.set_xlabel('Day X Tuning')
        ax1.set_ylabel('Day X+1 Tuning')
        ax1.legend()

    if ax2 is not None:
        ax2.bar(np.arange(sort_delta_tunings.shape[0]), sort_delta_tunings, color='k', width=1.0)
        ax2.bar(np.where(delta_tuning_sort_idxs == current_cn_idx)[0], delta_tunings[current_cn_idx], color=c_vals[0],
                width=1.0, zorder=5)
        ax2.scatter(np.where(delta_tuning_sort_idxs == current_cn_idx)[0], delta_tunings[current_cn_idx], color=c_vals[0],
                    marker='o', zorder=5, label='Day X CN')
        if session_idx > 0: # Prev CN idx
            ax2.bar(np.where(delta_tuning_sort_idxs == prev_cn_idx)[0], delta_tunings[prev_cn_idx], color=c_vals[2],
                    width=1.0, zorder=5)
            ax2.scatter(np.where(delta_tuning_sort_idxs == prev_cn_idx)[0], delta_tunings[prev_cn_idx], color=c_vals[2],
                        marker='o', zorder=5, label='Day X-1 CN')

        ax2.set_xlabel('Neuron idx')
        ax2.set_ylabel('$\Delta$ Tuning (X+1 - X)')
        ax2.legend()

    return current_cn_perc, prev_cn_perc

session_idxs = (0,)

session_idx = 1

output_corrs_flat = []

delta_W_rec_flat = []

def run_sequence(net_input, W_inp, W_rec, task_params, net_params):

    output = np.zeros((task_params['n_steps'], task_params['n_neurons'],))
    output_pre_act = np.zeros((task_params['n_steps'], task_params['n_neurons'],))

    for step_idx in range(task_params['n_steps']):

        if step_idx == 0: # Initialization values
            prev_activity = np.zeros((task_params['n_neurons'],))
            prev_activity_pre_act = np.zeros((task_params['n_neurons'],))
        else:
            prev_activity = output[step_idx-1]
            prev_activity_pre_act = output_pre_act[step_idx-1]

        output[step_idx], output_pre_act[step_idx] = net_forward(
            W_inp, net_input[step_idx], W_rec, prev_activity, prev_activity_pre_act, net_params
        )

        # Clip total activity for numerical stability
        if 'max_activity' in task_params:
            output[step_idx] = np.clip(output[step_idx], None, task_params['max_activity'])

    return output, output_pre_act

session_idx = 1

step_idx_i = 900

step_idx_f = 500

tuning_activity_type = 'raw_fl'

session_idx = 0

n_other_cells = 5

n_trial_avg_window = 10

session_idx = 0

exemplar_group_idx = 5

exemplar_neuron_idx2 = 22

ps_activity_type = 'raw_fl'

omit_ps_times=True

n_prev_dir_omit = 1

direct_mask = []

indirect_mask = []

T_POST_STIM = 300

T_PRE_STIM = 200

plot_1_exemplar = False

W_rec_indirect = []

cc_pre_indirect = []

cc_post_indirect = []

cc_no_pre_indirect = []

cc_indirect = []

session_idx = 0

causal_connectivity_mode = 'groups_to_neurons'

causal_connectivity_neurons_mode = 2

ps_activity_type = 'raw_fl'

ps_computation_type = 'cc'

n_prev_dir_omit = 1

activity_type = 'raw_fl'

ps_activity_type = 'raw_fl'

activity_division = 'split_training'

session_idx = 0

indirect_correlations_init = []

indirect_correlations_final = []

W_init_indirect = []

W_final_indirect = []

change_to_explain = 'post'

causal_connectivity_mode = 'groups_to_neurons'

causal_connectivity_neurons_mode = 2

ps_activity_type = 'raw_fl'

t_window = 60

trial_idx = 0

pretrial_idx = 0

tuning_activity_type = 'dff'

session_idx = 0

trial_idx = 10

neuron_idx = 79

stim_idx = 2

trial_idx = 55

neuron_idx = 79

tuning_activity_type = 'raw_activity'

change_to_explain = 'tuning'

session_idx = 0

delta_pre_to_post_flat = []

delta_post_to_pre_flat = []

init_pre_to_post_flat = []

init_post_to_pre_flat = []

final_pre_to_post_flat = []

final_post_to_pre_flat = []

n_offset = 10

session_idx = 0

session_idx = 0

session_idx = 0

n_smooth = 100

start_idx = 6000

end_idx = 45000

session_idx = 0

n_smooth = 500

pert_times = []

del_keys = ('bci_mean_activity', 'output', 'avg_reward', 'avg_activity',
            'loss_steps', 'W_inp_vals', 'W_rec_vals', 'bci_activity',
            'avg_bci_activity', 'total_rpes', 'mean_stabilize_actvities',
            'sol_step', 'delta_W_inp', 'delta_W_rec', 'output_fl',
            'estimated_tunings', 'avg_reward_stims')

session_idx = 0

save_path_mode = 'raw_path'

save_path = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/omp_example_'

session_idx = 0

compute_rewards = True

perm_mode = 'wm'

tolerance = 0.08

max_perts = 100000

session_idx = 0

pert_idx = 25

stim_idx = 3

session_idx = 0

n_window = 500

def compare_activity(first_activity, second_activity, first_space, second_space,
                     first_label='1st', second_label='2nd', plot_visualization=False):

    assert first_activity.shape == second_activity.shape

    #### PC of each data ####
    first_pca = PCA()
    first_pca.fit(first_activity)

    # first_activity_first_pca = first_pca.transform(first_activity)
    # second_activity_first_pca = first_pca.transform(second_activity)
    first_activity_first_pca = np.matmul(first_activity, first_pca.components_.T)
    second_activity_first_pca = np.matmul(second_activity, first_pca.components_.T)

    second_pca = PCA()
    second_pca.fit(second_activity)

    # first_activity_second_pca = second_pca.transform(first_activity)
    # second_activity_second_pca = second_pca.transform(second_activity)
    first_activity_second_pca = np.matmul(first_activity, second_pca.components_.T)
    second_activity_second_pca = np.matmul(second_activity, second_pca.components_.T)

    #### Projections onto BCI masks ###

    first_activity_first_bci_masks = np.matmul(first_activity, first_space.T)
    second_activity_first_bci_masks = np.matmul(second_activity, first_space.T)

    first_activity_second_bci_masks = np.matmul(first_activity, second_space.T)
    second_activity_second_bci_masks = np.matmul(second_activity, second_space.T)

    # first_proj = np.matmul(np.matmul(first_space.T, np.linalg.inv(np.matmul(
    #                 first_space, first_space.T
    #             ))), first_space) # P_A = A (A.T A)^(-1) A.T (n_neurons, n_neurons)

    # second_proj = np.matmul(np.matmul(second_space.T, np.linalg.inv(np.matmul(
    #                 second_space, second_space.T
    #             ))), second_space) # P_A = A (A.T A)^(-1) A.T (n_neurons, n_neurons)

    if plot_visualization:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

        ax1.scatter(first_activity_first_pca[:, 0], first_activity_first_pca[:, 1],
                    color=c_vals[0], label=first_label, marker='.', alpha=0.1)
        ax1.scatter(second_activity_first_pca[:, 0], second_activity_first_pca[:, 1],
                    color=c_vals[1], label=second_label, marker='.', alpha=0.1)
        ax1.set_xlabel('{} - PC1'.format(first_label))
        ax1.set_ylabel('{} - PC2'.format(first_label))

        ax2.scatter(first_activity_second_pca[:, 0], first_activity_second_pca[:, 1],
                    color=c_vals[0], label=first_label, marker='.', alpha=0.1)
        ax2.scatter(second_activity_second_pca[:, 0], second_activity_second_pca[:, 1],
                    color=c_vals[1], label=second_label, marker='.', alpha=0.1)
        ax2.set_xlabel('{} - PC1'.format(second_label))
        ax2.set_ylabel('{} - PC2'.format(second_label))

        if first_space.shape[0] == 1: # Histogram plots along space
            ax3.hist(first_activity_first_bci_masks[:, 0], bins=50,
                     color=c_vals[0], label=first_label, alpha=0.5)
            ax3.hist(second_activity_first_bci_masks[:, 0], bins=50,
                     color=c_vals[1], label=second_label, alpha=0.5)

            ax4.hist(first_activity_second_bci_masks[:, 0], bins=50,
                     color=c_vals[0], label=first_label, alpha=0.5)
            ax4.hist(second_activity_second_bci_masks[:, 0], bins=50,
                     color=c_vals[1], label=second_label, alpha=0.5)

            ax3.set_xlabel('Proj onto {} BCI mask'.format(first_label))
            ax4.set_xlabel('Proj onto {} BCI mask'.format(second_label))

        for ax in (ax1, ax2, ax3, ax4):
            ax.legend()

import copy

seed = 2018

save_run = True

save_sessions = False

load_run = False

load_sessions = False

save_path_mode = 'raw_path'

save_path = '/scratch/'

scan_type = 'learning_rule_comparison_rl_vs_sl'

n_scan = 10

n_repeats = 4

output_vars = ['output', 'reward', 'avg_reward', 'avg_activity', 'loss_steps', 'total_rpes', 'W_inp_vals', 'W_rec_vals']

metric_name = '(activity - threshold)/std (CN initial)'

import statsmodels.api as sm

def plot_scan_metric(ax, scan_metric, scan_vals=None, color_idx=0, plot_std=False, scatter_raw=False, scatter_raw_connect=False):

    if scan_vals is None:
        scan_vals = np.arange(scan_metric.shape[0])

    mean_scan_metric = np.mean(scan_metric, axis=-1)
    std_scan_metric = np.std(scan_metric, axis=-1) / np.sqrt(scan_metric.shape[-1])
    
    ax.plot(scan_vals, mean_scan_metric, color=c_vals_d[color_idx], zorder=10, linewidth=5.0)
    if plot_std:
        ax.fill_between(scan_vals, mean_scan_metric - std_scan_metric,
                        mean_scan_metric + std_scan_metric, color=c_vals[color_idx],
                        zorder=5, alpha=0.3)

    for scan_idx, scan_val in enumerate(scan_vals):
        if scatter_raw:
            ax.scatter(scan_val * np.ones_like(scan_metric[scan_idx]), scan_metric[scan_idx],
                       color=c_vals[color_idx], marker=None, zorder=-5, alpha=0.3)
    for repeat_idx in range(scan_metric.shape[1]):
        if scatter_raw_connect:
            ax.plot(scan_vals, scan_metric[:, repeat_idx],
                    color=c_vals[color_idx], marker='.', zorder=-1, alpha=0.5)

import copy

seed = 2012

save_run = False

save_sessions = False

load_run = False

load_sessions = False

save_path_mode = 'raw_path'

save_path = '/scratch/'

save_path_mod = 'dyn_thresh_'

scan_type = 'bci_train_speed'

n_scan = 7

n_repeats = 2

output_vars = ['reward', 'session_summary']

from sklearn.linear_model import LogisticRegression

N_INITIAL_TRIAL = 2

start_hit = []

end_hit = []

hit_diff = []

initial_hit_diff = []

fraction_hit_diff = []

later_hit_diff = []

trial_length_start = []

trial_length_end = []

trial_length_diff = []

initial_trial_length_diff = []

fraction_tl_diff = []

later_tl_diff = []

dt_si = 0.05

ax_it = 4

scan_idx = 0

n_bootstrap = 100

elim_outliers = True

OUTLIER_THRESH = 3

out_str_1 = ''

out_str_2 = ''

out_str_3 = ''

out_str_4 = ''

ts_metric_names = ('Pre', 'Early', 'Late', 'Trial', 'Reward',)

ts_metric_names = ('Pre', 'Early', 'Late', 'Trial', 'Reward',)

scan_idx = 0

ts_metrics = ('pre_start', 'post_start_nopre', 'pre_reward', 'post_start', 'post_reward',)

ts_metric_names = ('Pre', 'Early', 'Late', 'Trial', 'Rew')

scan_idx = 0

session_idx = 0

causal_connectivity_mode = 'groups_to_neurons'

ps_activity_type = 'raw_fl'

ps_computation_type = 'post_only'

W_init_indirect_all = []

W_final_indirect_all = []

delta_W_indirect_all = []

cc_init_indirect_all = []

cc_final_indirect_all = []

delta_cc_indirect_all = []

W_to_cc_init_slopes = []

W_to_cc_final_slopes = []

delta_W_to_cc_slopes = []

corr_init_indirect_all = []

corr_final_indirect_all = []

delta_corr_indirect_all = []

corr_to_cc_init_slopes = []

corr_to_cc_final_slopes = []

delta_corr_to_cc_slopes = []

from scipy.stats import sem

ts_metric1 = 'post_start'

ts_metric2 = 'pre_start'

MARKER = '.'

MARKER = '.'

scan_idx = 0

n_trial_avg_window = 10

error_shading_mode = 'mse'

n_max_trials_plot = 100

n_bootstrap = 100

ax_it = 0

ts_metrics = ('pre_start', 'post_start_nopre', 'pre_reward', 'post_start', 'post_reward',)

ts_metric_names = ('Pre', 'Early', 'Late', 'Trial', 'Rew')

ax_it = 0

stim_names = ('uniform', 'tone', 'reward', 'port move')

n_stims = 4

load_path_base = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/'

scan_type = 'bci_lr_scan'

avg_window = 200

plot_type = 'full'

sum_to_idx = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

rpe_slopes = []

delta_W_rec_slopes = []

hebbian_idx_slopes = []

save_path_mode = 'raw_path'

save_path = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/test_bci_activity'

session_idx = 0

load_path = '/content/drive/MyDrive/neuro_research/BCI/saved_nets/test_bci_activity'

task_syncs = ('task_scale', 't_step', 'simple_states', 'noise_type', 'noise_timescale',
              'reward_mode', 'state_mode', 'stim_to_noise_ratio', 'add_fl',)

n_neurons = 200

n_low_dim = 20

task_params = {'t_step': 50}

theta = 0.25

inp_idx = 5

ACTIVITY_SCALE = 1e1

REWARD_SCALE = 1e2

WEIGHT_SCALE = 1e3

session_idx = 0

x_axis_label = 'Time (s)'

n_cells = 3

label = 'not CN examples'

show_trial_starts = False

TICK_START = 100

TICK_END = 300

PRE_SES_START = 0

PRE_SES_END = 100

WINDOW_START = 12

WINDOW_END = 129

session_idx = 0

WEIGHT_SCALE = 1e3

RPE_DEV_SCALE = 1e2

n_offsets = 40

from matplotlib.colors import LinearSegmentedColormap

REWARD_SCALE = 1e3

ACTIVITY_SCALE = 1e1

n_samples = 10000

session_idx = 0

time_ranges = {
    'early': (0, 60,),
    'late': (600, 660,),
}

x_axis_label = 'Time (s)'

n_cells = 5

n_trial_avg_window = 10

early_trial_starts = []

late_trial_starts = []

activity_type = 'raw_fl'

trial_bounds=(-2, 10)

reward_bounds=(-5, 5)

stim_names = ('uniform', 'tone', 'reward', 'port move')

trial_lens_avg = []

ts_metric = 'post_start'

scan_idx = 0

n_trial_avg_window = 10

n_max_trials_plot = 100

error_shading_mode = 'mse'

n_bootstrap = 1000

ts_metrics = ('pre_start', 'post_start_nopre', 'pre_reward', 'post_start', 'post_reward',)

ts_metric_names = ('Pre', 'Early', 'Late', 'Trial', 'Rew')

ts_metric = 'post_start'

SESSION_START = 0

N_TRIALS_CHANGE = 10

MARKER = '.'

n_bins = 7

perc_range = (5, 95)

z_score = True

save_path = '/scratch/'

seed = 2012

n_repeats = 6

session_idx = 0

scan_vals_1 = ('backprop_sl', '3factor_sl', 'backprop', '3factor',)

scan_vals_2_all = []

scan_idx_1 = 1

scan_idx_2 = 4

repeat_idx = 4

lr_scan_2_idxs = (3, 3, 3, 4,)

scan_2_idx = 0

n_smooth = 500

exemplar_scan_2_idx = 4

scan_1_idx = 3

predictors_fit_all = []

W_init_to_cn_all = []

delta_W_in_all = []

delta_W_out_all = []

from sklearn.linear_model import LinearRegression

from scipy.stats import linregress

import statsmodels.api as sm

import time

run = True

minimal = False

seed = 2010

def net_forward(W_inp, input, W_rec, prev_activity, prev_activity_pre_act, net_params):
    ### Network forward pass ###
    W_rec_h = np.matmul(W_rec, prev_activity)

    if net_params['direct_input']: # input already takes into account W_inp
        output_pre_act = np.copy(input + W_rec_h)
    else:
        W_inp_x = np.matmul(W_inp, input)
        output_pre_act = np.copy(W_inp_x + W_rec_h)

    if net_params['leak_type'] in ('activity',):
        leak_term = net_params['leak_term'] * prev_activity
        output = leak_term + (1. - net_params['leak_term']) * act_fn(output_pre_act)
    elif net_params['leak_type'] in ('membrane',):
        leak_term = net_params['leak_term'] * prev_activity_pre_act
        output_pre_act = leak_term + (1. - net_params['leak_term']) * output_pre_act
        output = act_fn(output_pre_act)

    return output, output_pre_act

output_vars = ['output', 'reward', 'avg_reward', 'avg_activity', 'loss_steps',
               'W_inp_vals', 'W_rec_vals', 'bci_activity', 'avg_bci_activity',
               'total_rpes', 'mean_stabilize_actvities', 'sol_step',
               'delta_W_inp', 'delta_W_rec',]

import time

def compute_bci_masks_diff_center_out(bci_masks, task, task_params, verbose=False):

    def get_angle(v1, v2):
        cosine_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        return 180 / np.pi * np.arccos(cosine_angle)

    mean_bci_act_stims = np.matmul(task_params['mean_act_stims'], bci_masks.T)

    speeds = np.linalg.norm(mean_bci_act_stims, axis=-1)

    intuitive_velocities = np.matmul(task_params['mean_act_stims'], task_params['bci_masks'].T)

    angles = np.zeros((task.n_stim,))
    for stim_idx in range(task.n_stim):
        angles[stim_idx] =  get_angle(
            mean_bci_act_stims[stim_idx, :],
            intuitive_velocities[stim_idx, :]
        )

    if verbose:
        print('Speeds:', speeds)
        print('Angles:', angles)

    return speeds, angles

perm_mode = 'om_raw'

pert_found = False

count = 0

max_count = 10000

all_angles = []

all_speeds = []

max_angle = 50

min_angle = 12

max_speed = 3

min_speed = 0.3
