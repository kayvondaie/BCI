import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve

import warnings # Suppress certain annoying warnings

from scipy.stats import linregress

import statsmodels.api as sm
from net_helpers import accumulate_decay

# # For some plotting visualization
# 0 Red, 1 blue, 2 green, 3 purple, 4 orange, 5 teal, 6 gray, 7 pink, 8 yellow
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5','#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e',]
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]


def reproduce_output_fl(output, fl_kernel):
    """
    Since this does not clip early times, this can yield to some transient effects at the start of the sequence.
    """
    output_fl_reproduce = fftconvolve(output, fl_kernel[:, np.newaxis], axes=(0,), mode='full')
    return output_fl_reproduce[:-fl_kernel.shape[0] + 1] # Clip the end

def compute_dff(raw_fl, task_params, f0_CUTOFF=None, verbose=False):
    
    if f0_CUTOFF is None:
        f0_CUTOFF = 0.1 * task_params['task_scale']

    assert len(raw_fl.shape) == 2 # (n_seq, n_neurons)

    f0 = np.percentile(raw_fl, 50, axis=0, keepdims=True) # Median over sequence dimension (1, n_neurons)
    perc_cutoff = np.sum(np.where(f0 < f0_CUTOFF, 1, 0)) / raw_fl.shape[-1]
    if verbose:
        print('Perc. under cutoff: {:.2f}'.format(perc_cutoff))
    f0 = np.where(f0 < f0_CUTOFF, f0_CUTOFF, f0)

    return (raw_fl - f0) / f0, perc_cutoff

def pad_rew_idxs(task_hist):
    """ Adds nans to rew_idxs to keep them in sync with trial_start_idxs """

    trial_starts = task_hist['trial_starts']
    trial_outcomes = task_hist['trial_outcomes'] # Might get padded, so create local version
    
    if len(trial_starts) != len(trial_outcomes):
        if len(trial_starts) == len(trial_outcomes) + 1:
            print('Trial_starts len: {}, trial_outcomes len: {}, padding!'.format(len(trial_starts), len(trial_outcomes)))
            trial_outcomes = (*trial_outcomes, 0) # Pad trial_outcomes with miss, should results in nan for reward below
        else:
            print('Trial_starts len: {}, trial_outcomes len: {}!'.format(len(trial_starts), len(trial_outcomes)))
            raise NotImplementedError('This is an edge case not yet accounted for.')
    padded_rew_idxs = []

    rew_idx_current = 0
    for trial_idx in range(len(trial_starts)):
        if trial_outcomes[trial_idx] == 1:
            padded_rew_idxs.append(task_hist['rewards'][rew_idx_current])
            rew_idx_current += 1
        else:
            padded_rew_idxs.append(np.nan)

    # Make sure we got through all the rewards
    assert rew_idx_current == len(task_hist['rewards'])
    assert len(padded_rew_idxs) == len(trial_starts)

    return padded_rew_idxs

def get_our_bci_aligned_activities(
    activities, task_hist, task_params,
    trial_bounds=(-2, 10), reward_bounds=(-5, 5), pre_tuning_time=None, verbose=False
):
    """
    Computes activity aligned to BOTH trial start or the reward start.
    This is a newer version of "get_our_bci_trial_aligned_activities" more
    similar to how we do it in the experimental analysis.
    
    Note that the task times are the absolute times the transition occurs,
    but if there is a delay the transitions can be further offset because
    it references activity[step_idx - n_delay]. So if n_delay = 5, then
    the trial starts are reacting to what happened 5 time steps ago,
    and thus all activity should be moved up 5 times steps to be correctly
    aligned with the actual activity it influences.

    Pads with nans when things are clipped.
    """
    
    # Clips trials-aligned at next reward, clip reward-aligned at next trial
    CLIP_POST_TUNING = True
    # Clips trial-aligned at last reward, clip reward-aligned at last trial
    CLIP_PRE_TUNING = True 
    
    dt = task_params['t_step'] / 1000 # ms -> s
    
    trial_bounds_steps = np.astype(np.array(trial_bounds) / dt, np.int32)
    reward_bounds_steps = np.astype(np.array(reward_bounds) / dt, np.int32)

    assert len(activities.shape) == 2
    assert activities.shape[0] == task_params['n_steps'] - task_params['n_steps_stabilize'] # This code assumes the stabilizie time has been clipped already
    assert activities.shape[1] == task_params['n_neurons']
    
    n_time_steps = activities.shape[0]
    n_neurons = activities.shape[1]
    
    n_trial_bounds_steps = trial_bounds_steps[1] - trial_bounds_steps[0]
    n_reward_bounds_steps = reward_bounds_steps[1] - reward_bounds_steps[0]

    start_idxs = task_hist['trial_starts']
    rew_idxs = pad_rew_idxs(task_hist) # Adds nan pads to rew_idxs, should be same length as trial start idxs now

    ### TRIAL START ###
    trial_aligned = np.nan * np.ones((len(task_hist['trial_ends']), n_trial_bounds_steps, n_neurons,)) # (n_trials, n_post_tuning, n_neurons)
    trial_aligned_lens = np.nan * np.ones((len(rew_idxs),))
    
    for trial_idx, start_idx in enumerate(start_idxs):

        pre_idx = start_idx + trial_bounds_steps[0] # Often negative, so subtracts
        post_idx = start_idx + trial_bounds_steps[1] 
        
        if pre_idx < task_params['n_steps_stabilize']: # Start of sequence clipping
            pre_idx = task_params['n_steps_stabilize']
        if post_idx > n_time_steps: # End of sequence clipping
            post_idx = n_time_steps
            
        if CLIP_POST_TUNING:
            if post_idx > rew_idxs[trial_idx]: # Note this is False is rew_idx is nan
                post_idx = int(rew_idxs[trial_idx])
        if CLIP_PRE_TUNING and trial_idx > 0: # 1st trial can't be clipped from previous reward
            if pre_idx < rew_idxs[trial_idx-1]:
                pre_idx = int(rew_idxs[trial_idx-1])
        trial_aligned_lens[trial_idx] = post_idx - pre_idx
        rel_start = int(pre_idx - (start_idx + trial_bounds_steps[0])) # If equal, =0, else >0 since pre > trial_bounds_steps[0]
        rel_end = int(rel_start + trial_aligned_lens[trial_idx])
        
        # print('rel_start {}, rel_end {}, pre_idx {}, post_idx {}'.format(rel_start, rel_end, pre_idx, post_idx))
        
        trial_aligned[trial_idx, rel_start:rel_end, :] = activities[pre_idx:post_idx, :]

    ### REWARD ###
    rew_aligned = np.nan * np.ones((len(rew_idxs), n_reward_bounds_steps, n_neurons,)) # (n_trials, n_post_tuning, n_neurons)
    rew_aligned_lens = np.nan * np.ones((len(rew_idxs),))
    
    for trial_idx, rew_idx in enumerate(rew_idxs):
        
        if np.isnan(rew_idx):
            continue
        
        pre_idx = rew_idx + reward_bounds_steps[0] # Often negative, so subtracts
        post_idx = rew_idx + reward_bounds_steps[1] 
        
        if pre_idx < task_params['n_steps_stabilize']: # Start of sequence clipping
            pre_idx = task_params['n_steps_stabilize']
        if post_idx > n_time_steps: # End of sequence clipping
            post_idx = n_time_steps
            
        if CLIP_POST_TUNING and trial_idx < len(rew_idxs) - 1: # Last reward cant be clipped to next trial
            if post_idx > start_idxs[trial_idx+1]:
                post_idx = int(start_idxs[trial_idx+1])
        if CLIP_PRE_TUNING: 
            if pre_idx < start_idxs[trial_idx]:
                pre_idx = int(start_idxs[trial_idx])
        rew_aligned_lens[trial_idx] = post_idx - pre_idx
        rel_start = int(pre_idx - (rew_idx + reward_bounds_steps[0])) # If equal, =0, else >0 since pre > trial_bounds_steps[0]
        rel_end = int(rel_start + rew_aligned_lens[trial_idx])
        
        # print('rel_start {}, rel_end {}, pre_idx {}, post_idx {}'.format(rel_start, rel_end, pre_idx, post_idx))
        
        rew_aligned[trial_idx, rel_start:rel_end, :] = activities[pre_idx:post_idx, :]

    return trial_aligned, rew_aligned, trial_aligned_lens, rew_aligned_lens

N_MIN_TRIALS = 0
N_MAX_TRIALS = 1000 # Set this to a large value for now, since our networks do not become satiated

N_PRE_TRIAL = 40 # Kayvon just hard coded this number in, so use same definition as him for simplicity
N_PRE_TRIAL_END = 20 # End of pre-trial from trial start
N_MAX_TRIAL = 200 # Kayvon just hard coded this number in, so use same definition as him for simplicity
N_PRE_REWARD = 20
N_REWARD_START = 0
N_REWARD_END = 60

def compute_task_metric_aligned_values(
    activity, hist, task_params, tuning_mode=None, mean_mode='time_first', fit_changes=False, 
    trial_subset=None, return_trial_ts_metrics=False, return_task_metric_aligned_responses=False, 
    return_task_metric_stds=False,
):
    """
    This is the model equivalent of experiment's "compute_task_metric_aligned_values".
    
    Computes responses aligned to certain BCI task-relevant metrics.
    
    By default, does this for all period types except for a few that are excluded within.
    
    INPUTS:
    task_metric_type: why type of task metric this is computing, used to properly index F
        - start_pre_post: trial start aligned
    mean_mode: time_first or trials_first
        time_first: Take mean over time, then trials (equal trial weight)
        trials_first: Take mean over trials, then time (upweights long trials)
    trial_response_mode: pre+post or even
        pre+post: Same as tuning, but adds pre and post. Upweights pre since its generally fewer time steps
        even: Even weighting over all trial time steps
    compute_shifts: Whether or not shift corrections to indices need to be computed
    trial_subset: Compute values only a subset of trials, used for control to filter even/odd trials only.
    return_task_metric_alinged_responses: return the raw task metric aligned responses
    
    """
    
    def fit_change_over_trials(task_metric_trials):
        """
        task_metric_trials: (n_neurons, n_trials)
        """
        
        n_neurons = task_metric_trials.shape[0]
        trial_idxs = np.arange(N_MIN_TRIALS, min(N_MAX_TRIALS, task_metric_trials.shape[-1])) # Session could have less than N_MAX_TRIALS
        
        neuron_slopes = np.zeros((n_neurons,))
        neuron_intercepts = np.zeros((n_neurons,))
        neuron_rvalues = np.zeros((n_neurons,))
        neuron_ses = np.zeros((n_neurons,))
        for neuron_idx in range(n_neurons):
            nonnan_mask = ~np.isnan(task_metric_trials[neuron_idx, :]) # Non-nan trials

            nonnan_trial_idxs = trial_idxs[nonnan_mask]

            if nonnan_trial_idxs.shape[0] < 2: # Can't fit if we onlyonly have 0 or 1 point
                neuron_slopes[neuron_idx] = np.nan
                neuron_intercepts[neuron_idx] = np.nan
                neuron_rvalues[neuron_idx] = np.nan
                neuron_ses [neuron_idx] = np.nan
                continue
            else:
                neuron_slopes[neuron_idx], neuron_intercepts[neuron_idx], neuron_rvalues[neuron_idx], pvalue, neuron_ses[neuron_idx] = linregress(
                    nonnan_trial_idxs, task_metric_trials[neuron_idx, nonnan_mask]
                )
        
        return {'slope': neuron_slopes, 'intercept': neuron_intercepts, 
                'r_squared': neuron_rvalues**2, 'se': neuron_ses,}
     
    task_metric_aligned = get_task_metric_aligned_responses(activity, hist, task_params, tuning_mode=tuning_mode)
    
    task_metrics = {}
    ts_extras = {}
    
    # Equivalent to old way, just clearer
    if mean_mode == 'trials_first': # Mean over trials first, then mean over time. Upweights long trials. THIS IS NOT REALLY USED ANYMORE
        raise NotImplementedError('Depricated experiment mode, not implemented in models')
    elif mean_mode == 'time_first': # Mean over each trial's times first, then mean over trials
        ### Special case for tuning/trial_resp calculation ###
        if tuning_mode is not None:            
            if tuning_mode == 'omit_misses':
                pre_key = 'pre_start'
            elif tuning_mode == 'keep_misses':
                pre_key = 'pre_start_all'
            
            # Check if its even possible to compute tuning
            if (task_metric_aligned[pre_key] is None) or (task_metric_aligned['post_start'] is None): # This can happen if there are no occurences of said task metric
                print('Skipping tuning computation, invalid post_start or {}'.format(pre_key))
                task_metrics['tuning'] = None
            else:

                with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    f_pre_trial = np.nanmean(task_metric_aligned[pre_key][:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across pre-reward times
                    f_post_trial = np.nanmean(task_metric_aligned['post_start'][:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across post-reward times

                # Performs mean across trials in all these (n_neurons, n_trials,) -> (n_neurons,)
                with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    task_metrics['tuning'] = np.nanmean(f_post_trial - f_pre_trial, axis=-1) # Tuning   
                if return_trial_ts_metrics:
                    ts_extras['tuning_trials'] = f_post_trial - f_pre_trial       
                if fit_changes:
                    ts_extras['tuning'] = fit_change_over_trials(f_post_trial - f_pre_trial)
        
        ### Special case for reward_tuning ##
        # Check if its even possible to compute reward tuning
        if (task_metric_aligned['pre_reward'] is None) or (task_metric_aligned['post_reward'] is None): # This can happen if there are no occurences of said task metric
            print('Skipping tuning computation, invalid post_reward or pre_reward')
            task_metrics['tuning'] = None
        else:
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                f_pre_reward = np.nanmean(task_metric_aligned['pre_reward'][:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across pre-reward times
                f_post_reward = np.nanmean(task_metric_aligned['post_reward'][:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # Mean across post-reward times

            # Important to take sum/difference before nanmean, so if pre-/post-trial response is nan, tuning for trial is nan
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                task_metrics['reward_tuning'] = np.nanmean(f_post_reward - f_pre_reward, axis=-1) # Mean across trials
            if return_trial_ts_metrics:
                ts_extras['reward_tuning_trials'] = f_post_reward - f_pre_reward
            if fit_changes:             
                ts_extras['reward_tuning'] = fit_change_over_trials(f_post_reward - f_pre_reward)
        
        ### Now that special cases are covered, do everything as usual ###
        # All period keys by default
        task_metric_names = list(task_metric_aligned.keys())
        if 'start_aligned' in task_metric_names:
            task_metric_names.remove('start_aligned')
        
        if return_task_metric_stds: # NOTE THIS IS DIFFERENT THAN HOW WE PACKAGE THIS IN THE DATA ANALYSIS
            ts_extras['ts_metric_stds'] = {}
            
        for task_metric_name in task_metric_names:
            task_metric_aligned_name = task_metric_aligned[task_metric_name]
            
            if task_metric_aligned_name is None: # This can happen if there are no occurences of said task metric
                task_metrics[task_metric_name] = None
                continue
            
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                mean_task_metric_aligned = np.nanmean(task_metric_aligned_name[:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) 
            
            # Some filters for computing control metrics where only even or odd trials are chosen 
            if trial_subset in ('even', 'odd',):
                n_trials_eval = mean_task_metric_aligned.shape[-1]
                trial_idxs_filter = np.arange(n_trials_eval)
                if trial_subset == 'even':
                    trial_idxs_filter = trial_idxs_filter[0::2]
                elif trial_subset == 'odd':
                    trial_idxs_filter = trial_idxs_filter[1::2]

                print('Filtering trials for control!!')
                mean_task_metric_aligned = mean_task_metric_aligned[:, trial_idxs_filter]
            elif trial_subset is not None:
                raise ValueError('Trial_subset {} not recognized.'.format(trial_subset))
                
            # Performs mean across trials (n_neurons, n_trials,) -> (n_neurons,)
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                warnings.simplefilter('ignore', category=RuntimeWarning)
                task_metrics[task_metric_name] = np.nanmean(mean_task_metric_aligned, axis=-1) 
            
            if return_trial_ts_metrics:
                ts_extras[task_metric_name + '_trials'] = mean_task_metric_aligned
            if return_task_metric_stds:
                with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they just return nans as expected
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    # ts_extras[task_metric_name + '_stds'] = np.nanstd(task_metric_aligned_name[:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) 
                    ts_extras['ts_metric_stds'][task_metric_name] = np.nanstd(task_metric_aligned_name[:, :, N_MIN_TRIALS:N_MAX_TRIALS], axis=0) # NEW CONVENTION OF STORING THIS FOR MODELS
                    
            if return_task_metric_aligned_responses: # This takes up a lot of memory
                ts_extras[task_metric_name + '_responses'] = task_metric_aligned_name[:, :, N_MIN_TRIALS:N_MAX_TRIALS]
            
            if fit_changes:             
                ts_extras[task_metric_name] = fit_change_over_trials(mean_task_metric_aligned)
        
        return task_metrics, ts_extras 
    else:
        raise ValueError('Mean mode {} not recognized.'.format(mean_mode))

def get_task_metric_aligned_responses(activity, hist, task_params, tuning_mode=None, verbose=False):
    """
    This is the model equivalent of experiment's "get_task_metric_aligned_responses".
    
    This builds task-metric-aligned response arrays to be used for computing mean/change in various types of responses.
    
    Wraps around the "divide_df_closedloop_into_periods" function and then uses the various periods to construct the
    task-metric-aligned arrays.
    
    By default, creates task_metric_aligned responses for all period types except for a few that are excluded below.
    
    INPUTS:
    - tuning_mode: keep_misses or omit_misses
    
    OUTPUTS: 
    - task_metric_aligned: dictionary of the task-metric-aligned arrays
    
    """
    period_divisions = divide_activity_into_periods(activity, hist, task_params, verbose=verbose)
    
    df_closedloop = activity # Convenient reuse of a variable from experiment
    
    n_trials = len(period_divisions['post_start'])
    n_neurons = df_closedloop.shape[-1]
    
    task_metric_aligned = {}
      
    ### Special case for trial start tuning ###
    if tuning_mode is not None: # Constructs one F that contains both pre_start and post_start
        start_aligned = np.empty((N_PRE_TRIAL + N_MAX_TRIAL, n_neurons, n_trials))
        start_aligned[:] = np.nan

        for trial_idx in range(n_trials):
            if tuning_mode == 'omit_misses':
                pre_key = 'pre_start'
            elif tuning_mode == 'keep_misses':
                pre_key = 'pre_start_all'
            else:
                raise ValueError('Tuning_mode {} not recognized'.format(tuning_mode))
                
            if len(period_divisions[pre_key][trial_idx]) > 0: # This can still be zero for keep_misses because of first trial                  
                start_aligned[:N_PRE_TRIAL-N_PRE_TRIAL_END, :, trial_idx] = df_closedloop[period_divisions[pre_key][trial_idx], :]
            n_post_start = len(period_divisions['post_start'][trial_idx])
            start_aligned[N_PRE_TRIAL:N_PRE_TRIAL+n_post_start, :, trial_idx] = df_closedloop[period_divisions['post_start'][trial_idx], :]
            
        task_metric_aligned['start_aligned'] = start_aligned
        
    # All period keys except 'other' because not separated by trials
    task_metric_names = list(period_divisions.keys())
    task_metric_names.remove('other')
    # These are often so spare that they're not worth constructing
    task_metric_names.remove('post_start_extra')
    # task_metric_names.remove('miss')
    
    for task_metric_name in task_metric_names:
        # Find maximum length of this particular task_metric
        max_length = 0
        for trial_idx in range(n_trials):
            if len(period_divisions[task_metric_name][trial_idx]) > max_length:
                max_length = len(period_divisions[task_metric_name][trial_idx])
        # print('{} max_length = {}'.format(task_metric_name, max_length))
        if max_length == 0:
            print('All trials of type {} are length 0!'.format(task_metric_name))
            task_metric_aligned[task_metric_name] = None
        else:
            # Initialize and fill
            task_metric_aligned_name = np.nan * np.ones((max_length, n_neurons, n_trials))
            for trial_idx in range(n_trials):
                n_idxs_trial = len(period_divisions[task_metric_name][trial_idx])
                task_metric_aligned_name[:n_idxs_trial, :, trial_idx] = df_closedloop[period_divisions[task_metric_name][trial_idx], :]

            task_metric_aligned[task_metric_name] = task_metric_aligned_name
        
    return task_metric_aligned

def divide_activity_into_periods(activity, hist, task_params, omit_evaluation=True, verbose=False):
    """
    This is the model equivalent of experiment's "divide_df_closedloop_into_periods".
    
    INPUTS:
    - omit_evaluation: Remove trials that fall into the evaluation period
    
    Assign all times of the BCI task to distinct periods based on task-relevant stimuli that occur
    around said periods. This is done by going through all time steps of the BCI task, so can be
    relatively slow compared to going through all trials and constructing it from that.
    
    MUTUALLY EXCLUSIVE DIVISIONS:
    post_trial_start: times immediately after trial start until next reward OR N_MAX_TRIAL has elapsed
    pre_trial_start_me: times immediately before trial start AFTER a hit trial, cannot have overlap with post_reward/post_reward_high_act
    pre_trial_start_gap: pretrial times before a trial start not captured by pre_trial_start_me, between N_PRE_TRIAL_END and trial start
    post_reward_idxs: times immediately after reward licking begins on a hit trial, takes precidence over pre_trial
    post_reward_high_act_idxs: times after post_reward_idxs and before pretrial times have started
    miss_idxs: times after trial time has expired on miss trials
    other: times that occur before the first trial start or after the final trial reward because unknown when next trial start began
    
    NON-MUTUALLY EXCLUSIVE DIVISIONS:
    pre_trial_start_idxs: times immediately before trial start AFTER a hit trial, specifically between N_PRE_TRIAL and N_PRE_TRIAL_END, may have
        overlap with post_reward/post_reward_high_act
    pre_trial_start_all_idxs: same as above, but after EVERY trial instead of just hit trials
    post_trial_start_rew_idxs: same as post_start, but also includes times until next tiral start. Kayvon's old definition of post trial start response
    pre_reward_idxs: Time immediately before reward. What we call "late trial".
    post_start_nopre: Trial idxs that do not include the 'pre_reward_idxs'. What we call "early trial".
    trial_idxs: All idxs within each trial, from start to beginning of next trial
    """
    

    df_closedloop = activity # Convenient reuse of a variable from experiment
    n_times = df_closedloop.shape[0]
    
    trial_start_idxs = np.array(hist['trial_starts'])
    rew_idxs = np.array(pad_rew_idxs(hist))
    
    if task_params['n_steps_evaluate'] > 0 and omit_evaluation:
        step_idx_eval = task_params['n_steps'] - task_params['n_steps_evaluate']
        eval_mask = np.logical_or(trial_start_idxs >= step_idx_eval, rew_idxs >= step_idx_eval)
        trial_start_idxs = trial_start_idxs[~eval_mask]
        rew_idxs = rew_idxs[~eval_mask]
        print('Removed {}/{} trials for evaluation.'.format(np.sum(eval_mask), len(eval_mask))) 
        n_times = df_closedloop.shape[0] - task_params['n_steps_evaluate'] # Indexes don't go into evaluation period
            
    # Assumes activity has already been clipped to accomodate for stabilization, so needs to do same for trial_start and rew_idxs
    trial_start_idxs = trial_start_idxs - task_params['n_steps_stabilize']
    rew_idxs = rew_idxs - task_params['n_steps_stabilize']
    
    n_trials = trial_start_idxs.shape[0]
    
    # This set of indices are mutually exclusive
    post_trial_start_idxs = []
    post_trial_start_extra_idxs = [] # On hit trials, can go beyond N_MAX_TRIAL, so collect extras in here
    pre_trial_start_me_idxs = [[],] # There is never a pre-trial start for first trial, so always start with one empty.
    pre_trial_start_gap_idxs = [[],] # There is never a pre-trial start for first trial, so always start with one empty.
    post_reward_idxs = []
    post_reward_high_act_idxs = []
    miss_idxs = []
    other_idxs = []
    
    # These indices overlap with many of the previous
    pre_trial_start_idxs = [[],] # There is never a pre-trial start for first trial, so always start with one empty.
    pre_trial_start_all_idxs = [[],] # There is never a pre-trial start for first trial, so always start with one empty.
    post_trial_start_rew_idxs = [] 
    pre_reward_idxs = []
    post_trial_start_nopre_idxs = [] # Post trial start, with pre-reward steps excluded
    trial_idxs = [] # All idxs within each trial, from start to beginning of next trial
    
    # Initialize this as not in any trial to begin with just in case first trial start takes some time
    trial_idx = -1
    
    for time_idx in range(n_times):
        if trial_idx < n_trials - 1: 
            if time_idx == trial_start_idxs[trial_idx + 1]: # Started next trial, append to all except pre_trial since handled below
                trial_idx += 1
                
                post_trial_start_idxs.append([])
                post_trial_start_extra_idxs.append([])
                post_reward_idxs.append([])
                post_reward_high_act_idxs.append([])
                miss_idxs.append([])
                
                post_trial_start_rew_idxs.append([])
                pre_reward_idxs.append([])
                post_trial_start_nopre_idxs.append([])
                trial_idxs.append([])

        if trial_idx < 0:
            other_idxs.append(time_idx)
            continue
        else:
            trial_idxs[trial_idx].append(time_idx)
            
            # Adds to pre_trials, which are special because they occur before the trial start
            if trial_idx < n_trials - 1: # If its the last trial, no need because no pre-trial defined anyway
                if len(pre_trial_start_idxs) == trial_idx + 1: # for trial_idx = 1 (second trial), needs to be length 2
                    pre_trial_start_idxs.append([])
                    pre_trial_start_me_idxs.append([])
                    pre_trial_start_gap_idxs.append([])
                    pre_trial_start_all_idxs.append([])
            
            if ~np.isnan(rew_idxs[trial_idx]): # Reward in this trial
                
                if trial_idx < n_trials - 1: # Add pre-trial to the NEXT trial, because this was a hit. No need if last trial
                    if time_idx >= trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL and time_idx < trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL_END: # Next minus pre-trial
                        pre_trial_start_idxs[trial_idx+1].append(time_idx)
                        pre_trial_start_all_idxs[trial_idx+1].append(time_idx)
                        # No continue here because this can be repetitive with the other metrics, so still need to compute below
                
                if time_idx >= trial_start_idxs[trial_idx]: # Old post definition
                    if trial_idx < n_trials - 1:
                        if time_idx < trial_start_idxs[trial_idx] + N_MAX_TRIAL and time_idx < trial_start_idxs[trial_idx + 1]: # Not beyond and not into next trial
                            post_trial_start_rew_idxs[trial_idx].append(time_idx)
                            # No continue here because this can be repetitive with the other metrics, so still need to compute below
                    else: # Final trial case
                        if time_idx < trial_start_idxs[trial_idx] + N_MAX_TRIAL:
                            post_trial_start_rew_idxs[trial_idx].append(time_idx)
                            # No continue here because this can be repetitive with the other metrics, so still need to compute below
                
                if time_idx >= rew_idxs[trial_idx] - N_PRE_REWARD  and time_idx < rew_idxs[trial_idx]:
                    pre_reward_idxs[trial_idx].append(time_idx)
                    # No continue here because this can be repetitive with the other metrics, so still need to compute below
                
                # In trial (after trial start, before reward), before N_MAX_TRIAL
                if time_idx >= trial_start_idxs[trial_idx] and time_idx < rew_idxs[trial_idx] and time_idx < trial_start_idxs[trial_idx] + N_MAX_TRIAL:
                    if time_idx < rew_idxs[trial_idx] - N_PRE_REWARD:
                        post_trial_start_nopre_idxs[trial_idx].append(time_idx)
                        # No continue here because this can be repetitive with the other metrics, so still need to compute below
                    
                    post_trial_start_idxs[trial_idx].append(time_idx)
                    continue
                # In trial (after trial start, before reward), after N_MAX_TRIAL
                elif time_idx >= trial_start_idxs[trial_idx] + N_MAX_TRIAL and time_idx < rew_idxs[trial_idx]: # In trial, but beyond N_MAX_TRIAL
                    post_trial_start_extra_idxs[trial_idx].append(time_idx)
                    continue
                # In reward (after reward, before next trial start), before N_REWARD_END
                elif time_idx >= rew_idxs[trial_idx] and time_idx < rew_idxs[trial_idx] + N_REWARD_END: # In reward
                    if time_idx < rew_idxs[trial_idx] + N_REWARD_START: 
                        # Don't collect these indices yet, but to keep mutually exclusive still continue
                        continue
                    elif time_idx >= rew_idxs[trial_idx] + N_REWARD_START:
                        post_reward_idxs[trial_idx].append(time_idx)
                        continue
                # After reward (after N_REWARD_END, before next trial start)
                elif time_idx >= rew_idxs[trial_idx] + N_REWARD_END: # Post-reward
                    if trial_idx < n_trials - 1: 
                        if time_idx >= trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL and time_idx < trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL_END: # Next minus pre-trial
                            pre_trial_start_me_idxs[trial_idx+1].append(time_idx)
                            continue
                        elif time_idx >= trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL_END: # Next minus pre-trial
                            pre_trial_start_gap_idxs[trial_idx+1].append(time_idx)
                            continue
                        else:
                            post_reward_high_act_idxs[trial_idx].append(time_idx)
                            continue
                    else: # Final trial case, just terminate since unknown if low activity from pre or not
                        other_idxs.append(time_idx)
                        continue
                else:
                    raise ValueError('This shouldnt happen!')
                    
                    
            else: # Miss trial, no reward, only trial and other
                if trial_idx < n_trials - 1:
                    if time_idx >= trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL and time_idx < trial_start_idxs[trial_idx + 1] - N_PRE_TRIAL_END: # Next minus pre-trial
                        pre_trial_start_all_idxs[trial_idx+1].append(time_idx)
                        # No continue here because this can be repetitive with the other metrics, so still need to compute below
                
                if time_idx >= trial_start_idxs[trial_idx] and time_idx < trial_start_idxs[trial_idx] + N_MAX_TRIAL:
                    post_trial_start_idxs[trial_idx].append(time_idx)
                    post_trial_start_nopre_idxs[trial_idx].append(time_idx)
                    post_trial_start_rew_idxs[trial_idx].append(time_idx)
                    continue
                else: # Beyond normal trial times
                    miss_idxs[trial_idx].append(time_idx)
                    continue

        raise ValueError('This shouldnt happen!')
        
    return {
        # Mutually exclusive measures
        'post_start': post_trial_start_idxs,
        'post_start_extra': post_trial_start_extra_idxs,
        'pre_start_me': pre_trial_start_me_idxs,
        'pre_start_gap': pre_trial_start_gap_idxs,
        'post_reward': post_reward_idxs,
        'post_reward_high_act': post_reward_high_act_idxs,
        'miss': miss_idxs,
        'other': other_idxs,
        
        # Measures with potential overlap with others
        'pre_start': pre_trial_start_idxs,
        'pre_start_all': pre_trial_start_all_idxs,
        'post_start_rew': post_trial_start_rew_idxs,
        'pre_reward': pre_reward_idxs,
        'post_start_nopre': post_trial_start_nopre_idxs,
        'trial': trial_idxs,
    }

def get_our_bci_trial_aligned_activities(
    activities, task, session_idx, tuning_type='trial',
    post_tuning_time=None, pre_tuning_time=None, verbose=False
):
    """
    Computes activity aligned to a particular trial event, e.g. the trial
    start or the reward start. Note the way this computes post and pre is quite
    a bit different than how we compute trial start tuning in bci_1: the post terminates
    only at the next trial and the pre is included on miss trials too.
    
    Note that the task times are the absolute times the transition occurs,
    but if there is a delay the transitions can be further offset because
    it references activity[step_idx - n_delay]. So if n_delay = 5, then
    the trial starts are reacting to what happened 5 time steps ago,
    and thus all activity should be moved up 5 times steps to be correctly
    aligned with the actual activity it influences.

    Pads with nans when things are clipped.
    """

    POST_TUNING_TIME = 10000 # ms
    PRE_TUNING_TIME = 2000 # ms
    CLIP_POST_TUNING = True
    CLIP_PRE_TUNING = False

    assert len(activities.shape) == 2

    if post_tuning_time is None:
        post_tuning_time = np.copy(POST_TUNING_TIME)
    if pre_tuning_time is None:
        pre_tuning_time = np.copy(PRE_TUNING_TIME)

    n_post_tuning = int(post_tuning_time / (task.dt * 1000))
    n_pre_tuning = int(pre_tuning_time / (task.dt * 1000))
    n_seq = activities.shape[0]
    n_cells = activities.shape[1]

    task_hist = task.hists[session_idx]

    if tuning_type in ('trial',):
        n_trials = len(task_hist['trial_ends']) # May lose one incomplete trial at end
        start_idxs = task_hist['trial_starts']
    elif tuning_type in ('reward',):
        n_trials = len(task_hist['rewards'][:-1]) # Discard the last one just to avoid clipping
        start_idxs = task_hist['rewards']
    else:
        raise ValueError('Tuning type {} not recoginized.'.format(tuning_type))

    post_activities = np.nan * np.ones((n_trials, n_post_tuning, n_cells,)) # (n_trials, n_post_tuning, n_cells)
    pre_activities = np.nan * np.ones((n_trials, n_pre_tuning, n_cells,)) # (n_trials, n_pre_tuning, n_cells)
    tuning_lengths = np.zeros((n_trials,))

    for trial_idx in range(n_trials):

        trial_seq_start = start_idxs[trial_idx]

        # Trial specific n_pre and n_post to be modified
        n_post_trial = np.copy(n_post_tuning)
        n_pre_trial = np.copy(n_pre_tuning)

        if CLIP_PRE_TUNING and trial_idx > 0: # Clipping for overlap with previous trial
            if trial_seq_start - n_pre_trial < start_idxs[trial_idx-1]:
                n_pre_trial = trial_seq_start - (start_idxs[trial_idx-1])
        elif trial_seq_start - n_pre_trial < 0: # Clipping for first trial
            n_pre_trial =  trial_seq_start

        if CLIP_POST_TUNING and trial_idx < n_trials - 1:
            if trial_seq_start + n_post_trial > start_idxs[trial_idx+1]: # Overlap into next trial
                n_post_trial =  start_idxs[trial_idx+1] - trial_seq_start
                if verbose: print('Clipped post trial {} -> {}'.format(n_post_tuning, n_post_trial))
        elif trial_seq_start + n_post_trial > n_seq: # Clipping for last trial
            n_post_trial = n_seq - trial_seq_start
            if verbose: print('Clipped final trial {} -> {}'.format(n_post_tuning, n_post_trial))

        tuning_lengths[trial_idx] = n_pre_trial + n_post_trial
        post_activities[trial_idx, :n_post_trial, :] = (
            activities[np.newaxis, trial_seq_start:trial_seq_start + n_post_trial, :]
        ) # (1, n_post_trial, n_cells)
        pre_activities[trial_idx, -n_pre_trial:, :] = (
            activities[np.newaxis, trial_seq_start - n_pre_trial:trial_seq_start, :]
        ) # (1, n_post_trial, n_cells)
               
    return post_activities, pre_activities, tuning_lengths

def get_our_bci_tuning(post_activities, pre_activities):
    return np.nanmean(post_activities, axis=1) - np.nanmean(pre_activities, axis=1)

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

def compute_eligibility_approx(
    post_activity, pre_activity, train_params, elig_type='hebb', n_baseline=None, n_elig=None
):
    """
    Builds an approximation of the eligibility trace, given some sort of activity and specifications 
    for what that eligibility should look like

    INPUTS:
     - pre_activity, shape: (n_steps, n_presyn), note this can be either raw activity or fluorescence
     - post_activity, shape: (n_steps, n_neurons), note this can be either raw activity or fluorescence
     - elig_type
         - hebb: post * pre, no running average 
         - dpost_pre: (post - <post>) * pre, no running average
    
         - dpost_pre_acc: (post - <post>) * pre, running average eligibility
     - n_baseline: window size of mean post or pre activity
     - n_elig: window size of eligibility accumulation
    OUTPUTS:
    - eligibility, shape: (n_steps, n_neurons, n_presyn)), estimate of eligibility trace from activity.
    """
    
    assert pre_activity.shape[0] == post_activity.shape[0]
    n_steps = pre_activity.shape[0]
    n_presyn = pre_activity.shape[-1]
    n_neurons = post_activity.shape[-1]

    eligibility = np.nan * np.ones((n_steps, n_neurons, n_presyn))

    if n_baseline is None: # If not passed, defaults to actual value used in training
        n_baseline = train_params['n_window_baseline']
    if n_elig is None: # If not passed, defaults to actual value used in training
        n_elig = train_params['n_window_elig']
        
    if elig_type in ('dpost_pre', 'dpost_dpre', 'dpost_pre_acc',): # Requires mean post activity
        mean_post_activity = np.nan * np.ones_like(post_activity)
        for step_idx in range(n_steps):
            prev_post = mean_post_activity[step_idx-1] if step_idx > 0 else np.zeros_like(post_activity[step_idx])
            mean_post_activity[step_idx] = accumulate_decay(prev_post, post_activity[step_idx], n_window=n_baseline)
    if elig_type in ('post_dpre', 'dpost_dpre',): # Requires mean pre activity
        mean_pre_activity = np.nan * np.ones_like(pre_activity)
        for step_idx in range(n_steps):
            prev_pre = mean_pre_activity[step_idx-1] if step_idx > 0 else np.zeros_like(pre_activity[step_idx])
            mean_pre_activity[step_idx] = accumulate_decay(prev_pre, pre_activity[step_idx], n_window=n_baseline)
    
    if elig_type in ('hebb'):
        eligibility[:] = np.einsum('ij, ik -> ijk', post_activity, pre_activity) 
    elif elig_type in ('dpost_pre'):
        eligibility[:] = np.einsum('ij, ik -> ijk', post_activity - mean_post_activity, pre_activity) 
    elif elig_type in ('post_dpre'):
        eligibility[:] = np.einsum('ij, ik -> ijk', post_activity, pre_activity - mean_pre_activity) 
    elif elig_type in ('dpost_dpre'):
        eligibility[:] = np.einsum('ij, ik -> ijk', post_activity - mean_post_activity, pre_activity - mean_pre_activity) 
    elif elig_type in ('dpost_pre_acc'):
        inst_eligibility = np.einsum('ij, ik -> ijk', post_activity - mean_post_activity, pre_activity) 
        for step_idx in range(n_steps):
            prev_elig = eligibility[step_idx-1] if step_idx > 0 else np.zeros_like(eligibility[step_idx])
            eligibility[step_idx] = accumulate_decay(prev_elig, inst_eligibility[step_idx], n_window=n_elig)
        del inst_eligibility
    else:
        raise NotImplementedError('Elig_type {} not recognized.'.format(elig_type))
    
    return eligibility

def compute_local_hebbian_indexes(
    train_outputs, params, n_divisions=20, elig_types=('true',), return_elig_divs=False, run_mlr=False):
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

    # Divide indexes of loss/weight values into equal sizes
    loss_step_divs = np.linspace(0, len(train_outputs['loss_steps'])-1, n_divisions+1).astype(np.int32)

    div_step_idxs = []

    delta_W_divs = np.zeros((n_divisions, task_params['n_neurons'], n_presyn)) # Change in W over each division
    rpes_divs = np.zeros((n_divisions,)) # Total RPE in each division

    # Different eligibility types
    n_elig_types = len(elig_types)
    elig_divs = np.zeros((n_elig_types, n_divisions, task_params['n_neurons'], n_presyn)) # Sum of each elig in each division
    elig_full = np.zeros((n_elig_types, task_params['n_neurons'], n_presyn)) # Across full session, so no divisions

    ### Full session values for entire session Hebbian index ###
    idx_start = loss_step_divs[0]
    idx_end = loss_step_divs[-1]
    delta_W = W_vals[idx_end] - W_vals[idx_start]

    # Translate the loss steps 
    step_idx_start = train_outputs['loss_steps'][idx_start + 1] - train_params['n_steps_per_loss'] + 1
    step_idx_end = train_outputs['loss_steps'][idx_end] + 1

    # Hebbian index fits, eligibility to change in weights
    div_slopes = np.zeros((n_elig_types, n_divisions,)) # Elig in division to weight change in division
    div_slopes_full_delta = np.zeros((n_elig_types, n_divisions,)) # Elig in division to total weight change
    full_slopes = np.zeros((n_elig_types,)) # Over entire session

    for elig_type_idx, elig_type in enumerate(elig_types):
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
    for elig_type_idx, elig_type in enumerate(elig_types):

        # Compute eligibility for the relevant type
        if elig_type in ('hebb', 'dpost_pre', 'post_dpre', 'dpost_dpre', 'dpost_pre_acc',):
            eligibility = compute_eligibility_approx(output, output, train_params, elig_type=elig_type)
        elif elig_type not in ('true',): # Already saved in this place, so no need to compute.
            raise ValueError('Eligibility type {} not recognized!'.format(elig_type))

        for division_idx in range(n_divisions):
            # Start and end of division in loss steps
            idx_start = loss_step_divs[division_idx]
            idx_end = loss_step_divs[division_idx+1]

            # Translate loss steps intro raw steps, idx_start + 1 because the first loss_steps idx is 0
            step_idx_start = train_outputs['loss_steps'][idx_start + 1] - train_params['n_steps_per_loss'] + 1
            step_idx_end = train_outputs['loss_steps'][idx_end] + 1

            # print('Loss idxs: {} to {}\tStep idxs: {} to {}'.format(idx_start, idx_end, step_idx_start, step_idx_end))

            if elig_type_idx == 0: # These things are not eligibility dependent, so do only on first pass through
                delta_W_divs[division_idx] =  W_vals[idx_end] - W_vals[idx_start]
                div_step_idxs.append(step_idx_start)
                rpes_divs[division_idx] = np.sum(np.array(total_rpes[idx_start+1:idx_end+1]))
                
            if elig_type == 'true': 
                # Division relevant eligibility and RPE
                W_elg_trunc = W_elg[idx_start+1:idx_end+1, :, :] # +1 because first entry is all zeros
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

        for elig_type_idx, elig_type in enumerate(elig_types):
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
            
    return rpes_divs, div_slopes, div_slopes_full_delta, full_slopes, elig_divs, div_slopes_mlr


def get_hebbian_idx(train_outputs, params, n_bins=10, corr_activity_type='dff', 
                    mode='neurons_to_neurons', passed_delta_connectivity=None, group_conversion=None,
                    plot_visualization=False, step_idxs=None, pre_post_offset=True,
                    delta_connectivity_clip=None, normalize_connectivity=False):
    """
    For a given session, computes the Hebbian idx by fitting various measures
    of neuronal correlation to the change in connectivity between the two neurons.
    The change in connectivity can be either the actual weights of a recurrent n
    network or the output of a causal connectivity measure. Additionally, the 
    Hebbian index computation can be done either in group space (for comparing 
    to experiment) or in neuron space (for model comparison).

    INPUTS:
    mode: neurons_to_neurons or groups_to_neurons - determines whether or not 
        Hebbian idx is computed across (n_neurons, n_neurons,) OR (n_neurons, n_groups,)
    delta_W_rec: Compute with respect to passed delta_W_rec, used if Hebbian_idx
        is computed for a particular change in W_rec that doesn't correspond to
        that in train_outputs['delta_W_rec']
    step_idxs: Compute with respect to corresponding step_idxs, used to analyze
        Hebbian idx for a particular subset of times

    delta_connectivity_clip: None or float - excludes all connectivies whose absolute
        value is larger than this
    """

    task_params, train_params, net_params = params

    if corr_activity_type in ('raw_activity',):
        output = train_outputs['output']
    elif corr_activity_type in ('raw_fl',):
        if 'output_fl' not in train_outputs:
            train_outputs['output_fl'] = reproduce_output_fl(
                train_outputs['output'], task_params['fl_kernel']
            )
        output = train_outputs['output_fl']
    elif corr_activity_type in ('dff',):
        print('Using dff for correlations')
        if 'output_fl' not in train_outputs:
            train_outputs['output_fl'] = reproduce_output_fl(
                train_outputs['output'], task_params['fl_kernel']
            )
        if 'output_dff' not in train_outputs:
            train_outputs['output_dff'], train_outputs['output_dff_perc_cutoff_ps'] = compute_dff(
                train_outputs['output_fl'], task_params, verbose=False
            )
        output = train_outputs['output_dff']

    if step_idxs is None: # Full current session by default (not including stabilization)
        output_trunc = output[task_params['n_steps_stabilize']:]
    else: # Local part of time steps
        output_trunc = output[step_idxs[0]:step_idxs[1]]
        # print('Clipping from {} to {}'.format(step_idxs[0], step_idxs[1]))
   
    if pre_post_offset: # This ends up having an overall pretty minor effect
        # Offset by one for (post <- pre)
        output_corrs = custom_corrcoef(output_trunc.T[:, 1:], output_trunc.T[:, :-1]) # (n_neurons, n_neurons)
    else:
        output_corrs = custom_corrcoef(output_trunc.T, output_trunc.T) # (n_neurons, n_neurons)

    if mode in ('groups_to_neurons',): # This needs to be directly passed in this case
        assert passed_delta_connectivity is not None
        assert group_conversion is not None
        delta_connectivity = np.copy(passed_delta_connectivity)

        assert delta_connectivity.shape == (task_params['n_neurons'], task_params['n_groups'],)

        # Convert output_corrs into shape (n_neurons, n_groups) <- (n_neurons, n_neurons) x (n_neurons, n_groups)
        output_corrs = np.matmul(output_corrs, group_conversion)

    elif mode in ('neurons_to_neurons',):
        if passed_delta_connectivity is None:
            delta_connectivity = train_outputs['delta_W_rec'] # (n_neurons, n_neurons), already (post, pre)
        else:
            delta_connectivity = np.copy(passed_delta_connectivity)

        assert delta_connectivity.shape == (task_params['n_neurons'], task_params['n_neurons'],)

    assert delta_connectivity.shape == output_corrs.shape

    output_corrs_flat = []
    output_corrs_sum = np.zeros((task_params['n_neurons'],))
    delta_connectivity_flat = []
    delta_connectivity_sum = np.zeros((task_params['n_neurons'],))
    for neuron_idx in range(output_corrs.shape[0]):
        for input_idx in range(output_corrs.shape[1]):
            if mode in ('neurons_to_neurons',) and neuron_idx == input_idx: # Skip diagonals
                continue
            if np.isnan(output_corrs[neuron_idx, input_idx]):
                continue
            if delta_connectivity_clip is not None:
                if np.abs(delta_connectivity[neuron_idx, input_idx]) > delta_connectivity_clip:
                    continue
            output_corrs_flat.append(output_corrs[neuron_idx, input_idx])
            delta_connectivity_flat.append(delta_connectivity[neuron_idx, input_idx])
            output_corrs_sum[neuron_idx] += output_corrs[neuron_idx, input_idx]
            delta_connectivity_sum[neuron_idx] += delta_connectivity[neuron_idx, input_idx]

    if normalize_connectivity: # Normalize connectivity, making units arbitrary
        print('Normalizing connectivity.')
        if np.max(np.abs(delta_connectivity_flat)) > 0.: # Avoids case where connectivity hasn't changed
            delta_connectivity_flat = delta_connectivity_flat / np.max(np.abs(delta_connectivity_flat)) # Normalizes so maximum is 1

    # print('Max connectivity: {}'.format(np.max(np.abs(delta_connectivity_flat))))

    assert not np.isnan(output_corrs_flat).any()
    assert not np.isnan(delta_connectivity_flat).any()

    # print('Output corrs mean: {:.1e}'.format(np.mean(output_corrs_flat)))
    # test_slope, _, _, _, _ = linregress( # Linear regression over bins with elements
    #     output_corrs_flat,
    #     delta_connectivity_flat,
    # )
    # print('Corrs - delta_influence slope: {:.1e}'.format(test_slope))
    # # Different ways to compute the Hebbian index
    # corr_coef = np.corrcoef(output_corrs_flat, delta_connectivity_flat)[0, 1]
    # corr_coef = np.corrcoef(output_corrs_sum, delta_connectivity_sum)[0, 1]
    # if np.linalg.norm(delta_connectivity_flat) > 0.: # Avoids case where connectivity hasn't changed
    #     corr_coef = np.dot(output_corrs_flat, delta_connectivity_flat) / (
    #         np.linalg.norm(output_corrs_flat) * np.linalg.norm(delta_connectivity_flat)
    #     ) # Cosine angle
    # else:
    #     corr_coef = 0.

    # print('Corr_coef:', corr_coef)
    corr_coef, _, _, _, _ = linregress(output_corrs_flat, delta_connectivity_flat)

    # print('Corr_coef slope:', corr_coef)

    if np.isnan(corr_coef):
        print('output corrs:', output_corrs_flat)
        print('delta conn:', delta_connectivity_flat)
        print('dot', np.dot(output_corrs_flat, delta_connectivity_flat))
        print('output corr mag', np.linalg.norm(output_corrs_flat))
        print('delta W mag', np.linalg.norm(delta_connectivity_flat))

    if plot_visualization:
        corr_bins = np.linspace(np.min(output_corrs_flat), np.max(output_corrs_flat), n_bins)
        corr_bin_idxs = np.digitize(output_corrs_flat, corr_bins) - 1

        W_rec_bins_mean = []
        W_rec_bins_std = []
        W_rec_bins_sem = []

        for bin_idx in range(n_bins-1):
            if delta_connectivity_flat[corr_bin_idxs == bin_idx].shape[0] == 0: # If bin is empty
                W_rec_bins_mean.append(np.nan)
                W_rec_bins_std.append(np.nan)
                W_rec_bins_sem.append(np.nan)
            else:
                W_rec_bins_mean.append(np.mean(delta_connectivity_flat[corr_bin_idxs == bin_idx]))

                std = np.std(delta_connectivity_flat[corr_bin_idxs == bin_idx])
                W_rec_bins_std.append(std)
                W_rec_bins_sem.append(std / np.sqrt(delta_connectivity_flat[corr_bin_idxs == bin_idx].shape[0]))

        slope, intercept, _, _, _ = linregress( # Linear regression over bins with elements
            corr_bins[:-1][np.invert(np.isnan(W_rec_bins_mean))],
            np.array(W_rec_bins_mean)[np.invert(np.isnan(W_rec_bins_mean))]
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.scatter(output_corrs_flat, delta_connectivity_flat, color=c_vals_l[4], marker='.', alpha=0.5, s=1)
        ax.scatter(corr_bins[:-1], W_rec_bins_mean, color=c_vals_d[4], zorder=5)
        ax.errorbar(corr_bins[:-1], W_rec_bins_mean, W_rec_bins_sem, color=c_vals_d[4], zorder=5)

        y_plot = slope * corr_bins[:-1] + intercept

        ax.plot(corr_bins[:-1], y_plot, color=c_vals_l[4], zorder=-5)

        ax.set_xlabel('Neuron i,j output correlation')
        ax.set_ylabel('Delta connectivity (a.u.)')
        ax.set_title('Slope {:.2f}'.format(slope))

    return corr_coef, output_corrs

#######################################
########## PHOTOSTIM RELATED ##########
#######################################

def get_photostim_results(train_outputs_ps, params, task_ps, session_idx, 
                          ps_activity_type='raw_fl', ps_computation_type='cc', n_prev_dir_omit=0, 
                          omit_ps_times=True, verbose=False):
    """
    Get the average response of every neuron to every group, separated into
    three distinct periods of time. Each response is averaged over several
    distinct presentations of the group.

    INPUTS:
    train_outputs_ps: From photostim session, must contain 'output_ps' 
    params = task_params, train_params, net_params
    ps_activity_type: either 'raw_activity', 'raw_fl', 'dff'
        The type of neuronal activity to use in computing the photostim responses.
        Experiment uses raw fluorescence.
    ps_computation_type: cc, post-pre, post_only, pre_only
        How to combine the post and pre information into the PS response
        Experiment uses cc
    session_idx: None or int, used to reference appropriate session of photostim
        If None, use task_ps.session_idx (latest session)
    n_prev_dir_omit: when a neuron was recently directly stimulated, ignore its 
        responses in the following few photostimulation measurements since its still
        elevated
    omit_ps_times: when computing mean responses, ignore the time where the
        photostim laser is on. This is true in experiment because the photostim
        laser corrups other fluorescence. False matches Matt's initial setup.

    OUTPUTS:
    causal_connectivity (n_neurons, n_groups): defined as the difference in
        response of a neuron to a photostim group. Usually a functoon of the
        pre- and post-photostimulation response
    group_inputs (n_groups, (n_neurons, n_during_stim): during the photostim input
    group_post_responses (n_groups, n_neurons, n_post_stim): after the photostim input
    group_pre_responses (n_groups, n_neurons, n_pre_stim): before the photostim input
    """

    task_params, _, _ = params
    n_groups = task_ps.n_groups
    n_neurons = task_params['n_neurons']

    if 'net_input_ps' in train_outputs_ps: # Somtimes this wasn't saved, so skip corresponding computation if it wasn't
        net_input_ps = train_outputs_ps['net_input_ps']

    # Sometimes fluorescence/dff won't be saved for efficiency, so need to recompute in these cases
    if ps_activity_type in ('raw_activity',):
        output_ps = train_outputs_ps['output_ps']
    elif ps_activity_type in ('raw_fl',):
        print('Using raw_fl for PS')
        if 'output_fl_ps' not in train_outputs_ps:
            train_outputs_ps['output_fl_ps'] = reproduce_output_fl(
                train_outputs_ps['output_ps'], task_params['fl_kernel']
            )
        output_ps = train_outputs_ps['output_fl_ps']
        # print('Adding random constant to fluorescence!')
        # output_ps += np.random.uniform(low=5.0, high=10.0)
    elif ps_activity_type in ('dff',):
        if 'output_fl_ps' not in train_outputs_ps:
            train_outputs_ps['output_fl_ps'] = reproduce_output_fl(
                train_outputs_ps['output_ps'], task_params['fl_kernel']
            )
        if 'output_dff_ps' not in train_outputs_ps: # Needs fl to compute
            train_outputs_ps['output_dff_ps'], _  = compute_dff(
                train_outputs_ps['output_fl_ps'], task_params, #f0_CUTOFF= 0.1 * task_params['task_scale'],
                verbose=verbose
            )
        output_ps = train_outputs_ps['output_dff_ps']
    else:
        raise ValueError('This shouldnt happen!')

    T_STIM = int(np.round(task_ps.t_max_stim_on * 1000)) # s -> ms
    T_POST_STIM = 300 # ms, matched to what Kayvon uses
    T_PRE_STIM = 200 # ms, matched to what Kayvon uses
    n_during_stim = int(np.round(T_STIM / task_params['t_step']))
    n_post_stim = int(np.round(T_POST_STIM / task_params['t_step']))
    n_pre_stim = int(np.round(T_PRE_STIM / task_params['t_step']))
    
    assert n_pre_stim > 0

    # Means across all group presentation
    group_inputs = np.zeros((task_ps.n_groups, task_params['n_neurons'], n_during_stim,)) # Input into cell during photostim
    group_post_responses = np.zeros((task_ps.n_groups, task_params['n_neurons'], n_post_stim,)) # Response of cell to photostim
    group_pre_responses = np.zeros((task_ps.n_groups, task_params['n_neurons'], n_pre_stim,)) # Response of cell before photostim

    task_ps_hist = task_ps.hists[session_idx]

    # print('Stim starts {} end {}'.format(len(task_ps_hist['stim_starts']), len(task_ps_hist['stim_ends'])))

    raw_group_inputs = [[] for _ in range(task_ps.n_groups)]
    raw_group_post_responses = [[] for _ in range(task_ps.n_groups)]
    raw_group_pre_responses = [[] for _ in range(task_ps.n_groups)]

    # These don't care about specifically what group is being stimulated, only
    # whether or not neuron is in group
    within_group_response_mag = []
    outside_group_response_mag = []

    # Need to account for some weird off by one errors
    n_stim_starts = len(task_ps.hists[session_idx]['stim_starts'])
    n_stim_ends = len(task_ps.hists[session_idx]['stim_ends'])
    n_stim_group_idxs = len(task_ps.hists[session_idx]['stim_group_idxs'])
    assert n_stim_ends == n_stim_starts + 1
    assert n_stim_ends == n_stim_group_idxs + 1

    for stim_idx in range(1, len(task_ps_hist['stim_ends']) - 1): # 1 since no preresponse at start, -1 to avoid end cutoffs
        group_idx = task_ps_hist['stim_group_idxs'][stim_idx-1] # Correct some dumb off by one errors in saving history
        stim_start = task_ps_hist['stim_starts'][stim_idx-1] # Correct some dumb off by one errors in saving history
        stim_end = task_ps_hist['stim_ends'][stim_idx]
        
        stim_start += 1 # Again need to correct off by one indexing errors, verified this on raw PS responses
        stim_end += 1 # Again need to correct off by one indexing errors, verified this on raw PS responses
        
        assert stim_start < stim_end
        assert group_idx == task_ps.group_order[stim_idx]
        # if stim_idx == 1:
        #     print('Start: {}, end: {}'.format(stim_start, stim_end))

        if 'net_input_ps' in train_outputs_ps:
            raw_group_inputs[group_idx].append(net_input_ps[stim_start:stim_start + n_during_stim, task_ps.state_dim - task_ps.n_neurons:])
        
        # Copies here because the nan masking from n_prev_dir_omit was affecting output_ps
        if omit_ps_times: # From end onwards
            post_response = np.copy(output_ps[stim_end:stim_end + n_post_stim])
        else: # From start onwards
            post_response = np.copy(output_ps[stim_start:stim_start + n_post_stim])
        pre_response = np.copy(output_ps[stim_start-n_pre_stim:stim_start])

        if n_prev_dir_omit > 0: # Set responses to nans if recently directly stimulated
            prev_dir_mask = np.zeros((n_neurons,), dtype=bool)
            n_steps_back = np.min((n_prev_dir_omit, stim_idx)) # If stim_idx = 1 and n_prev_dir_omit = 2, can only go back 1
            for rel_stim_idx in range(1, n_steps_back+1): # Inclusive of n_steps_back, skip 0 steps back
                prev_group_idx = task_ps_hist['stim_group_idxs'][stim_idx-1-rel_stim_idx] # Correct some dumb off by one errors in saving history
                # print('Current step {} (group {}) prev step {} (group {})'.format(
                #     stim_idx, group_idx, stim_idx-rel_stim_idx, prev_group_idx
                # ))
                prev_dir_mask[task_ps.groups[prev_group_idx]] = True

            post_response[:, prev_dir_mask] = np.nan # Set all timesteps to nans
            pre_response[:, prev_dir_mask] = np.nan # Set all timesteps to nans
        
        raw_group_post_responses[group_idx].append(post_response)
        raw_group_pre_responses[group_idx].append(pre_response)
        
        with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they return nan as expected
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            # Mean over time and difference of post - pre, shape: (n_neurons,)
            mean_response = np.nanmean(post_response, axis=0) - np.nanmean(pre_response, axis=0)

        neuron_idxs_in_group_one_hot = np.zeros((task_params['n_neurons'],), dtype=np.int32)
        neuron_idxs_in_group_one_hot[task_ps.groups[group_idx]] = 1
        within_group_response_mag.append(np.where(
            neuron_idxs_in_group_one_hot, mean_response, np.nan
        ))
        outside_group_response_mag.append(np.where(
            neuron_idxs_in_group_one_hot, np.nan, mean_response
        ))

    # Now take mean over repeats of each group photostimulus -> (n_neurons, n_time)
    for group_idx in range(n_groups):

        if len(raw_group_post_responses) == 0:
            raise ValueError('Group does not have reponses!!')

        # (n_group_pres, n_seq, n_neurons) -> (n_neurons, n_seq)
        if 'net_input_ps' in train_outputs_ps:
            group_inputs[group_idx] = np.mean(
                np.array(raw_group_inputs[group_idx]), axis=0
            ).T
        
        group_post_responses[group_idx] = np.mean(
            np.array(raw_group_post_responses[group_idx]), axis=0
        ).T
        group_pre_responses[group_idx] = np.mean(
            np.array(raw_group_pre_responses[group_idx]), axis=0
        ).T
    
    # Mean over time and combines post and pre to determine causal conn.: (n_groups, n_neurons,)
    if ps_computation_type in ('cc',): # Pre acts as effective normalization, match how we do it in BCI-1
        causal_connectivity = np.nan * np.ones((n_neurons, n_groups))
        for group_idx in range(n_groups):
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they return nan as expected
                warnings.simplefilter("ignore", category=RuntimeWarning) 
            
                pre_mean = np.nanmean(np.nanmean(raw_group_pre_responses[group_idx], axis=1), axis=0) # Mean over time, then repeats (n_neurons,)
                if np.where(pre_mean == 0.)[0].shape[0] > 0:
                    print('{} neurons with pre_mean = 0, setting to nonzero minimum.'.format(np.where(pre_mean == 0.)[0].shape[0]))
                    min_pre_mean = np.min(pre_mean[np.where(pre_mean > 0.)])
                    pre_mean = np.where(pre_mean > 0., pre_mean, min_pre_mean)
                
                repeat_responses = ( # Mean over time for post and pre
                    np.nanmean(raw_group_post_responses[group_idx], axis=1) - np.nanmean(raw_group_pre_responses[group_idx], axis=1)
                ) / pre_mean[None, :]

                causal_connectivity[:, group_idx] = np.nanmean(repeat_responses, axis=0) / np.nanstd(repeat_responses, axis=0) # Now mean over groups
            
    elif ps_computation_type in ('post-pre',): # Same as above, with no pre normalization, match how we do it in BCI-1
        causal_connectivity = np.nan * np.ones((n_neurons, n_groups))
        for group_idx in range(n_groups):
            with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they return nan as expected
                warnings.simplefilter("ignore", category=RuntimeWarning) 
                repeat_responses = ( # Mean over time for post and pre
                    np.nanmean(raw_group_post_responses[group_idx], axis=1) - np.nanmean(raw_group_pre_responses[group_idx], axis=1)
                )
                causal_connectivity[:, group_idx] = np.nanmean(repeat_responses, axis=0) # Now mean over groups
    elif ps_computation_type in ('post_only',): # Just post
        causal_connectivity = np.nan * np.ones((n_neurons, n_groups))
        for group_idx in range(n_groups):
             with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they return nan as expected
                warnings.simplefilter("ignore", category=RuntimeWarning) 
                repeat_responses = np.nanmean(raw_group_post_responses[group_idx], axis=1) # Mean over time
                causal_connectivity[:, group_idx] = np.nanmean(repeat_responses, axis=0) # Now mean over groups
    elif ps_computation_type in ('pre_only',): # Just pre, mostly as a sanity check
        causal_connectivity = np.nan * np.ones((n_neurons, n_groups))
        for group_idx in range(n_groups):
             with warnings.catch_warnings(): # Suppress mean of empty slice warnings, since they return nan as expected
                warnings.simplefilter("ignore", category=RuntimeWarning) 
                repeat_responses = np.nanmean(raw_group_pre_responses[group_idx], axis=1) # Mean over time
                causal_connectivity[:, group_idx] = np.nanmean(repeat_responses, axis=0) # Now mean over groups
    elif ps_computation_type in ('raw_fl_old',): # Pre acts as effective normalization 
        # This is the old way of doing the averaging, which is slightly mismatched from how we do it in the paper
        # Here, the averages are taken across repeats first, then time average, THEN the difference is taken
        pre_response_not_norm = np.mean(group_pre_responses, axis=-1)
        # print('Minimum pre responses', np.sort(pre_response_not_norm.flatten())[:5])
        print('Mean fl across all neurons:', np.mean(output_ps))
        print('Mean fl pre:', np.mean(group_pre_responses))
        print('Mean fl post:', np.mean(group_post_responses))
        print('Cutoff is currently:', 0.02)

        # mean_group_pre_response = np.clip(np.mean(group_pre_responses, axis=-1), a_min=0.0001 * task_params['task_scale'], a_max=1e8)
        mean_group_pre_response = np.clip(np.nanmean(group_pre_responses, axis=-1), a_min= 0.02, a_max=1e8)
        causal_connectivity = (
            (np.nanmean(group_post_responses, axis=-1) - mean_group_pre_response) /
            mean_group_pre_response
        )

        # max_sort_idxs = np.argsort(causal_connectivity.flatten())[::-1]
        # n_print = 5
        # print('Max values:', causal_connectivity.flatten()[max_sort_idxs][:5])
        # print('Max posts:', np.mean(group_post_responses, axis=-1).flatten()[max_sort_idxs][:5])
        # print('Max pres:', mean_group_pre_response.flatten()[max_sort_idxs][:5])
        # print('Max pres unclipped:', pre_response_not_norm.flatten()[max_sort_idxs][:5])

    elif ps_computation_type in ('raw_fl_no_pre_div_old',): # Just post minus pre, no pre normalization
        causal_connectivity = np.nanmean(group_post_responses, axis=-1) - np.nanmean(group_pre_responses, axis=-1)
    # elif ps_computation_type in ('raw_fl_post_only',): # Just post
    #     causal_connectivity = np.mean(group_post_responses, axis=-1)
    # elif ps_computation_type in ('dff',): # Old way of computing
    #     causal_connectivity = np.nanmean(group_post_responses, axis=-1) - np.nanmean(group_pre_responses, axis=-1)
    else:
        raise ValueError('ps_computation_type {} not recognized'.format(ps_computation_type))

    # For old ways of calculating, converts to be same format as W_rec (post, pre)
    if ps_computation_type in ('raw_fl_old', 'raw_fl_no_pre_div_old', 'dff',):
        causal_connectivity = causal_connectivity.T # (n_neurons, n_groups,) <- (n_groups, n_neurons,)

    if verbose:
        within_group_count = np.sum(~np.isnan(np.array(within_group_response_mag)), axis=0)
        outside_group_count = np.sum(~np.isnan(np.array(outside_group_response_mag)), axis=0)
        print('Times within group: {:.1f} (std: {:.1f})'.format(
            within_group_count.mean(), within_group_count.std()
        ))
        print('Times outside group: {:.1f} (std: {:.1f})'.format(
            outside_group_count.mean(), outside_group_count.std()
        ))

    return causal_connectivity, (
        group_inputs,
        group_post_responses,
        group_pre_responses,
    ), (
        np.array(within_group_response_mag),
        np.array(outside_group_response_mag),
    )

def compute_delta_causal_connectivity(
    train_outputs_all, params, task_ps, session_idx, ps_activity_type='raw_fl', ps_computation_type='cc',
    n_prev_dir_omit=0, omit_ps_times=True, causal_connectivity_mode='groups_to_neurons', 
    causal_connectivity_neurons_mode='2', train_outputs_ps_init=None, train_outputs_ps_final=None,
):
    """
    Compute the change in causal connectivity. This can either return the change
    of the true causal connectivity (n_groups, n_neurons), or the causal
    connectivity by neuron (n_neurons, n_neurons). The latter uses the internal
    conversion to causal_connectivity_neurons

    INPUTS:
    - train_outputs_all: This is just used to get the train_outputs_ps for individual sessions, if this is None can 
        directly pass them instead.
    - omit_ps_times: True matches experiment, False matches Matt's setup
    - causal_connectivity_mode:
        - groups_to_neurons: same as experiment
        - neurons_to_neurons: some additional computation to get individual neuron responses, 
          several different methods, see below
    - train_outputs_ps_init: Directly passed ps, so we don't need to keep train_outputs_all across multiple inits
    - train_outputs_ps_final: Directly passed ps, so we don't need to keep train_outputs_all across multiple inits
    """

    # For special single session case, modify session_idxs correspondingly
    task_params, _, _ = params 
    if task_params['n_sessions'] == 1:
        session_idx_init = 0
        session_idx_final = 1
        if train_outputs_all is not None:
            train_outputs_ps_init = train_outputs_all[0]['train_outputs_ps_init']
            train_outputs_ps_final = train_outputs_all[0]['train_outputs_ps']
        else:
            assert train_outputs_ps_init is not None
            assert train_outputs_ps_final is not None
    else:
        session_idx_init = session_idx - 1
        session_idx_final = session_idx
        if train_outputs_all is not None:
            train_outputs_ps_init = train_outputs_all[session_idx_init]['train_outputs_ps']
            train_outputs_ps_final = train_outputs_all[session_idx_final]['train_outputs_ps']
        else:
            assert train_outputs_ps_init is not None
            assert train_outputs_ps_final is not None

    ### Initialization values ###
    causal_connectivity_init, _, neuron_ps_responses_init = get_photostim_results(
        train_outputs_ps_init, params, task_ps, session_idx_init, ps_computation_type=ps_computation_type,
        ps_activity_type=ps_activity_type, n_prev_dir_omit=n_prev_dir_omit, omit_ps_times=omit_ps_times, 
        verbose=False,
    )

    ### End of session values ###
    causal_connectivity, _, neuron_ps_responses = get_photostim_results(
        train_outputs_ps_final, params, task_ps, session_idx_final, ps_computation_type=ps_computation_type,
        ps_activity_type=ps_activity_type, n_prev_dir_omit=n_prev_dir_omit, omit_ps_times=omit_ps_times, 
        verbose=False,
    )

    if causal_connectivity_mode in ('neurons_to_neurons',):
        causal_connectivity_neurons_init = compute_causal_connectivity_neurons(
            causal_connectivity_init, train_outputs_ps_init, params, task_ps,
            mode=causal_connectivity_neurons_mode
        )
        causal_connectivity_neurons = compute_causal_connectivity_neurons(
            causal_connectivity, train_outputs_ps, params, task_ps,
            mode=causal_connectivity_neurons_mode
        )

        return causal_connectivity_neurons - causal_connectivity_neurons_init, (causal_connectivity_neurons_init, causal_connectivity_neurons)
    elif causal_connectivity_mode in ('groups_to_neurons',):
        return causal_connectivity - causal_connectivity_init, (causal_connectivity_init, causal_connectivity)
    
def compute_causal_connectivity_neurons(causal_connectivity, train_outputs, params, task_ps, mode=3):
    """
    Compute the causal connectivity between individual neurons from the
    causal_connectivity, which is a function of how each neuron responds to
    each GROUP.

    3 different methods of doing so

    INPUTS:
    causal_connectivity (n_neurons, n_groups,)

    task_ps: note this doesn't use any session-dependent quantities, so does not need to know session_idx

    OUTPUTS:
    causal_connectivity_neurons (n_neurons, n_neurons)
    """

    task_params, _, _ = params

    if mode == 1: # Option 1: Average over groups the given neuron appears in
        neuron_to_group_mask = np.zeros((task_ps.n_neurons, task_ps.n_groups), dtype=np.int32)
        for group_idx in range(task_ps.n_groups):
            for neuron_idx in task_ps.groups[group_idx]:
                neuron_to_group_mask[neuron_idx, group_idx] = 1
        n_groups_per_neuron = np.sum(neuron_to_group_mask, axis=-1) + 1e-3
        neuron_to_group_mask = neuron_to_group_mask / n_groups_per_neuron[:, np.newaxis]

        causal_connectivity_neurons = np.zeros((task_ps.n_neurons, task_ps.n_neurons,))
        for neuron_idx in range(task_ps.n_neurons):
            if n_groups_per_neuron[neuron_idx] > 0:
                causal_connectivity_neurons[neuron_idx] = np.matmul(
                    neuron_to_group_mask[neuron_idx], causal_connectivity.T,
                )
        causal_connectivity_neurons = causal_connectivity_neurons.T # (pre, post) -> (post, pre)
        np.fill_diagonal(causal_connectivity_neurons, 0.) # Zero diagonal

    elif mode == 2: # Option 2: Moore-Penrose inverse
        groups_onehot = np.zeros((task_ps.n_groups, task_ps.n_neurons,))
        for group_idx in range(task_ps.n_groups):
            groups_onehot[group_idx, task_ps.groups[group_idx]] = 1.0
        neuron_to_group_mask = np.linalg.pinv(groups_onehot) # Moore-Penrose inverse

        causal_connectivity_neurons = np.matmul(neuron_to_group_mask, causal_connectivity.T)
        causal_connectivity_neurons = causal_connectivity_neurons.T # (pre, post) -> (post, pre)
        np.fill_diagonal(causal_connectivity_neurons, 0.) # Zero diagonal

    elif mode == 3: # Option 3: Matt's method
        # Note: this uses the convention (post, pre) so no need for transpose at
        # the end like the other two methods
        causal_connectivity_neurons = np.zeros((task_ps.n_neurons, task_ps.n_neurons,))
        n_groups_per_neuron = np.zeros_like(causal_connectivity_neurons) + 1e-3
        n_costim = np.zeros_like(causal_connectivity_neurons) # Will be symmetric

        for group_idx in range(task_ps.n_groups):
            group_neuron_idxs = task_ps.groups[group_idx]

            # Response of all neurons to all neurons within the group
            causal_connectivity_neurons[:, group_neuron_idxs] += causal_connectivity[:, group_idx][:, np.newaxis]
            n_groups_per_neuron[:, group_neuron_idxs] += np.ones((task_ps.n_neurons, 1))

            # Amount of times each neuron pair was co-stimulated
            group_one_hot = np.zeros((task_ps.n_neurons, 1))
            group_one_hot[group_neuron_idxs] = 1.0
            n_costim[:, group_neuron_idxs] += group_one_hot

        # Average implied connectivity by number of groups each neuron has appeared in
        causal_connectivity_neurons = causal_connectivity_neurons / n_groups_per_neuron

        # How often each neuron is costimulated with every other neuron
        # No longer symmetric, because one neuron could be part of more groups than other, so have lower rate
        costim_rate = n_costim / n_groups_per_neuron # (Diagonal is 1 by definition)

        causal_connectivity_neurons = (
            (causal_connectivity_neurons - costim_rate * np.diag(causal_connectivity_neurons)[:, np.newaxis]) / (1 - costim_rate)
        )
        causal_connectivity_neurons[costim_rate > 0.9] = 0 # Zero out heavy costims (usually only diagonals)
    else:
        raise ValueError('Casual connectivity mode not recognized.')
    

    return causal_connectivity_neurons