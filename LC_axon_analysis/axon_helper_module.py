# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:06:42 2025

@author: kayvon.daie
"""
import os
import numpy as np
def get_axon_data_dict(sessions,i):
    mouse = sessions['Mouse'][i]
    session = sessions['Session'][i]
    try:
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        suffix = 'BCI'
        main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
        main_npy_path = os.path.join(folder, main_npy_filename)
        data = np.load(main_npy_path, allow_pickle=True)
    except:
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/'
        suffix = 'BCI'
        main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
        main_npy_path = os.path.join(folder, main_npy_filename)
        data = np.load(main_npy_path, allow_pickle=True)
    return data, folder

import numpy as np

def reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4)):
    # Parameters
    pre_s, post_s = window
    pre_pts  = int(np.round(pre_s / dt_si))
    post_pts = int(np.round(post_s / dt_si))
    total_pts = post_pts - pre_pts  # total points in window

    # Find reward times
    reward_times = np.where(reward_vector > 0)[0]

    # Only keep rewards with enough buffer on both sides
    valid_rewards = reward_times[(reward_times + post_pts < dfaxon.shape[1]) &
                                 (reward_times + pre_pts >= 0)]

    # Preallocate output: time x neuron x trial
    out = np.zeros((total_pts, dfaxon.shape[0], len(valid_rewards)))

    for i, rt in enumerate(valid_rewards):
        inds = np.arange(rt + pre_pts, rt + post_pts)
        out[:, :, i] = dfaxon[:, inds].T  # transpose to (time, neuron)

    return out  # shape: (time, neuron, trial)

import numpy as np

def trial_aligned_responses(dfaxon, trial_start_vector, reward_vector, rt, dt_si,
                            window=(-2, 4), pad_overlap=True):
    """
    Align dfaxon responses to either real rewards (hits) or pseudo-rewards (misses).
    
    Parameters
    ----------
    dfaxon : array (neurons x time)
        Activity traces.
    trial_start_vector : array (time,)
        Binary vector with trial start times.
    reward_vector : array (time,)
        Binary vector with reward times.
    rt : array (trials,)
        Reaction times per trial (in seconds). rt==20 marks misses.
    dt_si : float
        Sampling step (s).
    window : tuple
        Time window around alignment point (s).
    pad_overlap : bool, default=True
        If True, timepoints after the next trial start are padded with NaN.
    
    Returns
    -------
    out : array (time, neuron, trial)
        Time-aligned responses (NaN-padded if overlap).
    align_times : array (trial,)
        Frame indices of alignment events (real or pseudo).
    """
    pre_s, post_s = window
    pre_pts  = int(np.round(pre_s / dt_si))
    post_pts = int(np.round(post_s / dt_si))
    total_pts = post_pts - pre_pts

    # Trial starts
    trial_starts = np.where(trial_start_vector > 0)[0]
    n_trials = len(trial_starts)

    # Reward times
    reward_times = np.where(reward_vector > 0)[0]

    # Alignment times: actual reward if hit, else 10s after trial start
    align_times = np.zeros(n_trials, dtype=int)
    for i, ts in enumerate(trial_starts):
        if rt[i] != 20:  # hit
            idx = np.searchsorted(reward_times, ts)
            align_times[i] = reward_times[idx] if idx < len(reward_times) else ts + int(10/dt_si)
        else:  # miss
            align_times[i] = ts + int(10/dt_si)

    # Valid trials (enough buffer in recording)
    valid = (align_times + post_pts < dfaxon.shape[1]) & (align_times + pre_pts >= 0)
    align_times = align_times[valid]
    trial_starts = trial_starts[valid]

    # Preallocate with NaN (for padding)
    out = np.full((total_pts, dfaxon.shape[0], len(align_times)), np.nan, float)

    for j, (at, ts) in enumerate(zip(align_times, trial_starts)):
        inds = np.arange(at + pre_pts, at + post_pts)

        # clip to recording bounds
        in_bounds = (inds >= 0) & (inds < dfaxon.shape[1])

        # assign values
        out[in_bounds, :, j] = dfaxon[:, inds[in_bounds]].T

        if pad_overlap and j < len(trial_starts) - 1:
            next_trial = trial_starts[j+1]
            overlap = inds >= next_trial
            out[overlap, :, j] = np.nan  # pad overlap region

    return out, align_times

