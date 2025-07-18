# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:21:19 2025

Utility functions for BCI photostim data processing and population analysis.

@author: kayvon.daie
"""

import numpy as np
import re

def parse_hdf5_array_string(array_raw, trl):
    """
    Parse array-like string or list from HDF5 file into list of numpy arrays.

    Parameters
    ----------
    array_raw : str or list
        Raw string or object representing a list of arrays from HDF5.
    trl : int
        Number of trials; output list will be padded to this length if needed.

    Returns
    -------
    parsed_array : np.ndarray (dtype=object)
        Array of np.arrays, one per trial.
    """
    if isinstance(array_raw, str):
        pattern = r'array\(\[([^\]]*)\](?:, dtype=float64)?\)'
        matches = re.findall(pattern, array_raw.replace('\n', ''))

        parsed = []
        for match in matches:
            try:
                if match.strip() == '':
                    parsed.append(np.array([]))
                else:
                    arr = np.fromstring(match, sep=',')
                    parsed.append(arr)
            except Exception as e:
                print("Skipping array due to error:", e)

        pad_len = trl - len(parsed)
        if pad_len > 0:
            parsed += [np.array([])] * pad_len

        return np.array(parsed, dtype=object)

    else:
        if len(array_raw) < trl:
            pad_len = trl - len(array_raw)
            return np.array(list(array_raw) + [np.array([])] * pad_len, dtype=object)
        return array_raw

def get_indices_around_steps(tsta, steps, pre=0, post=0):
    """
    Return list of time indices around a list of step times.

    Parameters
    ----------
    tsta : np.ndarray
        Time vector.
    steps : array-like
        Step times.
    pre : int
        Number of indices to include before each step.
    post : int
        Number of indices to include after each step.

    Returns
    -------
    all_indices : np.ndarray
        Sorted, unique array of indices around step events.
    """
    indices = np.searchsorted(tsta, steps)
    all_indices = []

    for idx in indices:
        start = max(idx - pre, 0)
        end = min(idx + post + 1, len(tsta))  # +1 to include the last point
        all_indices.extend(range(start, end))
    
    return np.unique(all_indices)

def centered_dot(A):
    """
    Compute dot product between row-centered versions of A.

    Parameters
    ----------
    A : np.ndarray, shape (N, T)
        Matrix of N neurons over T timepoints.

    Returns
    -------
    dot : np.ndarray, shape (N, N)
        Dot product of mean-subtracted rows.
    """
    A_centered = A - A.mean(axis=1, keepdims=True)
    return A_centered @ A_centered.T

def compute_rpe(rt, baseline=3.0, window=10, fill_value=np.nan):
    """
    Compute reward prediction error based on running average of past rewards.

    Parameters
    ----------
    rt : np.ndarray
        Array of reward times or reward outcomes.
    baseline : float
        Initial baseline expectation if no past rewards.
    window : int
        Number of past trials to average.
    fill_value : float
        Value to use for NaNs during averaging.

    Returns
    -------
    rpe : np.ndarray
        RPE at each trial: avg_past - current.
    """
    rpe = np.full_like(rt, np.nan, dtype=np.float64)
    rt_clean = np.where(np.isnan(rt), fill_value, rt)

    for i in range(len(rt)):
        if i == 0:
            avg = baseline
        else:
            start = max(0, i - window)
            avg = np.nanmean(rt_clean[start:i]) if i > start else baseline
        rpe[i] = avg - rt_clean[i]
    return rpe

def compute_rpe_standard(rt, baseline=3.0, window=10, fill_value=np.nan):

    rpe = np.full_like(rt, np.nan, dtype=np.float64)
    rt_clean = np.where(np.isnan(rt), fill_value, rt)

    for i in range(len(rt)):
        if i == 0:
            avg = baseline
        else:
            start = max(0, i - window)
            avg = np.nanmean(rt_clean[start:i]) if i > start else baseline
        rpe[i] = avg - rt_clean[i]
    return rpe

def compute_amp_from_photostim(mouse, data, folder):
    """
    Compute AMP values (average ΔF/F in response to photostim) for each epoch.

    Parameters
    ----------
    mouse : str
        Mouse ID (e.g., 'BCI103') — affects stimulus timing.
    data : dict
        Dictionary containing photostim data, e.g. favg_raw, dt_si.
    folder : str
        Full path to suite2p folder containing siHeader.npy.

    Returns
    -------
    AMP : list of np.ndarray
        List of AMP values for each epoch.
    stimDist : np.ndarray
        Distance from stim site for each cell (in microns).
    """
    AMP = []

    # Load scan parameters to compute microns per pixel
    siHeader_path = folder + r'/suite2p_BCI/plane0/siHeader.npy'
    siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

    for epoch_i in range(2):
        if epoch_i == 0:
            stimDist = data['photostim']['stimDist'] * umPerPix
            favg_raw = data['photostim']['favg_raw']
        else:
            stimDist = data['photostim2']['stimDist'] * umPerPix
            favg_raw = data['photostim2']['favg_raw']

        # Normalize ΔF/F traces by 3-frame baseline
        favg = np.zeros_like(favg_raw)
        for i in range(favg.shape[1]):
            baseline = np.nanmean(favg_raw[0:3, i])
            favg[:, i] = (favg_raw[:, i] - baseline) / baseline

        dt_si = data['dt_si']
        after = int(np.floor(0.2 / dt_si))
        before = int(np.floor(0.2 / dt_si))
        if mouse == "BCI103":
            after = int(np.floor(0.5 / dt_si))

        # Detect stimulation artifact
        artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > 0.5)[0]
        artifact = artifact[artifact < 40]

        if artifact.size == 0:
            AMP.append(np.full(favg_raw.shape[1:], np.nan))
            continue

        # Define pre- and post-stim windows
        pre = (int(artifact[0] - before), int(artifact[0] - 2))
        post = (int(artifact[-1] + 2), int(artifact[-1] + after))

        # Mask artifact
        favg[artifact, :, :] = np.nan

        # Interpolate early frames to remove NaNs
        favg[0:30, :] = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
            ),
            axis=0,
            arr=favg[0:30, :]
        )

        # Compute AMP as Δ(mean_post - mean_pre)
        amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
        AMP.append(amp)

    return AMP, stimDist

import numpy as np
import matplotlib.pyplot as plt
def compute_trial_mismatch_score(
    df_closedloop,
    target_activity_pattern,
    trial_start_vector,
    step_vector,
    tau=400,
    target_window=1,
    scale_factor=5,
    normalized = 0
):
    """
    Computes a trial-averaged mismatch score between expected step rate and neural-target similarity.

    Parameters:
    - df_closedloop: np.ndarray [n_neurons x n_timepoints], observed activity
    - target_activity_pattern: np.ndarray [n_neurons x n_trials], ideal pattern per trial
    - trial_start_vector: np.ndarray [n_timepoints], binary vector with 1s marking trial starts
    - step_vector: np.ndarray [n_timepoints], binary step signal
    - tau: float, smoothing constant for expected step rate (default: 400)
    - target_window: int, number of trials to average when smoothing target pattern (default: 1)
    - scale_factor: float, multiplier for the smoothed step signal (default: 5)

    Returns:
    - trial_mismatch_score: np.ndarray [n_trials], mismatch score per trial
    """
    
    # === Smooth the target pattern across trials ===
    recent_target_pattern = np.zeros_like(target_activity_pattern)
    for i in range(target_activity_pattern.shape[1]):
        ind = np.arange(i - target_window, i)
        ind = ind[ind > -1]
        recent_target_pattern[:, i] = np.nanmean(target_activity_pattern[:, ind], axis=1)

    # === Compute similarity to expected target pattern ===
    trial_num = np.cumsum(trial_start_vector) - 1
    neural_target_similarity = np.zeros(df_closedloop.shape[1])
    for ti in range(df_closedloop.shape[1]):
        x = df_closedloop[:, ti]
        y = recent_target_pattern[:, trial_num[ti]]
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        neural_target_similarity[ti] = np.dot(x, y)
        if normalized == 1:
            #neural_target_similarity[ti] = np.dot(x - np.nanmean(x), y - np.nanmean(y)) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)
            neural_target_similarity[ti] = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)

        

    neural_target_similarity_raw = np.copy(neural_target_similarity)
    # === Normalize similarity between 0 and 1 ===
    neural_target_similarity -= np.nanmin(neural_target_similarity)
    neural_target_similarity /= np.nanmax(neural_target_similarity)

    # === Smooth step signal ===
    kernel = np.exp(-np.arange(0, int(10 * tau)) / tau)
    kernel /= kernel.sum()
    smooth_step = np.convolve(step_vector, kernel, mode='same')

    # === Mismatch = (expected drive) - (neural response) ===
    target_drive_mismatch = scale_factor * smooth_step - neural_target_similarity

    # === Trial-averaged mismatch ===
    trial_start_inds = np.where(trial_start_vector)[0]
    n_trials = len(trial_start_inds)
    trial_mismatch_score = np.zeros(n_trials)
    smooth_step_trial = np.zeros(n_trials)
    target_similarity_trial = np.zeros(n_trials)
    target_similarity_trial_raw = np.zeros(n_trials)
    for i in range(n_trials):
        start = trial_start_inds[i]
        end = trial_start_inds[i + 1] if i < n_trials - 1 else len(target_drive_mismatch)
        trial_mismatch_score[i] = np.nanmean(target_drive_mismatch[start:end])
        smooth_step_trial[i] = np.nanmean(smooth_step[start:end])
        target_similarity_trial[i] = np.nanmean(neural_target_similarity[start:end])
        target_similarity_trial_raw[i] = np.nanmean(neural_target_similarity_raw[start:end])
    
    return trial_mismatch_score, smooth_step_trial, target_similarity_trial, target_similarity_trial_raw
