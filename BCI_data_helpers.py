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

def compute_rpe(x, baseline=3.0, tau=10, fill_value=np.nan):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    x_clean = np.where(np.isnan(x), fill_value, x)
    for i in range(len(x_clean)):
        if i == 0:
            avg = baseline
        else:
            start = max(0, i - tau)
            avg = np.nanmean(x_clean[start:i]) if i > start else baseline
        out[i] = x_clean[i] - avg
    return out

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
    try:
        siHeader_path = folder + r'/suite2p_BCI/plane0/siHeader.npy'
        siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
        dt_si = data['dt_si']
    except:
        siHeader_path = folder + r'/suite2p_photostim_single/plane0/siHeader.npy'
        siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
        dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
    n_epochs = 2 if 'photostim2' in data else 1
    for epoch_i in range(n_epochs):
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
            
        #dt_si = data['dt_si']
        after = int(np.floor(0.4 / dt_si))
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


def get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 2)):
     
    nC, nFrames = df.shape
    reward_frames = np.where(reward_vector > 0)[0]

    pre_frames  = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_reward = np.arange(-pre_frames, post_frames) * dt_si

    F_reward = np.full((win_len, nC, len(reward_frames)), np.nan)

    for ri, r in enumerate(reward_frames):
        start = r - pre_frames
        stop  = r + post_frames
        if start < 0 or stop > nFrames:
            continue
        F_reward[:, :, ri] = df[:, start:stop].T  # (time, cells)

    return F_reward, t_reward

def get_trial_aligned_df_padded(df, trial_start_vector, reward_vector, dt_si, window=(-20, 10)):
    """
    Aligns all trials so t=0 (trial start) occurs at the same frame index for every trial.
    Pads both pre-trial and post-trial regions with NaNs when data are missing
    (e.g., short ITIs or truncated recording).

    Parameters
    ----------
    df : ndarray (nCells, nFrames)
    trial_start_vector : binary ndarray, 1 at trial start frames
    reward_vector : binary ndarray, 1 at reward frames
    dt_si : float
        Frame duration (s)
    window : tuple
        Time window (sec) around trial start, e.g. (-20, 10)

    Returns
    -------
    F_trial : ndarray (time, cells, trials)
        Each trial aligned to t=0 at the same frame index.
    t_trial : ndarray
        Relative time (s) to trial start.
    """
    nC, nFrames = df.shape
    trial_starts = np.where(trial_start_vector > 0)[0]
    reward_frames = np.where(reward_vector > 0)[0]

    pre_frames = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_trial = np.arange(-pre_frames, post_frames) * dt_si

    F_trial = np.full((win_len, nC, len(trial_starts)), np.nan)

    for ti, tstart in enumerate(trial_starts):
        # Identify neighboring rewards (avoid contamination)
        prev_reward = reward_frames[reward_frames < tstart]
        next_reward = reward_frames[reward_frames > tstart]
        prev_reward_frame = prev_reward[-1] if len(prev_reward) > 0 else 0
        next_reward_frame = next_reward[0] if len(next_reward) > 0 else nFrames

        # Desired absolute frame indices for the full window
        desired_start = tstart - pre_frames
        desired_stop = tstart + post_frames

        # Actual available region within constraints
        actual_start = max(desired_start, prev_reward_frame)
        actual_stop = min(desired_stop, next_reward_frame, nFrames)

        # Compute how much to pad on left (pre) and right (post)
        left_pad = actual_start - desired_start
        right_pad = desired_stop - actual_stop

        # Extract valid data
        valid_data = df[:, actual_start:actual_stop]

        # Place valid data within padded window
        start_idx = int(left_pad)
        stop_idx = start_idx + valid_data.shape[1]
        F_trial[start_idx:stop_idx, :, ti] = valid_data.T

    return F_trial, t_trial




import numpy as np

def get_reward_aligned_df_truncated(df, reward_vector, trial_start_vector, dt_si, window=(-2, 2)):
    """
    Extracts reward-aligned ΔF/F windows, truncating post-reward response at the next trial start.

    Parameters
    ----------
    df : ndarray
        (nCells, nFrames)
    reward_vector : ndarray
        Binary vector marking reward times (1 at reward frame).
    trial_start_vector : ndarray
        Binary vector marking trial start times (1 at trial start frame).
    dt_si : float
        Frame duration (s)
    window : tuple
        Time window around reward (sec), e.g. (-2, 10)

    Returns
    -------
    F_reward : ndarray
        (time, cells, rewards)
    t_reward : ndarray
        Relative time (sec) to reward.
    """
    nC, nFrames = df.shape
    reward_frames = np.where(reward_vector > 0)[0]
    trial_start_frames = np.where(trial_start_vector > 0)[0]

    pre_frames  = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_reward = np.arange(-pre_frames, post_frames) * dt_si

    F_reward = np.full((win_len, nC, len(reward_frames)), np.nan)

    for ri, r in enumerate(reward_frames):
        # find next trial start after this reward
        next_trial = trial_start_frames[trial_start_frames > r]
        next_trial_frame = next_trial[0] if len(next_trial) > 0 else nFrames

        start = max(r - pre_frames, 0)
        stop = min(r + post_frames, next_trial_frame)

        # compute actual length for truncation
        window_len = stop - start
        if window_len <= 0 or stop > nFrames:
            continue

        # place truncated trace into preallocated array
        F_reward[:window_len, :, ri] = df[:, start:stop].T

    return F_reward, t_reward

def zscore_mat(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1  # avoid div by zero
    return (X - mu) / sd



import numpy as np

def build_pairwise_XY_per_bin(
    CCstep, CCrew, CCts, CCpre,
    stimDist, AMP,
    nb, hit_bin,
    dist_target_lt=10,
    dist_nontarg_min=30,
    dist_nontarg_max=1000,
    amp0_thr=0.1,
    amp1_thr=0.1,
):
    """
    Builds:
      - Y_T: (n_pairs,) target = AMP1 - AMP0 for nontarget pairs (pair ordering fixed)
      - Xstep_T, Xrew_T, Xts_T, Xpre_T: (n_pairs, n_bins_used) regressors per bin

    Notes:
      - Pair ordering is defined by iterating gi in order, then nontarg indices in order.
      - Only bins with finite hit_bin[i] are included (same as your code).
      - Raises if no valid pairs, or if X/Y pair counts mismatch.
    """

    # ---- Build Y_T once (per-pair), independent of bin ----
    Y_list, Yo_list = [], []
    X_list, Xo_list = [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where(
            (stimDist[:, gi] < dist_target_lt)
            & (AMP[0][:, gi] > amp0_thr)
            & (AMP[1][:, gi] > amp1_thr)
        )[0]
        if cl.size == 0:
            continue

        nontarg = np.where(
            (stimDist[:, gi] > dist_nontarg_min) & (stimDist[:, gi] < dist_nontarg_max)
        )[0]
        if nontarg.size == 0:
            continue

        y  = AMP[1][nontarg, gi]
        yo = AMP[0][nontarg, gi]
        Y_list.append(y)
        Yo_list.append(yo)
        
        x  = np.nanmean(AMP[1][cl, gi],0)
        xo = np.nanmean(AMP[0][cl, gi],0)
        X_list.append(np.full(y.shape, x, dtype=float))
        Xo_list.append(np.full(y.shape, xo, dtype=float))


    if len(Y_list) == 0:
        raise ValueError("No valid nontarget pairs found to build Y_T.")

    Y_pair  = np.concatenate(Y_list)
    Yo_pair = np.concatenate(Yo_list)

    Y_T = np.nan_to_num((Y_pair - Yo_pair).ravel(), nan=0.0)
    
    X_pair  = np.concatenate(X_list)
    Xo_pair = np.concatenate(Xo_list)

    X_T = np.nan_to_num((X_pair - Xo_pair).ravel(), nan=0.0)

    # ---- Build X matrices per bin (same pairs ordering as Y_T) ----
    XXstep, XXrew, XXts, XXpre = [], [], [], []
    used_bins = []

    for i in range(nb):
        if not np.isfinite(hit_bin[i]):
            continue

        Xstep_list, Xrew_list, Xts_list, Xpre_list = [], [], [], []

        for gi in range(stimDist.shape[1]):
            cl = np.where(
                (stimDist[:, gi] < dist_target_lt)
                & (AMP[0][:, gi] > amp0_thr)
                & (AMP[1][:, gi] > amp1_thr)
            )[0]
            if cl.size == 0:
                continue

            xstep = np.nanmean(CCstep[cl, :, i], axis=0)
            xrew  = np.nanmean(CCrew[cl,  :, i], axis=0)
            xts   = np.nanmean(CCts[cl,   :, i], axis=0)
            xpre  = np.nanmean(CCpre[cl,  :, i], axis=0)

            nontarg = np.where(
                (stimDist[:, gi] > dist_nontarg_min) & (stimDist[:, gi] < dist_nontarg_max)
            )[0]
            if nontarg.size == 0:
                continue

            Xstep_list.append(xstep[nontarg])
            Xrew_list.append(xrew[nontarg])
            Xts_list.append(xts[nontarg])
            Xpre_list.append(xpre[nontarg])

        if len(Xstep_list) == 0:
            continue

        XXstep.append(np.concatenate(Xstep_list))
        XXrew.append(np.concatenate(Xrew_list))
        XXts.append(np.concatenate(Xts_list))
        XXpre.append(np.concatenate(Xpre_list))
        used_bins.append(i)

    Xstep = np.asarray(XXstep)
    Xrew  = np.asarray(XXrew)
    Xts   = np.asarray(XXts)
    Xpre  = np.asarray(XXpre)

    # transpose to (n_pairs, n_bins_used)
    Xstep_T = np.nan_to_num(Xstep, nan=0.0).T
    Xrew_T  = np.nan_to_num(Xrew,  nan=0.0).T
    Xts_T   = np.nan_to_num(Xts,   nan=0.0).T
    Xpre_T  = np.nan_to_num(Xpre,  nan=0.0).T

    # sanity check: pairs match target length
    if Xstep_T.shape[0] != Y_T.shape[0]:
        raise ValueError(f"X/Y mismatch: X has {Xstep_T.shape[0]} pairs, Y has {Y_T.shape[0]} pairs.")

    return Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_T, np.asarray(used_bins, dtype=int)


import numpy as np

def compute_binned_behaviors_and_pairwise(
    *,
    nb,
    trial_bins,
    hit,
    rt_filled,
    rt_rpe,
    miss_rpe,
    hit_rpe,
    BCI_thresholds,
    pairwise_mode,
    krewards,
    kstep,
    k,
    kpre,
    CCrew,
    CCstep,
    CCts,
    CCpre,
    cc=None,
    centered_dot=None,
    zscore_mat=None,
    return_interleaved_CC=True,
):
    """
    Fills per-bin behavioral summaries + per-bin pairwise matrices.

    Inputs:
      - nb: number of bins
      - trial_bins: array-like length nb+1 of bin edges (trial indices)
      - hit: boolean array per trial
      - rt_filled, rt_rpe, miss_rpe, hit_rpe: arrays per trial
      - BCI_thresholds: typically shape (2, n_trials) or similar; may be missing/wrong length
      - pairwise_mode: 'noise_corr' | 'dot_prod' | 'dot_prod_no_mean' | 'dot_prod_z'
      - krewards, kstep, k, kpre: arrays shaped (n_cells, n_trials)
      - CCrew, CCstep, CCts, CCpre: preallocated arrays shaped (n_cells, n_cells, nb)
      - cc: only used to size interleaved CC if return_interleaved_CC=True
      - centered_dot: required if pairwise_mode=='dot_prod_no_mean'
      - zscore_mat: required if pairwise_mode=='dot_prod_z'

    Returns:
      hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin, hit_rpe_bin, miss_rpe_bin,
      CCrew, CCstep, CCts, CCpre,
      (optional) CC interleaved as [step, rew, ts, pre] repeating
    """

    # Bin-averaged behavior features
    hit_bin      = np.full((nb,), np.nan)
    miss_bin     = np.full((nb,), np.nan)
    rt_bin       = np.full((nb,), np.nan)
    avg_dot_bin  = np.full((nb,), np.nan)  # kept for compatibility (not filled in your snippet)
    thr_bin      = np.full((nb,), np.nan)
    rpe_bin      = np.full((nb,), np.nan)
    hit_rpe_bin  = np.full((nb,), np.nan)
    miss_rpe_bin = np.full((nb,), np.nan)

    trial_bins = np.asarray(trial_bins)

    # Fill per-bin pairwise matrices + per-bin behaviors
    for i in range(nb):
        # skip empty / degenerate bins
        if trial_bins[i + 1] <= trial_bins[i]:
            continue

        ind = np.arange(trial_bins[i], trial_bins[i + 1])
        if ind.size == 0:
            continue

        # behaviors
        hit_bin[i]      = np.nanmean(hit[ind])
        miss_bin[i]     = np.nanmean((~hit)[ind])
        rt_bin[i]       = np.nanmean(rt_filled[ind])
        rpe_bin[i]      = np.nanmean(rt_rpe[ind])
        miss_rpe_bin[i] = np.nanmean(miss_rpe[ind])
        hit_rpe_bin[i]  = np.nanmean(hit_rpe[ind])

        # thresholds may be missing / wrong length for some sessions
        try:
            thr_bin[i] = np.nanmean(BCI_thresholds[1, ind])
        except Exception:
            thr_bin[i] = np.nan

        # pairwise
        if pairwise_mode == "noise_corr":
            CCrew[:, :, i]  = np.corrcoef(krewards[:, ind])
            CCstep[:, :, i] = np.corrcoef(kstep[:, ind])
            CCts[:, :, i]   = np.corrcoef(k[:, ind])
            CCpre[:, :, i]  = np.corrcoef(kpre[:, ind])

        elif pairwise_mode == "dot_prod":
            CCrew[:, :, i]  = krewards[:, ind] @ krewards[:, ind].T
            CCstep[:, :, i] = kstep[:, ind]    @ kstep[:, ind].T
            CCts[:, :, i]   = k[:, ind]        @ k[:, ind].T
            CCpre[:, :, i]  = kpre[:, ind]     @ kpre[:, ind].T

        elif pairwise_mode == "dot_prod_no_mean":
            if centered_dot is None:
                raise ValueError("centered_dot must be provided when pairwise_mode=='dot_prod_no_mean'.")
            CCrew[:, :, i]  = centered_dot(krewards[:, ind])
            CCstep[:, :, i] = centered_dot(kstep[:, ind])
            CCts[:, :, i]   = centered_dot(k[:, ind])
            CCpre[:, :, i]  = centered_dot(kpre[:, ind])

        elif pairwise_mode == "dot_prod_z":
            if zscore_mat is None:
                raise ValueError("zscore_mat must be provided when pairwise_mode=='dot_prod_z'.")
            kr = zscore_mat(krewards[:, ind])
            ks = zscore_mat(kstep[:, ind])
            kt = zscore_mat(k[:, ind])
            kp = zscore_mat(kpre[:, ind])
            CCrew[:, :, i]  = kr @ kr.T
            CCstep[:, :, i] = ks @ ks.T
            CCts[:, :, i]   = kt @ kt.T
            CCpre[:, :, i]  = kp @ kp.T

        else:
            raise ValueError(f"Unknown pairwise_mode: {pairwise_mode}")

    if return_interleaved_CC:
        # Interleave if you still want CC (optional)
        if cc is None:
            # fall back to CCstep shape if you don't pass cc
            n0, n1, _ = CCstep.shape
            CC = np.full((n0, n1, nb * 4), np.nan)
        else:
            CC = np.full((cc.shape[0], cc.shape[1], nb * 4), np.nan)

        CC[:, :, 0::4] = CCstep
        CC[:, :, 1::4] = CCrew
        CC[:, :, 2::4] = CCts
        CC[:, :, 3::4] = CCpre

        return (hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin, hit_rpe_bin, miss_rpe_bin,
                CCrew, CCstep, CCts, CCpre, CC)

    return (hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin, hit_rpe_bin, miss_rpe_bin,
            CCrew, CCstep, CCts, CCpre)


def bin_mean(x_trial, trial_bins):
    x_trial = np.asarray(x_trial, dtype=float).squeeze()
    nb = len(trial_bins) - 1
    out = np.full(nb, np.nan)
    for i in range(nb):
        a, b = trial_bins[i], trial_bins[i + 1]
        if b <= a:
            continue
        out[i] = np.nanmean(x_trial[a:b])
    return out

def build_behavior_matrix(binned_dict, trial_dict, trial_bins, used_bins, behavior_specs, zscore_cols_idx):
    used_bins = np.asarray(used_bins)
    is_mask = (used_bins.dtype == bool)
    behavior_cols = []
    behavior_names = []

    for spec in behavior_specs:
        name = spec["name"]
        source = spec["source"]
        level = spec.get("level", "bin")
        if level == "trial":
            if callable(source):
                vec_trial = np.asarray(source(trial_dict), dtype=float)
            else:
                vec_trial = np.asarray(trial_dict[source], dtype=float)
            vec = bin_mean(vec_trial, trial_bins)
        else:
            if callable(source):
                vec = np.asarray(source(binned_dict), dtype=float)
            else:
                vec = np.asarray(binned_dict[source], dtype=float)
        if is_mask:
            vec_k = vec[used_bins]
        else:
            vec_k = vec[used_bins.astype(int)]
        behavior_cols.append(vec_k)
        behavior_names.append(name)

    B = np.column_stack(behavior_cols).astype(float)

    if len(zscore_cols_idx) > 0 and B.shape[0] > 0:
        mu = np.nanmean(B[:, zscore_cols_idx], axis=0, keepdims=True)
        sd = np.nanstd(B[:, zscore_cols_idx], axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        B[:, zscore_cols_idx] = (B[:, zscore_cols_idx] - mu) / sd

    return B, behavior_names

def build_design_matrix(epoch_to_XT, B, epoch_order):
    X_parts = []
    for epoch_name in epoch_order:
        X_parts.append(epoch_to_XT[epoch_name] @ B)
    return np.hstack(X_parts)