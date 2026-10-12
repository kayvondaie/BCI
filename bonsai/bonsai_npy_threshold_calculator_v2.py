"""
Bonsai version of Bpod_npy_threshold_calculator2.py — v2.

Same as v1 EXCEPT the dummy_hit / dummy_rt prediction now uses the actual
BCI transfer function (clip at 0, saturate at 3.3) applied per-trial,
instead of the linear `alpha` shortcut that ignores saturation.

Effect:
- Trials whose fluorescence never reaches the *new* lower threshold are
  predicted to miss (speed_new == 0).
- Trials whose fluorescence stays saturated above the *new* upper threshold
  no longer have RT under-predicted, because the per-trial mean speed is
  capped at 3.3.

Requires: data, folder (from ddc.main)
"""
import sys
sys.path.insert(0, r'C:\Users\kayvon.daie\Documents\claude_code\bonsai')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os, json

def _draw_switch_ticks(switches, frac=0.05):
    """Vertical tick marks at threshold-switch trials, sized to the bottom
    fraction of the current y-range so they never inflate the y-limits."""
    ax = plt.gca()
    ylo, yhi = ax.get_ylim()
    th = ylo + frac * (yhi - ylo)
    for s in switches:
        ax.plot((s, s), (ylo, th), 'k', linewidth=0.6)
    ax.set_ylim(ylo, yhi)


def _bci_speed(fluor, lo, hi, max_speed=3.3):
    """BCI transfer function: threshold-linear with saturation.

    speed = clip((fluor - lo) / (hi - lo) * max_speed, 0, max_speed)
    Matches `fun` defined elsewhere in this script and threshold_analysis2's
    transfer_fun.
    """
    gain = hi - lo
    if gain <= 0:
        return np.zeros_like(fluor)
    s = (fluor - lo) / gain * max_speed
    return np.clip(s, 0, max_speed)


ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
len_files = ops['frames_per_file']
cn_ind = data['cn_csv_index'][0]

# Threshold crossing time relative to trial start
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
st = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
rt = rt - st

rew = ~np.isnan(rt)

# ROI CSV processing
roi = np.copy(data['roi_csv'])
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)

inds = np.where(np.diff(roi[:,1]) < 0)[0]
for i in range(len(inds)):
    ind = inds[i]
    roi[ind+1:, 1] = roi[ind+1:, 1] + roi[ind, 1]
    roi[ind+1:, 0] = roi[ind+1:, 0] + roi[ind, 0]

interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)

# --- Load thresholds ---
# Try .mat files first (same as bpod version), fall back to Bonsai Trial.json
BCI_thresholds = data.get('BCI_thresholds', None)

if BCI_thresholds is None or np.all(np.isnan(BCI_thresholds)):
    # Extract thresholds from Bonsai Trial.json
    parent = os.path.dirname(folder.rstrip('/\\'))
    trial_json = os.path.join(parent, 'behavior', 'SoftwareEvents', 'Trial.json')
    bonsai_thresholds = []
    if os.path.isfile(trial_json):
        with open(trial_json, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                rp = ev['data']['response_period']['action']
                lo = rp['lower_action_threshold']
                if isinstance(lo, dict):
                    lo = lo['distribution_parameters']['value']
                hi = rp['upper_action_threshold']
                if isinstance(hi, dict):
                    hi = hi['distribution_parameters']['value']
                bonsai_thresholds.append([lo, hi])
        bonsai_thresholds = np.array(bonsai_thresholds).T  # shape (2, n_bonsai_trials)
        # Need to align to matched trials - use same alignment as create_bonsai_info
        # For now, assume data already has the right number of trials
        n = min(bonsai_thresholds.shape[1], len(rew))
        BCI_thresholds = np.full((2, len(rew)), np.nan)
        BCI_thresholds[:, :n] = bonsai_thresholds[:, :n]
    else:
        BCI_thresholds = np.full((2, len(rew)), np.nan)
        print('WARNING: no BCI_thresholds found')

# If a lower-threshold change occurs in the first 10 trials, drop everything
# before that change (applied only to the bottom-row plots below).
_k_lower = np.diff(BCI_thresholds[0, :])
_early_lower = np.where((_k_lower != 0) & (~np.isnan(_k_lower)))[0]
_early_lower = _early_lower[_early_lower < 10]
trim_start = int(_early_lower[-1] + 1) if len(_early_lower) > 0 else 0

# Voltage mapping function (used for observed avg below)
fun = lambda x: np.minimum(
    (x > BCI_thresholds[0, 0]) * ((x - BCI_thresholds[0, 0]) / (BCI_thresholds[1, 0] - BCI_thresholds[0, 0])) * 3.3,
    3.3
)

# Initialize
strt = 0
dt_si = np.median(np.diff(roi[:, 0]))
fcn = np.empty((350, len(len_files) - 1))
FCN = np.empty((350, len(len_files) - 1))
t_si = np.empty((350, len(len_files) - 1))

# Find threshold switches
ind = np.where(~np.isnan(BCI_thresholds[0, :]))[0]
if len(ind) == 0:
    print('No valid thresholds found')
else:
    k = np.diff(BCI_thresholds[1, :])
    switchesu = np.where((k != 0) & (~np.isnan(k)))[0]
    k = np.diff(BCI_thresholds[0, :])
    switchesl = np.where((k != 0) & (~np.isnan(k)))[0]
    switches = switchesu
    switches = np.concatenate(([0], switches))
    avg = np.empty((len(len_files) - 1, len(switches)))
    avg_raw = np.empty((len(len_files) - 1, len(switches)))

    for si in range(len(switches)):
        strt = 0
        switch = switches[si]
        strts = np.empty(len(len_files) - 1, dtype=int)

        BCI_threshold = BCI_thresholds[:, switch + 2]

        fun = lambda x, thr=BCI_threshold: np.minimum(
            (x > thr[0]) * ((x - thr[0]) / (thr[1] - thr[0])) * 3.3, 3.3
        )
        t = roi_interp[:, 0]
        trl_frm = np.zeros(len(t),)
        thr_time = np.full((len(t), 2), np.nan)

        for i in range(len(rew) - 1):
            strts[i] = strt
            ind = np.arange(strt, strt + len_files[i], dtype=int)
            ind = np.clip(ind, 0, len(roi_interp) - 1)
            a = roi_interp[ind.astype(int), cn_ind + 2]
            thr_time[ind, 0] = BCI_thresholds[0, i]
            thr_time[ind, 1] = BCI_thresholds[1, i]
            a_padded = np.concatenate([a, np.full(400, np.nan)])
            fcn[:, i] = a_padded[:350]
            FCN[:, i] = a_padded[:350]

            a = roi_interp[ind.astype(int), 0]
            a = a - a[0]
            a_padded = np.concatenate([a, np.full(400, np.nan)])
            t_si[:, i] = a_padded[:350]

            strt = strt + len_files[i]

            if rew[i]:
                valid = np.where(t_si[:, i] < rt[i])[0]
                stp = np.max(valid) if len(valid) > 0 else t_si.shape[0]
            else:
                stp = t_si.shape[0]

            avg[i, si] = np.nanmean(fun(fcn[:stp, i]))
            avg_raw[i, si] = np.nanmean(fcn[:stp, i])
            FCN[stp:, i] = np.nan

    # --- Plotting ---
    fig = plt.figure(figsize=(6, 4))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8

    plt.subplot(231)
    epochs = np.concatenate((switches, [len(rew)]))
    dummy_hit = np.zeros(len(rew),)
    dummy_rt = np.zeros(len(switches),)
    actual_rt = np.zeros(len(switches),)
    upr = BCI_thresholds[1, switches + 1]
    lwr_arr = BCI_thresholds[0, switches + 1]

    # ====== v2 prediction: per-trial transfer function ======
    # Reference epoch = first epoch, ref_trials = trials [0, first_epoch_end).
    # For each ref trial, compute the trajectory's mean speed under
    #   (a) the first epoch's thresholds (lo_first, hi_first)  → speed_old
    #   (b) the current epoch's thresholds (lo_cur, hi_cur)    → speed_new
    # Predicted RT for that trial under the new thresholds:
    #   rt_pred = rt[t] * speed_old / speed_new
    # Predicted hit: rt_pred < 10
    # If speed_new == 0 (trajectory never reaches the new lower threshold),
    # the trial is predicted to time out → hit=False, rt_pred capped at 10.
    # ========================================================
    first_epoch_end = switches[1] if len(switches) > 1 else len(rew)
    ref_trials = np.arange(first_epoch_end)
    lo_first = BCI_thresholds[0, 0]
    hi_first = BCI_thresholds[1, 0]

    # Per-trial truncation index: where the trial ended under old thresholds.
    def _trial_stp(t_idx):
        if rew[t_idx]:
            valid = np.where(t_si[:, t_idx] < rt[t_idx])[0]
            return int(np.max(valid)) + 1 if len(valid) > 0 else t_si.shape[0]
        return t_si.shape[0]

    # Precompute speed_old per ref trial (depends only on first-epoch thresholds).
    ref_speed_old = np.full(first_epoch_end, np.nan)
    ref_fluor = [None] * first_epoch_end
    ref_stp = np.zeros(first_epoch_end, dtype=int)
    for t in ref_trials:
        f = fcn[:, t]
        f = f[~np.isnan(f)]
        if len(f) == 0:
            continue
        ref_stp[t] = min(_trial_stp(t), len(f))
        f = f[:ref_stp[t]]
        if len(f) == 0:
            continue
        ref_fluor[t] = f
        ref_speed_old[t] = np.nanmean(_bci_speed(f, lo_first, hi_first))

    for si in range(len(switches)):
        ind = np.arange(epochs[si], epochs[si + 1])

        # Current epoch thresholds.
        lo_cur = BCI_thresholds[0, switches[si] + 1] if switches[si] + 1 < BCI_thresholds.shape[1] else lo_first
        hi_cur = BCI_thresholds[1, switches[si] + 1] if switches[si] + 1 < BCI_thresholds.shape[1] else hi_first

        expected_hits = []
        expected_rts = []
        n_zero_speed = 0
        n_saturated = 0
        for t in ref_trials:
            if ref_fluor[t] is None:
                continue
            if not rew[t] or np.isnan(ref_speed_old[t]) or ref_speed_old[t] <= 0:
                # Skip trials that weren't a hit under old thresholds — matches
                # threshold_analysis2's "if hit[t] and speed_old > 0" gate.
                continue
            spd_new = _bci_speed(ref_fluor[t], lo_cur, hi_cur)
            mean_new = np.nanmean(spd_new)
            # Diagnostic: count saturation (frames at 3.3) under new thresholds.
            if np.any(spd_new >= 3.3 - 1e-9):
                n_saturated += 1
            if mean_new <= 0:
                # New thresholds zero out the trajectory entirely → infinite
                # predicted RT. Counts as a miss for hit-rate, but we don't
                # have a finite number to contribute to the RT mean, so skip
                # from expected_rts. n_zero_speed is reported separately so
                # you can see how often this happens.
                n_zero_speed += 1
                expected_hits.append(False)
                continue
            # No cap on rt_pred — a value > 10 is informative ("avg speed is
            # only fast enough to cross threshold in N seconds, which is past
            # the trial timeout"). Even if it's never observable in real data,
            # the magnitude tells you how slow the trajectory is.
            rt_pred = rt[t] * ref_speed_old[t] / mean_new
            expected_hits.append(rt_pred < 10.0)
            expected_rts.append(rt_pred)

        if len(expected_hits) > 0:
            dummy_hit[ind] = np.mean(expected_hits)
            dummy_rt[si] = np.nanmean(expected_rts)
        else:
            dummy_hit[ind] = np.nan
            dummy_rt[si] = np.nan

        # Optional debug: compare against the old alpha-based prediction.
        if (upr[si] - lwr_arr[0]) != 0:
            alpha = (upr[0] - lwr_arr[0]) / (upr[si] - lwr_arr[0])
        else:
            alpha = 1.0
        alpha_rt = np.nanmean(rt[ref_trials] / alpha)
        n_total = len(expected_hits) + n_zero_speed
        print(f'epoch {si}: alpha_pred_rt={alpha_rt:.3f}  '
              f'tf_pred_rt={dummy_rt[si]:.3f}  '
              f'n_zero_speed={n_zero_speed}/{n_total}  '
              f'n_saturated={n_saturated}/{n_total}')

        # actual_rt: mean RT within this epoch
        epoch_start = switches[si]
        epoch_end = switches[si + 1] if si + 1 < len(switches) else len(rew)
        actual_rt[si] = np.nanmean(rt[epoch_start:epoch_end])
    # ====== end v2 prediction ======

    k = 10
    y_hit = np.convolve(rew[:], np.ones(k) / k, mode='valid')
    x_hit = np.arange(len(y_hit)) + (k - 1) / 2
    plt.plot(x_hit, y_hit, 'k')
    plt.xlim(0, len(rew))
    plt.plot(dummy_hit, color='gray')
    plt.xlabel('Trial #')
    plt.ylabel('Hit rate')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _draw_switch_ticks(switches)

    plt.subplot(232)
    switch_frame = np.cumsum(len_files)[switch]
    plt.plot(t, roi_interp[:, cn_ind + 2], 'k', linewidth=.04)
    plt.plot(t, thr_time, 'b')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Raw fluorescence')
    plt.title(data['mouse'] + '  ' + data['session'])

    plt.subplot(233)
    F = data['F']
    cn = data['conditioned_neuron'][0][0]
    plt.imshow(F[:, cn, :].T, aspect='auto')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    plt.xticks([120, 720], ['0', '10'])
    plt.xlabel('Time from trial start (s)')
    plt.ylabel('Trial #')

    plt.subplot(234)
    k = 10
    cn_trial = np.nanmean(F[:, cn, trim_start:], axis=0)
    y_cn = np.convolve(cn_trial, np.ones(k) / k, mode='valid')
    x_cn = np.arange(len(y_cn)) + (k - 1) / 2 + trim_start
    plt.plot(x_cn, y_cn, 'k')
    plt.xlabel('Trial #')
    plt.ylabel('CN activity')
    _draw_switch_ticks(switches[switches >= trim_start])

    plt.subplot(235)
    ff = F[:, cn, :].copy()  # copy: the loop below subtracts the baseline in place
    for ti in range(ff.shape[1]):
        ff[:, ti] = ff[:, ti] - np.nanmean(ff[0:20, ti])
    switches_b = switches[switches >= trim_start]
    n = int(switches_b[1] - trim_start) if len(switches_b) > 1 else max(1, ff.shape[1] - trim_start)
    tuning_trial = np.nanmean(ff[60:, trim_start:], axis=0)
    y_tun = np.convolve(tuning_trial, np.ones(n) / n, mode='valid')
    x_tun = np.arange(len(y_tun)) + (n - 1) / 2 + trim_start
    plt.plot(x_tun, y_tun, 'k')
    plt.xlabel('Trial #')
    plt.ylabel('CN Tuning')
    _draw_switch_ticks(switches_b)

    plt.subplot(236)
    keep_epochs = switches >= trim_start
    actual_rt_b = actual_rt[keep_epochs]
    dummy_rt_b = dummy_rt[keep_epochs]
    x = np.arange(0, len(actual_rt_b) * 3, 3)
    plt.bar(x, actual_rt_b, color='k')
    x = np.arange(1, len(actual_rt_b) * 3 + 1, 3)
    plt.bar(x, dummy_rt_b, color='gray')
    plt.legend(['Real', 'expected'])
    plt.xlabel('Epoch')
    plt.ylabel('Time to reward (s)')
    plt.tight_layout()

    # --- Epoch analysis ---
    fig = plt.figure(figsize=(5, 2.5))
    rt_epoch = np.zeros((50, len(switches)))
    tuning_epoch = np.zeros((50, len(switches)))
    tuning = np.nanmean(ff[60:, :], axis=0)
    for i in range(len(switches) - 1):
        ind = np.arange(switches[i] - 3, switches[i + 1])
        ind[ind < 0] = 0
        b = rt[ind]
        b = b - np.nanmean(b[0:3])
        if len(b) > 50:
            b = b[0:50]
        a = np.zeros(50,)
        a[0:len(b)] = b
        rt_epoch[:, i] = a

        b = tuning[ind]
        b = b - np.nanmean(b[0:3])
        a = np.zeros(50,)
        if len(b) > 50:
            b = b[0:50]
        a[0:len(b)] = b
        tuning_epoch[:, i] = a
    rt_epoch[rt_epoch == 0] = np.nan
    tuning_epoch[tuning_epoch == 0] = np.nan

    x = np.arange(0, rt_epoch.shape[0])
    x = x - 3
    plt.subplot(121)
    plt.plot(x, np.nanmean(rt_epoch, axis=1), 'k.-')
    plt.xlabel('Trials since Thr change')
    plt.ylabel('$\Delta$ Time to reward (s)')

    plt.subplot(122)
    plt.plot(x, np.nanmean(tuning_epoch, axis=1), 'k.-')
    plt.xlabel('Trials since Thr change')
    plt.ylabel('$\Delta$ CN Tuning')
    plt.tight_layout()
