#%% ============================================================================
# CELL 0: Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import session_counting
import data_dict_create_module_test as ddct
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(_THIS_DIR, 'meta_analysis_results')
PANEL_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written\3-factor learning paper\claude code 032226\meta_analysis_results\panels'
os.makedirs(PANEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'svg.fonttype': 'none',
})

# QC failures
_qc_fail = {
    ('BCI104', '012325'),
    ('BCI105', '012125'),
    ('BCI105', '012425'),
}

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

# --- Per-mouse low_floor for the empirical transfer function ---
# The Zaber rate code has a noise-driven stepping floor that the original
# linear-ramp model didn't capture, causing expected RT to be underestimated
# at hard thresholds. Floors below are per-mouse medians of single-session
# fits (see threshold_debug.py / out/threshold_debug_summary.txt), computed
# after dropping ~3 sessions where the fit pegged at the optimizer bound.
MOUSE_LOW_FLOOR = {
    'BCI102': 0.36,
    'BCI103': 0.16,
    'BCI104': 0.22,
    'BCI105': 0.17,
    'BCI106': 0.24,
    'BCI109': 0.24,
}
DEFAULT_LOW_FLOOR = 0.23   # overall median across mice

def low_floor_for(mouse):
    """Return the empirical noise-driven stepping floor for a given mouse."""
    return MOUSE_LOW_FLOOR.get(mouse, DEFAULT_LOW_FLOOR)

# Value assigned to a miss when averaging reward time. A miss timed out at the
# 10s boundary but the "true" reward time is longer/undefined, so we penalize
# it with a value > the timeout rather than pinning it at 10s. This is the
# RT-averaging penalty ONLY; the hit/miss boundary stays at 10s.
MISS_RT = 20.0

# F[:, :, t] starts PRE_TRIAL_S seconds BEFORE the trial-start go cue (pre-trial
# buffer). To map a behavioral time t (seconds from trial start, e.g. reward
# time) to a frame index: frame = int((t + PRE_TRIAL_S) / dt_si).
PRE_TRIAL_S = 2.0

#%% ============================================================================
# CELL 1: Load single session and compute threshold epochs
# ============================================================================
mouse = "BCI102"
session_inds = np.where(
    (list_of_dirs['Mouse'] == mouse) &
    (list_of_dirs['Has data_main.npy'] == True)
)[0]
si = session_inds[7]
session = list_of_dirs['Session'][si]
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + '/' + session + '/pophys/'
print(f"Loading {mouse} {session}")

bci_keys = ['F', 'mouse', 'session', 'conditioned_neuron', 'dt_si',
            'step_time', 'reward_time', 'BCI_thresholds',
            'roi_csv', 'cn_csv_index', 'threshold_crossing_time',
            'SI_start_times']
data = ddct.load_hdf5(folder, bci_keys, [])

F = data['F']
trl = F.shape[2]
dt_si = data['dt_si']
cn = data['conditioned_neuron'][0][0]

# Parse behavioral arrays
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['threshold_crossing_time'] = parse_hdf5_array_string(
    data['threshold_crossing_time'], trl)
rt = np.array([x[0] if hasattr(x, '__len__') and len(x) > 0 else np.nan
               for x in data['threshold_crossing_time']], dtype=float)
hit = np.isfinite(rt)

# BCI thresholds — forward-fill NaNs in upper threshold
BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
thr_lower = BCI_thresholds[0, :].copy()
thr_upper = BCI_thresholds[1, :].copy()
for i in range(1, thr_upper.size):
    if np.isnan(thr_upper[i]):
        thr_upper[i] = thr_upper[i - 1]
    if np.isnan(thr_lower[i]):
        thr_lower[i] = thr_lower[i - 1]
if np.isnan(thr_upper[0]) and np.any(np.isfinite(thr_upper)):
    thr_upper[0] = thr_upper[np.isfinite(thr_upper)][0]
if np.isnan(thr_lower[0]) and np.any(np.isfinite(thr_lower)):
    thr_lower[0] = thr_lower[np.isfinite(thr_lower)][0]

# Pad to trl length with last valid value if thresholds are shorter
if len(thr_upper) < trl:
    thr_upper = np.concatenate([thr_upper, np.full(trl - len(thr_upper), thr_upper[-1])])
    thr_lower = np.concatenate([thr_lower, np.full(trl - len(thr_lower), thr_lower[-1])])

# Detect threshold switch trials (changes in upper threshold)
d_upper = np.diff(thr_upper)
switches = np.where((d_upper != 0) & np.isfinite(d_upper))[0] + 1  # trial indices where new threshold starts
switches = np.concatenate(([0], switches))  # include epoch 0
n_epochs = len(switches)
epoch_ends = np.concatenate((switches[1:], [trl]))

print(f"{trl} trials, {n_epochs} threshold epochs")
for ei in range(n_epochs):
    t0, t1 = switches[ei], epoch_ends[ei]
    print(f"  Epoch {ei}: trials {t0}-{t1-1}, "
          f"lower={thr_lower[t0]:.1f}, upper={thr_upper[t0]:.1f}, "
          f"n_trials={t1-t0}")

#%% ============================================================================
# CELL 2: Transfer function and per-trial CN fluorescence → speed
# ============================================================================

def transfer_fun(fluorescence, lower, upper, max_speed=3.3, low_floor=0.0):
    """Apply the BCI transfer function: threshold-linear with saturation,
    plus an empirical noise-driven stepping floor.

    Parameters
    ----------
    fluorescence : array — raw CN fluorescence values
    lower : float — lower threshold (speed = 0 below this in linear model)
    upper : float — upper threshold (speed = max_speed above this)
    max_speed : float — saturation speed (default 3.3)
    low_floor : float — noise-driven minimum speed (default 0). Pass
        low_floor_for(mouse) to use the empirical per-mouse floor.

    Returns
    -------
    speed : array — same shape as fluorescence, values in [0, max_speed]
    """
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(fluorescence)
    speed = (fluorescence - lower) / gain * max_speed
    speed = np.clip(speed, 0, max_speed)
    return np.maximum(speed, low_floor)


# Build continuous CN fluorescence from roi_csv
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
frames_per_file = ops['frames_per_file']
cn_ind = data['cn_csv_index'][0]

roi = np.copy(data['roi_csv'])
# Fix roi frame counter wrapping
inds_wrap = np.where(np.diff(roi[:, 1]) < 0)[0]
for i in range(len(inds_wrap)):
    ind = inds_wrap[i]
    roi[ind+1:, 1] = roi[ind+1:, 1] + roi[ind, 1]
    roi[ind+1:, 0] = roi[ind+1:, 0] + roi[ind, 0]

# Interpolate to uniform frame grid
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)

# Extract per-trial CN fluorescence and compute speed under each epoch's transfer function
cn_fluor_per_trial = []  # list of arrays, one per trial
cn_fluor_stp = []        # frame index of reward (or end of trial for misses)
speed_actual = np.full(trl, np.nan)  # mean speed under actual thresholds
strt = 0
for i in range(min(trl, len(frames_per_file))):
    ind = np.arange(strt, strt + frames_per_file[i], dtype=int)
    ind = np.clip(ind, 0, len(roi_interp) - 1)
    fluor = roi_interp[ind, cn_ind + 2]
    cn_fluor_per_trial.append(fluor)

    # Compute mean speed under this trial's actual thresholds (up to reward or end)
    if hit[i]:
        # Find frame index corresponding to reward time
        t_trial = roi_interp[ind, 0] - roi_interp[ind[0], 0]
        stp = np.searchsorted(t_trial, rt[i])
        stp = min(stp, len(fluor))
    else:
        stp = len(fluor)
    cn_fluor_stp.append(stp)

    spd = transfer_fun(fluor[:stp], thr_lower[i], thr_upper[i],
                       low_floor=low_floor_for(mouse))
    speed_actual[i] = np.nanmean(spd)
    strt += frames_per_file[i]

print(f"Extracted fluorescence for {len(cn_fluor_per_trial)} trials")

#%% ============================================================================
# CELL 3: Compute expected hit rate & speed — correct (with saturation)
# ============================================================================
# For each epoch after the first, ask: if we take the CN fluorescence from the
# PREVIOUS epoch and pass it through the NEW transfer function, what would
# the hit rate and speed have been?
#
# Also compute the flawed linear estimate for comparison.

# We need a hit criterion: trial is a "hit" if mean speed > some threshold.
# Use the actual criterion: reward is given when lickport reaches target.
# Simpler proxy: a trial "hits" if the animal was rewarded. For the counterfactual,
# we check whether mean speed under the new transfer function exceeds the
# minimum speed needed to reach reward in time.

# For each epoch, compute:
#   actual_hit_rate: fraction of trials rewarded
#   expected_hit_rate_correct: pass previous epoch's fluorescence through new transfer function
#   expected_hit_rate_linear: the flawed linear scaling

N_REF = 10  # number of reference trials from previous epoch to use

epoch_stats = []

for ei in range(n_epochs):
    t0, t1 = switches[ei], epoch_ends[ei]
    n_ep = t1 - t0
    lower_cur = thr_lower[t0]
    upper_cur = thr_upper[t0]

    actual_hr = np.nanmean(hit[t0:t1])
    actual_speed = np.nanmean(speed_actual[t0:t1])

    # CN fluorescence: mean over pre-reward period for this epoch
    cn_mean = np.nanmean([np.nanmean(cn_fluor_per_trial[t])
                          for t in range(t0, min(t1, len(cn_fluor_per_trial)))])

    # Reference: first 10 trials (or fewer if first switch < 10)
    n_ref = min(10, epoch_ends[0])
    ref_trials = list(range(0, n_ref))
    lower_ref = thr_lower[0]
    upper_ref = thr_upper[0]

    if ei == 0:
        # First epoch — expected = actual (same thresholds as reference)
        epoch_stats.append({
            'epoch': ei,
            'trial_start': t0,
            'trial_end': t1,
            'n_trials': n_ep,
            'lower': lower_cur,
            'upper': upper_cur,
            'actual_hr': actual_hr,
            'actual_speed': actual_speed,
            'cn_mean': cn_mean,
            'expected_hr_correct': actual_hr,
            'expected_hr_linear': np.nan,
            'expected_speed_correct': actual_speed,
            'expected_speed_linear': np.nan,
            'frac_saturated': np.nan,
        })
        continue

    lower_prev = lower_ref
    upper_prev = upper_ref

    # For each reference trial, compute speed under NEW thresholds
    expected_speeds_correct = []
    expected_hits_correct = []
    expected_speeds_linear = []
    frac_sat_list = []

    for t in ref_trials:
        if t >= len(cn_fluor_per_trial):
            continue
        fluor_use = cn_fluor_per_trial[t][:cn_fluor_stp[t]]

        # Pass fluorescence through BOTH old and new transfer functions
        spd_old = transfer_fun(fluor_use, lower_prev, upper_prev,
                               low_floor=low_floor_for(mouse))
        spd_new = transfer_fun(fluor_use, lower_cur, upper_cur,
                               low_floor=low_floor_for(mouse))
        mean_old = np.nanmean(spd_old)
        mean_new = np.nanmean(spd_new)
        expected_speeds_correct.append(mean_new)

        # Fraction of frames above saturation under OLD thresholds
        frac_sat = np.mean(fluor_use > upper_prev)
        frac_sat_list.append(frac_sat)

    # Expected hit rate: scale the previous epoch's actual hit rate by the
    # ratio of mean speeds. This asks: if the lickport moved this fraction
    # as fast, what fraction of trials would still finish in time?
    # Use per-trial speed ratios to get a more accurate estimate.
    speed_old_all = []
    speed_new_all = []
    for t in ref_trials:
        if t >= len(cn_fluor_per_trial):
            continue
        fluor_use = cn_fluor_per_trial[t][:cn_fluor_stp[t]]
        speed_old_all.append(np.nanmean(transfer_fun(
            fluor_use, lower_prev, upper_prev,
            low_floor=low_floor_for(mouse))))
        speed_new_all.append(np.nanmean(transfer_fun(
            fluor_use, lower_cur, upper_cur,
            low_floor=low_floor_for(mouse))))
    speed_old_all = np.array(speed_old_all)
    speed_new_all = np.array(speed_new_all)

    # For hit trials: scale RT by speed ratio, check if still < timeout
    # For miss trials: they remain misses (already timed out under old)
    prev_hit_rate = np.nanmean(hit[ref_trials])
    for t in ref_trials:
        ti = t - ref_trials[0]
        if ti >= len(speed_old_all):
            continue
        if hit[t] and speed_old_all[ti] > 0:
            speed_ratio = speed_new_all[ti] / speed_old_all[ti]
            if speed_ratio > 0:
                expected_hits_correct.append(rt[t] / speed_ratio < 10.0)
            else:
                expected_hits_correct.append(False)
        else:
            expected_hits_correct.append(False)

    # Expected HR = fraction of reference trials that would hit under new thresholds
    if expected_hits_correct:
        expected_hr_correct = np.nanmean(expected_hits_correct)
    else:
        expected_hr_correct = np.nan

    expected_speed_correct = np.nanmean(expected_speeds_correct) if expected_speeds_correct else np.nan
    expected_speed_linear = np.nan  # not computing linear estimate
    expected_hr_linear = np.nan
    frac_saturated = np.nanmean(frac_sat_list) if frac_sat_list else np.nan

    epoch_stats.append({
        'epoch': ei,
        'trial_start': t0,
        'trial_end': t1,
        'n_trials': n_ep,
        'lower': lower_cur,
        'upper': upper_cur,
        'actual_hr': actual_hr,
        'actual_speed': actual_speed,
        'cn_mean': cn_mean,
        'expected_hr_correct': expected_hr_correct,
        'expected_hr_linear': expected_hr_linear,
        'expected_speed_correct': expected_speed_correct,
        'expected_speed_linear': expected_speed_linear,
        'frac_saturated': frac_saturated,
    })

print(f"\n{'Epoch':>5} {'Trials':>8} {'Lower':>6} {'Upper':>6} {'ActHR':>6} "
      f"{'ExpHR_c':>8} {'ExpHR_l':>8} {'FracSat':>8}")
print("-" * 72)
for s in epoch_stats:
    print(f"{s['epoch']:5d} {s['trial_start']:3d}-{s['trial_end']-1:<4d} "
          f"{s['lower']:6.0f} {s['upper']:6.0f} {s['actual_hr']:6.2f} "
          f"{s['expected_hr_correct']:8.2f} {s['expected_hr_linear']:8.2f} "
          f"{s['frac_saturated']:8.2f}")

#%% ============================================================================
# CELL 4: Single-session figure (matches original Bpod layout)
# ============================================================================
# Layout 2x3:
#   (231) Hit rate + expected (correct)    (232) Raw CN fluor + thresholds   (233) CN heatmap
#   (234) CN activity vs trial             (235) CN tuning vs trial          (236) Actual vs expected RT

# Build expected hit rate trace (step function across epochs, correct transfer fn)
expected_hr_trace = np.full(trl, np.nan)
for s in epoch_stats:
    if np.isfinite(s['expected_hr_correct']):
        expected_hr_trace[s['trial_start']:s['trial_end']] = s['expected_hr_correct']

# Compute actual vs expected RT per epoch using correct transfer function
# For each epoch, compute mean RT. For expected: scale previous-epoch RT by speed ratio.
actual_rt_epoch = np.full(n_epochs, np.nan)
expected_rt_epoch = np.full(n_epochs, np.nan)
for ei in range(n_epochs):
    t0, t1 = switches[ei], epoch_ends[ei]
    actual_rt_epoch[ei] = np.nanmean(rt[t0:t1])

    if ei > 0:
        prev_t0 = switches[ei-1]
        prev_t1 = epoch_ends[ei-1]
        ref_start = max(prev_t0, prev_t1 - N_REF)
        ref_trials = list(range(ref_start, prev_t1))
        lower_prev = thr_lower[ref_trials[0]]
        upper_prev = thr_upper[ref_trials[0]]
        lower_cur = thr_lower[t0]
        upper_cur = thr_upper[t0]

        ratios = []
        for t in ref_trials:
            if t >= len(cn_fluor_per_trial):
                continue
            fl = cn_fluor_per_trial[t][:cn_fluor_stp[t]]
            spd_old = np.nanmean(transfer_fun(fl, lower_prev, upper_prev,
                                              low_floor=low_floor_for(mouse)))
            spd_new = np.nanmean(transfer_fun(fl, lower_cur, upper_cur,
                                              low_floor=low_floor_for(mouse)))
            if spd_old > 0:
                ratios.append(spd_new / spd_old)
        if ratios:
            mean_ratio = np.nanmean(ratios)
            if mean_ratio > 0:
                expected_rt_epoch[ei] = np.nanmean(rt[ref_start:prev_t1]) / mean_ratio

# Build continuous time axis and threshold traces from roi_interp
t_cont = roi_interp[:, 0]
cn_cont = roi_interp[:, cn_ind + 2]
thr_time_lower = np.full(len(t_cont), np.nan)
thr_time_upper = np.full(len(t_cont), np.nan)
strt = 0
for i in range(min(trl, len(frames_per_file))):
    idx = np.arange(strt, strt + frames_per_file[i], dtype=int)
    idx = np.clip(idx, 0, len(thr_time_lower) - 1)
    thr_time_lower[idx] = thr_lower[i]
    thr_time_upper[idx] = thr_upper[i]
    strt += frames_per_file[i]

fig = plt.figure(figsize=(8, 4))

# --- (241) Hit rate + expected ---
ax = plt.subplot(241)
win = 10
hr_smooth = np.convolve(hit.astype(float), np.ones(win)/win, mode='valid')
ax.plot(np.arange(win-1, win-1+len(hr_smooth)), hr_smooth, 'k', linewidth=0.8)
ax.plot(expected_hr_trace, color='gray', linewidth=1.0)
for sw in switches[1:]:
    ax.axvline(sw, ymin=0, ymax=0.08, color='k', linewidth=1)
ax.set_xlim(win-1, trl)
ax.set_xlabel('Trial #')
ax.set_ylabel('Hit rate')
ax.set_ylim(-0.05, 1.05)
ax.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (242) Raw CN fluorescence + threshold lines ---
ax = plt.subplot(242)
ax.plot(t_cont, cn_cont, 'k', linewidth=0.04)
ax.plot(t_cont, thr_time_lower, 'b', linewidth=0.5)
ax.plot(t_cont, thr_time_upper, 'r', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Raw fluorescence')
ax.set_title(f'{mouse}  {session}', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (243) CN heatmap (trials x time) ---
ax = plt.subplot(243)
ax.imshow(F[:, cn, :].T, aspect='auto', interpolation='nearest')
ax.set_xlabel('Time from trial start (s)')
ax.set_ylabel('Trial #')
# Approximate tick marks: 0 and 10s
n_frames_trial = F.shape[0]
frames_10s = int(10.0 / dt_si)
ax.set_xticks([0, min(frames_10s, n_frames_trial-1)])
ax.set_xticklabels(['0', '10'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (244) Peak activity per trial (95th percentile of raw fluorescence) ---
ax = plt.subplot(244)
# Also compute speed heatmap under hardest thresholds (needed for subplot 248)
hardest_lower = np.nanmax(thr_lower)
hardest_upper = np.nanmax(thr_upper)
n_frames_max = F.shape[0]
speed_heatmap = np.full((n_frames_max, trl), np.nan)
peak_activity = np.full(trl, np.nan)
for ti in range(trl):
    raw_fl = cn_fluor_per_trial[ti]
    n_fr = min(len(raw_fl), n_frames_max)
    speed_heatmap[:n_fr, ti] = transfer_fun(raw_fl[:n_fr], hardest_lower,
                                            hardest_upper,
                                            low_floor=low_floor_for(mouse))
    if len(raw_fl) > 0:
        peak_activity[ti] = np.nanpercentile(raw_fl, 100)
peak_smooth = np.convolve(peak_activity[np.isfinite(peak_activity)],
                          np.ones(win)/win, mode='valid')
valid_pk = np.where(np.isfinite(peak_activity))[0]
ax.plot(valid_pk[win-1:win-1+len(peak_smooth)], peak_smooth, 'k', linewidth=0.8)
for sw in switches[1:]:
    ax.axvline(sw, ymin=0, ymax=0.08, color='k', linewidth=1)
ax.set_xlabel('Trial #')
ax.set_ylabel('Peak activity (100th pctl)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (245) CN activity vs trial ---
ax = plt.subplot(245)
cn_trial_mean = np.nanmean(F[:, cn, :], axis=0)
cn_smooth = np.convolve(cn_trial_mean, np.ones(win), mode='valid') / win
ax.plot(np.arange(win-1, trl), cn_smooth, 'k', linewidth=0.8)
for sw in switches[1:]:
    ax.axvline(sw, ymin=0, ymax=0.08, color='k', linewidth=1)
ax.set_xlabel('Trial #')
ax.set_ylabel('CN activity')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (246) CN tuning vs trial (baseline-subtracted, smoothed) ---
ax = plt.subplot(246)
ff = F[:, cn, :].copy()
for ti in range(ff.shape[1]):
    ff[:, ti] = ff[:, ti] - np.nanmean(ff[0:20, ti])
tuning = np.nanmean(ff[60:, :], axis=0)
n_smooth = max(switches[1], 1) if len(switches) > 1 else 10
tuning_smooth = np.convolve(tuning, np.ones(n_smooth), mode='valid') / n_smooth
ax.plot(np.arange(n_smooth-1, trl), tuning_smooth, 'k', linewidth=0.8)
for sw in switches[1:]:
    ax.axvline(sw, ymin=0, ymax=0.08, color='k', linewidth=1)
ax.set_xlabel('Trial #')
ax.set_ylabel('CN Tuning')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (247) Actual vs expected RT per epoch (bar plot) ---
ax = plt.subplot(247)
x_bar = np.arange(n_epochs)
w = 0.35
ax.bar(x_bar - w/2, actual_rt_epoch, width=w, color='k', label='Actual')
ax.bar(x_bar + w/2, expected_rt_epoch, width=w, color='gray', label='Expected')
ax.set_xlabel('Epoch')
ax.set_ylabel('Time to reward (s)')
ax.set_xticks(x_bar)
ax.legend(frameon=False, fontsize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (248) Cursor speed vs trial (under hardest thresholds) ---
ax = plt.subplot(248)
speed_trial_mean = np.nanmean(speed_heatmap, axis=0)
speed_smooth = np.convolve(speed_trial_mean, np.ones(win)/win, mode='valid')
ax.plot(np.arange(win-1, trl), speed_smooth, 'k', linewidth=0.8)
for sw in switches[1:]:
    ax.axvline(sw, ymin=0, ymax=0.08, color='k', linewidth=1)
ax.set_xlabel('Trial #')
ax.set_ylabel('Cursor speed\n(hardest thresholds)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

LOCAL_FIG_DIR = os.path.join(RESULTS_DIR, 'threshold_figs')
os.makedirs(LOCAL_FIG_DIR, exist_ok=True)
fname = f'threshold_analysis_{mouse}_{session}'
fig.savefig(os.path.join(LOCAL_FIG_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {os.path.join(LOCAL_FIG_DIR, fname)}.png")

#%% ============================================================================
# CELL 5: Loop over all sessions — collect threshold epoch stats
# ============================================================================
all_epoch_stats = []
all_session_trials = {}   # (mouse, session) -> {'hit': array, 'rt': array, 'switches': array}
LOCAL_FIG_DIR = os.path.join(RESULTS_DIR, 'threshold_figs')
os.makedirs(LOCAL_FIG_DIR, exist_ok=True)

for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        mouse = mice[mi]
        session = list_of_dirs['Session'][session_inds[sii]]
        if (mouse, session) in _qc_fail:
            continue

        folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                  + mouse + '/' + session + '/pophys/')

        try:
            bci_keys_loop = ['F', 'mouse', 'session', 'conditioned_neuron',
                             'dt_si', 'reward_time', 'BCI_thresholds',
                             'roi_csv', 'cn_csv_index']
            data = ddct.load_hdf5(folder, bci_keys_loop, [])

            F = data['F']
            trl = F.shape[2]
            cn = data['conditioned_neuron'][0][0]

            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt_sess = np.array([x[0] if len(x) > 0 else np.nan
                                for x in data['reward_time']], dtype=float)
            hit_sess = np.isfinite(rt_sess)

            # Thresholds
            BCI_thr = np.asarray(data['BCI_thresholds'], dtype=float)
            thr_l = BCI_thr[0, :].copy()
            thr_u = BCI_thr[1, :].copy()
            for i in range(1, thr_u.size):
                if np.isnan(thr_u[i]): thr_u[i] = thr_u[i-1]
                if np.isnan(thr_l[i]): thr_l[i] = thr_l[i-1]
            if np.isnan(thr_u[0]) and np.any(np.isfinite(thr_u)):
                thr_u[0] = thr_u[np.isfinite(thr_u)][0]
            if np.isnan(thr_l[0]) and np.any(np.isfinite(thr_l)):
                thr_l[0] = thr_l[np.isfinite(thr_l)][0]

            # Pad to trl length with last valid value
            if len(thr_u) < trl:
                thr_u = np.concatenate([thr_u, np.full(trl - len(thr_u), thr_u[-1])])
                thr_l = np.concatenate([thr_l, np.full(trl - len(thr_l), thr_l[-1])])

            # Detect switches
            d_u = np.diff(thr_u)
            sw = np.where((d_u != 0) & np.isfinite(d_u))[0] + 1
            sw = np.concatenate(([0], sw))
            ep_ends = np.concatenate((sw[1:], [trl]))

            # Build roi_interp for CN fluorescence
            ops_s = np.load(folder + r'/suite2p_BCI/plane0/ops.npy',
                            allow_pickle=True).tolist()
            fpf = ops_s['frames_per_file']
            cn_idx = data['cn_csv_index'][0]

            roi_s = np.copy(data['roi_csv'])
            wraps = np.where(np.diff(roi_s[:, 1]) < 0)[0]
            for wi in range(len(wraps)):
                ww = wraps[wi]
                roi_s[ww+1:, 1] += roi_s[ww, 1]
                roi_s[ww+1:, 0] += roi_s[ww, 0]

            frm = np.arange(1, int(np.max(roi_s[:, 1])) + 1)
            ifunc = interp1d(roi_s[:, 1], roi_s, axis=0,
                             kind='linear', fill_value='extrapolate')
            roi_i = ifunc(frm)

            # Per-trial fluorescence and truncation at reward
            cn_fluor = []
            cn_stp = []
            strt = 0
            for ti in range(min(trl, len(fpf))):
                idx = np.arange(strt, strt + fpf[ti], dtype=int)
                idx = np.clip(idx, 0, len(roi_i) - 1)
                fluor_ti = roi_i[idx, cn_idx + 2]
                cn_fluor.append(fluor_ti)
                if hit_sess[ti]:
                    t_trial = roi_i[idx, 0] - roi_i[idx[0], 0]
                    stp = min(np.searchsorted(t_trial, rt_sess[ti]), len(fluor_ti))
                else:
                    stp = len(fluor_ti)
                cn_stp.append(stp)
                strt += fpf[ti]

            # Per-epoch stats
            for ei in range(len(sw)):
                t0, t1 = sw[ei], ep_ends[ei]
                lower_c = thr_l[t0]
                upper_c = thr_u[t0]
                actual_hr = np.nanmean(hit_sess[t0:t1])
                cn_mean = np.nanmean(F[:, cn, t0:t1])

                rec = {
                    'mouse': mouse, 'session': session,
                    'epoch': ei, 'n_epochs': len(sw),
                    'trial_start': t0, 'trial_end': t1,
                    'n_trials': t1 - t0,
                    'lower': lower_c, 'upper': upper_c,
                    'actual_hr': actual_hr,
                    'cn_mean': cn_mean,
                }

                # Reference: first 10 trials of epoch 0 (or fewer if first switch < 10)
                n_ref = min(10, ep_ends[0])
                ref_trials = list(range(0, n_ref))
                lower_ref = thr_l[0]
                upper_ref = thr_u[0]

                if ei > 0:
                    # Replay reference activity through current epoch's thresholds
                    exp_hits = []
                    exp_rts = []
                    frac_sats = []
                    for t in ref_trials:
                        if t >= len(cn_fluor):
                            continue
                        fl = cn_fluor[t][:cn_stp[t]]
                        spd_new = np.nanmean(transfer_fun(
                            fl, lower_c, upper_c,
                            low_floor=low_floor_for(mouse)))
                        spd_old = np.nanmean(transfer_fun(
                            fl, lower_ref, upper_ref,
                            low_floor=low_floor_for(mouse)))

                        if hit_sess[t] and spd_old > 0 and spd_new > 0:
                            ratio = spd_new / spd_old
                            scaled_rt = rt_sess[t] / ratio
                            # 10s is the hit/miss boundary, NOT a cap on the
                            # RT estimate: expected_rt measures HOW FAR OFF the
                            # animal would be, which can exceed 10s.
                            exp_hits.append(scaled_rt < 10.0)
                            exp_rts.append(scaled_rt)
                        else:
                            # Reference-epoch miss: no finite RT to scale.
                            # Penalize at MISS_RT (timed out, would be longer).
                            exp_hits.append(False)
                            exp_rts.append(MISS_RT)

                        frac_sats.append(np.mean(fl > upper_ref))

                    rec['expected_hr_correct'] = np.nanmean(exp_hits) if exp_hits else np.nan
                    rec['frac_saturated'] = np.nanmean(frac_sats) if frac_sats else np.nan
                    rec['expected_rt'] = np.nanmean(exp_rts) if exp_rts else np.nan

                    # Actual RT (misses penalized at MISS_RT, not the timeout)
                    rt_epoch = rt_sess[t0:t1].copy()
                    rt_epoch[~np.isfinite(rt_epoch)] = MISS_RT
                    rec['actual_rt'] = np.nanmean(rt_epoch)

                    # Actual RPE: difference between actual hit rate and correct expected
                    rec['actual_rpe'] = actual_hr - rec['expected_hr_correct']

                    # RT recovery ratio: how much of the expected slowdown
                    # the animal avoids. 1 = fully recovered, 0 = matched expected,
                    # negative = worse than expected
                    if rec['expected_rt'] > 0:
                        # Use epoch 0's actual RT as the baseline
                        ep0_rec = [r for r in all_epoch_stats
                                   if r['mouse'] == mouse and r['session'] == session
                                   and r['epoch'] == 0]
                        if ep0_rec:
                            ep0_actual_rt = ep0_rec[0]['actual_rt']
                            expected_delta_rt = rec['expected_rt'] - ep0_actual_rt
                            actual_delta_rt = rec['actual_rt'] - ep0_actual_rt
                            if abs(expected_delta_rt) > 0.01:
                                rec['rt_recovery_ratio'] = 1.0 - (actual_delta_rt / expected_delta_rt)
                            else:
                                rec['rt_recovery_ratio'] = np.nan
                            rec['expected_delta_rt'] = expected_delta_rt
                            rec['actual_delta_rt'] = actual_delta_rt
                        else:
                            rec['rt_recovery_ratio'] = np.nan
                            rec['expected_delta_rt'] = np.nan
                            rec['actual_delta_rt'] = np.nan
                    else:
                        rec['rt_recovery_ratio'] = np.nan
                        rec['expected_delta_rt'] = np.nan
                        rec['actual_delta_rt'] = np.nan

                else:
                    # Epoch 0: expected = actual (same thresholds as reference)
                    # Use the same reference window (first 10 trials) for consistency
                    ref_rt = rt_sess[ref_trials].copy()
                    ref_rt[~np.isfinite(ref_rt)] = MISS_RT
                    rec['expected_hr_correct'] = np.nanmean(hit_sess[ref_trials])
                    rec['frac_saturated'] = np.nan
                    rt_epoch0 = rt_sess[t0:t1].copy()
                    rt_epoch0[~np.isfinite(rt_epoch0)] = MISS_RT
                    rec['actual_rt'] = np.nanmean(rt_epoch0)
                    rec['expected_rt'] = np.nanmean(ref_rt)
                    rec['actual_rpe'] = 0.0
                    rec['rt_recovery_ratio'] = np.nan
                    rec['expected_delta_rt'] = np.nan
                    rec['actual_delta_rt'] = np.nan

                all_epoch_stats.append(rec)

            # Store trial-level data for switch-aligned analysis
            # CN activity: trial start to reward (or timeout at 10s).
            # F[0] is PRE_TRIAL_S before trial start, so both the window start
            # (trial-start go cue) and end (reward) are offset by the buffer.
            dt_si_s = data['dt_si']
            pre_trial_f = int(PRE_TRIAL_S / dt_si_s)
            cn_trial_mean = np.full(trl, np.nan)
            for ti_cn in range(trl):
                t_end = rt_sess[ti_cn] if np.isfinite(rt_sess[ti_cn]) else 10.0
                end_frame = min(int((t_end + PRE_TRIAL_S) / dt_si_s), F.shape[0])
                if end_frame > pre_trial_f:
                    cn_trial_mean[ti_cn] = np.nanmean(
                        F[pre_trial_f:end_frame, cn, ti_cn])
            # Compute per-trial RPE matching sliding_window_temporal_offset.py
            rt_filled_sess = rt_sess.copy()
            rt_filled_sess[~np.isfinite(rt_filled_sess)] = 30.0
            rt_rpe_sess = -compute_rpe(rt_filled_sess, baseline=2.0,
                                       tau=10, fill_value=10.0)
            hit_rpe_sess = compute_rpe(hit_sess.astype(float), baseline=1.0,
                                       tau=10, fill_value=0.0)

            # Peak activity (100th percentile of raw fluorescence per trial)
            peak_activity_s = np.full(trl, np.nan)
            for ti_pk in range(min(trl, len(cn_fluor))):
                fl_pk = cn_fluor[ti_pk]
                if len(fl_pk) > 0:
                    peak_activity_s[ti_pk] = np.nanpercentile(fl_pk, 100)

            # Cursor speed under hardest thresholds
            hardest_lower_s = np.nanmax(thr_l)
            hardest_upper_s = np.nanmax(thr_u)
            cursor_speed_s = np.full(trl, np.nan)
            for ti_sp in range(min(trl, len(cn_fluor))):
                fl_sp = cn_fluor[ti_sp]
                if len(fl_sp) > 0:
                    cursor_speed_s[ti_sp] = np.nanmean(
                        transfer_fun(fl_sp, hardest_lower_s, hardest_upper_s,
                                     low_floor=low_floor_for(mouse)))

            all_session_trials[(mouse, session)] = {
                'hit': hit_sess.copy(),
                'rt': rt_sess.copy(),
                'switches': sw.copy(),
                'thr_u': thr_u.copy(),
                'cn': cn_trial_mean.copy(),
                'rt_rpe': rt_rpe_sess.copy(),
                'hit_rpe': hit_rpe_sess.copy(),
                'peak_activity': peak_activity_s.copy(),
                'cursor_speed': cursor_speed_s.copy(),
            }

            # --- Save per-session figure (same layout as Cell 4) ---
            dt_si_s = data['dt_si']
            n_epochs_s = len(sw)

            # Build expected HR trace and expected RT per epoch
            exp_hr_trace = np.full(trl, np.nan)
            actual_rt_ep = np.full(n_epochs_s, np.nan)
            expected_rt_ep = np.full(n_epochs_s, np.nan)
            sess_recs = [r for r in all_epoch_stats
                         if r['mouse'] == mouse and r['session'] == session]
            for r in sess_recs:
                if np.isfinite(r.get('expected_hr_correct', np.nan)):
                    exp_hr_trace[r['trial_start']:r['trial_end']] = r['expected_hr_correct']
                actual_rt_ep[r['epoch']] = r.get('actual_rt', np.nan)
                expected_rt_ep[r['epoch']] = r.get('expected_rt', np.nan)

            # Continuous threshold traces
            t_cont_s = roi_i[:, 0]
            cn_cont_s = roi_i[:, cn_idx + 2]
            thr_time_l = np.full(len(t_cont_s), np.nan)
            thr_time_u = np.full(len(t_cont_s), np.nan)
            strt_f = 0
            for ti2 in range(min(trl, len(fpf))):
                idx2 = np.arange(strt_f, strt_f + fpf[ti2], dtype=int)
                idx2 = np.clip(idx2, 0, len(thr_time_l) - 1)
                thr_time_l[idx2] = thr_l[ti2]
                thr_time_u[idx2] = thr_u[ti2]
                strt_f += fpf[ti2]

            fig_s = plt.figure(figsize=(8, 4))

            win = 10

            # (241) Hit rate + expected
            ax = plt.subplot(241)
            hr_sm = np.convolve(hit_sess.astype(float), np.ones(win)/win, mode='valid')
            ax.plot(np.arange(win-1, trl), hr_sm, 'k', linewidth=0.8)
            ax.plot(exp_hr_trace, color='gray', linewidth=1.0)
            for s_sw in sw[1:]:
                ax.axvline(s_sw, ymin=0, ymax=0.08, color='k', linewidth=1)
            ax.set_xlim(win-1, trl)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('Hit rate')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (242) Raw CN fluorescence + thresholds
            ax = plt.subplot(242)
            ax.plot(t_cont_s, cn_cont_s, 'k', linewidth=0.04)
            ax.plot(t_cont_s, thr_time_l, 'b', linewidth=0.5)
            ax.plot(t_cont_s, thr_time_u, 'r', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Raw fluorescence')
            ax.set_title(f'{mouse}  {session}', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (243) CN heatmap
            ax = plt.subplot(243)
            ax.imshow(F[:, cn, :].T, aspect='auto', interpolation='nearest')
            ax.set_xlabel('Time from trial start (s)')
            ax.set_ylabel('Trial #')
            n_frames_t = F.shape[0]
            frames_10s = int(10.0 / dt_si_s)
            ax.set_xticks([0, min(frames_10s, n_frames_t-1)])
            ax.set_xticklabels(['0', '10'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (244) Peak activity per trial (100th percentile)
            ax = plt.subplot(244)
            pk_valid = peak_activity_s[np.isfinite(peak_activity_s)]
            if len(pk_valid) >= win:
                pk_sm = np.convolve(pk_valid, np.ones(win)/win, mode='valid')
                pk_idx = np.where(np.isfinite(peak_activity_s))[0]
                ax.plot(pk_idx[win-1:win-1+len(pk_sm)], pk_sm, 'k', linewidth=0.8)
            for s_sw in sw[1:]:
                ax.axvline(s_sw, ymin=0, ymax=0.08, color='k', linewidth=1)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('Peak activity (100th pctl)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (245) CN activity vs trial
            ax = plt.subplot(245)
            cn_tmean = np.nanmean(F[:, cn, :], axis=0)
            cn_sm = np.convolve(cn_tmean, np.ones(win), mode='valid') / win
            ax.plot(np.arange(win-1, trl), cn_sm, 'k', linewidth=0.8)
            for s_sw in sw[1:]:
                ax.axvline(s_sw, ymin=0, ymax=0.08, color='k', linewidth=1)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('CN activity')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (246) CN tuning vs trial
            ax = plt.subplot(246)
            ff_s = F[:, cn, :].copy()
            for ti2 in range(ff_s.shape[1]):
                ff_s[:, ti2] -= np.nanmean(ff_s[0:20, ti2])
            tuning_s = np.nanmean(ff_s[60:, :], axis=0)
            n_sm = max(sw[1], 1) if len(sw) > 1 else 10
            tun_sm = np.convolve(tuning_s, np.ones(n_sm), mode='valid') / n_sm
            ax.plot(np.arange(n_sm-1, trl), tun_sm, 'k', linewidth=0.8)
            for s_sw in sw[1:]:
                ax.axvline(s_sw, ymin=0, ymax=0.08, color='k', linewidth=1)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('CN Tuning')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (247) Actual vs expected RT per epoch
            ax = plt.subplot(247)
            x_bar = np.arange(n_epochs_s)
            w_bar = 0.35
            ax.bar(x_bar - w_bar/2, actual_rt_ep, width=w_bar, color='k', label='Actual')
            ax.bar(x_bar + w_bar/2, expected_rt_ep, width=w_bar, color='gray', label='Expected')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time to reward (s)')
            ax.set_xticks(x_bar)
            ax.legend(frameon=False, fontsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # (248) Cursor speed vs trial (under hardest thresholds)
            ax = plt.subplot(248)
            cs_valid = cursor_speed_s[np.isfinite(cursor_speed_s)]
            if len(cs_valid) >= win:
                cs_sm = np.convolve(cs_valid, np.ones(win)/win, mode='valid')
                cs_idx = np.where(np.isfinite(cursor_speed_s))[0]
                ax.plot(cs_idx[win-1:win-1+len(cs_sm)], cs_sm, 'k', linewidth=0.8)
            for s_sw in sw[1:]:
                ax.axvline(s_sw, ymin=0, ymax=0.08, color='k', linewidth=1)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('Cursor speed\n(hardest thresholds)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            fname_s = f'threshold_analysis_{mouse}_{session}'
            fig_s.savefig(os.path.join(LOCAL_FIG_DIR, f'{fname_s}.png'), dpi=300)
            plt.close(fig_s)

            print(f"  {mouse} {session}: {len(sw)} epochs, "
                  f"{trl} trials — saved {fname_s}.png")

        except Exception as e:
            print(f"  FAILED {mouse} {session}: {e}")
            continue

# Post-process: add RPE metrics per epoch from trial-level RPE
RPE_WINDOW = 15  # first N trials after switch to measure acute RPE
for rec in all_epoch_stats:
    key = (rec['mouse'], rec['session'])
    if key not in all_session_trials:
        rec['rpe_integral'] = np.nan
        rec['rpe_mean_acute'] = np.nan
        continue
    tdata = all_session_trials[key]
    t0, t1 = rec['trial_start'], rec['trial_end']
    epoch_rpe = tdata['rt_rpe'][t0:t1]
    rec['rpe_integral'] = np.nansum(epoch_rpe)
    # Mean RPE over first RPE_WINDOW trials (acute response)
    acute_rpe = epoch_rpe[:RPE_WINDOW]
    rec['rpe_mean_acute'] = np.nanmean(acute_rpe) if len(acute_rpe) > 0 else np.nan

print(f"\nCollected {len(all_epoch_stats)} epoch records "
      f"from {len(set((s['mouse'],s['session']) for s in all_epoch_stats))} sessions")

# --- Save to pickle so downstream scripts can load without re-running CELL 5 ---
import pickle as _pickle
_pkl_path = os.path.join(RESULTS_DIR, 'all_epoch_stats.pkl')
with open(_pkl_path, 'wb') as _fh:
    _pickle.dump({'all_epoch_stats': all_epoch_stats,
                  'all_session_trials': all_session_trials}, _fh)
print(f"Saved {_pkl_path}")

#%% ============================================================================
# CELL 6: Population summary — expected vs actual hit rate at threshold changes
# ============================================================================
# Filter to epochs > 0 (where we have expected values)
switch_epochs = [s for s in all_epoch_stats if s['epoch'] > 0
                 and np.isfinite(s['expected_hr_correct'])]

actual_hrs = np.array([s['actual_hr'] for s in switch_epochs])
expected_hrs = np.array([s['expected_hr_correct'] for s in switch_epochs])
frac_sats = np.array([s['frac_saturated'] for s in switch_epochs])
rpes = np.array([s['actual_rpe'] for s in switch_epochs])

from scipy.stats import wilcoxon, pearsonr

fig6, axes6 = plt.subplots(1, 3, figsize=(7, 2.5),
                            gridspec_kw={'wspace': 0.45, 'left': 0.08,
                                         'right': 0.96, 'bottom': 0.18,
                                         'top': 0.90})

# --- (a) Actual vs expected hit rate ---
ax = axes6[0]
ax.scatter(expected_hrs, actual_hrs, s=15, c='k', alpha=0.5, edgecolors='none')
mn = min(np.nanmin(expected_hrs), np.nanmin(actual_hrs)) - 0.05
mx = 1.05
ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Expected hit rate\n(correct transfer fn)')
ax.set_ylabel('Actual hit rate')
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Stats
n_above = np.sum(actual_hrs > expected_hrs)
stat, p_val = wilcoxon(actual_hrs - expected_hrs)
ax.set_title(f'{n_above}/{len(actual_hrs)} above unity\np={p_val:.4f}', fontsize=7)

# --- (b) Fraction saturated vs RPE ---
ax = axes6[1]
ok = np.isfinite(frac_sats) & np.isfinite(rpes)
ax.scatter(frac_sats[ok], rpes[ok], s=15, c='#ea580c', alpha=0.5, edgecolors='none')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Fraction saturated\n(prev epoch)')
ax.set_ylabel('RPE (actual - expected HR)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
if np.sum(ok) > 3:
    r_sat, p_sat = pearsonr(frac_sats[ok], rpes[ok])
    ax.set_title(f'r={r_sat:.3f}, p={p_sat:.4f}', fontsize=7)

# --- (c) Distribution of RPE ---
ax = axes6[2]
ax.hist(rpes[np.isfinite(rpes)], bins=15, color='k', alpha=0.7, edgecolor='white')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('RPE (actual - expected HR)')
ax.set_ylabel('Count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
med_rpe = np.nanmedian(rpes)
ax.set_title(f'median={med_rpe:.3f}', fontsize=7)

fname6 = 'threshold_population_summary'
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.png'), dpi=300)
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.svg'))
plt.show()
print(f"Saved {fname6}")

#%% ============================================================================
# CELL 7: Save summary text
# ============================================================================
txt_path = os.path.join(RESULTS_DIR, 'threshold_analysis_summary.txt')
with open(txt_path, 'w') as f:
    f.write("THRESHOLD ANALYSIS — CORRECT TRANSFER FUNCTION\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"N_REF = {N_REF} trials from previous epoch\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total epoch transitions: {len(switch_epochs)}\n")
    f.write(f"Sessions: {len(set((s['mouse'],s['session']) for s in switch_epochs))}\n\n")

    f.write("POPULATION SUMMARY\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Actual HR:   mean={np.nanmean(actual_hrs):.3f}, "
            f"median={np.nanmedian(actual_hrs):.3f}\n")
    f.write(f"  Expected HR: mean={np.nanmean(expected_hrs):.3f}, "
            f"median={np.nanmedian(expected_hrs):.3f}\n")
    f.write(f"  RPE:         mean={np.nanmean(rpes):.3f}, "
            f"median={np.nanmedian(rpes):.3f}\n")
    f.write(f"  Actual > Expected: {n_above}/{len(actual_hrs)}\n")
    f.write(f"  Wilcoxon signed-rank p = {p_val:.6f}\n")
    f.write(f"  Frac saturated:  mean={np.nanmean(frac_sats):.3f}, "
            f"median={np.nanmedian(frac_sats):.3f}\n\n")

    f.write("PER-EPOCH DETAIL\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Mouse':>8} {'Session':>8} {'Epoch':>5} {'Trials':>8} "
            f"{'Lower':>6} {'Upper':>6} {'ActHR':>6} {'ExpHR':>6} "
            f"{'RPE':>6} {'FracSat':>8}\n")
    f.write("-" * 90 + "\n")
    for s in all_epoch_stats:
        f.write(f"{s['mouse']:>8} {s['session']:>8} {s['epoch']:5d} "
                f"{s['trial_start']:3d}-{s['trial_end']-1:<4d} "
                f"{s['lower']:6.0f} {s['upper']:6.0f} {s['actual_hr']:6.2f} "
                f"{s.get('expected_hr_correct', np.nan):6.2f} "
                f"{s.get('actual_rpe', np.nan):+6.2f} "
                f"{s.get('frac_saturated', np.nan):8.2f}\n")

print(f"Saved {txt_path}")

#%% ============================================================================
# CELL 8: Switch-aligned hit rate and reward time
# ============================================================================
PRE = 5    # trials before threshold change
POST = 15   # trials after threshold change
MAX_SWITCH_TRIAL = 400   # only include switches within first N trials
# Exclude trials from neighboring epochs (NaN-mask) vs. include everything in
# the window (constant composition across lags, but post-window blends later
# epochs at long lags). Default off — see survivorship note.
MASK_NEIGHBOR_SWITCHES = False

hr_aligned = []   # each entry: array of length PRE+POST
rt_aligned = []
rt_hits_aligned = []  # hit trials only (NaN for misses)
cn_aligned = []
cn_sess_std_aligned = []   # full-session std of CN for each transition's session
rt_rpe_aligned = []
hit_rpe_aligned = []
exp_hr_aligned = []   # expected step functions per transition
exp_rt_aligned = []
transition_ids = []   # (mouse, session, epoch) for each aligned transition
thr_direction = []    # +1 = increase, -1 = decrease
thr_change_mag = []   # signed Δ upper threshold (post − pre)

# Build lookup from (mouse, session, epoch) -> epoch_stat record
epoch_stat_lookup = {}
for rec in all_epoch_stats:
    epoch_stat_lookup[(rec['mouse'], rec['session'], rec['epoch'])] = rec

for (mouse, session), tdata in all_session_trials.items():
    hit = tdata['hit']
    rt = tdata['rt'].copy()
    rt_hits = tdata['rt'].copy()   # NaN for misses
    rt[~np.isfinite(rt)] = 10.0   # fill miss trials with 10s
    cn_act = tdata['cn']
    rt_rpe_act = tdata['rt_rpe']
    hit_rpe_act = tdata['hit_rpe']
    switches = tdata['switches']
    thr_u_sess = tdata['thr_u']
    n_trials = len(hit)

    for si_idx in range(1, len(switches)):   # skip epoch 0 (no switch)
        sw_trial = switches[si_idx]

        if sw_trial >= MAX_SWITCH_TRIAL:
            continue

        # Direction & magnitude of threshold change
        d_thr = thr_u_sess[sw_trial] - thr_u_sess[sw_trial - 1]
        direction = 1 if d_thr > 0 else -1

        # Define the window
        t0 = sw_trial - PRE
        t1 = sw_trial + POST

        # Optional: restrict the window to the two epochs adjacent to THIS
        # switch. With MASK_NEIGHBOR_SWITCHES = False, every in-bounds trial is
        # included (constant composition across lags; double-counting is fine).
        if MASK_NEIGHBOR_SWITCHES:
            other_sw = np.concatenate([switches[:si_idx], switches[si_idx+1:]])
            prev_sw = max([s for s in other_sw if s < sw_trial], default=0)
            next_sw = min([s for s in other_sw if s > sw_trial], default=n_trials)
        else:
            prev_sw, next_sw = 0, n_trials

        hr_row = np.full(PRE + POST, np.nan)
        rt_row = np.full(PRE + POST, np.nan)
        rt_hits_row = np.full(PRE + POST, np.nan)
        cn_row = np.full(PRE + POST, np.nan)
        rt_rpe_row = np.full(PRE + POST, np.nan)
        hit_rpe_row = np.full(PRE + POST, np.nan)

        for k in range(PRE + POST):
            trial_idx = t0 + k
            if trial_idx < 0 or trial_idx >= n_trials:
                continue
            if trial_idx < prev_sw or trial_idx >= next_sw:
                continue
            hr_row[k] = float(hit[trial_idx])
            rt_row[k] = rt[trial_idx]
            rt_hits_row[k] = rt_hits[trial_idx]   # NaN for misses
            cn_row[k] = cn_act[trial_idx]
            rt_rpe_row[k] = rt_rpe_act[trial_idx]
            hit_rpe_row[k] = hit_rpe_act[trial_idx]

        hr_aligned.append(hr_row)
        rt_aligned.append(rt_row)
        rt_hits_aligned.append(rt_hits_row)
        cn_aligned.append(cn_row)
        cn_sess_std_aligned.append(float(np.nanstd(cn_act)))
        rt_rpe_aligned.append(rt_rpe_row)
        hit_rpe_aligned.append(hit_rpe_row)

        # Build expected step function for this transition
        pre_hr = np.nanmean(hr_row[:PRE])
        pre_rt = np.nanmean(rt_row[:PRE])

        rec_post = epoch_stat_lookup.get((mouse, session, si_idx), None)
        if rec_post is not None and np.isfinite(rec_post.get('expected_hr_correct', np.nan)):
            post_hr_exp = rec_post['expected_hr_correct']
            # Use the transfer-function replay expected RT from epoch stats
            post_rt_exp = rec_post.get('expected_rt', np.nan)
        else:
            post_hr_exp = np.nan
            post_rt_exp = np.nan

        exp_hr_row = np.full(PRE + POST, np.nan)
        exp_rt_row = np.full(PRE + POST, np.nan)
        exp_hr_row[:PRE] = pre_hr
        exp_hr_row[PRE:] = post_hr_exp
        exp_rt_row[:PRE] = pre_rt
        exp_rt_row[PRE:] = post_rt_exp

        exp_hr_aligned.append(exp_hr_row)
        exp_rt_aligned.append(exp_rt_row)
        transition_ids.append((mouse, session, si_idx))
        thr_direction.append(direction)
        thr_change_mag.append(d_thr)

thr_direction = np.array(thr_direction)
thr_change_mag = np.array(thr_change_mag)
hr_aligned = np.array(hr_aligned)   # (n_switches, PRE+POST)
rt_aligned = np.array(rt_aligned)
rt_hits_aligned = np.array(rt_hits_aligned)
cn_aligned = np.array(cn_aligned)
cn_sess_std_aligned = np.array(cn_sess_std_aligned)
rt_rpe_aligned = np.array(rt_rpe_aligned)
hit_rpe_aligned = np.array(hit_rpe_aligned)
exp_hr_aligned = np.array(exp_hr_aligned)
exp_rt_aligned = np.array(exp_rt_aligned)

# Speed = 1/RT (misses have RT=10s → speed=0.1; miss hits stay NaN)
speed_aligned = 1.0 / rt_aligned
speed_hits_aligned = 1.0 / rt_hits_aligned
exp_speed_aligned = 1.0 / exp_rt_aligned

trial_axis = np.arange(-PRE, POST)

# --- Helper to compute mean/sem and plot a direction subset ---
def _mean_sem(arr, axis=0):
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis)
    s = np.nanstd(arr, axis=axis) / np.sqrt(np.clip(n, 1, None))
    return m, s

# Aggregate over threshold-increase transitions (the meaningful protocol push).
# (The previous "recovering vs deteriorating" split by post-switch RT slope was
# removed — the post_rt_slope-based subgrouping wasn't substantively useful.)
for dir_label, dir_mask, dir_suffix in [('Threshold increases',
                                         thr_direction > 0, 'inc')]:
    if np.sum(dir_mask) < 2:
        print(f"Skipping {dir_label}: only {np.sum(dir_mask)} transitions")
        continue

    hr_sub = hr_aligned[dir_mask]
    rt_sub = rt_aligned[dir_mask]
    rt_hits_sub = rt_hits_aligned[dir_mask]
    cn_sub = cn_aligned[dir_mask]
    rt_rpe_sub = rt_rpe_aligned[dir_mask]
    exp_hr_sub = exp_hr_aligned[dir_mask]
    exp_rt_sub = exp_rt_aligned[dir_mask]
    n_sub = np.sum(dir_mask)

    hr_m, hr_s = _mean_sem(hr_sub)
    rt_m, rt_s = _mean_sem(rt_sub)
    rt_hits_m, rt_hits_s = _mean_sem(rt_hits_sub)
    exp_hr_m = np.nanmean(exp_hr_sub, axis=0)
    exp_rt_m = np.nanmean(exp_rt_sub, axis=0)

    # IQR of hit-only RT across transitions at each trial position
    rt_hits_q25 = np.nanpercentile(rt_hits_sub, 25, axis=0)
    rt_hits_q75 = np.nanpercentile(rt_hits_sub, 75, axis=0)
    rt_hits_iqr = rt_hits_q75 - rt_hits_q25

    # --- Raw aligned ---
    fig8, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),
                                          gridspec_kw={'wspace': 0.40, 'left': 0.08,
                                                       'right': 0.96, 'bottom': 0.18,
                                                       'top': 0.88})

    ax1.fill_between(trial_axis, hr_m - hr_s, hr_m + hr_s, color='k', alpha=0.15)
    ax1.plot(trial_axis, hr_m, 'k', linewidth=1.2)
    ax1.plot(trial_axis, exp_hr_m, color='cornflowerblue', linewidth=1.2)
    ax1.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax1.set_xlabel('Trials from threshold change')
    ax1.set_ylabel('Hit rate')
    ax1.set_xlim(-PRE, POST - 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
    ax1.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    ax2.fill_between(trial_axis, rt_m - rt_s, rt_m + rt_s, color='k', alpha=0.15)
    ax2.plot(trial_axis, rt_m, 'k', linewidth=1.2)
    ax2.plot(trial_axis, rt_hits_m, color='#e07b00', linewidth=1.2, linestyle='--')
    ax2.plot(trial_axis, exp_rt_m, color='cornflowerblue', linewidth=1.2)
    ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Trials from threshold change')
    ax2.set_ylabel('Time to reward (s)')
    ax2.set_xlim(-PRE, POST - 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(['Actual (all)', 'Hits only', 'Expected'], frameon=False, fontsize=6)
    ax2.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    ax3.plot(trial_axis, rt_hits_iqr, color='#e07b00', linewidth=1.2)
    ax3.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax3.set_xlabel('Trials from threshold change')
    ax3.set_ylabel('IQR of hit RT (s)')
    ax3.set_xlim(-PRE, POST - 1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Hit RT spread (IQR)', fontsize=8)

    fname8 = f'switch_aligned_hr_rt_{dir_suffix}'
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.png'), dpi=300)
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.svg'))
    plt.show()
    print(f"Saved {fname8}")

    # --- Delta version: subtract pre-transition mean ---
    hr_delta = hr_sub - np.nanmean(hr_sub[:, :PRE], axis=1, keepdims=True)
    rt_delta = rt_sub - np.nanmean(rt_sub[:, :PRE], axis=1, keepdims=True)
    rt_hits_delta = rt_hits_sub - np.nanmean(rt_hits_sub[:, :PRE], axis=1, keepdims=True)
    exp_hr_delta = exp_hr_sub - np.nanmean(exp_hr_sub[:, :PRE], axis=1, keepdims=True)
    exp_rt_delta = exp_rt_sub - np.nanmean(exp_rt_sub[:, :PRE], axis=1, keepdims=True)

    hr_d_m, hr_d_s = _mean_sem(hr_delta)
    rt_d_m, rt_d_s = _mean_sem(rt_delta)
    rt_hits_d_m, rt_hits_d_s = _mean_sem(rt_hits_delta)
    exp_hr_d_m = np.nanmean(exp_hr_delta, axis=0)
    exp_rt_d_m = np.nanmean(exp_rt_delta, axis=0)

    # Fractional RT change: RT / pre_RT - 1  (baseline = 0, positive = slower)
    pre_rt_mean = np.nanmean(rt_sub[:, :PRE], axis=1, keepdims=True)
    pre_rt_hits_mean = np.nanmean(rt_hits_sub[:, :PRE], axis=1, keepdims=True)
    rt_frac = rt_sub / pre_rt_mean - 1.0
    rt_hits_frac = rt_hits_sub / pre_rt_hits_mean - 1.0
    rt_frac_m, rt_frac_s = _mean_sem(rt_frac)
    rt_hits_frac_m, rt_hits_frac_s = _mean_sem(rt_hits_frac)

    rt_rpe_m, rt_rpe_s = _mean_sem(rt_rpe_sub)

    fig8b, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(13, 3),
                                                gridspec_kw={'wspace': 0.40, 'left': 0.06,
                                                             'right': 0.97, 'bottom': 0.18,
                                                             'top': 0.88})

    ax1.fill_between(trial_axis, hr_d_m - hr_d_s, hr_d_m + hr_d_s, color='k', alpha=0.15)
    ax1.plot(trial_axis, hr_d_m, 'k', linewidth=1.2)
    ax1.plot(trial_axis, exp_hr_d_m, color='cornflowerblue', linewidth=1.2)
    ax1.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax1.set_xlabel('Trials from threshold change')
    ax1.set_ylabel('\u0394 Hit rate')
    ax1.set_xlim(-PRE, POST - 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
    ax1.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    ax2.fill_between(trial_axis, rt_d_m - rt_d_s, rt_d_m + rt_d_s, color='k', alpha=0.15)
    ax2.plot(trial_axis, rt_d_m, 'k', linewidth=1.2)
    ax2.plot(trial_axis, rt_hits_d_m, color='#e07b00', linewidth=1.2, linestyle='--')
    ax2.plot(trial_axis, exp_rt_d_m, color='cornflowerblue', linewidth=1.2)
    ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax2.set_xlabel('Trials from threshold change')
    ax2.set_ylabel('\u0394 Time to reward (s)')
    ax2.set_xlim(-PRE, POST - 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(['Actual (all)', 'Hits only', 'Expected'], frameon=False, fontsize=6)
    ax2.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    ax3.fill_between(trial_axis, rt_rpe_m - rt_rpe_s, rt_rpe_m + rt_rpe_s, color='k', alpha=0.15)
    ax3.plot(trial_axis, rt_rpe_m, 'k', linewidth=1.2)
    ax3.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax3.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax3.set_xlabel('Trials from threshold change')
    ax3.set_ylabel('RT RPE (s)')
    ax3.set_xlim(-PRE, POST - 1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title(f'RPE: \u2212(RT \u2212 trailing avg)', fontsize=8)

    ax4.fill_between(trial_axis, rt_frac_m - rt_frac_s, rt_frac_m + rt_frac_s,
                     color='k', alpha=0.15)
    ax4.plot(trial_axis, rt_frac_m, 'k', linewidth=1.2)
    ax4.plot(trial_axis, rt_hits_frac_m, color='#e07b00', linewidth=1.2, linestyle='--')
    ax4.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax4.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax4.set_xlabel('Trials from threshold change')
    ax4.set_ylabel('Fractional \u0394RT (RT/pre\u2013RT \u22121)')
    ax4.set_xlim(-PRE, POST - 1)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.legend(['Actual (all)', 'Hits only'], frameon=False, fontsize=6)
    ax4.set_title('Fractional RT change', fontsize=8)

    fname8b = f'switch_aligned_delta_{dir_suffix}'
    fig8b.savefig(os.path.join(PANEL_DIR, f'{fname8b}.png'), dpi=300)
    fig8b.savefig(os.path.join(PANEL_DIR, f'{fname8b}.svg'))
    plt.show()
    print(f"Saved {fname8b}")

    # --- CN activity aligned, baseline-subtracted per transition ---
    # Subtraction (not division) avoids blowups when the per-transition
    # pre-switch mean is near zero (CN dF/F crosses zero).
    cn_pre_mean = np.nanmean(cn_sub[:, :PRE], axis=1, keepdims=True)

    # Left: difference from pre-switch mean (robust)
    cn_diff = cn_sub - cn_pre_mean
    cn_diff_m, cn_diff_s = _mean_sem(cn_diff)

    # Right: pre-switch-centered, scaled by each session's FULL-session CN std
    # (stable denominator; avoids the pre-window variance collapse).
    cn_sess_std_sub = cn_sess_std_aligned[dir_mask][:, None]
    cn_z = (cn_sub - cn_pre_mean) / np.where(cn_sess_std_sub > 1e-6,
                                             cn_sess_std_sub, np.nan)
    cn_z_m, cn_z_s = _mean_sem(cn_z)

    fig8c, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),
                                       gridspec_kw={'wspace': 0.35, 'left': 0.10,
                                                    'right': 0.96, 'bottom': 0.18,
                                                    'top': 0.88})

    ax1.fill_between(trial_axis, cn_diff_m - cn_diff_s, cn_diff_m + cn_diff_s,
                     color='k', alpha=0.15)
    ax1.plot(trial_axis, cn_diff_m, 'k', linewidth=1.2)
    ax1.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax1.set_xlabel('Trials from threshold change')
    ax1.set_ylabel('CN activity (Δ from pre-switch)')
    ax1.set_xlim(-PRE, POST - 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    ax2.fill_between(trial_axis, cn_z_m - cn_z_s, cn_z_m + cn_z_s,
                     color='k', alpha=0.15)
    ax2.plot(trial_axis, cn_z_m, 'k', linewidth=1.2)
    ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax2.set_xlabel('Trials from threshold change')
    ax2.set_ylabel('CN activity (z-score)')
    ax2.set_xlim(-PRE, POST - 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    fname8c = f'switch_aligned_cn_{dir_suffix}'
    fig8c.savefig(os.path.join(PANEL_DIR, f'{fname8c}.png'), dpi=300)
    fig8c.savefig(os.path.join(PANEL_DIR, f'{fname8c}.svg'))
    plt.show()
    print(f"Saved {fname8c}")

    # --- RT heatmap: one row per transition, sorted by pre-switch mean RT ---
    rt_heat = rt_aligned[dir_mask].copy()   # (n_transitions, PRE+POST), misses=10s
    sort_order = np.argsort(np.nanmean(rt_heat[:, :PRE], axis=1))
    rt_heat_sorted = rt_heat[sort_order]

    fig8d, ax = plt.subplots(1, 1, figsize=(5, 4),
                              gridspec_kw={'left': 0.12, 'right': 0.92,
                                           'bottom': 0.14, 'top': 0.90})
    im = ax.imshow(rt_heat_sorted, aspect='auto', interpolation='nearest',
                   cmap='viridis_r', vmin=0, vmax=10,
                   extent=[trial_axis[0] - 0.5, trial_axis[-1] + 0.5,
                            n_sub - 0.5, -0.5])
    ax.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Trials from threshold change')
    ax.set_ylabel('Transition (sorted by pre-switch RT)')
    ax.set_title(f'{dir_label} — RT heatmap (n={n_sub})', fontsize=8)
    plt.colorbar(im, ax=ax, label='RT (s)', shrink=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fname8d = f'switch_aligned_rt_heatmap_{dir_suffix}'
    fig8d.savefig(os.path.join(PANEL_DIR, f'{fname8d}.png'), dpi=300)
    fig8d.savefig(os.path.join(PANEL_DIR, f'{fname8d}.svg'))
    plt.show()
    print(f"Saved {fname8d}")

#%% ============================================================================
# CELL 8f: Per-transition slope/CN-change vs expected ΔRT (threshold increases)
# ============================================================================
# Expected ΔRT comes from the step function in exp_rt_aligned:
#   pre = pre-switch actual RT, post = transfer-function-replay expected RT
exp_pre_rt = np.nanmean(exp_rt_aligned[:, :PRE], axis=1)
exp_post_rt = np.nanmean(exp_rt_aligned[:, PRE:], axis=1)
expected_drt = exp_post_rt - exp_pre_rt

# Post-switch RT slope per transition (re-compute since we replaced earlier definition)
post_rt = rt_aligned[:, PRE:PRE + POST]
x_post = np.arange(POST, dtype=float)
post_rt_slope = np.full(rt_aligned.shape[0], np.nan)
for i in range(rt_aligned.shape[0]):
    y = post_rt[i]
    valid = np.isfinite(y)
    if valid.sum() >= 3:
        post_rt_slope[i] = np.polyfit(x_post[valid], y[valid], 1)[0]

# CN change: z-scored (mean post - mean pre) / pre-window std, matching aligned plots
cn_pre_per = np.nanmean(cn_aligned[:, :PRE], axis=1)
cn_post_per = np.nanmean(cn_aligned[:, PRE:PRE + POST], axis=1)
cn_pre_std_per = np.nanstd(cn_aligned[:, :PRE], axis=1)
cn_change = (cn_post_per - cn_pre_per) / np.where(cn_pre_std_per > 1e-6,
                                                  cn_pre_std_per, np.nan)

# Restrict to threshold increases
inc_mask_all = thr_change_mag > 0
ok = inc_mask_all & np.isfinite(expected_drt) & np.isfinite(post_rt_slope) & np.isfinite(cn_change)

x = expected_drt[ok]
y_slope = post_rt_slope[ok]
y_cn = cn_change[ok]

from scipy.stats import pearsonr
r_slope, p_slope = pearsonr(x, y_slope)
r_cn, p_cn = pearsonr(x, y_cn)

fig8f, (axA, axB) = plt.subplots(1, 2, figsize=(7, 3),
                                  gridspec_kw={'wspace': 0.40, 'left': 0.10,
                                               'right': 0.96, 'bottom': 0.18,
                                               'top': 0.86})

axA.scatter(x, y_slope, s=14, c='k', alpha=0.5, edgecolors='none')
axA.axhline(0, color='gray', linewidth=0.5, linestyle=':')
axA.axvline(0, color='gray', linewidth=0.5, linestyle=':')
axA.set_xlabel('Expected ΔRT (s)')
axA.set_ylabel('Post-switch RT slope (s/trial)')
axA.set_title(f'r={r_slope:.2f}, p={p_slope:.3g}, n={ok.sum()}', fontsize=8)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

axB.scatter(x, y_cn, s=14, c='#ea580c', alpha=0.5, edgecolors='none')
axB.axhline(0, color='gray', linewidth=0.5, linestyle=':')
axB.axvline(0, color='gray', linewidth=0.5, linestyle=':')
axB.set_xlabel('Expected ΔRT (s)')
axB.set_ylabel('Δ CN (z-scored, post − pre)')
axB.set_title(f'r={r_cn:.2f}, p={p_cn:.3g}, n={ok.sum()}', fontsize=8)
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)

fname8f = 'slope_and_cn_vs_expected_drt'
fig8f.savefig(os.path.join(PANEL_DIR, f'{fname8f}.png'), dpi=300)
fig8f.savefig(os.path.join(PANEL_DIR, f'{fname8f}.svg'))
plt.show()
print(f"Saved {fname8f}")

# --- CN heatmap aligned to switch, sorted by expected ΔRT ---
# z-score CN within each transition using its own pre-switch window
cn_pre_mean_all = np.nanmean(cn_aligned[:, :PRE], axis=1, keepdims=True)
cn_pre_std_all = np.nanstd(cn_aligned[:, :PRE], axis=1, keepdims=True)
cn_z_all = (cn_aligned - cn_pre_mean_all) / np.where(cn_pre_std_all > 1e-6,
                                                     cn_pre_std_all, np.nan)

ok_inc = (thr_change_mag > 0) & np.isfinite(expected_drt)
cn_z_inc = cn_z_all[ok_inc]
exp_drt_inc = expected_drt[ok_inc]
sort_order = np.argsort(exp_drt_inc)
cn_z_sorted = cn_z_inc[sort_order]
n_inc = cn_z_sorted.shape[0]

fig8g, ax = plt.subplots(1, 1, figsize=(5, 5),
                          gridspec_kw={'left': 0.14, 'right': 0.92,
                                       'bottom': 0.12, 'top': 0.92})
vmax = np.nanpercentile(np.abs(cn_z_sorted), 95)
im = ax.imshow(cn_z_sorted, aspect='auto', interpolation='nearest',
               cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               extent=[trial_axis[0] - 0.5, trial_axis[-1] + 0.5,
                       n_inc - 0.5, -0.5])
ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel('Trials from threshold change')
ax.set_ylabel('Transition (sorted by expected ΔRT, small→large)')
ax.set_title(f'CN z-score, threshold ↑ (n={n_inc})', fontsize=8)
plt.colorbar(im, ax=ax, label='CN (z)', shrink=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname8g = 'cn_heatmap_sorted_by_expected_drt'
fig8g.savefig(os.path.join(PANEL_DIR, f'{fname8g}.png'), dpi=300)
fig8g.savefig(os.path.join(PANEL_DIR, f'{fname8g}.svg'))
plt.show()
print(f"Saved {fname8g}")

#%% ============================================================================
# CELL 8h: Per-session regression — what predicts ΔCN_z (trial-by-trial learning)?
# ============================================================================
# Outcome: ΔCN_z[t] = CN_z[t] − CN_z[t−1]  (z-scored within session by first 10 trials)
# Predictors (all at trial t−1, standardized within session):
#   trial_num, rt_rpe, hit_rpe, rt, hit_rate (10-trial smooth), thr_change at t−1
predictor_names = ['trial_num', 'rt_rpe', 'hit_rpe', 'rt', 'hit_rate', 'thr_change']
session_betas_dcn = []   # outcome: ΔCN_z (predictors lagged at t−1)
session_betas_cn = []    # outcome: CN_z level (predictors same trial t)

for (mouse, session), tdata in all_session_trials.items():
    cn = tdata['cn'].astype(float)
    rt = tdata['rt'].copy()
    rt_rpe_s = tdata['rt_rpe']
    hit_rpe_s = tdata['hit_rpe']
    hit_s = tdata['hit'].astype(float)
    switches_s = tdata['switches']
    thr_u_s = tdata['thr_u']
    n_t = len(cn)
    if n_t < 30:
        continue

    cn_base_mean = np.nanmean(cn[:10])
    cn_base_std = np.nanstd(cn[:10])
    if not np.isfinite(cn_base_std) or cn_base_std < 1e-6:
        continue
    cn_z_sess = (cn - cn_base_mean) / cn_base_std

    dcn = np.diff(cn_z_sess)   # length n_t-1

    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 10.0

    hr_smooth = np.full(n_t, np.nan)
    for t in range(n_t):
        a = max(0, t - 9)
        hr_smooth[t] = np.nanmean(hit_s[a:t + 1])

    thr_change_per = np.zeros(n_t)
    for s in switches_s[1:]:
        if 0 < s < n_t:
            thr_change_per[s] = thr_u_s[s] - thr_u_s[s - 1]

    trial_num = np.arange(n_t, dtype=float)

    def _fit(X_full, y_full):
        valid = np.all(np.isfinite(X_full), axis=1) & np.isfinite(y_full)
        if valid.sum() < 20:
            return None
        Xv = X_full[valid]
        yv = y_full[valid]
        Xs = Xv.std(axis=0)
        if np.any(Xs < 1e-9):
            return None
        Xz = (Xv - Xv.mean(axis=0)) / Xs
        X_design = np.column_stack([np.ones(Xz.shape[0]), Xz])
        coef, *_ = np.linalg.lstsq(X_design, yv, rcond=None)
        return coef[1:]

    # ΔCN_z regression: predictors at t−1
    X_d = np.column_stack([
        trial_num[:-1], rt_rpe_s[:-1], hit_rpe_s[:-1],
        rt_filled[:-1], hr_smooth[:-1], thr_change_per[:-1],
    ])
    b_d = _fit(X_d, dcn)

    # CN_z level regression: predictors also at t−1 (avoid same-trial CN-RT coupling)
    X_l = np.column_stack([
        trial_num[:-1], rt_rpe_s[:-1], hit_rpe_s[:-1],
        rt_filled[:-1], hr_smooth[:-1], thr_change_per[:-1],
    ])
    b_l = _fit(X_l, cn_z_sess[1:])

    if b_d is not None and b_l is not None:
        session_betas_dcn.append(b_d)
        session_betas_cn.append(b_l)

session_betas_dcn = np.array(session_betas_dcn)
session_betas_cn = np.array(session_betas_cn)
print(f"Fit {session_betas_dcn.shape[0]} sessions for CN regressions")

from scipy.stats import wilcoxon

def _summary(betas):
    m = np.nanmean(betas, axis=0)
    s = np.nanstd(betas, axis=0) / np.sqrt(betas.shape[0])
    p = np.array([wilcoxon(betas[:, j]).pvalue for j in range(betas.shape[1])])
    return m, s, p

m_d, s_d, p_d = _summary(session_betas_dcn)
m_l, s_l, p_l = _summary(session_betas_cn)

fig8h, (axA, axB) = plt.subplots(1, 2, figsize=(9, 3),
                                  gridspec_kw={'wspace': 0.40, 'left': 0.08,
                                               'right': 0.97, 'bottom': 0.30,
                                               'top': 0.86})
xpos = np.arange(len(predictor_names))

def _plot_bars(ax, m, s, p, ylabel, title):
    ax.bar(xpos, m, yerr=s, color='#888', edgecolor='black',
           linewidth=0.8, capsize=3)
    ax.axhline(0, color='k', linewidth=0.5)
    for j, pv in enumerate(p):
        if pv < 0.001: tag = '***'
        elif pv < 0.01: tag = '**'
        elif pv < 0.05: tag = '*'
        else: tag = ''
        if tag:
            offset = (s[j] * 1.5 + 0.005) * np.sign(m[j] if m[j] != 0 else 1)
            ax.text(xpos[j], m[j] + offset, tag, ha='center', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(predictor_names, rotation=35, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

_plot_bars(axA, m_d, s_d, p_d,
           'Standardized β (ΔCN_z)',
           f'Trial-by-trial CN change (n={session_betas_dcn.shape[0]})')
_plot_bars(axB, m_l, s_l, p_l,
           'Standardized β (CN_z)',
           f'CN level (n={session_betas_cn.shape[0]})')

fname8h = 'cn_learning_regression'
fig8h.savefig(os.path.join(PANEL_DIR, f'{fname8h}.png'), dpi=300)
fig8h.savefig(os.path.join(PANEL_DIR, f'{fname8h}.svg'))
plt.show()
print(f"Saved {fname8h}")

#%% ============================================================================
# CELL 8i: Univariate β for each predictor (collinearity check)
# ============================================================================
# Fit one predictor at a time, per session, then compare to multivariate β.
session_uni_dcn = []
session_uni_cn = []

for (mouse, session), tdata in all_session_trials.items():
    cn = tdata['cn'].astype(float)
    rt = tdata['rt'].copy()
    rt_rpe_s = tdata['rt_rpe']
    hit_rpe_s = tdata['hit_rpe']
    hit_s = tdata['hit'].astype(float)
    switches_s = tdata['switches']
    thr_u_s = tdata['thr_u']
    n_t = len(cn)
    if n_t < 30:
        continue

    cn_base_mean = np.nanmean(cn[:10])
    cn_base_std = np.nanstd(cn[:10])
    if not np.isfinite(cn_base_std) or cn_base_std < 1e-6:
        continue
    cn_z_sess = (cn - cn_base_mean) / cn_base_std
    dcn = np.diff(cn_z_sess)

    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 10.0
    hr_smooth = np.full(n_t, np.nan)
    for t in range(n_t):
        a = max(0, t - 9)
        hr_smooth[t] = np.nanmean(hit_s[a:t + 1])
    thr_change_per = np.zeros(n_t)
    for s in switches_s[1:]:
        if 0 < s < n_t:
            thr_change_per[s] = thr_u_s[s] - thr_u_s[s - 1]
    trial_num = np.arange(n_t, dtype=float)

    pred_lag = [trial_num[:-1], rt_rpe_s[:-1], hit_rpe_s[:-1],
                rt_filled[:-1], hr_smooth[:-1], thr_change_per[:-1]]

    def _univariate_betas(predictors, y):
        betas = np.full(len(predictors), np.nan)
        for j, x in enumerate(predictors):
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 20:
                continue
            xv = x[valid]
            yv = y[valid]
            if xv.std() < 1e-9:
                continue
            xz = (xv - xv.mean()) / xv.std()
            X_design = np.column_stack([np.ones(xz.shape[0]), xz])
            coef, *_ = np.linalg.lstsq(X_design, yv, rcond=None)
            betas[j] = coef[1]
        return betas

    b_d = _univariate_betas(pred_lag, dcn)
    b_l = _univariate_betas(pred_lag, cn_z_sess[1:])   # same lag as ΔCN_z
    if np.all(np.isfinite(b_d)) and np.all(np.isfinite(b_l)):
        session_uni_dcn.append(b_d)
        session_uni_cn.append(b_l)

session_uni_dcn = np.array(session_uni_dcn)
session_uni_cn = np.array(session_uni_cn)
print(f"Univariate fits: {session_uni_dcn.shape[0]} sessions")

m_ud, s_ud, p_ud = _summary(session_uni_dcn)
m_ul, s_ul, p_ul = _summary(session_uni_cn)

# Side-by-side: multivariate vs univariate
fig8i, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.2),
                                  gridspec_kw={'wspace': 0.35, 'left': 0.07,
                                               'right': 0.97, 'bottom': 0.30,
                                               'top': 0.86})
xpos = np.arange(len(predictor_names))
w = 0.38

def _grouped_bars(ax, m_multi, s_multi, p_multi, m_uni, s_uni, p_uni, ylabel, title):
    ax.bar(xpos - w/2, m_multi, w, yerr=s_multi, color='#888',
           edgecolor='black', linewidth=0.6, capsize=2, label='Multivariate')
    ax.bar(xpos + w/2, m_uni, w, yerr=s_uni, color='#e07b00',
           edgecolor='black', linewidth=0.6, capsize=2, label='Univariate')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(predictor_names, rotation=35, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

_grouped_bars(axA, m_d, s_d, p_d, m_ud, s_ud, p_ud,
              'Standardized β (ΔCN_z)',
              f'ΔCN_z: multi vs uni (n={session_uni_dcn.shape[0]})')
_grouped_bars(axB, m_l, s_l, p_l, m_ul, s_ul, p_ul,
              'Standardized β (CN_z)',
              f'CN_z: multi vs uni (n={session_uni_cn.shape[0]})')

fname8i = 'cn_learning_regression_uni_vs_multi'
fig8i.savefig(os.path.join(PANEL_DIR, f'{fname8i}.png'), dpi=300)
fig8i.savefig(os.path.join(PANEL_DIR, f'{fname8i}.svg'))
plt.show()
print(f"Saved {fname8i}")

#%% ============================================================================
# CELL 8j: Multi-lag regression — β at each predictor × lag combination
# ============================================================================
LAGS = [1, 2, 3, 4, 5]
n_pred = len(predictor_names)
n_lag = len(LAGS)

# Per-session: for each lag L, fit multivariate regression of CN_z[t] (and ΔCN_z[t])
# on all 6 predictors at t−L. So one regression per lag, giving n_pred betas per lag.
session_lag_dcn = []   # (n_sessions, n_lag, n_pred)
session_lag_cn = []

for (mouse, session), tdata in all_session_trials.items():
    cn = tdata['cn'].astype(float)
    rt = tdata['rt'].copy()
    rt_rpe_s = tdata['rt_rpe']
    hit_rpe_s = tdata['hit_rpe']
    hit_s = tdata['hit'].astype(float)
    switches_s = tdata['switches']
    thr_u_s = tdata['thr_u']
    n_t = len(cn)
    if n_t < 30 + max(LAGS):
        continue
    cn_base_mean = np.nanmean(cn[:10])
    cn_base_std = np.nanstd(cn[:10])
    if not np.isfinite(cn_base_std) or cn_base_std < 1e-6:
        continue
    cn_z_sess = (cn - cn_base_mean) / cn_base_std
    dcn = np.diff(cn_z_sess)

    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 10.0
    hr_smooth = np.full(n_t, np.nan)
    for t in range(n_t):
        a = max(0, t - 9)
        hr_smooth[t] = np.nanmean(hit_s[a:t + 1])
    thr_change_per = np.zeros(n_t)
    for s in switches_s[1:]:
        if 0 < s < n_t:
            thr_change_per[s] = thr_u_s[s] - thr_u_s[s - 1]
    trial_num = np.arange(n_t, dtype=float)
    base_preds = [trial_num, rt_rpe_s, hit_rpe_s, rt_filled, hr_smooth, thr_change_per]

    sess_dcn = np.full((n_lag, n_pred), np.nan)
    sess_cn = np.full((n_lag, n_pred), np.nan)
    for li, L in enumerate(LAGS):
        Xr = np.column_stack([p[:-L] for p in base_preds]) if L > 0 else None
        # Outcome at t for predictors at t−L means y indexed from L to end
        y_dcn = dcn[L - 1:] if L >= 1 else None   # dcn[t-1] corresponds to ΔCN at trial t
        y_cn = cn_z_sess[L:]

        # Align lengths: both X and y need same number of rows
        n_rows = min(Xr.shape[0], len(y_cn), len(y_dcn))
        Xr = Xr[-n_rows:]
        y_cn_a = y_cn[-n_rows:]
        y_dcn_a = y_dcn[-n_rows:]

        for outcome, store in [(y_dcn_a, sess_dcn), (y_cn_a, sess_cn)]:
            valid = np.all(np.isfinite(Xr), axis=1) & np.isfinite(outcome)
            if valid.sum() < 20:
                continue
            Xv = Xr[valid]
            yv = outcome[valid]
            Xs = Xv.std(axis=0)
            if np.any(Xs < 1e-9):
                continue
            Xz = (Xv - Xv.mean(axis=0)) / Xs
            X_design = np.column_stack([np.ones(Xz.shape[0]), Xz])
            coef, *_ = np.linalg.lstsq(X_design, yv, rcond=None)
            store[li] = coef[1:]

    if np.all(np.isfinite(sess_dcn)) and np.all(np.isfinite(sess_cn)):
        session_lag_dcn.append(sess_dcn)
        session_lag_cn.append(sess_cn)

session_lag_dcn = np.array(session_lag_dcn)   # (n_sessions, n_lag, n_pred)
session_lag_cn = np.array(session_lag_cn)
print(f"Multi-lag fits: {session_lag_dcn.shape[0]} sessions, "
      f"{n_lag} lags × {n_pred} predictors")

# Mean and SEM across sessions
def _lag_summary(arr):
    m = np.nanmean(arr, axis=0)   # (n_lag, n_pred)
    s = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    return m, s

m_dlag, s_dlag = _lag_summary(session_lag_dcn)
m_llag, s_llag = _lag_summary(session_lag_cn)

# Per-(lag, predictor) Wilcoxon p-value across sessions
def _lag_pvals(arr):
    n_l, n_p = arr.shape[1], arr.shape[2]
    P = np.full((n_l, n_p), np.nan)
    for li in range(n_l):
        for pj in range(n_p):
            v = arr[:, li, pj]
            v = v[np.isfinite(v)]
            if len(v) > 5 and np.any(v != 0):
                P[li, pj] = wilcoxon(v).pvalue
    return P

p_dlag = _lag_pvals(session_lag_dcn)
p_llag = _lag_pvals(session_lag_cn)

# Highlight predictors whose lag-1 |β| is significant for ΔCN
LAG1_SIG = (p_dlag[0, :] < 0.05)

colors = {'trial_num': '#888888', 'rt_rpe': '#e07b00', 'hit_rpe': '#2ca02c',
          'rt': '#d62728', 'hit_rate': '#9467bd', 'thr_change': '#8c564b'}

# Two-panel figure: ΔCN_z by lag (focus), CN_z by lag (context)
fig8j, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4),
                                  gridspec_kw={'wspace': 0.32, 'left': 0.08,
                                               'right': 0.80, 'bottom': 0.16,
                                               'top': 0.88})

def _plot_lag(ax, m, s, p, ylabel, title):
    for j, name in enumerate(predictor_names):
        # Emphasize predictors with any significant lag for ΔCN
        is_emph = LAG1_SIG[j] if ax is axA else (np.any(p[:, j] < 0.05))
        lw = 2.2 if is_emph else 0.9
        ms_size = 7 if is_emph else 4
        alpha = 1.0 if is_emph else 0.5
        ax.errorbar(LAGS, m[:, j], yerr=s[:, j],
                    marker='o', color=colors[name], linewidth=lw,
                    markersize=ms_size, capsize=2, alpha=alpha, label=name)
        # Stars at significant lags
        for li, L in enumerate(LAGS):
            pv = p[li, j]
            if np.isfinite(pv) and pv < 0.05:
                tag = '***' if pv < 0.001 else ('**' if pv < 0.01 else '*')
                yy = m[li, j] + np.sign(m[li, j] if m[li, j] != 0 else 1) * \
                     (s[li, j] + 0.04)
                ax.text(L, yy, tag, ha='center', fontsize=8,
                        color=colors[name])
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Lag (trials before t)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(LAGS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

_plot_lag(axA, m_dlag, s_dlag, p_dlag,
          'Standardized β (ΔCN_z)',
          f'ΔCN_z(t) ~ predictors at t−lag (n={session_lag_dcn.shape[0]})')
_plot_lag(axB, m_llag, s_llag, p_llag,
          'Standardized β (CN_z)',
          f'CN_z(t) ~ predictors at t−lag (n={session_lag_cn.shape[0]})')

axA.legend(frameon=False, fontsize=8, loc='upper right')
axB.legend(frameon=False, fontsize=8, loc='center left',
           bbox_to_anchor=(1.05, 0.5))

fname8j = 'cn_learning_regression_lags'
fig8j.savefig(os.path.join(PANEL_DIR, f'{fname8j}.png'), dpi=300)
fig8j.savefig(os.path.join(PANEL_DIR, f'{fname8j}.svg'))
plt.show()
print(f"Saved {fname8j}")

#%% ============================================================================
# CELL 8k: Multi-lag regression with thr_change shifted back one trial
# ============================================================================
# Hypothesis: the negative β on thr_change at lag 1 is a selection artifact —
# the BCI algorithm raises the threshold *because* CN[S] was high, so ΔCN[S+1]
# = CN[S+1] − CN[S] is mechanically negative regardless of any neural response.
# Test: shift thr_change by an extra trial (use thr_change[t−L−1] instead of
# thr_change[t−L]) so the switch trial is no longer adjacent to the predicted
# trial. Other predictors stay at lag L. If the β goes positive (or vanishes),
# the original negative β was the selection artifact.

session_lag_dcn_shift = []   # (n_sessions, n_lag, n_pred) — same shape as 8j
session_lag_cn_shift = []

for (mouse, session), tdata in all_session_trials.items():
    cn = tdata['cn'].astype(float)
    rt = tdata['rt'].copy()
    rt_rpe_s = tdata['rt_rpe']
    hit_rpe_s = tdata['hit_rpe']
    hit_s = tdata['hit'].astype(float)
    switches_s = tdata['switches']
    thr_u_s = tdata['thr_u']
    n_t = len(cn)
    if n_t < 30 + max(LAGS) + 1:
        continue
    cn_base_mean = np.nanmean(cn[:10])
    cn_base_std = np.nanstd(cn[:10])
    if not np.isfinite(cn_base_std) or cn_base_std < 1e-6:
        continue
    cn_z_sess = (cn - cn_base_mean) / cn_base_std
    dcn = np.diff(cn_z_sess)

    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 10.0
    hr_smooth = np.full(n_t, np.nan)
    for t in range(n_t):
        a = max(0, t - 9)
        hr_smooth[t] = np.nanmean(hit_s[a:t + 1])
    thr_change_per = np.zeros(n_t)
    for s in switches_s[1:]:
        if 0 < s < n_t:
            thr_change_per[s] = thr_u_s[s] - thr_u_s[s - 1]
    trial_num = np.arange(n_t, dtype=float)
    base_preds = [trial_num, rt_rpe_s, hit_rpe_s, rt_filled, hr_smooth]
    # thr_change handled separately so it can be shifted by L+1

    sess_dcn = np.full((n_lag, n_pred), np.nan)
    sess_cn = np.full((n_lag, n_pred), np.nan)
    for li, L in enumerate(LAGS):
        # Other predictors at t−L; thr_change at t−L−1
        cols = [p[:n_t - L] for p in base_preds]
        thr_shift = thr_change_per[:n_t - L - 1]
        # Trim other predictors by one extra so all columns align to thr_shift length
        cols = [c[:len(thr_shift)] for c in cols]
        Xr = np.column_stack(cols + [thr_shift])
        # y indexed so that row i corresponds to outcome at trial L+1+i
        # (because thr_change at t-L-1 with t = L+1+i gives index i)
        y_dcn = dcn[L:][:Xr.shape[0]]    # ΔCN at trial t=L+1, ..., dcn[t-1] index
        y_cn = cn_z_sess[L + 1:][:Xr.shape[0]]

        n_rows = min(Xr.shape[0], len(y_cn), len(y_dcn))
        Xr = Xr[:n_rows]
        y_cn_a = y_cn[:n_rows]
        y_dcn_a = y_dcn[:n_rows]

        for outcome, store in [(y_dcn_a, sess_dcn), (y_cn_a, sess_cn)]:
            valid = np.all(np.isfinite(Xr), axis=1) & np.isfinite(outcome)
            if valid.sum() < 20:
                continue
            Xv = Xr[valid]
            yv = outcome[valid]
            Xs = Xv.std(axis=0)
            if np.any(Xs < 1e-9):
                continue
            Xz = (Xv - Xv.mean(axis=0)) / Xs
            X_design = np.column_stack([np.ones(Xz.shape[0]), Xz])
            coef, *_ = np.linalg.lstsq(X_design, yv, rcond=None)
            store[li] = coef[1:]

    if np.all(np.isfinite(sess_dcn)) and np.all(np.isfinite(sess_cn)):
        session_lag_dcn_shift.append(sess_dcn)
        session_lag_cn_shift.append(sess_cn)

session_lag_dcn_shift = np.array(session_lag_dcn_shift)
session_lag_cn_shift = np.array(session_lag_cn_shift)
print(f"Shifted thr_change fits: {session_lag_dcn_shift.shape[0]} sessions")

m_dlag_sh, s_dlag_sh = _lag_summary(session_lag_dcn_shift)
m_llag_sh, s_llag_sh = _lag_summary(session_lag_cn_shift)
p_dlag_sh = _lag_pvals(session_lag_dcn_shift)
p_llag_sh = _lag_pvals(session_lag_cn_shift)

# Side-by-side comparison: original (8j) vs shifted (8k) — focus on thr_change β
THR_IDX = predictor_names.index('thr_change')

fig8k, (axA2, axB2) = plt.subplots(1, 2, figsize=(11, 4),
                                    gridspec_kw={'wspace': 0.32, 'left': 0.08,
                                                 'right': 0.80, 'bottom': 0.16,
                                                 'top': 0.88})

# Panel A: ΔCN — overlay original vs shifted thr_change β only
axA2.errorbar(LAGS, m_dlag[:, THR_IDX], yerr=s_dlag[:, THR_IDX],
              marker='o', color=colors['thr_change'], linewidth=2.0,
              capsize=2, label='thr_change[t−L] (original 8j)')
axA2.errorbar(LAGS, m_dlag_sh[:, THR_IDX], yerr=s_dlag_sh[:, THR_IDX],
              marker='s', color=colors['thr_change'], linewidth=2.0,
              linestyle='--', capsize=2, alpha=0.6,
              label='thr_change[t−L−1] (shifted)')
for li, L in enumerate(LAGS):
    for m_arr, p_arr, marker_y in [(m_dlag, p_dlag, m_dlag[li, THR_IDX]),
                                     (m_dlag_sh, p_dlag_sh, m_dlag_sh[li, THR_IDX])]:
        pv = p_arr[li, THR_IDX]
        if np.isfinite(pv) and pv < 0.05:
            tag = '***' if pv < 0.001 else ('**' if pv < 0.01 else '*')
            axA2.text(L, marker_y + 0.03, tag, ha='center', fontsize=8,
                      color=colors['thr_change'])
axA2.axhline(0, color='k', linewidth=0.5)
axA2.set_xlabel('Lag (trials before t)')
axA2.set_ylabel('Standardized β on thr_change (ΔCN_z)')
axA2.set_title('thr_change β: original vs shifted-by-one', fontsize=9)
axA2.set_xticks(LAGS)
axA2.legend(frameon=False, fontsize=8, loc='best')
axA2.spines['top'].set_visible(False)
axA2.spines['right'].set_visible(False)

# Panel B: full shifted β profile across all predictors (ΔCN_z)
_plot_lag(axB2, m_dlag_sh, s_dlag_sh, p_dlag_sh,
          'Standardized β (ΔCN_z)',
          f'Shifted thr_change: ΔCN_z(t) ~ preds at t−L, thr at t−L−1 '
          f'(n={session_lag_dcn_shift.shape[0]})')
axB2.legend(frameon=False, fontsize=8, loc='center left',
            bbox_to_anchor=(1.05, 0.5))

fname8k = 'cn_learning_regression_thr_shifted'
fig8k.savefig(os.path.join(PANEL_DIR, f'{fname8k}.png'), dpi=300)
fig8k.savefig(os.path.join(PANEL_DIR, f'{fname8k}.svg'))
plt.show()
print(f"Saved {fname8k}")

# Print numeric comparison for thr_change
print("\nthr_change β (ΔCN_z) comparison:")
print(f"  {'Lag':<5}{'orig β':<12}{'orig p':<12}{'shifted β':<14}{'shifted p':<12}")
for li, L in enumerate(LAGS):
    print(f"  {L:<5}{m_dlag[li,THR_IDX]:<12.3f}{p_dlag[li,THR_IDX]:<12.3g}"
          f"{m_dlag_sh[li,THR_IDX]:<14.3f}{p_dlag_sh[li,THR_IDX]:<12.3g}")

#%% ============================================================================
# CELL 8e: Dummy-switch control — align CN to random trial positions
# ============================================================================
# For each session, pick the same number of "dummy switches" as real switches
# within the same MAX_SWITCH_TRIAL range, avoiding positions within ±POST trials
# of any real switch. Then run the same alignment analysis on CN.
np.random.seed(0)
EXCLUSION_RADIUS = POST   # how far from real switches dummies must stay

cn_dummy_aligned = []

for (mouse, session), tdata in all_session_trials.items():
    cn_act = tdata['cn']
    real_switches = tdata['switches']
    n_trials = len(tdata['hit'])

    # Number of real switches that fall within MAX_SWITCH_TRIAL (matches Cell 8)
    n_real = int(np.sum((real_switches[1:] < MAX_SWITCH_TRIAL)))
    if n_real == 0:
        continue

    # Candidate dummy positions: in [PRE, MAX_SWITCH_TRIAL), away from real switches
    candidates = []
    for t in range(PRE, min(MAX_SWITCH_TRIAL, n_trials - POST)):
        if np.all(np.abs(real_switches - t) > EXCLUSION_RADIUS):
            candidates.append(t)
    if len(candidates) == 0:
        continue

    n_pick = min(n_real, len(candidates))
    dummy_switches = np.random.choice(candidates, size=n_pick, replace=False)

    for sw_trial in dummy_switches:
        t0 = sw_trial - PRE
        t1 = sw_trial + POST
        cn_row = np.full(PRE + POST, np.nan)
        for k in range(PRE + POST):
            ti = t0 + k
            if 0 <= ti < n_trials:
                cn_row[k] = cn_act[ti]
        cn_dummy_aligned.append(cn_row)

cn_dummy_aligned = np.array(cn_dummy_aligned)
n_dummy = cn_dummy_aligned.shape[0]
print(f"Dummy alignment: {n_dummy} dummy switches")

# Same normalizations as Cell 8 CN analysis
cn_d_pre_mean = np.nanmean(cn_dummy_aligned[:, :PRE], axis=1, keepdims=True)
cn_d_pre_std = np.nanstd(cn_dummy_aligned[:, :PRE], axis=1, keepdims=True)

cn_d_frac = cn_dummy_aligned / np.where(np.abs(cn_d_pre_mean) > 1e-6,
                                         cn_d_pre_mean, np.nan)
cn_d_z = (cn_dummy_aligned - cn_d_pre_mean) / np.where(cn_d_pre_std > 1e-6,
                                                       cn_d_pre_std, np.nan)
cn_d_frac_m, cn_d_frac_s = _mean_sem(cn_d_frac)
cn_d_z_m, cn_d_z_s = _mean_sem(cn_d_z)

fig8e, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),
                                  gridspec_kw={'wspace': 0.35, 'left': 0.10,
                                               'right': 0.96, 'bottom': 0.18,
                                               'top': 0.88})

ax1.fill_between(trial_axis, cn_d_frac_m - cn_d_frac_s, cn_d_frac_m + cn_d_frac_s,
                 color='gray', alpha=0.25)
ax1.plot(trial_axis, cn_d_frac_m, color='gray', linewidth=1.2)
ax1.axvline(0, color='r', linewidth=0.8, linestyle='--')
ax1.axhline(1, color='gray', linewidth=0.5, linestyle=':')
ax1.set_xlabel('Trials from dummy switch')
ax1.set_ylabel('CN activity (frac. of pre)')
ax1.set_xlim(-PRE, POST - 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title(f'Dummy switches (n = {n_dummy})', fontsize=8)

ax2.fill_between(trial_axis, cn_d_z_m - cn_d_z_s, cn_d_z_m + cn_d_z_s,
                 color='gray', alpha=0.25)
ax2.plot(trial_axis, cn_d_z_m, color='gray', linewidth=1.2)
ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax2.set_xlabel('Trials from dummy switch')
ax2.set_ylabel('CN activity (z-score)')
ax2.set_xlim(-PRE, POST - 1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title(f'Dummy switches (n = {n_dummy})', fontsize=8)

fname8e = 'switch_aligned_cn_dummy'
fig8e.savefig(os.path.join(PANEL_DIR, f'{fname8e}.png'), dpi=300)
fig8e.savefig(os.path.join(PANEL_DIR, f'{fname8e}.svg'))
plt.show()
print(f"Saved {fname8e}")

#%% ============================================================================
# CELL 9: Expected vs actual RT per epoch (scatter)
# ============================================================================
switch_epochs_rt = [s for s in all_epoch_stats if s['epoch'] > 0
                    and np.isfinite(s.get('expected_rt', np.nan))
                    and np.isfinite(s.get('actual_rt', np.nan))]

exp_rts_all = np.array([s['expected_rt'] for s in switch_epochs_rt])
act_rts_all = np.array([s['actual_rt'] for s in switch_epochs_rt])

fig9, ax = plt.subplots(1, 1, figsize=(3.5, 3.5),
                         gridspec_kw={'left': 0.18, 'right': 0.94,
                                      'bottom': 0.16, 'top': 0.90})

ax.scatter(exp_rts_all, act_rts_all, s=15, c='k', alpha=0.5, edgecolors='none')
mn = min(np.nanmin(exp_rts_all), np.nanmin(act_rts_all)) - 0.2
mx = max(np.nanmax(exp_rts_all), np.nanmax(act_rts_all)) + 0.2
ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Expected RT (s)')
ax.set_ylabel('Actual RT (s)')
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

n_above = np.sum(act_rts_all > exp_rts_all)
from scipy.stats import wilcoxon
stat_rt, p_rt = wilcoxon(act_rts_all - exp_rts_all)
ax.set_title(f'{n_above}/{len(act_rts_all)} above unity\np={p_rt:.4f}', fontsize=7)

fname9 = 'expected_vs_actual_rt'
fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.png'), dpi=300)
fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.svg'))
plt.show()
print(f"Saved {fname9}")

#%% ============================================================================
# CELL 10: RT recovery split by threshold change magnitude
# ============================================================================
# For each transition, compute:
#   - expected RT change (expected_rt - pre_rt): how much harder the threshold made it
#   - RT improvement: expected_rt - actual_rt (positive = animal beat expectation)

# Build lookup for previous epoch's actual RT (pre-switch baseline)
prev_rt_lookup = {}
for rec in all_epoch_stats:
    if np.isfinite(rec.get('actual_rt', np.nan)):
        prev_rt_lookup[(rec['mouse'], rec['session'], rec['epoch'])] = rec['actual_rt']

exp_rt_change = []   # expected RT - previous epoch's actual RT
rt_improvement = []  # expected RT - actual RT (positive = beat expectation)
for s in switch_epochs_rt:
    prev_rt = prev_rt_lookup.get((s['mouse'], s['session'], s['epoch'] - 1), None)
    if prev_rt is not None:
        exp_rt_change.append(s['expected_rt'] - prev_rt)
        rt_improvement.append(s['expected_rt'] - s['actual_rt'])

exp_rt_change = np.array(exp_rt_change)
rt_improvement = np.array(rt_improvement)

# Split into tertiles by expected RT change
terts = np.percentile(exp_rt_change, [33.3, 66.7])
grp_labels = ['Small', 'Medium', 'Large']
grp_masks = [
    exp_rt_change <= terts[0],
    (exp_rt_change > terts[0]) & (exp_rt_change <= terts[1]),
    exp_rt_change > terts[1],
]

fig10, axes10 = plt.subplots(1, 3, figsize=(9, 3),
                              gridspec_kw={'wspace': 0.4, 'left': 0.08,
                                           'right': 0.96, 'bottom': 0.18,
                                           'top': 0.88})

# (a) Scatter: expected RT change vs RT improvement
ax = axes10[0]
ax.scatter(exp_rt_change, rt_improvement, s=15, c='k', alpha=0.5, edgecolors='none')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Expected \u0394RT (s)')
ax.set_ylabel('RT improvement (expected - actual, s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
if len(exp_rt_change) > 3:
    from scipy.stats import pearsonr
    r_du, p_du = pearsonr(exp_rt_change, rt_improvement)
    ax.set_title(f'r={r_du:.3f}, p={p_du:.4f}', fontsize=7)

# (b) Bar plot: RT improvement by tertile of expected RT change
ax = axes10[1]
means = [np.nanmean(rt_improvement[m]) for m in grp_masks]
sems = [np.nanstd(rt_improvement[m]) / np.sqrt(np.sum(m)) for m in grp_masks]
colors = ['#888888', '#555555', '#222222']
ax.bar(range(3), means, yerr=sems, color=colors, edgecolor='white',
       capsize=3, error_kw={'linewidth': 0.8})
ax.set_xticks(range(3))
ax.set_xticklabels([f'{l}\n(n={np.sum(m)})' for l, m in zip(grp_labels, grp_masks)],
                    fontsize=7)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Expected \u0394RT tertile')
ax.set_ylabel('RT improvement (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) Bar plot: actual vs expected RT by tertile
ax = axes10[2]
w = 0.3
for gi, (label, mask) in enumerate(zip(grp_labels, grp_masks)):
    act_g = np.array([switch_epochs_rt[i]['actual_rt'] for i in range(len(switch_epochs_rt))
                       if i < len(exp_rt_change) and mask[i]])
    exp_g = np.array([switch_epochs_rt[i]['expected_rt'] for i in range(len(switch_epochs_rt))
                       if i < len(exp_rt_change) and mask[i]])
    ax.bar(gi - w/2, np.nanmean(exp_g), width=w, color='cornflowerblue',
           yerr=np.nanstd(exp_g)/np.sqrt(len(exp_g)), capsize=2, error_kw={'linewidth': 0.8})
    ax.bar(gi + w/2, np.nanmean(act_g), width=w, color='k',
           yerr=np.nanstd(act_g)/np.sqrt(len(act_g)), capsize=2, error_kw={'linewidth': 0.8})
ax.set_xticks(range(3))
ax.set_xticklabels(grp_labels, fontsize=7)
ax.set_xlabel('Expected \u0394RT tertile')
ax.set_ylabel('Time to reward (s)')
ax.legend(['Expected', 'Actual'], frameon=False, fontsize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname10 = 'rt_recovery_by_threshold_change'
fig10.savefig(os.path.join(PANEL_DIR, f'{fname10}.png'), dpi=300)
fig10.savefig(os.path.join(PANEL_DIR, f'{fname10}.svg'))
plt.show()
print(f"Saved {fname10}")

#%% ============================================================================
# CELL 11: Learning metrics — RT improvement and acute RPE
# ============================================================================
from scipy.stats import pearsonr, spearmanr

# Filter: threshold increases with expected ΔRT > 0.5s (meaningful perturbations)
MIN_EXP_DRT = 0.5
learning_epochs = [s for s in all_epoch_stats if s['epoch'] > 0
                   and np.isfinite(s.get('expected_delta_rt', np.nan))
                   and np.isfinite(s.get('actual_delta_rt', np.nan))
                   and np.isfinite(s.get('rpe_mean_acute', np.nan))
                   and s.get('expected_delta_rt', 0) > MIN_EXP_DRT]

exp_drt = np.array([s['expected_delta_rt'] for s in learning_epochs])
act_drt = np.array([s['actual_delta_rt'] for s in learning_epochs])
rt_improve = exp_drt - act_drt  # positive = animal beat expectation
rpe_acute = np.array([s['rpe_mean_acute'] for s in learning_epochs])
n_trials_ep = np.array([s['n_trials'] for s in learning_epochs])
exp_rt = np.array([s['expected_rt'] for s in learning_epochs])
act_rt = np.array([s['actual_rt'] for s in learning_epochs])

fig11, axes11 = plt.subplots(2, 3, figsize=(10, 6),
                               gridspec_kw={'wspace': 0.4, 'hspace': 0.45,
                                            'left': 0.08, 'right': 0.96,
                                            'bottom': 0.10, 'top': 0.92})

n_ep = len(learning_epochs)

# --- Top row: RT improvement (expected - actual, in seconds) ---

# (a) RT improvement vs expected ΔRT
ax = axes11[0, 0]
ax.scatter(exp_drt, rt_improve, s=15, c='k', alpha=0.5, edgecolors='none')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Expected \u0394RT (s)')
ax.set_ylabel('RT improvement (expected \u2212 actual, s)')
if n_ep > 3:
    r_v, p_v = pearsonr(exp_drt, rt_improve)
    ax.set_title(f'Increases, exp \u0394RT>{MIN_EXP_DRT}s (n={n_ep})\n'
                 f'r={r_v:.3f}, p={p_v:.4f}', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) Distribution of RT improvement
ax = axes11[0, 1]
ax.hist(rt_improve, bins=20, color='k', edgecolor='white', alpha=0.7)
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(np.nanmedian(rt_improve), color='r', linewidth=1.2)
ax.set_xlabel('RT improvement (s)')
ax.set_ylabel('Count')
frac_pos = np.sum(rt_improve > 0) / n_ep
ax.set_title(f'Median={np.nanmedian(rt_improve):.2f}s\n'
             f'{np.sum(rt_improve > 0)}/{n_ep} beat expected ({frac_pos:.0%})',
             fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) Actual vs expected RT scatter
ax = axes11[0, 2]
ax.scatter(exp_rt, act_rt, s=15, c='k', alpha=0.5, edgecolors='none')
mn = min(np.nanmin(exp_rt), np.nanmin(act_rt)) - 0.2
mx = max(np.nanmax(exp_rt), np.nanmax(act_rt)) + 0.2
ax.plot([mn, mx], [mn, mx], 'k--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Expected RT (s)')
ax.set_ylabel('Actual RT (s)')
ax.set_aspect('equal')
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_title(f'Filtered to exp \u0394RT > {MIN_EXP_DRT}s', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Bottom row: Acute mean RPE (first 15 trials post-switch) ---

# (d) Acute RPE vs expected ΔRT
ax = axes11[1, 0]
ax.scatter(exp_drt, rpe_acute, s=15, c='k', alpha=0.5, edgecolors='none')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Expected \u0394RT (s)')
ax.set_ylabel(f'Mean RPE (first {RPE_WINDOW} trials)')
if n_ep > 3:
    r_ri, p_ri = pearsonr(exp_drt, rpe_acute)
    ax.set_title(f'n={n_ep}\nr={r_ri:.3f}, p={p_ri:.4f}', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (e) Acute RPE vs RT improvement
ax = axes11[1, 1]
ax.scatter(rpe_acute, rt_improve, s=15, c='k', alpha=0.5, edgecolors='none')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel(f'Mean RPE (first {RPE_WINDOW} trials)')
ax.set_ylabel('RT improvement (s)')
if n_ep > 3:
    r_rr, p_rr = pearsonr(rpe_acute, rt_improve)
    ax.set_title(f'r={r_rr:.3f}, p={p_rr:.4f}', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (f) Acute RPE distribution
ax = axes11[1, 2]
ax.hist(rpe_acute, bins=20, color='k', edgecolor='white', alpha=0.7)
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(np.nanmedian(rpe_acute), color='r', linewidth=1.2)
ax.set_xlabel(f'Mean RPE, first {RPE_WINDOW} trials (s)')
ax.set_ylabel('Count')
ax.set_title(f'Median={np.nanmedian(rpe_acute):.2f}s\n'
             f'Negative = worse than trailing avg', fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname11 = 'learning_metrics_recovery_and_rpe'
fig11.savefig(os.path.join(PANEL_DIR, f'{fname11}.png'), dpi=300)
fig11.savefig(os.path.join(PANEL_DIR, f'{fname11}.svg'))
plt.show()
print(f"Saved {fname11}")

# Print summary
print(f"\n--- Learning metric summary (threshold increases, exp ΔRT > {MIN_EXP_DRT}s) ---")
print(f"  N epochs: {n_ep}")
print(f"  RT improvement: median={np.nanmedian(rt_improve):.2f}s, "
      f"mean={np.nanmean(rt_improve):.2f}s")
print(f"    >0 (beat expected): {np.sum(rt_improve > 0)}/{n_ep}")
print(f"  Acute mean RPE: median={np.nanmedian(rpe_acute):.2f}s, "
      f"mean={np.nanmean(rpe_acute):.2f}s")
print(f"    <0 (net negative): {np.sum(rpe_acute < 0)}/{n_ep}")

#%% ============================================================================
# CELL 12: When and how much — refining the threshold-change protocol
# ============================================================================
# Manual threshold-increases were chosen by experimenter intuition. Treat that
# variability as natural-experiment perturbations and ask which (state at
# change time × magnitude of change) combinations produced the largest CN
# learning signal.
#
# Dependent variable (single outcome):
#   cn_gain = mean(CN_z over post window) − mean(CN_z over pre window)
#   CN_z is z-scored within session by first 10 trials (matches Cell 8h).
#
# State (last K_PRE trials of prior epoch): pre_hr, pre_cn_z, prior_epoch_len,
#                                            trials_since_start.
# Magnitude: expected_delta_rt, delta_upper, ratio_upper.

from scipy.stats import pearsonr

K_PRE = 10    # window for pre-switch state
K_POST = 10   # window for post-switch learning signal

# Group epoch records by session, ordered by epoch
sess_records = {}
for r in all_epoch_stats:
    sess_records.setdefault((r['mouse'], r['session']), []).append(r)
for k in sess_records:
    sess_records[k].sort(key=lambda x: x['epoch'])

# Per-session CN z-scoring (baseline = first 10 trials, matches Cell 8h)
sess_cn_z = {}
sess_rt_first10 = {}
for (mouse, session), tdata in all_session_trials.items():
    cn_arr = tdata['cn'].astype(float)
    base_mean = np.nanmean(cn_arr[:10])
    base_std = np.nanstd(cn_arr[:10])
    if not np.isfinite(base_std) or base_std < 1e-6:
        sess_cn_z[(mouse, session)] = None
    else:
        sess_cn_z[(mouse, session)] = (cn_arr - base_mean) / base_std
    rt0 = tdata['rt'][:10].astype(float).copy()
    rt0[~np.isfinite(rt0)] = 10.0
    sess_rt_first10[(mouse, session)] = float(np.nanmean(rt0))

transitions = []
for (mouse, session), recs in sess_records.items():
    tdata = all_session_trials.get((mouse, session))
    if tdata is None:
        continue
    cn_z_sess = sess_cn_z.get((mouse, session))
    if cn_z_sess is None:
        continue

    hit_s = tdata['hit'].astype(float)
    rt_s = tdata['rt'].copy()
    rt_s[~np.isfinite(rt_s)] = 10.0   # fill miss trials with 10 s
    hit_rpe_s = tdata['hit_rpe']

    for ei in range(1, len(recs)):
        cur, prev = recs[ei], recs[ei - 1]
        if not np.isfinite(cur.get('expected_delta_rt', np.nan)):
            continue

        delta_upper = cur['upper'] - prev['upper']
        if delta_upper <= 0:   # threshold INCREASES only
            continue

        t0 = cur['trial_start']
        pre_a = max(prev['trial_start'], t0 - K_PRE)
        pre_b = t0
        if pre_b - pre_a < 3:
            continue
        post_a = t0
        post_b = min(cur['trial_end'], t0 + K_POST)
        if post_b - post_a < 5:
            continue

        pre_hr = np.nanmean(hit_s[pre_a:pre_b])
        post_hr = np.nanmean(hit_s[post_a:post_b])
        hr_gain = post_hr - pre_hr
        pre_rt = np.nanmean(rt_s[pre_a:pre_b])
        pre_hit_rpe = np.nanmean(hit_rpe_s[pre_a:pre_b])
        pre_cn_z = np.nanmean(cn_z_sess[pre_a:pre_b])
        post_cn_z = np.nanmean(cn_z_sess[post_a:post_b])
        cn_gain = post_cn_z - pre_cn_z

        # Linear fit of CN_z over the pre window (origin at pre_a)
        pre_x = np.arange(pre_b - pre_a, dtype=float)
        pre_y = cn_z_sess[pre_a:pre_b]
        valid_pre = np.isfinite(pre_y)
        if valid_pre.sum() >= 3:
            Xpre = np.column_stack([np.ones(valid_pre.sum()), pre_x[valid_pre]])
            cf_pre, *_ = np.linalg.lstsq(Xpre, pre_y[valid_pre], rcond=None)
            intercept_pre, slope_pre = cf_pre[0], cf_pre[1]
        else:
            intercept_pre, slope_pre = np.nan, np.nan

        # Linear fit of CN_z over the post window (origin still at pre_a, so x continues)
        post_x = np.arange(pre_b - pre_a, post_b - pre_a, dtype=float)
        post_y = cn_z_sess[post_a:post_b]
        valid_post = np.isfinite(post_y)
        if valid_post.sum() >= 5:
            Xpost = np.column_stack([np.ones(valid_post.sum()), post_x[valid_post]])
            cf_post, *_ = np.linalg.lstsq(Xpost, post_y[valid_post], rcond=None)
            slope_post = cf_post[1]
            cn_slope = slope_post
        else:
            slope_post = np.nan
            cn_slope = np.nan

        # Boost in learning rate driven by the threshold change
        delta_slope = (slope_post - slope_pre
                       if np.isfinite(slope_post) and np.isfinite(slope_pre)
                       else np.nan)

        # Excess CN gain beyond the pre-switch trend extrapolated forward
        if (np.isfinite(slope_pre) and np.isfinite(intercept_pre)
                and valid_post.sum() >= 5):
            predicted_post = slope_pre * post_x + intercept_pre
            excess_cn = (np.nanmean(post_y) - np.nanmean(predicted_post))
        else:
            excess_cn = np.nan

        ratio_upper = (cur['upper'] / prev['upper']
                       if prev['upper'] > 0 else np.nan)

        transitions.append({
            'mouse': mouse, 'session': session, 'epoch': cur['epoch'],
            'trial_start': t0,
            # state at time of change
            'pre_hr': pre_hr,
            'pre_rt': pre_rt,
            'pre_hit_rpe': pre_hit_rpe,
            'pre_cn_z': pre_cn_z,
            'prior_epoch_len': float(prev['n_trials']),
            'trials_since_start': float(t0),
            'rt_first10': sess_rt_first10.get((mouse, session), np.nan),
            # magnitude
            'expected_delta_rt': cur['expected_delta_rt'],
            'delta_upper': float(delta_upper),
            'ratio_upper': float(ratio_upper),
            # outcomes
            'rt_improvement': cur['expected_rt'] - cur['actual_rt'],
            'rpe_mean_acute': cur.get('rpe_mean_acute', np.nan),
            'cn_gain': cn_gain,
            'cn_slope': cn_slope,
            'hr_gain': hr_gain,
            'excess_cn': excess_cn,
            'delta_slope': delta_slope,
            'slope_pre': slope_pre,
            'slope_post': slope_post,
        })

print(f"\nCELL 12: collected {len(transitions)} threshold-increase transitions "
      f"from {len(set((t['mouse'], t['session']) for t in transitions))} sessions")

# --- Distribution of pre-switch hit rate (verify the experimenter rule) ---
pre_hr_dist = np.array([t['pre_hr'] for t in transitions], dtype=float)
fig12_hr, ax = plt.subplots(1, 1, figsize=(4.0, 3.0),
                             gridspec_kw={'left': 0.16, 'right': 0.96,
                                          'bottom': 0.18, 'top': 0.88})
ax.hist(pre_hr_dist, bins=np.linspace(0, 1, 21),
        color='#444', edgecolor='white')
ax.axvline(np.nanmedian(pre_hr_dist), color='r', linewidth=1.2,
           label=f'median={np.nanmedian(pre_hr_dist):.2f}')
ax.axvline(1.0, color='cornflowerblue', linewidth=1.0, linestyle='--',
           label='nominal rule (HR=1.0)')
ax.set_xlabel(f'Pre-switch hit rate (last {K_PRE} trials of prior epoch)')
ax.set_ylabel('Number of transitions')
n_one = int(np.sum(pre_hr_dist >= 0.99))
n_total = len(pre_hr_dist)
ax.set_title(f'n={n_total} transitions; '
             f'{n_one} at HR≥0.99 ({n_one/n_total:.0%})', fontsize=8)
ax.legend(frameon=False, fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname12_hr = 'pre_switch_hr_distribution'
fig12_hr.savefig(os.path.join(PANEL_DIR, f'{fname12_hr}.png'), dpi=300)
fig12_hr.savefig(os.path.join(PANEL_DIR, f'{fname12_hr}.svg'))
plt.show()
print(f"Saved {fname12_hr}")
print(f"  pre_hr quartiles: "
      f"25%={np.nanpercentile(pre_hr_dist,25):.2f}, "
      f"50%={np.nanpercentile(pre_hr_dist,50):.2f}, "
      f"75%={np.nanpercentile(pre_hr_dist,75):.2f}")

state_features = ['pre_hr', 'pre_rt', 'pre_hit_rpe', 'pre_cn_z',
                  'prior_epoch_len', 'trials_since_start', 'rt_first10']
mag_features = ['expected_delta_rt', 'delta_upper', 'ratio_upper']
predictors = state_features + mag_features
OUTCOMES = [
    ('cn_gain', 'CN gain (post−pre, z)'),
    ('excess_cn', 'CN excess vs pre-trend (z)'),
    ('delta_slope', 'Δ slope (post − pre, z/trial)'),
    ('hr_gain', 'HR change (post−pre)'),
]

def _arr(name):
    return np.array([t[name] for t in transitions], dtype=float)

# --- Figure 12a: marginal scatter, each predictor vs each outcome ---
fig12a, axes12a = plt.subplots(len(OUTCOMES), len(predictors),
                                figsize=(1.9 * len(predictors),
                                         2.2 * len(OUTCOMES)),
                                gridspec_kw={'wspace': 0.55, 'hspace': 0.55,
                                             'left': 0.06, 'right': 0.98,
                                             'bottom': 0.10, 'top': 0.92})
for oi, (okey, olabel) in enumerate(OUTCOMES):
    y = _arr(okey)
    for pi, pname in enumerate(predictors):
        ax = axes12a[oi, pi]
        x = _arr(pname)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 5:
            ax.scatter(x[ok], y[ok], s=10, c='k', alpha=0.4, edgecolors='none')
            r, p = pearsonr(x[ok], y[ok])
            ax.set_title(f'r={r:.2f}, p={p:.3f}', fontsize=6)
        ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if oi == len(OUTCOMES) - 1:
            ax.set_xlabel(pname, fontsize=7)
        if pi == 0:
            ax.set_ylabel(olabel, fontsize=7)

fname12a = 'threshold_change_predictors_marginal'
fig12a.savefig(os.path.join(PANEL_DIR, f'{fname12a}.png'), dpi=300)
fig12a.savefig(os.path.join(PANEL_DIR, f'{fname12a}.svg'))
plt.show()
print(f"Saved {fname12a}")

# --- Figure 12b: 2D heatmaps — outcome in (state × magnitude) bins ---
N_BINS = 3

def _make_bins(v, n_bins):
    """Quantile bins with fallback to linspace if too many ties."""
    q = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(q) >= 3:
        return q
    rng = v.max() - v.min()
    if rng < 1e-9:
        return None
    return np.linspace(v.min(), v.max() + rng * 1e-3, n_bins + 1)


def _heatmap(ax, x, y, z, xlabel, ylabel, title):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(ok.sum())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}  (n={n})', fontsize=8)
    if n < 6:
        ax.text(0.5, 0.5, f'n={n} (too few)', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        return None, None
    x_ok, y_ok, z_ok = x[ok], y[ok], z[ok]
    print(f"    {xlabel}: range=[{x_ok.min():.3g}, {x_ok.max():.3g}], "
          f"unique={len(np.unique(x_ok))}")
    print(f"    {ylabel}: range=[{y_ok.min():.3g}, {y_ok.max():.3g}], "
          f"unique={len(np.unique(y_ok))}")

    xb = _make_bins(x_ok, N_BINS)
    yb = _make_bins(y_ok, N_BINS)
    if xb is None or yb is None:
        ax.scatter(x_ok, y_ok, c=z_ok, cmap='RdBu_r', s=25,
                   edgecolors='black', linewidths=0.3)
        ax.text(0.5, 0.95, 'no variance — scatter only', ha='center', va='top',
                transform=ax.transAxes, fontsize=7, color='gray')
        return None, None

    H = np.full((len(yb) - 1, len(xb) - 1), np.nan)
    N = np.zeros_like(H, dtype=int)
    xi = np.clip(np.digitize(x_ok, xb) - 1, 0, len(xb) - 2)
    yi = np.clip(np.digitize(y_ok, yb) - 1, 0, len(yb) - 2)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            mask = (xi == j) & (yi == i)
            if mask.sum() > 0:
                H[i, j] = np.nanmean(z_ok[mask])
                N[i, j] = int(mask.sum())

    if not np.any(np.isfinite(H)):
        ax.scatter(x_ok, y_ok, c=z_ok, cmap='RdBu_r', s=25,
                   edgecolors='black', linewidths=0.3)
        ax.text(0.5, 0.95, 'all bins empty — scatter only', ha='center', va='top',
                transform=ax.transAxes, fontsize=7, color='gray')
        return None, None

    vmax = np.nanmax(np.abs(H))
    if not np.isfinite(vmax) or vmax < 1e-9:
        vmax = 1.0
    im = ax.imshow(H, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[xb[0], xb[-1], yb[0], yb[-1]])
    # Overlay raw points
    ax.scatter(x_ok, y_ok, c='white', s=8, edgecolors='black',
               linewidths=0.3, alpha=0.6)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if N[i, j] > 0:
                cy = (yb[i] + yb[i + 1]) / 2
                cx = (xb[j] + xb[j + 1]) / 2
                ax.text(cx, cy, f'{N[i, j]}', ha='center', va='center',
                        fontsize=8, color='black', alpha=0.9, weight='bold')
    return im, (H, xb, yb)

pre_hr_t = _arr('pre_hr')
pre_rt_t = _arr('pre_rt')
X_full = np.column_stack([_arr(p) for p in predictors])
ok_rows = np.all(np.isfinite(X_full), axis=1)

heatmap_results_all = {}
fig12b, axes12b = plt.subplots(1, len(OUTCOMES), figsize=(4.5 * len(OUTCOMES), 4.0),
                                gridspec_kw={'wspace': 0.45,
                                             'left': 0.10, 'right': 0.93,
                                             'bottom': 0.14, 'top': 0.85})
if len(OUTCOMES) == 1:
    axes12b = [axes12b]

for ax, (okey, olabel) in zip(axes12b, OUTCOMES):
    out_t = _arr(okey)
    im, hres = _heatmap(ax, pre_hr_t, pre_rt_t, out_t,
                        'Pre-switch hit rate', 'Pre-switch RT (s)', olabel)
    if im is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    heatmap_results_all[okey] = [(olabel, 'pre_hr', 'pre_rt', hres)]

fname12b = 'threshold_change_state_pre_hr_x_pre_rt'
fig12b.savefig(os.path.join(PANEL_DIR, f'{fname12b}.png'), dpi=300)
fig12b.savefig(os.path.join(PANEL_DIR, f'{fname12b}.svg'))
plt.show()
print(f"Saved {fname12b}")

# --- Multivariate standardized regression: which predictors drive each outcome? ---
for okey, olabel in OUTCOMES:
    y_out = _arr(okey)
    ok = ok_rows & np.isfinite(y_out)
    print(f"\n--- Multivariate standardized β  (outcome = {okey}, "
          f"n_transitions={ok.sum()}) ---")
    if ok.sum() >= len(predictors) + 5:
        Xv = X_full[ok]
        yv = y_out[ok]
        Xz = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0)
        yz = (yv - yv.mean()) / (yv.std() if yv.std() > 0 else 1.0)
        Xd = np.column_stack([np.ones(Xz.shape[0]), Xz])
        coef, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
        for j, pn in enumerate(predictors):
            print(f"      {pn:>22}  β = {coef[j+1]:+.3f}")
    else:
        print(f"  skipped (n={ok.sum()} too small)")

# --- Recommendation: best (state × magnitude) bin per outcome ---
print(f"\n--- Recommendation: best-performing (state × magnitude) bin ---")
print(f"  K_PRE={K_PRE} (state window), K_POST={K_POST} (outcome window)")
for okey, olabel in OUTCOMES:
    print(f"  Outcome = {okey}:")
    for tt, xl, yl, hres in heatmap_results_all[okey]:
        if hres is None:
            continue
        H, xb, yb = hres
        if not np.any(np.isfinite(H)):
            continue
        i_max, j_max = np.unravel_index(np.nanargmax(H), H.shape)
        val = H[i_max, j_max]
        x_lo, x_hi = xb[j_max], xb[j_max + 1]
        y_lo, y_hi = yb[i_max], yb[i_max + 1]
        print(f"      best bin: {xl} in [{x_lo:.2f}, {x_hi:.2f}], "
              f"{yl} in [{y_lo:.2f}, {y_hi:.2f}]  → mean = {val:+.3f}")

#%% ============================================================================
# CELL 13: Post-transition RT slope — what predicts the dynamics?
# ============================================================================
# Cell 8 showed two qualitative patterns after a threshold change:
#   (a) "ideal" — RT keeps falling across the post-transition window
#   (b) "fast"  — RT drops fast on trial 1, then drifts back up
# In both cases mean(RT_post) < mean(RT_pre); the within-post slope
# distinguishes them. This cell asks which pre-transition variables
# (hit, rpe, rt, hit_rpe, trial #, CN features) predict that slope.

from scipy.stats import pearsonr, mannwhitneyu

K_POST_RT = 20      # max post window for fitting RT slope
MIN_POST_TRIALS = 8

rt_dynamics = []
for tr in transitions:
    mouse, session = tr['mouse'], tr['session']
    tdata = all_session_trials.get((mouse, session))
    if tdata is None:
        continue
    rt_s = tdata['rt'].copy()
    rt_s[~np.isfinite(rt_s)] = 10.0
    rt_rpe_s = tdata['rt_rpe']
    hit_s = tdata['hit'].astype(float)

    recs = sess_records[(mouse, session)]
    cur_idx = next((i for i, r in enumerate(recs)
                    if r['epoch'] == tr['epoch']), None)
    if cur_idx is None or cur_idx == 0:
        continue
    cur_rec = recs[cur_idx]
    prev_rec = recs[cur_idx - 1]

    t0 = tr['trial_start']
    post_a = t0
    post_b = min(cur_rec['trial_end'], t0 + K_POST_RT)
    if post_b - post_a < MIN_POST_TRIALS:
        continue
    rt_post = rt_s[post_a:post_b]

    x = np.arange(len(rt_post), dtype=float)
    valid = np.isfinite(rt_post)
    if valid.sum() < MIN_POST_TRIALS:
        continue
    Xfit = np.column_stack([np.ones(valid.sum()), x[valid]])
    cf, *_ = np.linalg.lstsq(Xfit, rt_post[valid], rcond=None)
    rt_intercept_post, rt_slope_post = float(cf[0]), float(cf[1])

    # Also report initial drop and late-window RT for context
    rt_init_drop = float(tr['pre_rt'] - np.nanmean(rt_post[:3]))
    late_idx = max(MIN_POST_TRIALS // 2, len(rt_post) - 5)
    rt_late = float(np.nanmean(rt_post[late_idx:]))

    # Pre-transition RPE and pre-RT slope (state dynamics, not just level)
    pre_a_w = max(prev_rec['trial_start'], t0 - 10)
    rt_rpe_pre = float(np.nanmean(rt_rpe_s[pre_a_w:t0]))
    rt_pre_window = rt_s[pre_a_w:t0]
    if np.isfinite(rt_pre_window).sum() >= 4:
        xp = np.arange(len(rt_pre_window), dtype=float)
        vp = np.isfinite(rt_pre_window)
        Xp = np.column_stack([np.ones(vp.sum()), xp[vp]])
        cfp, *_ = np.linalg.lstsq(Xp, rt_pre_window[vp], rcond=None)
        rt_slope_pre = float(cfp[1])
    else:
        rt_slope_pre = np.nan

    rt_dynamics.append({
        **tr,
        'rt_slope_post': rt_slope_post,
        'rt_intercept_post': rt_intercept_post,
        'rt_init_drop': rt_init_drop,
        'rt_late': rt_late,
        'rt_slope_pre': rt_slope_pre,
        'rt_rpe_pre': rt_rpe_pre,
        'cn_slope_pre': tr['slope_pre'],
        'cn_slope_post': tr['slope_post'],
        'n_post_rt': int(valid.sum()),
    })

print(f"\nCELL 13: {len(rt_dynamics)} transitions with valid post-RT slopes "
      f"(K_POST_RT={K_POST_RT}, min={MIN_POST_TRIALS})")

# Predictors: pre-transition state and perturbation magnitude
state_predictors = ['pre_hr', 'pre_rt', 'pre_hit_rpe', 'rt_rpe_pre',
                    'pre_cn_z', 'rt_slope_pre',
                    'trials_since_start', 'prior_epoch_len', 'rt_first10']
mag_predictors = ['delta_upper', 'expected_delta_rt']
predictors_rt = state_predictors + mag_predictors

def _arr_d(name):
    return np.array([t.get(name, np.nan) for t in rt_dynamics], dtype=float)

y_slope = _arr_d('rt_slope_post')
print(f"  rt_slope_post: median={np.nanmedian(y_slope):+.3f} s/trial, "
      f"frac<0 (ideal)={np.mean(y_slope < 0):.2f}, "
      f"frac>0 (decline)={np.mean(y_slope > 0):.2f}")

# --- Figure 13a: marginal scatter — each predictor vs RT slope ---
n_p = len(predictors_rt)
fig13a, axes = plt.subplots(1, n_p, figsize=(1.9 * n_p, 2.4),
                             gridspec_kw={'wspace': 0.55,
                                          'left': 0.05, 'right': 0.99,
                                          'bottom': 0.22, 'top': 0.88})
for ai, pname in enumerate(predictors_rt):
    ax = axes[ai]
    x_v = _arr_d(pname)
    ok = np.isfinite(x_v) & np.isfinite(y_slope)
    if ok.sum() >= 5:
        ax.scatter(x_v[ok], y_slope[ok], s=12, c='k', alpha=0.5,
                   edgecolors='none')
        r, p = pearsonr(x_v[ok], y_slope[ok])
        ax.set_title(f'r={r:+.2f}, p={p:.3f}', fontsize=7)
        m, b = np.polyfit(x_v[ok], y_slope[ok], 1)
        xf = np.array([x_v[ok].min(), x_v[ok].max()])
        ax.plot(xf, m * xf + b, color='C3', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.set_xlabel(pname, fontsize=7)
    if ai == 0:
        ax.set_ylabel('RT slope post (s/trial)', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fname13a = 'rt_slope_predictors_marginal'
fig13a.savefig(os.path.join(PANEL_DIR, f'{fname13a}.png'), dpi=300)
fig13a.savefig(os.path.join(PANEL_DIR, f'{fname13a}.svg'))
plt.show()
print(f"Saved {fname13a}")

# --- Figure 13b: standardized multivariate β with bootstrap 95% CI ---
X_mat = np.column_stack([_arr_d(p) for p in predictors_rt])
ok_rows_rt = np.all(np.isfinite(X_mat), axis=1) & np.isfinite(y_slope)
n_obs = int(ok_rows_rt.sum())

X_ok = X_mat[ok_rows_rt]
y_ok = y_slope[ok_rows_rt]
X_z = (X_ok - X_ok.mean(axis=0)) / np.where(X_ok.std(axis=0) > 1e-9,
                                             X_ok.std(axis=0), 1.0)
y_z = (y_ok - y_ok.mean()) / (y_ok.std() if y_ok.std() > 1e-9 else 1.0)
Xd = np.column_stack([np.ones(X_z.shape[0]), X_z])
betas, *_ = np.linalg.lstsq(Xd, y_z, rcond=None)
beta_pred = betas[1:]

N_BOOT = 500
boot = np.full((N_BOOT, len(predictors_rt)), np.nan)
rng = np.random.default_rng(0)
for bi in range(N_BOOT):
    idx = rng.integers(0, n_obs, n_obs)
    cf, *_ = np.linalg.lstsq(Xd[idx], y_z[idx], rcond=None)
    boot[bi] = cf[1:]
ci_lo = np.nanpercentile(boot, 2.5, axis=0)
ci_hi = np.nanpercentile(boot, 97.5, axis=0)

fig13b, ax = plt.subplots(1, 1, figsize=(0.55 * n_p + 1.5, 3.0),
                           gridspec_kw={'left': 0.18, 'right': 0.96,
                                        'bottom': 0.32, 'top': 0.90})
xs = np.arange(n_p)
ax.bar(xs, beta_pred, color='#888', edgecolor='black', linewidth=0.5)
ax.errorbar(xs, beta_pred,
            yerr=[beta_pred - ci_lo, ci_hi - beta_pred],
            fmt='none', color='black', linewidth=1.0, capsize=2.5)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xticks(xs)
ax.set_xticklabels(predictors_rt, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('β (standardized) → RT slope post')
ax.set_title(f'Multivariate β  (n={n_obs} transitions, 95% CI bootstrap)',
             fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname13b = 'rt_slope_betas'
fig13b.savefig(os.path.join(PANEL_DIR, f'{fname13b}.png'), dpi=300)
fig13b.savefig(os.path.join(PANEL_DIR, f'{fname13b}.svg'))
plt.show()
print(f"Saved {fname13b}")
print(f"  Predictor               β        95% CI")
for p_, b_, lo_, hi_ in zip(predictors_rt, beta_pred, ci_lo, ci_hi):
    sig = '*' if (lo_ > 0) or (hi_ < 0) else ''
    print(f"  {p_:22s} {b_:+.3f}    [{lo_:+.3f}, {hi_:+.3f}] {sig}")

# --- Figure 13c: split transitions into "ideal" vs "decline" by slope sign ---
slopes = _arr_d('rt_slope_post')
ideal_mask = slopes < 0
decline_mask = slopes > 0
n_ideal = int(ideal_mask.sum())
n_decline = int(decline_mask.sum())

fig13c, axes = plt.subplots(1, n_p, figsize=(1.7 * n_p, 2.6),
                             gridspec_kw={'wspace': 0.55,
                                          'left': 0.05, 'right': 0.99,
                                          'bottom': 0.22, 'top': 0.82})
for ai, pname in enumerate(predictors_rt):
    ax = axes[ai]
    vals = _arr_d(pname)
    a = vals[ideal_mask & np.isfinite(vals)]
    b = vals[decline_mask & np.isfinite(vals)]
    parts = ax.boxplot([a, b], widths=0.6, patch_artist=True,
                       showfliers=False, labels=['ideal', 'decline'])
    for patch, c in zip(parts['boxes'], ['#5b9bd5', '#e6794d']):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    if len(a) >= 3 and len(b) >= 3:
        _, pv = mannwhitneyu(a, b, alternative='two-sided')
        ax.set_title(f'{pname}\np={pv:.3f}', fontsize=7)
    else:
        ax.set_title(pname, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig13c.suptitle(f'Pre-state by post-RT-slope class '
                f'(ideal slope<0  n={n_ideal};  decline slope>0  n={n_decline})',
                fontsize=9)
fname13c = 'rt_slope_class_split'
fig13c.savefig(os.path.join(PANEL_DIR, f'{fname13c}.png'), dpi=300)
fig13c.savefig(os.path.join(PANEL_DIR, f'{fname13c}.svg'))
plt.show()
print(f"Saved {fname13c}")

# --- Figure 13d: CN-activity follow-up (does post-RT decline track CN dynamics?) ---
cn_predictors = ['pre_cn_z', 'cn_slope_pre', 'cn_slope_post', 'cn_gain',
                 'excess_cn']
fig13d, axes = plt.subplots(1, len(cn_predictors),
                             figsize=(1.9 * len(cn_predictors), 2.4),
                             gridspec_kw={'wspace': 0.55,
                                          'left': 0.06, 'right': 0.99,
                                          'bottom': 0.22, 'top': 0.85})
for ai, pname in enumerate(cn_predictors):
    ax = axes[ai]
    x_v = _arr_d(pname)
    ok = np.isfinite(x_v) & np.isfinite(y_slope)
    if ok.sum() >= 5:
        ax.scatter(x_v[ok], y_slope[ok], s=12, c='k', alpha=0.5,
                   edgecolors='none')
        r, p = pearsonr(x_v[ok], y_slope[ok])
        ax.set_title(f'r={r:+.2f}, p={p:.3f}', fontsize=7)
        m, b = np.polyfit(x_v[ok], y_slope[ok], 1)
        xf = np.array([x_v[ok].min(), x_v[ok].max()])
        ax.plot(xf, m * xf + b, color='C3', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.set_xlabel(pname, fontsize=7)
    if ai == 0:
        ax.set_ylabel('RT slope post (s/trial)', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig13d.suptitle('CN activity → post-RT slope', fontsize=9)
fname13d = 'rt_slope_cn_predictors'
fig13d.savefig(os.path.join(PANEL_DIR, f'{fname13d}.png'), dpi=300)
fig13d.savefig(os.path.join(PANEL_DIR, f'{fname13d}.svg'))
plt.show()
print(f"Saved {fname13d}")

# --- Summary block: paste-friendly + saved to file ---
import io
_summary_buf = io.StringIO()
def _sp(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_summary_buf)

all_predictors = predictors_rt + [p for p in cn_predictors
                                  if p not in predictors_rt]

uni_rows = []
for pname in all_predictors:
    x_v = _arr_d(pname)
    ok = np.isfinite(x_v) & np.isfinite(y_slope)
    if ok.sum() < 5:
        uni_rows.append((pname, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, int(ok.sum())))
        continue
    r, p = pearsonr(x_v[ok], y_slope[ok])
    a = x_v[ideal_mask & np.isfinite(x_v)]
    b = x_v[decline_mask & np.isfinite(x_v)]
    if len(a) >= 3 and len(b) >= 3:
        _, mwp = mannwhitneyu(a, b, alternative='two-sided')
        med_a, med_b = float(np.nanmedian(a)), float(np.nanmedian(b))
    else:
        mwp, med_a, med_b = np.nan, np.nan, np.nan
    uni_rows.append((pname, r, p, mwp, med_a, med_b,
                     len(a) + len(b), int(ok.sum())))

uni_rows.sort(key=lambda r: (np.inf if not np.isfinite(r[3]) else r[3]))

_sp("\n" + "=" * 78)
_sp(f"CELL 13 SUMMARY  (n={len(rt_dynamics)} transitions; "
    f"ideal={n_ideal}, decline={n_decline})")
_sp("=" * 78)
_sp(f"  rt_slope_post: median={np.nanmedian(y_slope):+.3f} s/trial, "
    f"frac<0={np.mean(y_slope<0):.2f}, frac>0={np.mean(y_slope>0):.2f}")
_sp("")
_sp("Univariate signal vs rt_slope_post  (sorted by Mann-Whitney p):")
_sp(f"  {'predictor':22s} {'pearson_r':>9s} {'p_corr':>7s}  "
    f"{'MW_p':>7s}  {'median_ideal':>12s} {'median_decline':>14s}")
for (pname, r, p, mwp, ma, mb, n_grp, n_corr) in uni_rows:
    sig_corr = '*' if (np.isfinite(p) and p < 0.05) else ' '
    sig_mw = '*' if (np.isfinite(mwp) and mwp < 0.05) else ' '
    r_str = f'{r:+.3f}' if np.isfinite(r) else '   nan'
    p_str = f'{p:.3f}' if np.isfinite(p) else ' nan '
    mw_str = f'{mwp:.3f}' if np.isfinite(mwp) else ' nan '
    ma_str = f'{ma:+.3f}' if np.isfinite(ma) else '   nan'
    mb_str = f'{mb:+.3f}' if np.isfinite(mb) else '   nan'
    _sp(f"  {pname:22s} {r_str:>9s}{sig_corr} {p_str:>6s}  "
        f"{mw_str:>6s}{sig_mw}  {ma_str:>12s} {mb_str:>14s}")

_sp("")
_sp("Multivariate standardized β  (bootstrap 95% CI):")
_sp(f"  {'predictor':22s} {'beta':>7s}  {'CI_lo':>7s} {'CI_hi':>7s}")
for p_, b_, lo_, hi_ in zip(predictors_rt, beta_pred, ci_lo, ci_hi):
    sig = '*' if (lo_ > 0) or (hi_ < 0) else ' '
    _sp(f"  {p_:22s} {b_:+.3f}  [{lo_:+.3f}, {hi_:+.3f}] {sig}")

_sp("")
_sp("Predictor correlation matrix (Pearson r):")
corr_predictors = predictors_rt
M = np.column_stack([_arr_d(p) for p in corr_predictors])
ok_all = np.all(np.isfinite(M), axis=1)
Mok = M[ok_all]
C = np.corrcoef(Mok, rowvar=False)
hdr = ' ' * 22 + ''.join(f'{p[:8]:>9s}' for p in corr_predictors)
_sp(hdr)
for i, p in enumerate(corr_predictors):
    row = f'  {p:22s}'
    for j in range(len(corr_predictors)):
        row += f'{C[i, j]:+9.2f}'
    _sp(row)
_sp("=" * 78)

_summary_path = os.path.join(PANEL_DIR, 'cell13_rt_slope_summary.txt')
with open(_summary_path, 'w', encoding='utf-8') as fh:
    fh.write(_summary_buf.getvalue())
print(f"\nSaved summary -> {_summary_path}")

# --- Per-mouse breakdown of ideal vs decline ---
from collections import defaultdict
per_mouse = defaultdict(lambda: {'ideal': 0, 'decline': 0,
                                  'slopes': [], 'sessions': set()})
for t in rt_dynamics:
    bucket = 'ideal' if t['rt_slope_post'] < 0 else 'decline'
    per_mouse[t['mouse']][bucket] += 1
    per_mouse[t['mouse']]['slopes'].append(t['rt_slope_post'])
    per_mouse[t['mouse']]['sessions'].add(t['session'])

print("\nPer-mouse breakdown:")
print(f"  {'mouse':10s} {'n_sess':>7s} {'ideal':>6s} {'decline':>8s} "
      f"{'frac_ideal':>11s} {'mean_slope':>11s}")
mouse_rows = []
for mouse in sorted(per_mouse.keys()):
    d = per_mouse[mouse]
    n_total = d['ideal'] + d['decline']
    frac_ideal = d['ideal'] / n_total if n_total else np.nan
    mean_slope = float(np.mean(d['slopes'])) if d['slopes'] else np.nan
    mouse_rows.append((mouse, len(d['sessions']), d['ideal'],
                       d['decline'], frac_ideal, mean_slope))
    print(f"  {mouse:10s} {len(d['sessions']):>7d} {d['ideal']:>6d} "
          f"{d['decline']:>8d} {frac_ideal:>11.2f} {mean_slope:>+11.3f}")

# Append to summary file
with open(_summary_path, 'a', encoding='utf-8') as fh:
    fh.write("\nPer-mouse breakdown:\n")
    fh.write(f"  {'mouse':10s} {'n_sess':>7s} {'ideal':>6s} {'decline':>8s} "
             f"{'frac_ideal':>11s} {'mean_slope':>11s}\n")
    for (m, ns, ni, nd, fi, ms) in mouse_rows:
        fh.write(f"  {m:10s} {ns:>7d} {ni:>6d} {nd:>8d} "
                 f"{fi:>11.2f} {ms:>+11.3f}\n")

# Stacked bar figure
fig13e, ax = plt.subplots(1, 1, figsize=(5.5, 3.0),
                           gridspec_kw={'left': 0.12, 'right': 0.96,
                                        'bottom': 0.18, 'top': 0.92})
mice_order = sorted(per_mouse.keys())
ideals = np.array([per_mouse[m]['ideal'] for m in mice_order])
declines = np.array([per_mouse[m]['decline'] for m in mice_order])
xs = np.arange(len(mice_order))
ax.bar(xs, ideals, color='#5b9bd5', label='ideal (slope<0)')
ax.bar(xs, declines, bottom=ideals, color='#e6794d',
       label='decline (slope>0)')
for xi, m in enumerate(mice_order):
    n_total = ideals[xi] + declines[xi]
    if n_total > 0:
        frac = ideals[xi] / n_total
        ax.text(xi, n_total + 0.3, f'{frac:.0%}', ha='center', va='bottom',
                fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(mice_order, rotation=0)
ax.set_ylabel('# transitions')
ax.set_title('Ideal vs decline transitions per mouse', fontsize=9)
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname13e = 'rt_slope_per_mouse'
fig13e.savefig(os.path.join(PANEL_DIR, f'{fname13e}.png'), dpi=300)
fig13e.savefig(os.path.join(PANEL_DIR, f'{fname13e}.svg'))
plt.show()
print(f"Saved {fname13e}")

#%% ============================================================================
# CELL 14: Threshold increases per session, by mouse
# ============================================================================
# For each (mouse, session), count the number of threshold INCREASES
# (i.e., epoch-to-epoch transitions where upper threshold went up).
# Plot one column per mouse with each session as a dot, plus the median.

from collections import defaultdict

incr_per_session = defaultdict(dict)   # mouse -> session -> count
for (mouse, session), recs in sess_records.items():
    n_incr = 0
    for ei in range(1, len(recs)):
        if recs[ei]['upper'] - recs[ei - 1]['upper'] > 0:
            n_incr += 1
    incr_per_session[mouse][session] = n_incr

mice_sorted = sorted(incr_per_session.keys())
print("\nThreshold increases per session, by mouse:")
print(f"  {'mouse':10s} {'n_sess':>7s} {'mean':>6s} {'median':>7s} "
      f"{'min':>4s} {'max':>4s}  counts")
for mouse in mice_sorted:
    counts = list(incr_per_session[mouse].values())
    print(f"  {mouse:10s} {len(counts):>7d} {np.mean(counts):>6.1f} "
          f"{int(np.median(counts)):>7d} {min(counts):>4d} {max(counts):>4d}  "
          f"{sorted(counts, reverse=True)}")

# --- Figure 14: per-mouse increases-per-session ---
fig14, ax = plt.subplots(1, 1, figsize=(5.5, 3.2),
                          gridspec_kw={'left': 0.12, 'right': 0.96,
                                       'bottom': 0.16, 'top': 0.92})
xs = np.arange(len(mice_sorted))
rng14 = np.random.default_rng(0)
for xi, mouse in enumerate(mice_sorted):
    counts = np.array(list(incr_per_session[mouse].values()), dtype=float)
    jitter = rng14.uniform(-0.18, 0.18, size=len(counts))
    ax.scatter(np.full_like(counts, xi) + jitter, counts,
               s=22, color='#666', alpha=0.7, edgecolors='none')
    # median bar
    ax.hlines(np.median(counts), xi - 0.30, xi + 0.30,
              colors='C3', linewidth=2.0, zorder=3)
    # n label
    ax.text(xi, ax.get_ylim()[1] if False else max(counts) + 0.4,
            f'n={len(counts)}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(xs)
ax.set_xticklabels(mice_sorted)
ax.set_ylabel('# threshold increases per session')
ax.set_title('Threshold increases per session — by mouse', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fname14 = 'threshold_increases_per_session_by_mouse'
fig14.savefig(os.path.join(PANEL_DIR, f'{fname14}.png'), dpi=300)
fig14.savefig(os.path.join(PANEL_DIR, f'{fname14}.svg'))
plt.show()
print(f"Saved {fname14}")

#%% ============================================================================
# CELL 15: Starting RT (first 10 trials) vs CN increase after threshold changes
# ============================================================================
# Predictor: rt_first10 (one value per session, baseline RT)
# Outcomes:
#   cn_gain     — post mean − pre mean (z)
#   excess_cn   — post mean − pre-trend extrapolated (z)
#   cn_slope    — slope of CN over post window (z/trial)
# Show two views:
#   (1) Per-transition scatter: each transition is one dot;
#       transitions in same session share x.
#   (2) Per-session aggregated scatter: mean(outcome) per session.

mice_set = sorted(set(t['mouse'] for t in transitions))
mouse_color = {m: c for m, c in zip(
    mice_set,
    plt.cm.tab10(np.linspace(0, 1, max(len(mice_set), 10)))
)}

CN_OUTCOMES = [
    ('cn_gain', 'CN gain (post−pre, z)'),
    ('excess_cn', 'CN excess vs pre-trend (z)'),
    ('cn_slope', 'CN slope post (z/trial)'),
]

# Group transitions by session
sess_groups = defaultdict(list)
for t in transitions:
    sess_groups[(t['mouse'], t['session'])].append(t)

# --- Figure 15a: per-transition scatter (one dot per threshold increase) ---
fig15a, axes = plt.subplots(1, len(CN_OUTCOMES),
                             figsize=(3.5 * len(CN_OUTCOMES), 3.2),
                             gridspec_kw={'wspace': 0.40,
                                          'left': 0.08, 'right': 0.98,
                                          'bottom': 0.16, 'top': 0.86})
for ai, (okey, olabel) in enumerate(CN_OUTCOMES):
    ax = axes[ai]
    x_v = np.array([t['rt_first10'] for t in transitions], dtype=float)
    y_v = np.array([t[okey] for t in transitions], dtype=float)
    cols = [mouse_color[t['mouse']] for t in transitions]
    ok = np.isfinite(x_v) & np.isfinite(y_v)
    if ok.sum() >= 5:
        ax.scatter(x_v[ok], y_v[ok], s=18, c=[cols[i] for i in np.where(ok)[0]],
                   alpha=0.7, edgecolors='none')
        r, p = pearsonr(x_v[ok], y_v[ok])
        m, b = np.polyfit(x_v[ok], y_v[ok], 1)
        xf = np.array([x_v[ok].min(), x_v[ok].max()])
        ax.plot(xf, m * xf + b, color='k', linewidth=1.0)
        ax.set_title(f'{olabel}\nr={r:+.3f}, p={p:.3f}, n={ok.sum()}',
                     fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.set_xlabel('rt_first10 (s, mean of first 10 trials)')
    if ai == 0:
        ax.set_ylabel('CN outcome')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend for mouse colors
handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=mouse_color[m],
                       label=m, markersize=6) for m in mice_set]
axes[-1].legend(handles=handles, frameon=False, fontsize=7,
                loc='upper right', bbox_to_anchor=(1.02, 1.02))

fname15a = 'rt_first10_vs_cn_gain_per_transition'
fig15a.savefig(os.path.join(PANEL_DIR, f'{fname15a}.png'), dpi=300)
fig15a.savefig(os.path.join(PANEL_DIR, f'{fname15a}.svg'))
plt.show()
print(f"Saved {fname15a}")

# --- Figure 15b: per-session aggregated (one dot per session) ---
sess_agg = []
for (mouse, session), tlist in sess_groups.items():
    rt0 = tlist[0]['rt_first10']
    if not np.isfinite(rt0):
        continue
    row = {'mouse': mouse, 'session': session, 'rt_first10': rt0,
           'n_transitions': len(tlist)}
    for okey, _ in CN_OUTCOMES:
        vals = np.array([t[okey] for t in tlist], dtype=float)
        row[okey + '_mean'] = float(np.nanmean(vals))
    sess_agg.append(row)

print(f"\nCELL 15: {len(sess_agg)} sessions in aggregate analysis")

fig15b, axes = plt.subplots(1, len(CN_OUTCOMES),
                             figsize=(3.5 * len(CN_OUTCOMES), 3.2),
                             gridspec_kw={'wspace': 0.40,
                                          'left': 0.08, 'right': 0.98,
                                          'bottom': 0.16, 'top': 0.86})
for ai, (okey, olabel) in enumerate(CN_OUTCOMES):
    ax = axes[ai]
    x_v = np.array([s['rt_first10'] for s in sess_agg], dtype=float)
    y_v = np.array([s[okey + '_mean'] for s in sess_agg], dtype=float)
    cols = [mouse_color[s['mouse']] for s in sess_agg]
    sizes = np.array([8 + 6 * s['n_transitions'] for s in sess_agg])
    ok = np.isfinite(x_v) & np.isfinite(y_v)
    if ok.sum() >= 5:
        ax.scatter(x_v[ok], y_v[ok], s=sizes[ok],
                   c=[cols[i] for i in np.where(ok)[0]],
                   alpha=0.75, edgecolors='black', linewidths=0.4)
        r, p = pearsonr(x_v[ok], y_v[ok])
        m, b = np.polyfit(x_v[ok], y_v[ok], 1)
        xf = np.array([x_v[ok].min(), x_v[ok].max()])
        ax.plot(xf, m * xf + b, color='k', linewidth=1.0)
        ax.set_title(f'{olabel}\nr={r:+.3f}, p={p:.3f}, n_sess={ok.sum()}',
                     fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.4, linestyle=':')
    ax.set_xlabel('rt_first10 (s)')
    if ai == 0:
        ax.set_ylabel('mean CN outcome (per session)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fname15b = 'rt_first10_vs_cn_gain_per_session'
fig15b.savefig(os.path.join(PANEL_DIR, f'{fname15b}.png'), dpi=300)
fig15b.savefig(os.path.join(PANEL_DIR, f'{fname15b}.svg'))
plt.show()
print(f"Saved {fname15b}")

# --- Print summary table ---
print(f"\nrt_first10 vs CN outcomes (per-transition / per-session):")
print(f"  {'outcome':18s} {'r_trans':>9s} {'p_trans':>8s}  "
      f"{'r_sess':>8s} {'p_sess':>7s}")
for okey, olabel in CN_OUTCOMES:
    x_t = np.array([t['rt_first10'] for t in transitions], dtype=float)
    y_t = np.array([t[okey] for t in transitions], dtype=float)
    ok = np.isfinite(x_t) & np.isfinite(y_t)
    r_t, p_t = (pearsonr(x_t[ok], y_t[ok]) if ok.sum() >= 5
                else (np.nan, np.nan))
    x_s = np.array([s['rt_first10'] for s in sess_agg], dtype=float)
    y_s = np.array([s[okey + '_mean'] for s in sess_agg], dtype=float)
    ok_s = np.isfinite(x_s) & np.isfinite(y_s)
    r_s, p_s = (pearsonr(x_s[ok_s], y_s[ok_s]) if ok_s.sum() >= 5
                else (np.nan, np.nan))
    print(f"  {okey:18s} {r_t:>+9.3f} {p_t:>8.3f}  "
          f"{r_s:>+8.3f} {p_s:>7.3f}")

# --- Median split at rt_first10 = 3 s, and outlier robustness ---
RT_SPLIT = 3.0
print(f"\nMedian split at rt_first10 = {RT_SPLIT}s:")
print(f"  {'outcome':18s} {'group':>10s} {'n':>4s} {'mean':>8s} "
      f"{'median':>8s}  MW_p")
for okey, olabel in CN_OUTCOMES:
    # Per-session aggregation
    x_s = np.array([s['rt_first10'] for s in sess_agg], dtype=float)
    y_s = np.array([s[okey + '_mean'] for s in sess_agg], dtype=float)
    ok_s = np.isfinite(x_s) & np.isfinite(y_s)
    lo = y_s[ok_s & (x_s < RT_SPLIT)]
    hi = y_s[ok_s & (x_s >= RT_SPLIT)]
    if len(lo) >= 3 and len(hi) >= 3:
        _, mwp = mannwhitneyu(lo, hi, alternative='two-sided')
    else:
        mwp = np.nan
    print(f"  {okey + ' (sess)':18s} {'<3s':>10s} {len(lo):>4d} "
          f"{np.mean(lo):>+8.3f} {np.median(lo):>+8.3f}")
    print(f"  {'':18s} {'>=3s':>10s} {len(hi):>4d} "
          f"{np.mean(hi):>+8.3f} {np.median(hi):>+8.3f}  p={mwp:.3f}")

# Robustness: drop top-rt_first10 outlier and refit cn_slope correlation
x_s = np.array([s['rt_first10'] for s in sess_agg], dtype=float)
y_s = np.array([s['cn_slope_mean'] for s in sess_agg], dtype=float)
ok_s = np.isfinite(x_s) & np.isfinite(y_s)
xs_ok = x_s[ok_s]
ys_ok = y_s[ok_s]
order = np.argsort(xs_ok)[::-1]
print(f"\nRobustness check on cn_slope (per-session):")
for k in [0, 1, 2, 3]:
    keep = np.ones(len(xs_ok), dtype=bool)
    if k > 0:
        keep[order[:k]] = False
    if keep.sum() >= 5:
        r_, p_ = pearsonr(xs_ok[keep], ys_ok[keep])
        rt_max_kept = xs_ok[keep].max()
        print(f"  drop top {k} rt_first10 (max kept = {rt_max_kept:.2f}s): "
              f"n={keep.sum()}, r={r_:+.3f}, p={p_:.3f}")

