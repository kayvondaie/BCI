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
rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
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

def transfer_fun(fluorescence, lower, upper, max_speed=3.3):
    """Apply the BCI transfer function: threshold-linear with saturation.

    Parameters
    ----------
    fluorescence : array — raw CN fluorescence values
    lower : float — lower threshold (speed = 0 below this)
    upper : float — upper threshold (speed = max_speed above this)
    max_speed : float — saturation speed (default 3.3)

    Returns
    -------
    speed : array — same shape as fluorescence, values in [0, max_speed]
    """
    gain = upper - lower
    if gain <= 0:
        return np.zeros_like(fluorescence)
    speed = (fluorescence - lower) / gain * max_speed
    speed = np.clip(speed, 0, max_speed)
    return speed


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

    spd = transfer_fun(fluor[:stp], thr_lower[i], thr_upper[i])
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
        spd_old = transfer_fun(fluor_use, lower_prev, upper_prev)
        spd_new = transfer_fun(fluor_use, lower_cur, upper_cur)
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
        speed_old_all.append(np.nanmean(transfer_fun(fluor_use, lower_prev, upper_prev)))
        speed_new_all.append(np.nanmean(transfer_fun(fluor_use, lower_cur, upper_cur)))
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
            spd_old = np.nanmean(transfer_fun(fl, lower_prev, upper_prev))
            spd_new = np.nanmean(transfer_fun(fl, lower_cur, upper_cur))
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
    speed_heatmap[:n_fr, ti] = transfer_fun(raw_fl[:n_fr], hardest_lower, hardest_upper)
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
                        spd_new = np.nanmean(transfer_fun(fl, lower_c, upper_c))
                        spd_old = np.nanmean(transfer_fun(fl, lower_ref, upper_ref))

                        if hit_sess[t] and spd_old > 0 and spd_new > 0:
                            ratio = spd_new / spd_old
                            scaled_rt = rt_sess[t] / ratio
                            exp_hits.append(scaled_rt < 10.0)
                            exp_rts.append(min(scaled_rt, 10.0))
                        else:
                            exp_hits.append(False)
                            exp_rts.append(10.0)

                        frac_sats.append(np.mean(fl > upper_ref))

                    rec['expected_hr_correct'] = np.nanmean(exp_hits) if exp_hits else np.nan
                    rec['frac_saturated'] = np.nanmean(frac_sats) if frac_sats else np.nan
                    rec['expected_rt'] = np.nanmean(exp_rts) if exp_rts else np.nan

                    # Actual RT (with 10s for misses)
                    rt_epoch = rt_sess[t0:t1].copy()
                    rt_epoch[~np.isfinite(rt_epoch)] = 10.0
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
                    ref_rt[~np.isfinite(ref_rt)] = 10.0
                    rec['expected_hr_correct'] = np.nanmean(hit_sess[ref_trials])
                    rec['frac_saturated'] = np.nan
                    rt_epoch0 = rt_sess[t0:t1].copy()
                    rt_epoch0[~np.isfinite(rt_epoch0)] = 10.0
                    rec['actual_rt'] = np.nanmean(rt_epoch0)
                    rec['expected_rt'] = np.nanmean(ref_rt)
                    rec['actual_rpe'] = 0.0
                    rec['rt_recovery_ratio'] = np.nan
                    rec['expected_delta_rt'] = np.nan
                    rec['actual_delta_rt'] = np.nan

                all_epoch_stats.append(rec)

            # Store trial-level data for switch-aligned analysis
            # CN activity: trial start to reward (or timeout at 10s)
            dt_si_s = data['dt_si']
            cn_trial_mean = np.full(trl, np.nan)
            for ti_cn in range(trl):
                t_end = rt_sess[ti_cn] if np.isfinite(rt_sess[ti_cn]) else 10.0
                end_frame = min(int(t_end / dt_si_s), F.shape[0])
                if end_frame > 0:
                    cn_trial_mean[ti_cn] = np.nanmean(F[:end_frame, cn, ti_cn])
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
                        transfer_fun(fl_sp, hardest_lower_s, hardest_upper_s))

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
POST = 10   # trials after threshold change

hr_aligned = []   # each entry: array of length PRE+POST
rt_aligned = []
cn_aligned = []
rt_rpe_aligned = []
hit_rpe_aligned = []
exp_hr_aligned = []   # expected step functions per transition
exp_rt_aligned = []
transition_ids = []   # (mouse, session, epoch) for each aligned transition
thr_direction = []    # +1 = increase, -1 = decrease

# Build lookup from (mouse, session, epoch) -> epoch_stat record
epoch_stat_lookup = {}
for rec in all_epoch_stats:
    epoch_stat_lookup[(rec['mouse'], rec['session'], rec['epoch'])] = rec

for (mouse, session), tdata in all_session_trials.items():
    hit = tdata['hit']
    rt = tdata['rt'].copy()
    rt[~np.isfinite(rt)] = 10.0   # fill miss trials with 10s
    cn_act = tdata['cn']
    rt_rpe_act = tdata['rt_rpe']
    hit_rpe_act = tdata['hit_rpe']
    switches = tdata['switches']
    thr_u_sess = tdata['thr_u']
    n_trials = len(hit)

    for si_idx in range(1, len(switches)):   # skip epoch 0 (no switch)
        sw_trial = switches[si_idx]

        # Direction of threshold change
        direction = 1 if thr_u_sess[sw_trial] > thr_u_sess[sw_trial - 1] else -1

        # Define the window
        t0 = sw_trial - PRE
        t1 = sw_trial + POST

        # Check for other switches that would contaminate the window
        other_sw = np.concatenate([switches[:si_idx], switches[si_idx+1:]])

        hr_row = np.full(PRE + POST, np.nan)
        rt_row = np.full(PRE + POST, np.nan)
        cn_row = np.full(PRE + POST, np.nan)
        rt_rpe_row = np.full(PRE + POST, np.nan)
        hit_rpe_row = np.full(PRE + POST, np.nan)

        for k in range(PRE + POST):
            trial_idx = t0 + k
            # Out of bounds → NaN
            if trial_idx < 0 or trial_idx >= n_trials:
                continue
            # Another switch falls on this trial → NaN
            if np.any(other_sw == trial_idx):
                # NaN-pad from here to the edge of the window on that side
                if k < PRE:
                    hr_row[:k+1] = np.nan
                    rt_row[:k+1] = np.nan
                    cn_row[:k+1] = np.nan
                    rt_rpe_row[:k+1] = np.nan
                    hit_rpe_row[:k+1] = np.nan
                else:
                    hr_row[k:] = np.nan
                    rt_row[k:] = np.nan
                    cn_row[k:] = np.nan
                    rt_rpe_row[k:] = np.nan
                    hit_rpe_row[k:] = np.nan
                continue
            hr_row[k] = float(hit[trial_idx])
            rt_row[k] = rt[trial_idx]
            cn_row[k] = cn_act[trial_idx]
            rt_rpe_row[k] = rt_rpe_act[trial_idx]
            hit_rpe_row[k] = hit_rpe_act[trial_idx]

        hr_aligned.append(hr_row)
        rt_aligned.append(rt_row)
        cn_aligned.append(cn_row)
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

thr_direction = np.array(thr_direction)
hr_aligned = np.array(hr_aligned)   # (n_switches, PRE+POST)
rt_aligned = np.array(rt_aligned)
cn_aligned = np.array(cn_aligned)
rt_rpe_aligned = np.array(rt_rpe_aligned)
hit_rpe_aligned = np.array(hit_rpe_aligned)
exp_hr_aligned = np.array(exp_hr_aligned)
exp_rt_aligned = np.array(exp_rt_aligned)

trial_axis = np.arange(-PRE, POST)

# --- Helper to compute mean/sem and plot a direction subset ---
def _mean_sem(arr, axis=0):
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis)
    s = np.nanstd(arr, axis=axis) / np.sqrt(np.clip(n, 1, None))
    return m, s

# Direction masks
mask_inc = thr_direction == 1    # threshold increases
mask_dec = thr_direction == -1   # threshold decreases

for dir_label, dir_mask, dir_suffix in [('Threshold increases', mask_inc, 'inc'),
                                         # ('Threshold decreases', mask_dec, 'dec'),
                                         ]:
    if np.sum(dir_mask) < 2:
        print(f"Skipping {dir_label}: only {np.sum(dir_mask)} transitions")
        continue

    hr_sub = hr_aligned[dir_mask]
    rt_sub = rt_aligned[dir_mask]
    cn_sub = cn_aligned[dir_mask]
    rt_rpe_sub = rt_rpe_aligned[dir_mask]
    exp_hr_sub = exp_hr_aligned[dir_mask]
    exp_rt_sub = exp_rt_aligned[dir_mask]
    n_sub = np.sum(dir_mask)

    hr_m, hr_s = _mean_sem(hr_sub)
    rt_m, rt_s = _mean_sem(rt_sub)
    exp_hr_m = np.nanmean(exp_hr_sub, axis=0)
    exp_rt_m = np.nanmean(exp_rt_sub, axis=0)

    # --- Raw aligned ---
    fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),
                                      gridspec_kw={'wspace': 0.35, 'left': 0.10,
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
    ax2.plot(trial_axis, exp_rt_m, color='cornflowerblue', linewidth=1.2)
    ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Trials from threshold change')
    ax2.set_ylabel('Time to reward (s)')
    ax2.set_xlim(-PRE, POST - 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
    ax2.set_title(f'{dir_label} (n = {n_sub})', fontsize=8)

    fname8 = f'switch_aligned_hr_rt_{dir_suffix}'
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.png'), dpi=300)
    fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.svg'))
    plt.show()
    print(f"Saved {fname8}")

    # --- Delta version: subtract pre-transition mean ---
    hr_delta = hr_sub - np.nanmean(hr_sub[:, :PRE], axis=1, keepdims=True)
    rt_delta = rt_sub - np.nanmean(rt_sub[:, :PRE], axis=1, keepdims=True)
    exp_hr_delta = exp_hr_sub - np.nanmean(exp_hr_sub[:, :PRE], axis=1, keepdims=True)
    exp_rt_delta = exp_rt_sub - np.nanmean(exp_rt_sub[:, :PRE], axis=1, keepdims=True)

    hr_d_m, hr_d_s = _mean_sem(hr_delta)
    rt_d_m, rt_d_s = _mean_sem(rt_delta)
    exp_hr_d_m = np.nanmean(exp_hr_delta, axis=0)
    exp_rt_d_m = np.nanmean(exp_rt_delta, axis=0)

    rt_rpe_m, rt_rpe_s = _mean_sem(rt_rpe_sub)

    fig8b, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),
                                            gridspec_kw={'wspace': 0.35, 'left': 0.07,
                                                         'right': 0.96, 'bottom': 0.18,
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
    ax2.plot(trial_axis, exp_rt_d_m, color='cornflowerblue', linewidth=1.2)
    ax2.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax2.set_xlabel('Trials from threshold change')
    ax2.set_ylabel('\u0394 Time to reward (s)')
    ax2.set_xlim(-PRE, POST - 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(['Actual', 'Expected'], frameon=False, fontsize=6)
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

    fname8b = f'switch_aligned_delta_{dir_suffix}'
    fig8b.savefig(os.path.join(PANEL_DIR, f'{fname8b}.png'), dpi=300)
    fig8b.savefig(os.path.join(PANEL_DIR, f'{fname8b}.svg'))
    plt.show()
    print(f"Saved {fname8b}")

    # --- CN activity aligned (normalized) ---
    # Normalize per transition by pre-switch window
    cn_pre_mean = np.nanmean(cn_sub[:, :PRE], axis=1, keepdims=True)
    cn_pre_std = np.nanstd(cn_sub[:, :PRE], axis=1, keepdims=True)

    # Option 3: divide by pre-mean (baseline = 1.0)
    cn_frac = cn_sub / np.where(np.abs(cn_pre_mean) > 1e-6, cn_pre_mean, np.nan)
    cn_frac_m, cn_frac_s = _mean_sem(cn_frac)

    # Option 1: z-score (baseline mean=0, std=1)
    cn_z = (cn_sub - cn_pre_mean) / np.where(cn_pre_std > 1e-6, cn_pre_std, np.nan)
    cn_z_m, cn_z_s = _mean_sem(cn_z)

    fig8c, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),
                                       gridspec_kw={'wspace': 0.35, 'left': 0.10,
                                                    'right': 0.96, 'bottom': 0.18,
                                                    'top': 0.88})

    ax1.fill_between(trial_axis, cn_frac_m - cn_frac_s, cn_frac_m + cn_frac_s,
                     color='k', alpha=0.15)
    ax1.plot(trial_axis, cn_frac_m, 'k', linewidth=1.2)
    ax1.axvline(0, color='r', linewidth=0.8, linestyle='--')
    ax1.axhline(1, color='gray', linewidth=0.5, linestyle=':')
    ax1.set_xlabel('Trials from threshold change')
    ax1.set_ylabel('CN activity (frac. of pre-switch)')
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

