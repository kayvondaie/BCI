#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

import csv as _csv
import re as _re

# Pick sessions from the inhibitory-photostim summary spreadsheet:
#   Type == 'Inhibitory'
#   Pre-training ∈ {'Good', 'Ok'}  OR  Post-training ∈ {'Good', 'Ok'}
#   Notes do NOT contain "offset = N" with N >= 1   (christina flagged these
#   as sessions that shouldn't have passed QC even if their pre/post quality
#   was Good/Ok)
# Subject and Type columns are sparse (filled only on first row per mouse) so
# forward-fill them. Date is MM/DD/YY → MMDDYY session-folder format.
_QC_CSV = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\Inhibitory & Pan-neuronal BCI summary - Sheet1.csv'
_OK_QUALITY    = {'good', 'ok'}
_BAD_OFFSET_RE = _re.compile(r'offset\s*=\s*[1-9]', _re.IGNORECASE)

_qc_keep = set()
_qc_dropped_offset = []
_current_mouse = None
_current_type  = None
with open(_QC_CSV, encoding='utf-8') as _f:
    _rows = list(_csv.reader(_f))
for _row in _rows[2:]:                                       # skip 2 header rows
    if not _row:
        continue
    _subj = _row[0].strip() if len(_row) > 0 else ''
    _type = _row[1].strip() if len(_row) > 1 else ''
    if _subj:
        _current_mouse = _subj
    if _type:
        _current_type = _type
    if (_current_type or '').lower() != 'inhibitory':
        continue
    _date  = _row[2].strip()        if len(_row) > 2 else ''
    _pre   = _row[4].strip().lower() if len(_row) > 4 else ''
    _post  = _row[5].strip().lower() if len(_row) > 5 else ''
    _notes = _row[6]                 if len(_row) > 6 else ''
    if (not _date or _current_mouse is None or
        (_pre not in _OK_QUALITY and _post not in _OK_QUALITY)):
        continue
    try:
        _m, _d, _y = _date.split('/')
        _session = f'{int(_m):02d}{int(_d):02d}{int(_y):02d}'
    except Exception:
        continue
    if _BAD_OFFSET_RE.search(_notes or ''):
        _qc_dropped_offset.append((_current_mouse, _session, _notes.strip()))
        continue
    _qc_keep.add((_current_mouse, _session))

if _qc_dropped_offset:
    print(f'QC CSV: dropped {len(_qc_dropped_offset)} sessions with non-zero offset notes:')
    for _m, _s, _n in _qc_dropped_offset:
        print(f'   {_m} {_s}  — "{_n}"')

mice = sorted({m for m, _ in _qc_keep})
print(f'QC CSV: kept {len(_qc_keep)} inhibitory sessions across mice {mice}')

list_of_dirs = session_counting.counter22(mice)
RESULTS_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Global plot style ----
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================


# Sliding window parameters
WIN_SIZE = 10    # trials per window
WIN_STEP = 5     # step between windows
tau_elig = 10

# Temporal offset in seconds (pre leads post)
OFFSET_SEC = 0

# Baseline trials for dev2 mode
N_BASELINE = 20

# CC modes:
#   'dot_prod_lag'     — sum_t pre(t) * post(t + lag)
#   'dev2_lag'         — sum_t pre(t) * (post(t + lag) - mean_post_baseline)
#   'target_only'      — sum_t pre(t)  (target activity only, no non-target term)
#   'nontarget_only'   — sum_t (post(t + lag) - mean_post_baseline)  (non-target only)
CC_MODES = ['dot_prod_lag', 'dev2_lag', 'target_only', 'nontarget_only']
CC_MODES = ['target_only']

# If True, regress out the per-group change in target neuron's direct
# photostim response (dTarget = mean(AMP[1][cl] - AMP[0][cl]) per group)
# from dW before fitting the slope on cc_pair. Controls for the confound
# where a target driven harder post inflates dW for its non-target neighbors.
# (Same correction as pan_neuronal_analysis2/sliding_window_temporal_offset_v3_by_group.py)
CONTROL_DTARGET = True

# If True, restrict the non-target pool to cells that are themselves responsive
# targets in some other group — i.e. focus on I→I connections only. The
# headline results suggest these are the cells with the strongest connections,
# so restricting can sharpen the signal.
RESTRICT_TO_RESPONSIVE_TARGETS = False

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s (pre leads post)")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop (mice & _qc_keep set up in cell 1 from the QC CSV)
# ============================================================================
import glob as _glob
import h5py as _h5py
import pickle as _pickle


def _load_stim_params_from_h5(folder, ps_epoch):
    """stim_params is an h5 Group of sub-datasets (incl. scalar `total_duration`)
    that ddct.load_hdf5 can't read. Pull it directly via h5py."""
    all_h5 = sorted(_glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5
                if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        return None
    sp = {}
    with _h5py.File(cand[0], 'r') as f:
        if 'stim_params' in f:
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][()]
                except Exception:
                    pass
    return sp


def _decode_event_times(v, trl):
    """For BCI111/BCI116-era data: ddct.load_hdf5 returns step_time / reward_time
    as a 0-d ndarray of pickled bytes. Decode via pickle and flatten each (1,n)
    per-trial entry to 1-d. Fall back to parse_hdf5_array_string for older sessions."""
    raw = v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            obj = _pickle.loads(raw)
            return [np.asarray(x).ravel() for x in obj]
        except Exception:
            pass
    return parse_hdf5_array_string(v, trl)


for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == False)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) not in _qc_keep:
                print(f"  Skipping {mouse} {session} -- not in QC CSV")
                continue
            folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                      + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time',
                'reward_time', 'BCI_thresholds',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            # Attach stim_params to each photostim epoch (artifact-free amp needs it)
            if 'photostim' in data:
                _sp1 = _load_stim_params_from_h5(folder, 'photostim')
                if _sp1 is not None:
                    data['photostim']['stim_params'] = _sp1
            if 'photostim2' in data:
                _sp2 = _load_stim_params_from_h5(folder, 'photostim2')
                if _sp2 is not None:
                    data['photostim2']['stim_params'] = _sp2

            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']  # (timepoints, neurons, trials)
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            # Temporal offset in frames
            lag_frames = int(round(OFFSET_SEC / dt_si))
            print(f"  dt_si={dt_si:.4f}s, lag={lag_frames} frames ({lag_frames*dt_si:.2f}s)")

            data['step_time']   = _decode_event_times(data['step_time'],   trl)
            data['reward_time'] = _decode_event_times(data['reward_time'], trl)

            # Per-trial behavioral variables
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

            # ---- Build pair selection ----
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.05
            amp1_thr = 0.05

            # If RESTRICT_TO_RESPONSIVE_TARGETS, build the per-cell mask of
            # neurons that pass the same target criteria for some group.
            n_cells_sess = AMP[0].shape[0]
            is_responsive_target = np.zeros(n_cells_sess, dtype=bool)
            if RESTRICT_TO_RESPONSIVE_TARGETS:
                for _gi in range(stimDist.shape[1]):
                    _cl = np.where(
                        (stimDist[:, _gi] < dist_target_lt) &
                        (AMP[0][:, _gi] > amp0_thr) &
                        (AMP[1][:, _gi] > amp1_thr)
                    )[0]
                    is_responsive_target[_cl] = True

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []
            dtarget_list = []

            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < dist_target_lt) &
                    (AMP[0][:, gi] > amp0_thr) &
                    (AMP[1][:, gi] > amp1_thr)
                )[0]
                if cl.size == 0:
                    continue
                nontarg = np.where(
                    (stimDist[:, gi] > dist_nontarg_min) &
                    (stimDist[:, gi] < dist_nontarg_max)
                )[0]
                if RESTRICT_TO_RESPONSIVE_TARGETS:
                    nontarg = nontarg[is_responsive_target[nontarg]]
                if nontarg.size == 0:
                    continue
                dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
                dw_list.append(dw)
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)
                # dTarget: mean change in target neuron's direct response for this group
                dt = np.mean(AMP[1][cl, gi] - AMP[0][cl, gi])
                dtarget_list.append(np.full(len(nontarg), dt))

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            dTarget = np.nan_to_num(np.concatenate(dtarget_list), nan=0.0)
            n_pairs = len(Y_T)

            # cl averaging weights: (n_pairs, n_neurons)
            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # ---- Prepare F for epoch computation ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            # ---- Epoch time indices ----
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            # ---- Compute lagged epoch averages ----
            # Same epoch structure as before, but pre and post use offset time windows.
            # Pre: average F over [epoch_start, epoch_end]
            # Post: average F over [epoch_start + lag, epoch_end + lag]
            # This gives one scalar per neuron per trial per epoch, same as before.

            EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
            n_epochs = len(EPOCH_ORDER)

            # Compute lagged epoch activity: pre neuron uses original window,
            # post neuron uses window shifted by lag_frames
            # For simplicity: average pre over epoch, average post over epoch+lag
            # Shape: (n_neurons, trl) for each

            epoch_pre_act = {}   # presynaptic: original epoch window
            epoch_post_act = {}  # postsynaptic: epoch window + lag

            for ep in ['pre', 'go_cue']:
                if ep == 'pre':
                    t0, t1 = ts_pre[0], ts_pre[-1]
                else:
                    t0, t1 = ts_go[0], ts_go[-1]
                t0_lag = max(0, min(t0 + lag_frames, n_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, n_frames - 1))
                epoch_pre_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)  # (N, trl)
                epoch_post_act[ep] = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

            # Late and reward: per-trial epoch windows
            epoch_pre_act['late'] = np.zeros((n_neurons, trl))
            epoch_post_act['late'] = np.zeros((n_neurons, trl))
            epoch_pre_act['reward'] = np.zeros((n_neurons, trl))
            epoch_post_act['reward'] = np.zeros((n_neurons, trl))

            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    # Late (pre-reward)
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_pre_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        indices_lag = indices + lag_frames
                        indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                        if len(indices_lag) > 0:
                            epoch_post_act['late'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

                    # Reward (post-reward)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_pre_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        indices_lag = indices + lag_frames
                        indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                        if len(indices_lag) > 0:
                            epoch_post_act['reward'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

            # Baseline post mean per epoch (for dev2)
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            baseline_post_mean_ep = {}
            for ep in EPOCH_ORDER:
                baseline_post_mean_ep[ep] = np.nanmean(
                    epoch_post_act[ep][:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # ---- Compute CC per window per epoch (same structure as v2) ----
            raw_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            dev2_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            target_only_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            nontarget_only_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)

            win_hit = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            for wi, ws in enumerate(win_starts):
                we = ws + WIN_SIZE
                trial_idx = np.arange(ws, we)
                win_center[wi] = (ws + we) / 2.0
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])

                for ei, ep in enumerate(EPOCH_ORDER):
                    # Pre activity: original epoch window; Post: shifted by lag
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]   # (n_pairs, win_size)
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]   # (n_pairs, win_size)
                    cc_raw = np.sum(pre_act * post_act, axis=1)              # (n_pairs,)
                    raw_cc[wi, :, ei] = cc_raw

                    # dev2: pre(t) * (post(t+lag) - mean_post_baseline)
                    post_dev = epoch_post_act[ep][all_nt, :][:, trial_idx] - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    cc_dev2 = np.sum(pre_act * post_dev, axis=1)
                    dev2_cc[wi, :, ei] = cc_dev2

                    target_only_cc[wi, :, ei] = np.sum(pre_act, axis=1)
                    nontarget_only_cc[wi, :, ei] = np.sum(post_dev, axis=1)


            # ---- Fit slope/intercept for each mode per epoch ----
            for mode in CC_MODES:
                hi_no_int = np.full((n_wins, n_epochs), np.nan)
                hi_with_int = np.full((n_wins, n_epochs), np.nan)
                hi_intercept = np.full((n_wins, n_epochs), np.nan)
                hi_corr = np.full((n_wins, n_epochs), np.nan)

                for ei, ep in enumerate(EPOCH_ORDER):
                    if mode == 'dot_prod_lag':
                        cc_all = raw_cc[:, :, ei]
                    elif mode == 'dev2_lag':
                        cc_all = dev2_cc[:, :, ei]
                    elif mode == 'target_only':
                        cc_all = target_only_cc[:, :, ei]
                    elif mode == 'nontarget_only':
                        cc_all = nontarget_only_cc[:, :, ei]

                    for wi in range(n_wins):
                        cc_pair = cc_all[wi, :]
                        if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                            continue

                        hi_no_int[wi, ei] = np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair)

                        if CONTROL_DTARGET:
                            A = np.column_stack([np.ones(n_pairs), cc_pair, dTarget])
                        else:
                            A = np.column_stack([np.ones(n_pairs), cc_pair])
                        coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                        hi_intercept[wi, ei] = coeffs[0]
                        hi_with_int[wi, ei] = coeffs[1]

                        hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

                def count_flips(arr):
                    signs = np.sign(arr)
                    valid = np.isfinite(signs)
                    s = signs[valid]
                    if len(s) < 2:
                        return 0
                    return int(np.sum(s[1:] != s[:-1]))

                result = {
                    'mouse': mouse,
                    'session': session,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
                    'n_frames': n_frames,
                    'lag_frames': lag_frames,
                    'lag_sec': lag_frames * dt_si,
                    'dt_si': dt_si,
                    'win_centers': win_center,
                    'hit_rate': np.nanmean(hit),
                    'hi_no_int': hi_no_int,
                    'hi_with_int': hi_with_int,
                    'hi_intercept': hi_intercept,
                    'hi_corr': hi_corr,
                    'win_hit': win_hit,
                    'win_rpe': win_rpe,
                    'win_rt': win_rt,
                    'win_hit_rpe': win_hit_rpe,
                }
                all_results[mode].append(result)

            print(f"  {n_wins} windows, {n_pairs} pairs")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

for mode in CC_MODES:
    print(f"{mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded modes: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 6: Compute within-session correlations
# ============================================================================
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

corr_slope = {}
corr_intercept = {}

for mode in CC_MODES:
    results = all_results[mode]
    n_s = len(results)
    cs = np.full((n_s, n_beh, n_epochs), np.nan)
    ci = np.full((n_s, n_beh, n_epochs), np.nan)

    for si, s in enumerate(results):
        for bi, bname in enumerate(beh_names):
            bvar = get_beh(s, bname)
            if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
                continue
            for ei in range(n_epochs):
                slope = s['hi_with_int'][:, ei]
                intercept = s['hi_intercept'][:, ei]
                ok = np.isfinite(bvar) & np.isfinite(slope)
                if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
                    cs[si, bi, ei], _ = spearmanr(bvar[ok], slope[ok])
                ok2 = np.isfinite(bvar) & np.isfinite(intercept)
                if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
                    ci[si, bi, ei], _ = spearmanr(bvar[ok2], intercept[ok2])

    corr_slope[mode] = cs
    corr_intercept[mode] = ci

print("Within-session correlations computed.")

#%% ============================================================================
# CELL 7: Coefficient matrices — behavior x epoch (same style as v2)
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(5 * len(CC_MODES), 6),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    n_s = len(all_results[mode])

    for row, (corr_arr, row_label) in enumerate([
        (corr_slope[mode], 'Slope'),
        (corr_intercept[mode], 'Intercept'),
    ]):
        ax = axes[row, col]
        mat_mean = np.full((n_beh, n_epochs), np.nan)
        mat_p = np.full((n_beh, n_epochs), np.nan)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                vals = corr_arr[:, bi, ei]
                v = vals[np.isfinite(vals)]
                if len(v) < 3:
                    continue
                mat_mean[bi, ei] = np.mean(v)
                try:
                    _, p = wilcoxon(v)
                except Exception:
                    p = 1.0
                mat_p[bi, ei] = p

        vmax = np.nanmax(np.abs(mat_mean)) if np.any(np.isfinite(mat_mean)) else 0.2
        vmax = max(vmax, 0.05)
        im = ax.imshow(mat_mean, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        # Annotate with values and significance stars
        for bi in range(n_beh):
            for ei in range(n_epochs):
                val = mat_mean[bi, ei]
                p = mat_p[bi, ei]
                if np.isnan(val):
                    continue
                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                txt = f'{val:+.3f}'
                if sig:
                    txt += f'\n{sig}'
                ax.text(ei, bi, txt, ha='center', va='center',
                        fontsize=9, fontweight='bold' if sig else 'normal')

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean rho')

        if row == 0:
            disp = mode.replace('_lag', f' (lag={OFFSET_SEC}s)')
            ax.set_title(f'{disp}\n({row_label})', fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=12)

lag_str = f"{OFFSET_SEC}s"
fig.suptitle(f'Temporal offset: pre leads post by {lag_str} (n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig12_temporal_offset_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 12 saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'temporal_offset_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("TEMPORAL OFFSET COACTIVITY ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s (pre leads post)\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write(f"  dot_prod_lag : sum_t pre(t) * post(t + lag), epoch-averaged\n")
    f.write(f"  dev2_lag     : sum_t pre(t) * (post(t+lag) - mean_post_baseline)\n\n")

    epoch_labels_rpt = ['pre', 'go_cue', 'late', 'reward']

    for mode in CC_MODES:
        n_s = len(all_results[mode])
        f.write(f"\n{'='*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'='*50}\n\n")

        for target, corr_arr, label in [
            ('slope', corr_slope[mode], 'BEHAVIOR vs SLOPE'),
            ('intercept', corr_intercept[mode], 'BEHAVIOR vs INTERCEPT'),
        ]:
            f.write(f"{label}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  {'beh x epoch':25s} {'mean':>7s} {'median':>7s} "
                    f"{'%>0':>5s} {'Wilcoxon p':>10s} {'sig':>4s}\n")

            for bi, bname in enumerate(beh_names):
                for ei, ep in enumerate(epoch_labels_rpt):
                    vals = corr_arr[:, bi, ei]
                    v = vals[np.isfinite(vals)]
                    m = np.mean(v) if len(v) > 0 else np.nan
                    md = np.median(v) if len(v) > 0 else np.nan
                    fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
                    try:
                        _, p = wilcoxon(v)
                    except Exception:
                        p = 1.0
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    row_name = f"{bname}_{ep}"
                    f.write(f"  {row_name:25s} {m:+7.3f} {md:+7.3f} "
                            f"{fpos:4.0f}% {p:10.4f} {sig:>4s}\n")
                f.write("\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 9: Binned scatter — within-session z-scored RPE vs Slope (dev2, pre epoch)
# ============================================================================
mode_plot = 'target_only'
ei_plot = 0  # pre epoch

# For inhibitory data the meaningful behavioral correlate of HI is reward TIME,
# not RPE. Pull win_rt and z-score within session.
all_rt_z = []
all_slope_z = []

for s in all_results[mode_plot]:
    rt = s['win_rpe']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rt) & np.isfinite(slope)
    if np.sum(ok) < 5:
        continue
    rt_ok = rt[ok]
    slope_ok = slope[ok]
    if np.std(rt_ok) == 0 or np.std(slope_ok) == 0:
        continue
    all_rt_z.append((rt_ok - np.mean(rt_ok)) / np.std(rt_ok))
    all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))

all_rt_z = np.concatenate(all_rt_z)
all_slope_z = np.concatenate(all_slope_z)

# Bin by z-scored reward time
n_bins = 3
bin_edges = np.percentile(all_rt_z, np.linspace(0, 100, n_bins + 1))
bin_centers = []
bin_means = []
bin_sems = []

for bi in range(n_bins):
    if bi < n_bins - 1:
        mask = (all_rt_z >= bin_edges[bi]) & (all_rt_z < bin_edges[bi + 1])
    else:
        mask = (all_rt_z >= bin_edges[bi]) & (all_rt_z <= bin_edges[bi + 1])
    if np.sum(mask) < 3:
        continue
    bin_centers.append(np.mean(all_rt_z[mask]))
    bin_means.append(np.mean(all_slope_z[mask]))
    bin_sems.append(np.std(all_slope_z[mask]) / np.sqrt(np.sum(mask)))

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_sems = np.array(bin_sems)

# Find session ranked by reward-time-slope correlation (1 = best, 2 = second best, etc.)
RANK = 21

all_corrs = []
for si, s in enumerate(all_results[mode_plot]):
    rt = s['win_rt']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rt) & np.isfinite(slope)
    if np.sum(ok) >= 5 and np.std(slope[ok]) > 0 and np.std(rt[ok]) > 0:
        r, _ = spearmanr(rt[ok], slope[ok])
        all_corrs.append((r, si))
    else:
        all_corrs.append((np.nan, si))

all_corrs.sort(key=lambda x: -x[0] if np.isfinite(x[0]) else np.inf)
best_corr, best_idx = all_corrs[RANK - 1]
best_s = all_results[mode_plot][best_idx]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: binned scatter
ax = axes[0]
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-',
            color='#2c3e50', capsize=5, linewidth=2, markersize=7)
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.axvline(0, color='k', ls='--', alpha=0.3)
ax.set_xlabel('RPE (within-session z-score)')
ax.set_ylabel('Slope (within-session z-score)')
ax.set_title(f'dev2 lag={OFFSET_SEC}s — Pre epoch\nReward time vs HI slope (n={len(all_results[mode_plot])} sessions)',
             fontsize=13, fontweight='bold')

# Right: time series for best session
ax2 = axes[1]
wc = best_s['win_centers']
rt_ts = best_s['win_rt']
slope_ts = best_s['hi_with_int'][:, ei_plot]

ax2.plot(wc, (rt_ts - np.nanmean(rt_ts)) / np.nanstd(rt_ts),
         'o-', color='#e74c3c', label='Reward time', linewidth=2, markersize=4)
ax2.plot(wc, (slope_ts - np.nanmean(slope_ts)) / np.nanstd(slope_ts),
         'o-', color='#2c3e50', label='HI slope', linewidth=2, markersize=4)
ax2.axhline(0, color='k', ls='-', alpha=0.3)
ax2.set_xlabel('Trial (window center)')
ax2.set_ylabel('z-score')
ax2.legend(loc='best')
ax2.set_title(f'{best_s["mouse"]} {best_s["session"]}\nrho={best_corr:.3f}',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig14_rt_vs_slope_binned.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("Figure 14 saved.")

#%% ============================================================================
# CELL 10: Single-session dW vs CC scatter, split by RPE (re-computes CC)
# ============================================================================
# Uses best_s from Cell 9 (the RANK-th best session)
ex_mouse = best_s['mouse']
ex_session = best_s['session']
ex_lag_sec = best_s['lag_sec']
ei_plot_10 = 0  # pre epoch

print(f"Re-computing CC for {ex_mouse} {ex_session}, lag={ex_lag_sec}s ...")

# --- Reload session data ---
folder = (r'//allen/aind/scratch/BCI/2p-raw/'
          + ex_mouse + r'/' + ex_session + '/pophys/')
photostim_keys = ['stimDist', 'favg_raw']
bci_keys = [
    'df_closedloop', 'F', 'mouse', 'session',
    'conditioned_neuron', 'dt_si', 'step_time',
    'reward_time', 'BCI_thresholds',
]
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)

BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
thr = BCI_thresholds[1, :]
for i in range(1, thr.size):
    if np.isnan(thr[i]):
        thr[i] = thr[i - 1]
if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
    thr[0] = thr[np.isfinite(thr)][0]
BCI_thresholds[1, :] = thr

AMP, stimDist = compute_amp_from_photostim(ex_mouse, data, folder)
dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
n_neurons = F.shape[1]
n_frames = F.shape[0]
tsta = np.arange(0, 12, dt_si)
tsta = tsta - tsta[int(2 / dt_si)]
lag_frames = int(round(ex_lag_sec / dt_si))

data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
hit = np.isfinite(rt)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)

# Pair selection
dw_list = []
pair_cl_list = []
pair_nt_list = []
for gi in range(stimDist.shape[1]):
    cl = np.where(
        (stimDist[:, gi] < 10) &
        (AMP[0][:, gi] > 0.1) &
        (AMP[1][:, gi] > 0.1)
    )[0]
    if cl.size == 0:
        continue
    nontarg = np.where(
        (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000)
    )[0]
    if nontarg.size == 0:
        continue
    dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
    pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
    pair_nt_list.append(nontarg)

Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
all_nt = np.concatenate(pair_nt_list)
n_pairs = len(Y_T)

cl_weights = np.zeros((n_pairs, n_neurons))
offset = 0
for gi_idx in range(len(dw_list)):
    n_nt = len(dw_list[gi_idx])
    cl_arr = pair_cl_list[gi_idx]
    for qi in range(n_nt):
        cl_neurons = cl_arr[qi]
        cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
    offset += n_nt

# Compute lagged epoch activity for pre epoch only
F_nan = F.copy()
F_nan[np.isnan(F_nan)] = 0
ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
t0e, t1e = ts_pre[0], ts_pre[-1]
t0_lag = max(0, min(t0e + lag_frames, n_frames - 1))
t1_lag = max(0, min(t1e + lag_frames, n_frames - 1))
epoch_pre = np.nanmean(F_nan[t0e:t1e+1, :, :], axis=0)   # (N, trl)
epoch_post = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

# Baseline for dev2
baseline_trials_arr = np.arange(min(N_BASELINE, trl))
bl_post_mean = np.nanmean(epoch_post[:, baseline_trials_arr], axis=1)  # (N,)

# Compute CC per window
win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
n_wins = len(win_starts)

cc_per_win = np.full((n_wins, n_pairs), np.nan)
rpe_per_win = np.full(n_wins, np.nan)

for wi, ws in enumerate(win_starts):
    trial_idx = np.arange(ws, ws + WIN_SIZE)
    rpe_per_win[wi] = np.nanmean(rt_rpe[trial_idx])
    pre_act = cl_weights @ epoch_pre[:, trial_idx]
    post_dev = epoch_post[all_nt, :][:, trial_idx] - bl_post_mean[all_nt, np.newaxis]
    cc_per_win[wi, :] = np.sum(pre_act * post_dev, axis=1)

# Split windows by RPE sign (within-session, relative to median)
med_rpe = np.nanmedian(rpe_per_win)
hi_rpe = rpe_per_win >= np.percentile(rpe_per_win,90)
lo_rpe = rpe_per_win < np.percentile(rpe_per_win,10)

# Average CC across windows in each group
cc_hi = np.nanmean(cc_per_win[hi_rpe, :], axis=0)
cc_lo = np.nanmean(cc_per_win[lo_rpe, :], axis=0)

n_bins_10 = 5

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

for ax, cc, label, color in [
    (axes[0], cc_hi, f'RPE > median ({np.sum(hi_rpe)} wins)', '#e74c3c'),
    (axes[1], cc_lo, f'RPE < median ({np.sum(lo_rpe)} wins)', '#3498db'),
]:
    ok = np.isfinite(cc) & np.isfinite(Y_T)
    cc_ok = cc[ok]
    dw_ok = Y_T[ok]

    if len(cc_ok) < n_bins_10:
        continue

    edges = np.percentile(cc_ok, np.linspace(0, 100, n_bins_10 + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins_10):
        if bi < n_bins_10 - 1:
            mask = (cc_ok >= edges[bi]) & (cc_ok < edges[bi + 1])
        else:
            mask = (cc_ok >= edges[bi]) & (cc_ok <= edges[bi + 1])
        if np.sum(mask) < 3:
            continue
        bx.append(np.mean(cc_ok[mask]))
        by.append(np.mean(dw_ok[mask]))
        be.append(np.std(dw_ok[mask]) / np.sqrt(np.sum(mask)))

    bx, by, be = np.array(bx), np.array(by), np.array(be)
    ax.errorbar(bx, by, yerr=be, fmt='o-', color=color,
                capsize=5, linewidth=2, markersize=7)

    # Fit line on raw data for stats
    if np.std(cc_ok) > 0:
        A = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A, dw_ok, rcond=None)[0]
        xr = np.array([bx[0], bx[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, '--', color='k', linewidth=1.5)
        r, p = spearmanr(cc_ok, dw_ok)
        ax.set_title(f'{label}\nslope={coeffs[1]:.4f}, r={r:.3f}, p={p:.3f}',
                     fontsize=12, fontweight='bold')

    ax.axhline(0, color='k', ls='-', alpha=0.2)
    ax.axvline(0, color='k', ls='--', alpha=0.2)
    ax.set_xlabel('CC (dev2, pre epoch)')
    ax.set_ylabel('dW')

fig.suptitle(f'{ex_mouse} {ex_session} — dW vs CC split by RPE\n(dev2, pre epoch, lag={ex_lag_sec}s)',
             fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig15_dw_vs_cc_rpe_split.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 15 saved.")

#%% ============================================================================
# CELL 16: Example traces — non-target responses to high-pretrial-activity targets
# in pre vs post photostim epochs.
#
# Interpretation of the headline result: target neurons with high pre-trial
# activity (and large RPE windows) increase their inhibition of downstream
# non-targets, so we expect more negative dW for those pairs. Visualize this
# directly by re-loading one session, scoring pairs by
#       pretrial_target × (−dW),
# and plotting the top-N non-target photostim responses (favg_raw) overlaid
# pre vs post. Style mirrors inhibitory_artifact_removal.py: smoothed mean
# trace per epoch, stim window shaded, one panel per pair.
# ============================================================================
from scipy.ndimage import median_filter as _medfilt

N_TARGETS       = 5                       # how many targets to average over
PRE_TRIAL_WIN_S = (-2.0, -1.0)            # paper's pretrial epoch
SMOOTH_K        = 1                      # median-filter kernel (odd)
EX_MODE         = 'target_only'           # the mode where the headline negative
                                          #   RPE→slope effect was strongest
EX_EI           = 0                        # epoch index (0 = pre)

# Pick the session with the most NEGATIVE Spearman correlation between
# win_rpe and the per-window HI slope — the session where the headline
# "high pretrial → more inhibition" relationship is strongest.
_ex_results = all_results.get(EX_MODE, [])
_ex_corrs = []
for _si, _s in enumerate(_ex_results):
    _rpe = _s.get('win_rpe', None)
    if _rpe is None:
        continue
    _slope = _s['hi_with_int'][:, EX_EI]
    _ok = np.isfinite(_rpe) & np.isfinite(_slope)
    if _ok.sum() < 5 or np.std(_rpe[_ok]) == 0 or np.std(_slope[_ok]) == 0:
        continue
    _r, _ = spearmanr(_rpe[_ok], _slope[_ok])
    _ex_corrs.append((_r, _si))

if not _ex_corrs:
    raise RuntimeError(
        f'Could not find any session with valid RPE/slope in all_results[{EX_MODE!r}] '
        '— run cell 3 first.'
    )
_ex_corrs.sort(key=lambda x: x[0])               # ascending → most negative first
_best_r, _best_si = _ex_corrs[3]
_best_s = _ex_results[_best_si]
EX_MOUSE   = _best_s['mouse']
EX_SESSION = _best_s['session']
print(f'Most-negative RPE→slope session ({EX_MODE}, epoch {EX_EI}): '
      f'{EX_MOUSE} {EX_SESSION}, ρ = {_best_r:+.3f}')

ex_folder = (r'//allen/aind/scratch/BCI/2p-raw/' + EX_MOUSE + '/' + EX_SESSION + '/pophys/')
print(f'\n--- Example traces for {EX_MOUSE} {EX_SESSION} ---')

ex_data = ddct.load_hdf5(ex_folder, bci_keys, photostim_keys)
for _ep in ('photostim', 'photostim2'):
    if _ep in ex_data:
        _sp = _load_stim_params_from_h5(ex_folder, _ep)
        if _sp is not None:
            ex_data[_ep]['stim_params'] = _sp

ex_F   = ex_data['F']
ex_dt  = ex_data['dt_si']
ex_trl = ex_F.shape[2]
ex_data['step_time']   = _decode_event_times(ex_data['step_time'],   ex_trl)
ex_data['reward_time'] = _decode_event_times(ex_data['reward_time'], ex_trl)

ex_AMP, ex_stimDist = compute_amp_from_photostim_artifact_free(
    EX_MOUSE, ex_data, ex_folder)

# Pre-trial activity per cell (mean dF/F in [-2, -1] s, averaged across trials)
ex_tsta = np.arange(0, 12, ex_dt)
ex_tsta = ex_tsta - ex_tsta[int(2 / ex_dt)]
ex_tsta = ex_tsta[: ex_F.shape[0]]
_pre_mask = (ex_tsta >= PRE_TRIAL_WIN_S[0]) & (ex_tsta <= PRE_TRIAL_WIN_S[1])
ex_pretrial = np.nanmean(np.nanmean(ex_F[_pre_mask, :, :], axis=0), axis=-1)

# favg_raw and stim_params time axis for both photostim epochs
favg_pre  = np.array(ex_data['photostim']['favg_raw'],  dtype=float)
favg_post = np.array(ex_data['photostim2']['favg_raw'], dtype=float)
sp_post = ex_data.get('photostim2', {}).get('stim_params', {}) or {}
if 'time' in sp_post:
    t_ps = np.asarray(sp_post['time']).ravel()
    h5_total = float(np.asarray(sp_post['total_duration']).ravel()[0])
    stim_start = float(t_ps[np.where(np.isclose(t_ps, 0.0))[0][0]])
    end_hits = np.where(t_ps >= h5_total)[0]
    stim_end = float(t_ps[end_hits[0]]) if len(end_hits) > 0 else stim_start + 0.5
else:
    t_ps = np.arange(favg_post.shape[0]) * ex_dt
    stim_start, stim_end = 0.0, 0.5
pre_axis_mask = t_ps < 0          # pre-stim baseline window in stim time

# Per-cell baseline averaged across stim groups — matches the denominator
# that compute_amp_from_photostim_artifact_free uses, so the displayed
# trace's post-stim amplitude corresponds to the AMP value.
def _F0_per_cell(favg, mask):
    bl_per_g = np.nanmean(favg[mask, :, :], axis=0)        # (n_cells, n_groups)
    return np.nanmean(bl_per_g, axis=-1)                   # (n_cells,)

F0_pre_pc  = _F0_per_cell(favg_pre,  pre_axis_mask)
F0_post_pc = _F0_per_cell(favg_post, pre_axis_mask)

# Score per TARGET group: mean dW across all its non-targets (>30 µm).
# We then include EVERY non-target pair from the top-N_TARGETS targets in
# the average — the postsynaptic identity isn't what matters here, the
# target identity is.
target_scores = []
for gi in range(ex_stimDist.shape[1]):
    cl = np.where(
        (ex_stimDist[:, gi] < 10) &
        (ex_AMP[0][:, gi] > TARGET_AMP_THR) &
        (ex_AMP[1][:, gi] > TARGET_AMP_THR)
    )[0]
    if cl.size == 0:
        continue
    pretrial_target = float(np.nanmean(ex_pretrial[cl]))
    if not np.isfinite(pretrial_target):
        continue
    nt = np.where(
        (ex_stimDist[:, gi] > NONTARG_DIST_LO) &
        (ex_stimDist[:, gi] < NONTARG_DIST_HI)
    )[0]
    if nt.size == 0:
        continue
    dw_per_nt = ex_AMP[1][nt, gi] - ex_AMP[0][nt, gi]
    valid = np.isfinite(dw_per_nt)
    if valid.sum() < 5:
        continue
    target_scores.append((float(np.mean(dw_per_nt[valid])),    # mean dW (signed)
                          gi, int(cl[0]), nt[valid], pretrial_target))

target_scores.sort(key=lambda x: x[0])                          # most-negative first
top_targets = target_scores[:N_TARGETS]
print(f'Top {N_TARGETS} targets by most-negative mean dW (over their non-targets):')
for mdw, gi, ti, nt_arr, pt in top_targets:
    print(f'  group={gi} target={ti}  n_nt={nt_arr.size}  '
          f'mean dW={mdw:+.3f}  pretrial(target)={pt:+.3f}')


def _bs_dff(trace, F0):
    """(F − per-trace baseline) / F0_per_cell. NO per-trace median filter —
    smoothing is applied once to the averaged trace at the end so NaN
    propagation and edge effects in median_filter can't leak into the
    leftmost frame."""
    bl = np.nanmean(trace[pre_axis_mask]) if pre_axis_mask.any() else 0.0
    if not np.isfinite(F0) or F0 <= 0:
        F0 = 1.0
    return (trace - bl) / F0


# Average across ALL non-target pairs from the top-N_TARGETS targets.
pre_stack  = []
post_stack = []
for mdw, gi, ti, nt_arr, pt in top_targets:
    for nti in nt_arr:
        pre_stack.append( _bs_dff(favg_pre[:,  nti, gi], F0_pre_pc[nti]))
        post_stack.append(_bs_dff(favg_post[:, nti, gi], F0_post_pc[nti]))
pre_stack  = np.vstack(pre_stack)        # (n_pairs_total, n_frames)
post_stack = np.vstack(post_stack)
print(f'Aggregated {pre_stack.shape[0]} (target, non-target) pairs from '
      f'{N_TARGETS} targets')

pre_mu  = np.nanmean(pre_stack,  axis=0)
post_mu = np.nanmean(post_stack, axis=0)
pre_se  = np.nanstd(pre_stack,  axis=0, ddof=1) / np.sqrt(pre_stack.shape[0])
post_se = np.nanstd(post_stack, axis=0, ddof=1) / np.sqrt(post_stack.shape[0])

# Single smoothing pass on the averaged traces (after stacking)
pre_mu  = _medfilt(pre_mu,  size=SMOOTH_K, mode='reflect')
post_mu = _medfilt(post_mu, size=SMOOTH_K, mode='reflect')
pre_se  = _medfilt(pre_se,  size=SMOOTH_K, mode='reflect')
post_se = _medfilt(post_se, size=SMOOTH_K, mode='reflect')

# Anchor the averaged trace so the pre-stim window has mean zero
# (cosmetic only — does not change the dW estimate inside the stim window).
if pre_axis_mask.any():
    pre_mu  -= np.nanmean(pre_mu[pre_axis_mask])
    post_mu -= np.nanmean(post_mu[pre_axis_mask])

# Sanity: integrate the displayed trace within the stim window and compare
# to the AMP values stored in ex_AMP. They should match.
_disp_mask = (t_ps >= stim_start) & (t_ps <= stim_end)
_disp_pre_amp  = float(np.nanmean(pre_mu[_disp_mask]))
_disp_post_amp = float(np.nanmean(post_mu[_disp_mask]))
_amp_pre_list  = []
_amp_post_list = []
for _mdw, _gi, _ti, _nt_arr, _pt in top_targets:
    for _nti in _nt_arr:
        _amp_pre_list.append(ex_AMP[0][_nti, _gi])
        _amp_post_list.append(ex_AMP[1][_nti, _gi])
_amp_pre_mean  = float(np.nanmean(_amp_pre_list))
_amp_post_mean = float(np.nanmean(_amp_post_list))
print(f'Trace integral in stim window: pre={_disp_pre_amp:+.3f}  post={_disp_post_amp:+.3f}'
      f'   |   AMP-mean: pre={_amp_pre_mean:+.3f}  post={_amp_post_mean:+.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ---- Left: averaged pre vs post photostim trace across the picked pairs ----
ax = axes[0]
ax.plot(t_ps, pre_mu,  color='#0096FF', linewidth=2, label=f'pre  (n={pre_stack.shape[0]})')
ax.fill_between(t_ps, pre_mu - pre_se, pre_mu + pre_se,
                color='#0096FF', alpha=0.25, edgecolor='none')
ax.plot(t_ps, post_mu, color='#FFA500', linewidth=2, label=f'post (n={post_stack.shape[0]})')
ax.fill_between(t_ps, post_mu - post_se, post_mu + post_se,
                color='#FFA500', alpha=0.25, edgecolor='none')
ax.axvspan(stim_start, stim_end, color='#FFBF00', alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Time from stim onset (s)', fontsize=12)
ax.set_ylabel('ΔF/F (mean ± SEM)', fontsize=12)
ax.set_title(
    f'{EX_MOUSE} {EX_SESSION}: avg across {pre_stack.shape[0]} non-target pairs\n'
    f'from top {N_TARGETS} targets (scored by mean dW)',
    fontsize=11,
)
ax.legend(fontsize=10, frameon=False, loc='best')
ax.set_xlim(t_ps[0], 1.0)
_disp_x_mask = (t_ps >= t_ps[0]) & (t_ps <= 2.0)
_y_lo = float(min(np.nanmin(pre_mu[_disp_x_mask]  - pre_se[_disp_x_mask]),
                  np.nanmin(post_mu[_disp_x_mask] - post_se[_disp_x_mask])))
_y_hi = float(max(np.nanmax(pre_mu[_disp_x_mask]  + pre_se[_disp_x_mask]),
                  np.nanmax(post_mu[_disp_x_mask] + post_se[_disp_x_mask])))
_y_pad = (_y_hi - _y_lo) * 0.1
ax.set_ylim(_y_lo - _y_pad, _y_hi + _y_pad)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---- Right: HI slope vs RPE for the selected session, time series style ----
ax2 = axes[1]
_ex_rpe   = _best_s.get('win_rpe', None)
_ex_slope = _best_s['hi_with_int'][:, EX_EI]
_ex_wc    = _best_s['win_centers']
if _ex_rpe is None:
    ax2.text(0.5, 0.5, 'win_rpe not available', ha='center', va='center',
             transform=ax2.transAxes)
else:
    _ok = np.isfinite(_ex_rpe) & np.isfinite(_ex_slope)
    if _ok.sum() >= 5:
        _r, _ = spearmanr(_ex_rpe[_ok], _ex_slope[_ok])
        _rpe_z   = (_ex_rpe   - np.nanmean(_ex_rpe))   / np.nanstd(_ex_rpe)
        _slope_z = (_ex_slope - np.nanmean(_ex_slope)) / np.nanstd(_ex_slope)
        ax2.plot(_ex_wc, _rpe_z,   'o-', color='#e74c3c',
                 label='RPE',     linewidth=2, markersize=4)
        ax2.plot(_ex_wc, _slope_z, 'o-', color='#2c3e50',
                 label='HI slope', linewidth=2, markersize=4)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.set_xlabel('Trial (window center)', fontsize=12)
        ax2.set_ylabel('z-score', fontsize=12)
        ax2.set_title(f'{EX_MODE}: RPE vs HI slope\nρ = {_r:+.3f}', fontsize=11)
        ax2.legend(fontsize=10, frameon=False, loc='best')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'not enough valid windows', ha='center', va='center',
                 transform=ax2.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'example_traces_pretrial_inhibition.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print('Example averaged trace saved.')
