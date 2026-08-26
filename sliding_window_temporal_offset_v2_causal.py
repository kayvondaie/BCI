#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Causal-baseline variant of sliding_window_temporal_offset_v2.py.

Difference from v2:
  v2 computes baseline_post_mean_fulltrial ONCE per session from trials
  [0, N_BASELINE), then subtracts the same per-neuron scalar at every
  sliding window.  For early windows this baseline overlaps with (or
  entirely covers) the trials inside the window — i.e., the baseline
  "leaks" into the dev2 quantity.

  This script replaces that fixed baseline with a rolling causal baseline:
  for a window starting at trial ws, the baseline is the per-neuron mean
  post activity over trials [max(0, ws - N_BASELINE), ws) — the up-to-N
  trials immediately preceding the window.  Default N_BASELINE = 10.

Windows with ws < 1 have no past trials and are skipped (CC = NaN).
For ws < N_BASELINE the baseline is computed over fewer than N_BASELINE
past trials.

CC modes (same names as v2 for downstream compatibility, but with the
rolling baseline now applied):
  dev2_fulltrial_baseline — epoch-resolved CC, rolling causal baseline
  full_trial              — CC over entire trial, rolling causal baseline
  full_trial_dot_prod     — CC over entire trial, raw dot product (no baseline)
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_1samp
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

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
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

# Sliding window parameters
WIN_SIZE = 10    # trials per window
WIN_STEP = 5     # step between windows
tau_elig = 15

# Temporal offset in seconds (pre leads post)
OFFSET_SEC = 0

# Baseline strategy.  Three options:
#   'rolling'           — causal rolling window: trials [max(0, ws-N), ws)
#                         (default; "previous N trials" for window at ws).
#   'fixed_v2'          — non-causal v2 baseline: trials [0, min(N, trl))
#                         computed once per session.  Includes "future"
#                         trials for early windows.  Reverts behavior to v2.
#   'expanding_causal'  — causal expanding window: trials [0, min(ws, N))
#                         At ws<N, baseline grows from 1..ws-1 past trials.
#                         At ws>=N, baseline is fixed at first N trials
#                         (same trials as fixed_v2, but only used once the
#                         window has moved past them, i.e. fully causal).
BASELINE_MODE = 'rolling'
N_BASELINE = 20

# Optionally restrict analysis to the first TRIAL_CAP trials of each session.
# Only windows whose end trial (ws + WIN_SIZE) lies within [0, TRIAL_CAP] are
# generated.  Baseline trials are NOT capped — they still come from whichever
# range BASELINE_MODE specifies (e.g. fixed_v2 still uses trials [0, N)).
# Set to None to use all trials.
TRIAL_CAP = None

# Statistical test used to determine if per-session rhos are different from 0
# across sessions.  One of:
#   'wilcoxon' — Wilcoxon signed-rank (non-parametric, default)
#   't_test'   — one-sample t-test against 0 (parametric, uses Fisher z-transform
#                of the per-session rhos before testing for stability).
TEST_TYPE = 'wilcoxon'


def _across_session_pval(v):
    """Return the two-sided p-value for "median(v) > 0 or < 0" across sessions,
    using the test type configured at the top of Cell 2."""
    if len(v) < 3:
        return 1.0
    if TEST_TYPE == 'wilcoxon':
        try:
            _, p = wilcoxon(v)
            return float(p)
        except Exception:
            return 1.0
    if TEST_TYPE == 't_test':
        # Fisher z-transform stabilizes variance of bounded rhos.  Clamp to
        # avoid arctanh(+/-1) blowing up to +/-inf.
        v_clipped = np.clip(v, -0.999999, 0.999999)
        z = np.arctanh(v_clipped)
        try:
            _, p = ttest_1samp(z, 0.0)
            return float(p)
        except Exception:
            return 1.0
    raise ValueError(f"Unknown TEST_TYPE: {TEST_TYPE}")

# CC modes.  Each mode controls (a) whether the dev2 baseline is computed
# from the post neuron's full-trial activity vs the epoch-specific activity,
# and (b) whether the CC itself is computed per epoch or over the full trial.
# All baseline-using modes draw their baseline trials according to
# BASELINE_MODE above.
#
#   'dev2_fulltrial_baseline' — epoch-resolved CC; baseline = mean over
#                               baseline trials of the post neuron's
#                               *full-trial* mean activity (one scalar per
#                               neuron, used for all epochs).
#   'dev2_epoch_baseline'     — epoch-resolved CC; baseline = mean over
#                               baseline trials of the post neuron's
#                               activity *within the same epoch* (one
#                               scalar per neuron per epoch).
#   'full_trial'              — single CC over the entire trial (no epoch
#                               splitting); baseline = full-trial mean.
#   'full_trial_dot_prod'     — single CC over the entire trial, raw dot
#                               product, no baseline subtraction.
CC_MODES = ['dev2_epoch_baseline',
            'full_trial']

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s (pre leads post)")
print(f"Baseline mode: {BASELINE_MODE} (N={N_BASELINE})")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
import csv
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")
else:
    print("WARNING: qc_summary.csv not found, no sessions excluded")

for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
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

            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
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

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

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
            amp0_thr = 0.1
            amp1_thr = 0.1

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []

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
                if nontarg.size == 0:
                    continue
                dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
                dw_list.append(dw)
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
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
            # If TRIAL_CAP is set, only generate windows that end at or before
            # trial TRIAL_CAP.  Baseline trials are still drawn per BASELINE_MODE
            # from the full session.
            trl_eff = min(TRIAL_CAP, trl) if TRIAL_CAP is not None else trl
            win_starts = np.arange(0, trl_eff - WIN_SIZE + 1, WIN_STEP)
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
            EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
            n_epochs = len(EPOCH_ORDER)

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

            # ==================================================================
            # Full-trial activity (no epochs)
            # Pre neuron: mean over all frames; Post neuron: mean over all frames + lag
            # Shape: (n_neurons, trl)
            # ==================================================================
            fulltrial_pre_act = np.nanmean(F_nan, axis=0)  # (n_neurons, trl)
            if lag_frames == 0:
                fulltrial_post_act = fulltrial_pre_act.copy()
            else:
                t0_ft = max(0, lag_frames)
                t1_ft = min(n_frames, n_frames + lag_frames)
                fulltrial_post_act = np.nanmean(
                    F_nan[t0_ft:t1_ft, :, :], axis=0)  # (n_neurons, trl)

            # Precompute fixed baselines only if BASELINE_MODE == 'fixed_v2'.
            # We compute BOTH the full-trial baseline (for
            # dev2_fulltrial_baseline + full_trial) and per-epoch baselines
            # (for dev2_epoch_baseline).
            if BASELINE_MODE == 'fixed_v2':
                _fixed_bl_trials = np.arange(min(N_BASELINE, trl))
                fixed_bl_post = np.nanmean(
                    fulltrial_post_act[:, _fixed_bl_trials], axis=1)
                fixed_bl_post_ep = {
                    ep: np.nanmean(
                        epoch_post_act[ep][:, _fixed_bl_trials], axis=1)
                    for ep in EPOCH_ORDER
                }
            else:
                fixed_bl_post = None
                fixed_bl_post_ep = None

            # ---- Compute CC per window ----
            dev2_ftb_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            dev2_epoch_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            fulltrial_cc = np.full((n_wins, n_pairs, 1), np.nan)
            fulltrial_dp_cc = np.full((n_wins, n_pairs, 1), np.nan)

            # Per-pair CC arrays for pre epoch only (for flip decomposition).
            raw_cc_pre = np.full((n_wins, n_pairs), np.nan)
            dev2_cc_pre = np.full((n_wins, n_pairs), np.nan)

            # Per-window drift diagnostics (pre epoch nontarget deviation from
            # the rolling baseline at that window).
            win_mean_dev = np.full(n_wins, np.nan)
            win_abs_dev = np.full(n_wins, np.nan)

            # How many past trials were available at each window (for diagnostics).
            win_baseline_n = np.zeros(n_wins, dtype=int)

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

                # --- Per-window baseline (selected by BASELINE_MODE) ---
                # bl_post: full-trial baseline (n_neurons,)
                # bl_post_ep: per-epoch baseline {ep: (n_neurons,)}
                if BASELINE_MODE == 'rolling':
                    bl_start = max(0, ws - N_BASELINE)
                    bl_end = ws        # exclusive; window's own trials NOT in baseline
                    n_past = bl_end - bl_start
                    win_baseline_n[wi] = n_past
                    if n_past < 1:
                        continue       # no past trials -> CC stays NaN
                    bl_post = np.nanmean(
                        fulltrial_post_act[:, bl_start:bl_end], axis=1)
                    bl_post_ep = {ep: np.nanmean(
                        epoch_post_act[ep][:, bl_start:bl_end], axis=1)
                        for ep in EPOCH_ORDER}

                elif BASELINE_MODE == 'expanding_causal':
                    bl_start = 0
                    bl_end = min(ws, N_BASELINE)  # exclusive; cap at first N
                    n_past = bl_end - bl_start
                    win_baseline_n[wi] = n_past
                    if n_past < 1:
                        continue
                    bl_post = np.nanmean(
                        fulltrial_post_act[:, bl_start:bl_end], axis=1)
                    bl_post_ep = {ep: np.nanmean(
                        epoch_post_act[ep][:, bl_start:bl_end], axis=1)
                        for ep in EPOCH_ORDER}

                elif BASELINE_MODE == 'fixed_v2':
                    bl_post = fixed_bl_post
                    bl_post_ep = fixed_bl_post_ep
                    win_baseline_n[wi] = min(N_BASELINE, trl)

                else:
                    raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")

                # --- deviation of nontarget post neurons from baseline (pre epoch) ---
                post_pre_ep = epoch_post_act['pre'][all_nt, :][:, trial_idx]
                dev_from_bl = post_pre_ep - bl_post[all_nt, np.newaxis]
                win_mean_dev[wi] = np.mean(dev_from_bl)
                win_abs_dev[wi] = np.mean(np.abs(dev_from_bl))

                # --- per-pair CC for pre epoch (raw and dev2, for flip analysis) ---
                pre_act_pre = cl_weights @ epoch_pre_act['pre'][:, trial_idx]
                post_act_pre = epoch_post_act['pre'][all_nt, :][:, trial_idx]
                raw_cc_pre[wi, :] = np.sum(pre_act_pre * post_act_pre, axis=1)
                post_dev_pre = post_act_pre - bl_post[all_nt, np.newaxis]
                dev2_cc_pre[wi, :] = np.sum(pre_act_pre * post_dev_pre, axis=1)

                # --- epoch-resolved dev2 CCs (full-trial AND epoch-specific baselines) ---
                for ei, ep in enumerate(EPOCH_ORDER):
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]
                    post_window = epoch_post_act[ep][all_nt, :][:, trial_idx]
                    # full-trial baseline
                    post_dev_ft = post_window - bl_post[all_nt, np.newaxis]
                    dev2_ftb_cc[wi, :, ei] = np.sum(pre_act * post_dev_ft, axis=1)
                    # epoch-specific baseline
                    post_dev_ep = post_window - bl_post_ep[ep][all_nt, np.newaxis]
                    dev2_epoch_cc[wi, :, ei] = np.sum(pre_act * post_dev_ep, axis=1)

                # --- full_trial: no epochs, entire trial (dev2-style) ---
                pre_act_ft = cl_weights @ fulltrial_pre_act[:, trial_idx]
                post_dev_ft = (fulltrial_post_act[all_nt, :][:, trial_idx]
                               - bl_post[all_nt, np.newaxis])
                cc_ft = np.sum(pre_act_ft * post_dev_ft, axis=1)
                fulltrial_cc[wi, :, 0] = cc_ft

                # --- full_trial_dot_prod: no epochs, raw dot product (no baseline) ---
                post_act_ft = fulltrial_post_act[all_nt, :][:, trial_idx]
                cc_ft_dp = np.sum(pre_act_ft * post_act_ft, axis=1)
                fulltrial_dp_cc[wi, :, 0] = cc_ft_dp

            # ---- Fit slope/intercept for each mode ----
            for mode in CC_MODES:
                if mode == 'dev2_fulltrial_baseline':
                    n_ep_mode = n_epochs
                    cc_data = dev2_ftb_cc
                elif mode == 'dev2_epoch_baseline':
                    n_ep_mode = n_epochs
                    cc_data = dev2_epoch_cc
                elif mode == 'full_trial':
                    n_ep_mode = 1
                    cc_data = fulltrial_cc
                elif mode == 'full_trial_dot_prod':
                    n_ep_mode = 1
                    cc_data = fulltrial_dp_cc

                hi_no_int = np.full((n_wins, n_ep_mode), np.nan)
                hi_with_int = np.full((n_wins, n_ep_mode), np.nan)
                hi_intercept = np.full((n_wins, n_ep_mode), np.nan)
                hi_corr = np.full((n_wins, n_ep_mode), np.nan)

                for ei in range(n_ep_mode):
                    cc_all = cc_data[:, :, ei]

                    for wi in range(n_wins):
                        cc_pair = cc_all[wi, :]
                        if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                            continue

                        hi_no_int[wi, ei] = np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair)

                        A = np.column_stack([np.ones(n_pairs), cc_pair])
                        coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                        hi_intercept[wi, ei] = coeffs[0]
                        hi_with_int[wi, ei] = coeffs[1]

                        hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

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
                    'n_baseline': N_BASELINE,
                    'baseline_mode': BASELINE_MODE,
                    'win_centers': win_center,
                    'win_baseline_n': win_baseline_n,
                    'hit_rate': np.nanmean(hit),
                    'hi_no_int': hi_no_int,
                    'hi_with_int': hi_with_int,
                    'hi_intercept': hi_intercept,
                    'hi_corr': hi_corr,
                    'win_hit': win_hit,
                    'win_rpe': win_rpe,
                    'win_rt': win_rt,
                    'win_hit_rpe': win_hit_rpe,
                    'win_mean_dev': win_mean_dev,
                    'win_abs_dev': win_abs_dev,
                    'sess_mean_dev': np.nanmean(win_mean_dev),
                    'sess_abs_dev': np.nanmean(win_abs_dev),
                    'raw_cc_pre': raw_cc_pre,
                    'dev2_cc_pre': dev2_cc_pre,
                    'Y_T': Y_T,
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
_cap_tag = f"_cap{TRIAL_CAP}" if TRIAL_CAP is not None else ""
np.save(os.path.join(RESULTS_DIR, f'sliding_window_temporal_offset_v2_causal_{BASELINE_MODE}_N{N_BASELINE}{_cap_tag}.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
_cap_tag = f"_cap{TRIAL_CAP}" if TRIAL_CAP is not None else ""
all_results = np.load(
    os.path.join(RESULTS_DIR, f'sliding_window_temporal_offset_v2_causal_{BASELINE_MODE}_N{N_BASELINE}{_cap_tag}.npy'),
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

    n_ep_mode = results[0]['hi_with_int'].shape[1] if n_s > 0 else n_epochs

    cs = np.full((n_s, n_beh, n_ep_mode), np.nan)
    ci = np.full((n_s, n_beh, n_ep_mode), np.nan)

    for si, s in enumerate(results):
        for bi, bname in enumerate(beh_names):
            bvar = get_beh(s, bname)
            if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
                continue
            for ei in range(n_ep_mode):
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
# CELL 7: Coefficient matrices — behavior x epoch
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(5 * len(CC_MODES), 6),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    n_s = len(all_results[mode])
    n_ep_mode = corr_slope[mode].shape[2]

    if mode in ('full_trial', 'full_trial_dot_prod'):
        ep_labels_mode = ['Full trial']
    else:
        ep_labels_mode = epoch_labels

    for row, (corr_arr, row_label) in enumerate([
        (corr_slope[mode], 'Slope'),
        (corr_intercept[mode], 'Intercept'),
    ]):
        ax = axes[row, col]
        mat_mean = np.full((n_beh, n_ep_mode), np.nan)
        mat_p = np.full((n_beh, n_ep_mode), np.nan)

        for bi in range(n_beh):
            for ei in range(n_ep_mode):
                vals = corr_arr[:, bi, ei]
                v = vals[np.isfinite(vals)]
                if len(v) < 3:
                    continue
                mat_mean[bi, ei] = np.mean(v)
                mat_p[bi, ei] = _across_session_pval(v)

        vmax = np.nanmax(np.abs(mat_mean)) if np.any(np.isfinite(mat_mean)) else 0.2
        vmax = max(vmax, 0.05)
        im = ax.imshow(mat_mean, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        for bi in range(n_beh):
            for ei in range(n_ep_mode):
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

        ax.set_xticks(range(n_ep_mode))
        ax.set_xticklabels(ep_labels_mode, rotation=30, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean rho')

        if row == 0:
            ax.set_title(f'{mode}\n({row_label})', fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=12)

_cap_label = f', first {TRIAL_CAP} trials' if TRIAL_CAP is not None else ''
fig.suptitle(
    f'Baseline mode: {BASELINE_MODE} (N={N_BASELINE}, n={n_s} sessions{_cap_label})',
    fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
_cap_tag = f"_cap{TRIAL_CAP}" if TRIAL_CAP is not None else ""
_test_tag = f"_{TEST_TYPE}"
plt.savefig(os.path.join(RESULTS_DIR, f'fig_temporal_offset_v2_causal_{BASELINE_MODE}_N{N_BASELINE}{_cap_tag}{_test_tag}_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
_cap_tag = f"_cap{TRIAL_CAP}" if TRIAL_CAP is not None else ""
_test_tag = f"_{TEST_TYPE}"
report_path = os.path.join(RESULTS_DIR, f'temporal_offset_v2_causal_{BASELINE_MODE}_N{N_BASELINE}{_cap_tag}{_test_tag}_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("TEMPORAL OFFSET COACTIVITY — V2 CAUSAL VARIANTS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s (pre leads post)\n")
    f.write(f"Baseline mode: {BASELINE_MODE}, N={N_BASELINE}\n")
    if TRIAL_CAP is not None:
        f.write(f"Trial cap: first {TRIAL_CAP} trials\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write(f"  dev2_fulltrial_baseline : epoch-resolved CC, baseline = full-trial\n"
            f"                            mean over BASELINE_MODE trials\n")
    f.write(f"  dev2_epoch_baseline     : epoch-resolved CC, baseline = per-epoch\n"
            f"                            mean over BASELINE_MODE trials\n")
    f.write(f"  full_trial              : CC over entire trial, baseline = full-trial\n"
            f"                            mean over BASELINE_MODE trials\n")
    f.write(f"  full_trial_dot_prod     : CC over entire trial, raw dot product\n\n")

    for mode in CC_MODES:
        n_s = len(all_results[mode])
        n_ep_mode = corr_slope[mode].shape[2]
        if mode in ('full_trial', 'full_trial_dot_prod'):
            ep_labels_rpt = ['full_trial']
        else:
            ep_labels_rpt = ['pre', 'go_cue', 'late', 'reward']

        f.write(f"\n{'='*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'='*50}\n\n")

        for target, corr_arr, label in [
            ('slope', corr_slope[mode], 'BEHAVIOR vs SLOPE'),
            ('intercept', corr_intercept[mode], 'BEHAVIOR vs INTERCEPT'),
        ]:
            f.write(f"{label}\n")
            f.write("-" * 40 + "\n")
            _ptitle = ('Wilcoxon p' if TEST_TYPE == 'wilcoxon'
                       else 't-test p (Fisher-z)')
            f.write(f"  {'beh x epoch':25s} {'mean':>7s} {'median':>7s} "
                    f"{'%>0':>5s} {_ptitle:>16s} {'sig':>4s}\n")

            for bi, bname in enumerate(beh_names):
                for ei, ep in enumerate(ep_labels_rpt):
                    vals = corr_arr[:, bi, ei]
                    v = vals[np.isfinite(vals)]
                    m = np.mean(v) if len(v) > 0 else np.nan
                    md = np.median(v) if len(v) > 0 else np.nan
                    fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
                    p = _across_session_pval(v) if len(v) >= 3 else 1.0
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    row_name = f"{bname}_{ep}"
                    f.write(f"  {row_name:25s} {m:+7.3f} {md:+7.3f} "
                            f"{fpos:4.0f}% {p:16.4f} {sig:>4s}\n")
                f.write("\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 9: Binned plots — HI (slope) vs RPE and HI vs Hit rate, all 3 modes
# ============================================================================
n_bins = 3
beh_plot = [('win_rpe', 'RPE'), ('win_hit', 'Hit rate')]

fig, axes = plt.subplots(len(beh_plot), len(CC_MODES),
                         figsize=(5 * len(CC_MODES), 4 * len(beh_plot)),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    results = all_results[mode]
    n_ep_mode = results[0]['hi_with_int'].shape[1] if len(results) > 0 else 1

    if mode in ('full_trial', 'full_trial_dot_prod'):
        ep_indices = [0]
        ep_names = ['Full trial']
        colors = ['#2c3e50']
    else:
        ep_indices = list(range(n_ep_mode))
        ep_names = ['Pre', 'Go cue', 'Late', 'Reward']
        colors = ['#c0392b', '#e67e22', '#27ae60', '#2980b9']

    for row, (beh_key, beh_label) in enumerate(beh_plot):
        ax = axes[row, col]

        for ei, ep_name, clr in zip(ep_indices, ep_names, colors):
            all_beh_z = []
            all_slope_z = []

            for s in results:
                bvar = s[beh_key]
                slope = s['hi_with_int'][:, ei]
                ok = np.isfinite(bvar) & np.isfinite(slope)
                if np.sum(ok) < 5:
                    continue
                bvar_ok = bvar[ok]
                slope_ok = slope[ok]
                if np.std(bvar_ok) == 0 or np.std(slope_ok) == 0:
                    continue
                all_beh_z.append((bvar_ok - np.mean(bvar_ok)) / np.std(bvar_ok))
                all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))

            if len(all_beh_z) == 0:
                continue
            all_beh_z = np.concatenate(all_beh_z)
            all_slope_z = np.concatenate(all_slope_z)

            bin_edges = np.percentile(all_beh_z, np.linspace(0, 100, n_bins + 1))
            bc, bm, bs = [], [], []
            for bi in range(n_bins):
                if bi < n_bins - 1:
                    mask = (all_beh_z >= bin_edges[bi]) & (all_beh_z < bin_edges[bi + 1])
                else:
                    mask = (all_beh_z >= bin_edges[bi]) & (all_beh_z <= bin_edges[bi + 1])
                if np.sum(mask) < 3:
                    continue
                bc.append(np.mean(all_beh_z[mask]))
                bm.append(np.mean(all_slope_z[mask]))
                bs.append(np.std(all_slope_z[mask]) / np.sqrt(np.sum(mask)))

            ax.errorbar(bc, bm, yerr=bs, fmt='o-', color=clr, capsize=5,
                        linewidth=2, markersize=7, label=ep_name)

        ax.axhline(0, color='k', ls='-', alpha=0.3)
        ax.axvline(0, color='k', ls='--', alpha=0.3)
        ax.set_xlabel(f'{beh_label} (within-session z)')
        ax.set_ylabel('Slope (within-session z)')
        if row == 0:
            ax.set_title(f'{mode}', fontsize=13, fontweight='bold')
        if len(ep_indices) > 1:
            ax.legend(fontsize=9, loc='best')

for row, (_, beh_label) in enumerate(beh_plot):
    axes[row, 0].set_ylabel(f'{beh_label}\nSlope (within-session z)')

_cap_label = f', first {TRIAL_CAP} trials' if TRIAL_CAP is not None else ''
_cap_tag = f"_cap{TRIAL_CAP}" if TRIAL_CAP is not None else ""
_test_tag = f"_{TEST_TYPE}"
fig.suptitle(
    f'Baseline mode: {BASELINE_MODE} N={N_BASELINE} '
    f'(n={len(results)} sessions, {n_bins} bins{_cap_label})',
    fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'fig_temporal_offset_v2_causal_{BASELINE_MODE}_N{N_BASELINE}{_cap_tag}{_test_tag}_binned.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Binned figure saved.")
