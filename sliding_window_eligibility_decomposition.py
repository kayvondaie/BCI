#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Eligibility-form decomposition test (copy of sliding_window_temporal_offset_v2.py).

Goal: on the DATA, compare three postsynaptic factors that all share the same
presynaptic factor, to see whether the raw form's behavior is inherited from
the mean-rate (mean-drive) term or from the co-fluctuation term.

  raw        = r_pre * r_post                  (full_trial_dot_prod)
  fluctuation= r_pre * (r_post - baseline)     (full_trial)
  mean-drive = r_pre * baseline                (full_trial_mean_drive)  <-- NEW

Because the baseline is a per-neuron constant over trials, the decomposition is
exact at the CC level:  raw = fluctuation + mean-drive, i.e.
  fulltrial_meandrive_cc = fulltrial_dp_cc - fulltrial_cc.
(The equality does NOT hold at the HI/slope level; the slope is nonlinear in CC.)

Metric of interest: within-session corr(HI(window), RPE). Prediction under the
"raw is swamped by mean rates" hypothesis: HI_raw tracks HI_mean_drive (RPE-blind)
while HI_fluctuation tracks RPE. Data-only; informs the matching model run.
"""
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

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

# Shared folder for model+data comparison figures (model scripts write here too).
COMPARE_DIR = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'
os.makedirs(COMPARE_DIR, exist_ok=True)

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
tau_elig = 10

# Temporal offset in seconds (pre leads post)
OFFSET_SEC = 0

# Baseline trials for dev2 modes
N_BASELINE = 20

# CC modes for this script:
#   'dev2_fulltrial_baseline' — epoch-resolved CC, but baseline = full-trial mean
#   'full_trial'              — CC over entire trial (no epochs), dev2-style (baseline subtracted) = FLUCTUATION
#   'full_trial_dot_prod'     — CC over entire trial (no epochs), raw dot product (no baseline) = RAW
#   'full_trial_mean_drive'   — CC over entire trial (no epochs), pre * baseline = MEAN-DRIVE (= raw - fluctuation)
CC_MODES = ['dev2_fulltrial_baseline', 'full_trial', 'full_trial_dot_prod', 'full_trial_mean_drive',
            # single-factor controls (no coactivity product): pre only, post only, deviation only
            'full_trial_pre_only', 'full_trial_post_only', 'full_trial_dev_only']

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s (pre leads post)")
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

            # ==================================================================
            # Baselines
            # ==================================================================
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))

            # Full-trial baseline: mean post activity across ALL frames,
            # averaged over the first N_BASELINE trials
            baseline_post_mean_fulltrial = np.nanmean(
                fulltrial_post_act[:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # ---- Compute CC per window ----
            # dev2_fulltrial_baseline: epoch-resolved, but subtract full-trial baseline
            dev2_ftb_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            # full_trial: single CC per window (no epoch dimension -> 1), dev2-style
            fulltrial_cc = np.full((n_wins, n_pairs, 1), np.nan)
            # full_trial_dot_prod: single CC per window, raw dot product (no baseline)
            fulltrial_dp_cc = np.full((n_wins, n_pairs, 1), np.nan)
            # single-factor controls (no product): pre only, post only, deviation only
            fulltrial_preonly_cc = np.full((n_wins, n_pairs, 1), np.nan)
            fulltrial_postonly_cc = np.full((n_wins, n_pairs, 1), np.nan)
            fulltrial_devonly_cc = np.full((n_wins, n_pairs, 1), np.nan)

            # Per-pair CC arrays for pre epoch only (for flip decomposition).
            # raw = pre*post, dev2 = pre*(post-baseline), correction = dev2 - raw
            raw_cc_pre = np.full((n_wins, n_pairs), np.nan)
            dev2_cc_pre = np.full((n_wins, n_pairs), np.nan)

            # Per-window deviation of post-synaptic (nontarget) neurons from
            # their baseline, in the pre epoch.  One scalar per window.
            #   win_mean_dev:  mean signed deviation (drift direction)
            #   win_abs_dev:   mean |deviation|   (drift magnitude)
            win_mean_dev = np.full(n_wins, np.nan)
            win_abs_dev = np.full(n_wins, np.nan)
            # Mean RAW post activity per window (crux probe: does the mean rate
            # itself track RPE? model says yes, prediction here is ~0).
            win_mean_post = np.full(n_wins, np.nan)

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

                # --- deviation of nontarget post neurons from baseline (pre epoch) ---
                post_pre_ep = epoch_post_act['pre'][all_nt, :][:, trial_idx]  # (n_pairs, win_size)
                dev_from_bl = post_pre_ep - baseline_post_mean_fulltrial[all_nt, np.newaxis]
                win_mean_dev[wi] = np.mean(dev_from_bl)
                win_abs_dev[wi] = np.mean(np.abs(dev_from_bl))
                # mean raw post activity (no baseline subtraction) this window
                win_mean_post[wi] = np.mean(post_pre_ep)

                # --- per-pair CC for pre epoch (raw and dev2, for flip analysis) ---
                pre_act_pre = cl_weights @ epoch_pre_act['pre'][:, trial_idx]
                post_act_pre = epoch_post_act['pre'][all_nt, :][:, trial_idx]
                raw_cc_pre[wi, :] = np.sum(pre_act_pre * post_act_pre, axis=1)
                post_dev_pre = post_act_pre - baseline_post_mean_fulltrial[all_nt, np.newaxis]
                dev2_cc_pre[wi, :] = np.sum(pre_act_pre * post_dev_pre, axis=1)

                # --- dev2_fulltrial_baseline: epoch-resolved, full-trial baseline ---
                for ei, ep in enumerate(EPOCH_ORDER):
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]   # (n_pairs, win_size)
                    post_dev = (epoch_post_act[ep][all_nt, :][:, trial_idx]
                                - baseline_post_mean_fulltrial[all_nt, np.newaxis])
                    cc_dev2_ftb = np.sum(pre_act * post_dev, axis=1)
                    dev2_ftb_cc[wi, :, ei] = cc_dev2_ftb

                # --- full_trial: no epochs, entire trial (dev2-style) ---
                pre_act_ft = cl_weights @ fulltrial_pre_act[:, trial_idx]    # (n_pairs, win_size)
                post_dev_ft = (fulltrial_post_act[all_nt, :][:, trial_idx]
                               - baseline_post_mean_fulltrial[all_nt, np.newaxis])
                cc_ft = np.sum(pre_act_ft * post_dev_ft, axis=1)
                fulltrial_cc[wi, :, 0] = cc_ft

                # --- full_trial_dot_prod: no epochs, raw dot product ---
                post_act_ft = fulltrial_post_act[all_nt, :][:, trial_idx]   # (n_pairs, win_size)
                cc_ft_dp = np.sum(pre_act_ft * post_act_ft, axis=1)
                fulltrial_dp_cc[wi, :, 0] = cc_ft_dp

                # --- single-factor controls (no product) ---
                fulltrial_preonly_cc[wi, :, 0] = np.sum(pre_act_ft, axis=1)    # pre only
                fulltrial_postonly_cc[wi, :, 0] = np.sum(post_act_ft, axis=1)  # post only
                fulltrial_devonly_cc[wi, :, 0] = np.sum(post_dev_ft, axis=1)   # deviation only

            # ---- Cross-neuron coupling: does mean rate predict fluctuation size? ----
            # Derived crux: mean-drive HI ~ cross-neuron corr(<r_post_i>, dev_i).
            # Predict positive in the model, ~0 here (baseline rate decoupled from
            # task modulation). Over unique nontarget (post) neurons, pre epoch.
            uniq_nt = np.unique(all_nt)
            post_sess = epoch_post_act['pre'][uniq_nt, :]        # (n_uniq, trl)
            neuron_mean = np.nanmean(post_sess, axis=1)
            neuron_std = np.nanstd(post_sess, axis=1)
            _okn = np.isfinite(neuron_mean) & np.isfinite(neuron_std)
            if np.sum(_okn) >= 5 and np.std(neuron_mean[_okn]) > 0:
                coupling_r = spearmanr(neuron_mean[_okn], neuron_std[_okn])[0]
            else:
                coupling_r = np.nan

            # ---- REFINED (confound-free) coupling: corr(mean rate, RPE-modulation) ----
            # RPE-modulation_i = |corr(neuron i's per-window activity, win_rpe)|,
            # not raw std -> strips the mean-variance mechanics.
            nwin_act = np.full((len(uniq_nt), n_wins), np.nan)
            for _wi, _ws in enumerate(win_starts):
                nwin_act[:, _wi] = np.nanmean(post_sess[:, _ws:_ws + WIN_SIZE], axis=1)
            rpe_mod = np.full(len(uniq_nt), np.nan)
            for _k in range(len(uniq_nt)):
                _col = nwin_act[_k]
                _okc = np.isfinite(_col) & np.isfinite(win_rpe)
                if np.sum(_okc) >= 5 and np.std(_col[_okc]) > 0 and np.std(win_rpe[_okc]) > 0:
                    rpe_mod[_k] = abs(spearmanr(_col[_okc], win_rpe[_okc])[0])
            _okr = np.isfinite(neuron_mean) & np.isfinite(rpe_mod)
            if np.sum(_okr) >= 5 and np.std(neuron_mean[_okr]) > 0:
                coupling_rpe = spearmanr(neuron_mean[_okr], rpe_mod[_okr])[0]
            else:
                coupling_rpe = np.nan

            # ---- DIMENSIONALITY: pre-post correlation + participation ratio ----
            # pre-post correlation (redundancy): per pair, corr over trials of
            # CN-weighted pre activity vs nontarget post activity (pre epoch).
            # Predict LOW here (distinct pops), HIGH in the model (same low-D pop).
            pre_full = cl_weights @ epoch_pre_act['pre']     # (n_pairs, trl)
            post_full = epoch_post_act['pre'][all_nt, :]      # (n_pairs, trl)
            _ppc = np.full(n_pairs, np.nan)
            for _p in range(n_pairs):
                _a = pre_full[_p]; _b = post_full[_p]
                _ok = np.isfinite(_a) & np.isfinite(_b)
                if np.sum(_ok) >= 5 and np.std(_a[_ok]) > 0 and np.std(_b[_ok]) > 0:
                    _ppc[_p] = np.corrcoef(_a[_ok], _b[_ok])[0, 1]
            prepost_corr = np.nanmean(_ppc)
            # participation ratio of the recorded population (pre epoch)
            try:
                _pop = epoch_post_act['pre']                  # (n_neurons, trl)
                _good = np.all(np.isfinite(_pop), axis=1) & (np.std(_pop, axis=1) > 0)
                _cov = np.cov(_pop[_good])
                _ev = np.linalg.eigvalsh(_cov); _ev = _ev[_ev > 1e-12]
                pr = float((_ev.sum() ** 2) / (_ev ** 2).sum())
                pr_frac = pr / int(np.sum(_good))
            except Exception:
                pr = np.nan; pr_frac = np.nan

            # ---- Mean-drive CC: raw - fluctuation = pre * baseline (exact) ----
            fulltrial_meandrive_cc = fulltrial_dp_cc - fulltrial_cc

            # ---- Fit slope/intercept for each mode ----
            for mode in CC_MODES:
                if mode == 'dev2_fulltrial_baseline':
                    n_ep_mode = n_epochs
                    cc_data = dev2_ftb_cc
                elif mode == 'full_trial':
                    n_ep_mode = 1
                    cc_data = fulltrial_cc
                elif mode == 'full_trial_dot_prod':
                    n_ep_mode = 1
                    cc_data = fulltrial_dp_cc
                elif mode == 'full_trial_mean_drive':
                    n_ep_mode = 1
                    cc_data = fulltrial_meandrive_cc
                elif mode == 'full_trial_pre_only':
                    n_ep_mode = 1
                    cc_data = fulltrial_preonly_cc
                elif mode == 'full_trial_post_only':
                    n_ep_mode = 1
                    cc_data = fulltrial_postonly_cc
                elif mode == 'full_trial_dev_only':
                    n_ep_mode = 1
                    cc_data = fulltrial_devonly_cc

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
                    'win_mean_dev': win_mean_dev,
                    'win_abs_dev': win_abs_dev,
                    'win_mean_post': win_mean_post,
                    'coupling_r': coupling_r,   # cross-neuron corr(mean rate, fluct size)
                    'coupling_rpe': coupling_rpe,  # cross-neuron corr(mean rate, RPE-modulation)
                    'prepost_corr': prepost_corr,  # per-pair corr(pre, post) over trials
                    'pr': pr, 'pr_frac': pr_frac,  # participation ratio + PR/N
                    'sess_mean_dev': np.nanmean(win_mean_dev),
                    'sess_abs_dev': np.nanmean(win_abs_dev),
                    # Per-pair CC arrays for pre epoch (n_wins x n_pairs)
                    'raw_cc_pre': raw_cc_pre,
                    'dev2_cc_pre': dev2_cc_pre,
                    'Y_T': Y_T,  # dW per pair (n_pairs,)
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
np.save(os.path.join(RESULTS_DIR, 'sliding_window_eligibility_decomposition.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_eligibility_decomposition.npy'),
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

    # full_trial has 1 "epoch"; dev2_fulltrial_baseline has 4
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

    if mode in ('full_trial', 'full_trial_dot_prod', 'full_trial_mean_drive',
                'full_trial_pre_only', 'full_trial_post_only', 'full_trial_dev_only'):
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

fig.suptitle(f'Control analyses (n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    plt.savefig(os.path.join(_d, 'fig_eligibility_decomposition_controls.png'),
                dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved (incl. comparison dir).")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'eligibility_decomposition_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("TEMPORAL OFFSET COACTIVITY — CONTROL ANALYSES\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s (pre leads post)\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write(f"  dev2_fulltrial_baseline : epoch-resolved CC, baseline = full-trial\n")
    f.write(f"                            mean (not epoch-specific mean)\n")
    f.write(f"  full_trial              : CC over entire trial, no epoch splitting\n")
    f.write(f"                            (dev2-style, baseline subtracted)\n")
    f.write(f"  full_trial_dot_prod     : CC over entire trial, raw dot product\n")
    f.write(f"                            (no baseline subtraction) = RAW\n")
    f.write(f"  full_trial_mean_drive   : CC over entire trial, pre * baseline\n")
    f.write(f"                            (mean-drive = raw - fluctuation)\n\n")

    for mode in CC_MODES:
        n_s = len(all_results[mode])
        n_ep_mode = corr_slope[mode].shape[2]
        if mode in ('full_trial', 'full_trial_dot_prod', 'full_trial_mean_drive',
                'full_trial_pre_only', 'full_trial_post_only', 'full_trial_dev_only'):
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
            f.write(f"  {'beh x epoch':25s} {'mean':>7s} {'median':>7s} "
                    f"{'%>0':>5s} {'Wilcoxon p':>10s} {'sig':>4s}\n")

            for bi, bname in enumerate(beh_names):
                for ei, ep in enumerate(ep_labels_rpt):
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
# CELL 9: Binned plots — HI (slope) vs RPE and HI vs Hit rate, all 3 modes
# ============================================================================
n_bins = 3

# For epoch-resolved mode, use all 4 epochs; for full_trial modes, use epoch 0
# We'll plot one row per behavioral variable (RPE, hit rate) and one column per mode
beh_plot = [('win_rpe', 'RPE'), ('win_hit', 'Hit rate')]

fig, axes = plt.subplots(len(beh_plot), len(CC_MODES),
                         figsize=(5 * len(CC_MODES), 4 * len(beh_plot)),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    results = all_results[mode]
    n_ep_mode = results[0]['hi_with_int'].shape[1] if len(results) > 0 else 1

    if mode in ('full_trial', 'full_trial_dot_prod', 'full_trial_mean_drive',
                'full_trial_pre_only', 'full_trial_post_only', 'full_trial_dev_only'):
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
            # Collect within-session z-scored behavior and slope
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

            # Bin by z-scored behavior
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

# Row labels on left
for row, (_, beh_label) in enumerate(beh_plot):
    axes[row, 0].set_ylabel(f'{beh_label}\nSlope (within-session z)')

fig.suptitle(f'Binned HI slope vs behavior (n={len(results)} sessions, {n_bins} bins)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    plt.savefig(os.path.join(_d, 'fig_eligibility_decomposition_binned.png'),
                dpi=150, bbox_inches='tight')
plt.show()
print("Binned figure saved (incl. comparison dir).")

#%% ============================================================================
# CELL 10: CRUX PROBE -- does mean post activity track RPE? (data side)
# ============================================================================
# Model analog: run_model_hi_multiseed.py computes corr(mean post activity per
# division, RPE) and predicts it is POSITIVE in the model. Here we test the same
# in the data, where the hypothesis predicts ~0: the outcome signal lives in the
# fluctuation, not the mean rate. Uses the fluctuation-mode results (win_mean_post
# and win_rpe are identical across modes).
_probe_mode = 'full_trial'
probe_rhos = []
for s in all_results[_probe_mode]:
    mp = s.get('win_mean_post')
    rpe = s.get('win_rpe')
    if mp is None or rpe is None:
        continue
    ok = np.isfinite(mp) & np.isfinite(rpe)
    if np.sum(ok) >= 5 and np.std(mp[ok]) > 0 and np.std(rpe[ok]) > 0:
        probe_rhos.append(spearmanr(mp[ok], rpe[ok])[0])
probe_rhos = np.array(probe_rhos, dtype=float)

_p = wilcoxon(probe_rhos)[1] if len(probe_rhos) >= 2 and np.any(probe_rhos != 0) else np.nan
print("\nCRUX PROBE  corr(mean post activity, RPE)  [data, {} sessions]".format(
    len(probe_rhos)))
print("  mean={:+.3f}  median={:+.3f}  sem={:.3f}  Wilcoxon p={:.3g}".format(
    np.mean(probe_rhos), np.median(probe_rhos),
    np.std(probe_rhos) / np.sqrt(len(probe_rhos)), _p))

figp, axp = plt.subplots(figsize=(4.5, 4))
axp.axhline(0, color='0.6', lw=0.8)
axp.scatter(np.random.uniform(-0.06, 0.06, len(probe_rhos)), probe_rhos,
            s=22, color='tab:blue', alpha=0.5)
axp.scatter([0], [np.mean(probe_rhos)], s=110, color='tab:blue',
            edgecolor='w', zorder=5)
axp.set_xlim(-0.4, 0.4); axp.set_xticks([])
axp.set_ylabel('corr(mean post activity, RPE)')
axp.set_title('DATA: does the mean rate track RPE?\nmean={:+.2f} (n={})'.format(
    np.mean(probe_rhos), len(probe_rhos)))
figp.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    figp.savefig(os.path.join(_d, 'fig_data_meanrate_vs_rpe.png'), dpi=150,
                 bbox_inches='tight')
plt.show()
print("Probe figure saved (incl. comparison dir).")

#%% ============================================================================
# CELL 11: CROSS-NEURON COUPLING -- does mean rate predict fluctuation size?
# ============================================================================
# Derived crux from the model comparison: the mean-drive HI equals, up to
# structure, the cross-neuron corr(<r_post_i>, deviation_i). The model predicts
# this coupling is POSITIVE (active neurons = task neurons); here we predict ~0
# (a neuron's baseline rate is decoupled from its BCI-task modulation), which is
# why raw/mean-drive fail and only the deviation form works.
# CAVEAT: a mean-variance relationship can inflate this; interpret model-vs-data
# as the contrast, not the absolute value.
coupling_vals = []
for s in all_results['full_trial']:
    c = s.get('coupling_r')
    if c is not None and np.isfinite(c):
        coupling_vals.append(c)
coupling_vals = np.array(coupling_vals, dtype=float)

_pc = wilcoxon(coupling_vals)[1] if len(coupling_vals) >= 2 and np.any(coupling_vals != 0) else np.nan
print("\nCOUPLING  cross-neuron corr(mean rate, fluct size)  [data, {} sessions]".format(
    len(coupling_vals)))
print("  mean={:+.3f}  median={:+.3f}  sem={:.3f}  Wilcoxon p={:.3g}".format(
    np.mean(coupling_vals), np.median(coupling_vals),
    np.std(coupling_vals) / np.sqrt(len(coupling_vals)), _pc))

figc, axc = plt.subplots(figsize=(4.5, 4))
axc.axhline(0, color='0.6', lw=0.8)
axc.scatter(np.random.uniform(-0.06, 0.06, len(coupling_vals)), coupling_vals,
            s=22, color='tab:green', alpha=0.5)
axc.scatter([0], [np.mean(coupling_vals)], s=110, color='tab:green',
            edgecolor='w', zorder=5)
axc.set_xlim(-0.4, 0.4); axc.set_xticks([]); axc.set_ylim(-1.05, 1.05)
axc.set_ylabel('corr(mean rate, fluct size) across neurons')
axc.set_title('DATA: is mean rate coupled to deviation?\nmean={:+.2f} (n={})'.format(
    np.mean(coupling_vals), len(coupling_vals)))
figc.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    figc.savefig(os.path.join(_d, 'fig_data_meanrate_vs_dev_coupling.png'), dpi=150,
                 bbox_inches='tight')
plt.show()
print("Coupling figure saved (incl. comparison dir).")

#%% ============================================================================
# CELL 12: REFINED coupling -- corr(mean rate, RPE-modulation) across neurons
# ============================================================================
# Confound-free version of CELL 11: uses each neuron's RPE-modulation
# (|corr(activity, win_rpe)|) instead of raw fluctuation size, stripping the
# mean-variance mechanics. Prediction: model stays high, data drops toward 0.
coupling_rpe_vals = []
for s in all_results['full_trial']:
    c = s.get('coupling_rpe')
    if c is not None and np.isfinite(c):
        coupling_rpe_vals.append(c)
coupling_rpe_vals = np.array(coupling_rpe_vals, dtype=float)

_pcr = wilcoxon(coupling_rpe_vals)[1] if len(coupling_rpe_vals) >= 2 and np.any(coupling_rpe_vals != 0) else np.nan
print("\nREFINED COUPLING  corr(mean rate, RPE-modulation)  [data, {} sessions]".format(
    len(coupling_rpe_vals)))
print("  mean={:+.3f}  median={:+.3f}  sem={:.3f}  Wilcoxon p={:.3g}".format(
    np.mean(coupling_rpe_vals), np.median(coupling_rpe_vals),
    np.std(coupling_rpe_vals) / np.sqrt(len(coupling_rpe_vals)), _pcr))

figr, axr = plt.subplots(figsize=(4.5, 4))
axr.axhline(0, color='0.6', lw=0.8)
axr.scatter(np.random.uniform(-0.06, 0.06, len(coupling_rpe_vals)), coupling_rpe_vals,
            s=22, color='tab:purple', alpha=0.5)
axr.scatter([0], [np.mean(coupling_rpe_vals)], s=110, color='tab:purple',
            edgecolor='w', zorder=5)
axr.set_xlim(-0.4, 0.4); axr.set_xticks([]); axr.set_ylim(-1.05, 1.05)
axr.set_ylabel('corr(mean rate, RPE-modulation) across neurons')
axr.set_title('DATA: mean rate vs RPE-modulation\nmean={:+.2f} (n={})'.format(
    np.mean(coupling_rpe_vals), len(coupling_rpe_vals)))
figr.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    figr.savefig(os.path.join(_d, 'fig_data_meanrate_vs_rpemod_coupling.png'),
                 dpi=150, bbox_inches='tight')
plt.show()
print("Refined coupling figure saved (incl. comparison dir).")

#%% ============================================================================
# CELL 13: DIMENSIONALITY -- pre-post correlation + participation ratio
# ============================================================================
# The "what's missing in the model" test: single factors proxy the coactivity
# product only when activity is low-D / pre and post are redundant. Predict the
# DATA has LOW pre-post correlation and HIGH participation ratio (distinct,
# high-D populations); the model the opposite.
ppc_vals, pr_vals, prf_vals = [], [], []
for s in all_results['full_trial']:
    if s.get('prepost_corr') is not None and np.isfinite(s['prepost_corr']):
        ppc_vals.append(s['prepost_corr'])
    if s.get('pr') is not None and np.isfinite(s['pr']):
        pr_vals.append(s['pr'])
        prf_vals.append(s['pr_frac'])
ppc_vals = np.array(ppc_vals); pr_vals = np.array(pr_vals); prf_vals = np.array(prf_vals)

print("\nDIMENSIONALITY (data, {} sessions)".format(len(ppc_vals)))
print("  pre-post correlation  = {:+.3f}  sem={:.3f}".format(
    np.mean(ppc_vals), np.std(ppc_vals) / np.sqrt(len(ppc_vals))))
print("  participation ratio   = {:.1f}  sem={:.1f}   PR/N = {:.3f}".format(
    np.mean(pr_vals), np.std(pr_vals) / np.sqrt(len(pr_vals)), np.mean(prf_vals)))

figd, axd = plt.subplots(1, 2, figsize=(8, 4))
for _ax, _v, _lab in [(axd[0], ppc_vals, 'pre-post correlation'),
                      (axd[1], prf_vals, 'participation ratio / N')]:
    _ax.axhline(0, color='0.6', lw=0.8)
    _ax.scatter(np.random.uniform(-0.06, 0.06, len(_v)), _v, s=22, color='tab:gray', alpha=0.5)
    _ax.scatter([0], [np.mean(_v)], s=110, color='k', edgecolor='w', zorder=5)
    _ax.set_xlim(-0.4, 0.4); _ax.set_xticks([]); _ax.set_ylabel(_lab)
axd[0].set_title('DATA (n={})'.format(len(ppc_vals)))
figd.tight_layout()
for _d in (RESULTS_DIR, COMPARE_DIR):
    figd.savefig(os.path.join(_d, 'fig_data_dimensionality.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Dimensionality figure saved (incl. comparison dir).")
