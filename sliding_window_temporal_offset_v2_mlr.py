#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
sliding_window_temporal_offset_v2 + an MLR (joint) option.

Motivation (from the toy model): the standard per-window HI regresses dW on ONE
window's coactivity at a time (hi_with_int). Because the windows' CC vectors are
highly collinear (shared pairs + overlapping trials), that univariate slope is
contaminated by every other window's contribution:
    proj_w ~ c_w + sum_{v!=w} c_v <CC_w,CC_v>/<CC_w,CC_w>.
A JOINT multiple-linear regression  dW ~ [CC_w1, CC_w2, ...]  inverts that overlap
and returns each window's *partial* slope -> cleaner recovery of the per-window
RPE-weighting. In the noiseless toy this is exact; on noisy data the Gram matrix
of collinear CC is near-singular, so OLS amplifies noise -> a RIDGE penalty
(MLR_LAMBDAS) is the bias/variance dial. lambda=0 is plain OLS.

This script computes hi_with_int (univariate, unchanged) AND hi_mlr (joint, per
ridge lambda) for every mode/epoch, then compares how well each tracks behavior.
Sacred file sliding_window_temporal_offset_v2.py is untouched.
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

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
OFFSET_SEC = 0
N_BASELINE = 20

# v3 eligibility-form modes (all epoch-resolved, per-epoch early-trial baselines):
#   dot_prod       — sum_t pre*post                         (raw)
#   dev2           — sum_t pre*(post-bl_post)               (post-dev)  <- sacred v3 dev2
#   pre_dev_only   — sum_t (pre-bl_pre)*post                (pre-dev)
#   pre_dev        — sum_t (pre-bl_pre)*(post-bl_post)      (both-dev)
#   phi_prime_dev2 — dev2 gated by post > 20th pctl
CC_MODES = ['dot_prod', 'dev2', 'pre_dev_only', 'pre_dev', 'phi_prime_dev2']
CC_MODES = ['dev2']
# ---- MLR settings ----
# Ridge penalty expressed as a FRACTION of the data's own scale (mean diag of X'X,
# ~n_pairs), so lambda is dimensionless: 0 = OLS; ~1 = shrinkage comparable to the
# data; >>1 -> heavy shrinkage toward the SLR slope. lambda=1000 is ~1000x the data
# scale, i.e. essentially the SLR-slope limit (the ceiling).
# NOTE: CELL 5 (load) reads lambdas back from the saved .npy, so re-run CELL 3->4
# after changing this list -- reloading alone will show the OLD lambdas.
MLR_LAMBDAS = [0.0, 0.01, 0.1,  30.0, 100.0, 1000.0, 100000.0]  # CV searches this grid

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"CC modes: {CC_MODES}")
print(f"MLR ridge lambdas: {MLR_LAMBDAS}")


def compute_mlr(cc_data, Y_T, n_wins, n_ep_mode, lambdas):
    """Joint (MLR) partial coefficient per window. cc_data: (n_wins, n_pairs, n_ep).
    Fit dW ~ all-windows jointly (ridge over `lambdas`) on standardized CC columns
    for numerical stability, then return TWO forms:
      out_std   : partial coefficient on standardized CC  (partial-correlation units;
                  lambda->inf limit == univariate hi_corr)
      out_slope : same fit converted to raw-CC units by dividing each window's
                  coefficient by that window's CC std  (partial-SLOPE units, directly
                  comparable to hi_with_int; lambda->inf limit == the SLR slope).
    Both: (n_wins, n_ep, n_lambda)."""
    n_pairs = Y_T.shape[0]
    out_std = np.full((n_wins, n_ep_mode, len(lambdas)), np.nan)
    out_slope = np.full((n_wins, n_ep_mode, len(lambdas)), np.nan)
    y = Y_T - np.mean(Y_T)
    for ei in range(n_ep_mode):
        cc = cc_data[:, :, ei]                              # (n_wins, n_pairs)
        valid = np.array([(not np.any(~np.isfinite(cc[wi]))) and np.std(cc[wi]) > 0
                          for wi in range(n_wins)])
        vw = np.where(valid)[0]
        if len(vw) < 3:
            continue
        stds = np.array([cc[wi].std() for wi in vw])
        X = np.empty((n_pairs, len(vw)))                   # standardized CC columns
        for j, wi in enumerate(vw):
            c = cc[wi]; X[:, j] = (c - c.mean()) / stds[j]
        XtX = X.T @ X; Xty = X.T @ y
        I = np.eye(len(vw))
        d = np.mean(np.diag(XtX))                          # data scale (~n_pairs) -> dimensionless lambda
        for li, lam in enumerate(lambdas):
            try:
                coef = np.linalg.solve(XtX + lam * d * I, Xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
            out_std[vw, ei, li] = coef
            out_slope[vw, ei, li] = coef / stds            # raw-slope units (matches hi_with_int)
    return out_std, out_slope


def mlr_cv(cc_data, Y_T, pair_group, n_ep_mode, lambdas, K=5, seed=0):
    """Group-K-fold cross-validation over photostim groups: at each lambda, fit the
    joint ridge on training pairs and measure held-out R^2 predicting dW on left-out
    GROUPS (same-target pairs share the pre neuron, so we leave whole groups out to
    avoid leakage). Standardization + y-centering use TRAIN stats only.
    Returns cv_r2 (n_ep, n_lambda): pooled held-out R^2 per lambda. This is what
    picks lambda -- the coefficients for the RPE comparison still come from the
    full-data fit (compute_mlr)."""
    groups = np.unique(pair_group)
    if len(groups) < K:
        return np.full((n_ep_mode, len(lambdas)), np.nan)
    folds = np.array_split(np.random.default_rng(seed).permutation(groups), K)
    cv_r2 = np.full((n_ep_mode, len(lambdas)), np.nan)
    for ei in range(n_ep_mode):
        cc = cc_data[:, :, ei]
        valid = np.array([(not np.any(~np.isfinite(cc[wi]))) and np.std(cc[wi]) > 0
                          for wi in range(cc.shape[0])])
        vw = np.where(valid)[0]
        if len(vw) < 3:
            continue
        Xraw = cc[vw].T                                    # (n_pairs, n_valid)
        sse = np.zeros(len(lambdas)); sst = 0.0
        for k in range(K):
            te = np.isin(pair_group, folds[k])
            tr = ~te
            if tr.sum() < len(vw) + 2 or te.sum() < 2:
                continue
            Xtr = Xraw[tr]; ytr = Y_T[tr]
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            Xtr_s = (Xtr - mu) / sd; ymu = ytr.mean(); ytr_c = ytr - ymu
            XtX = Xtr_s.T @ Xtr_s; Xty = Xtr_s.T @ ytr_c
            d = np.mean(np.diag(XtX)); I = np.eye(len(vw))
            Xte_s = (Xraw[te] - mu) / sd; yte_c = Y_T[te] - ymu
            sst += np.sum(yte_c ** 2)
            for li, lam in enumerate(lambdas):
                try:
                    coef = np.linalg.solve(XtX + lam * d * I, Xty)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(Xtr_s, ytr_c, rcond=None)[0]
                sse[li] += np.sum((yte_c - Xte_s @ coef) ** 2)
        if sst > 0:
            cv_r2[ei, :] = 1.0 - sse / sst
    return cv_r2


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
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) &
                            (list_of_dirs['Has data_main.npy'] == True))[0]
    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
                continue
            folder = (r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/')
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop', 'F', 'mouse', 'session', 'conditioned_neuron',
                        'dt_si', 'step_time', 'reward_time', 'BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print("  Skipping -- file not found."); continue

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
            F = data['F']
            trl = F.shape[2]; n_neurons = F.shape[1]; n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si); tsta = tsta - tsta[int(2 / dt_si)]
            lag_frames = int(round(OFFSET_SEC / dt_si))

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy(); rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)

            # ---- pair selection ----
            dw_list, pair_cl_list, pair_nt_list = [], [], []
            for gi in range(stimDist.shape[1]):
                cl = np.where((stimDist[:, gi] < 10) & (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
                if cl.size == 0:
                    continue
                nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
                if nontarg.size == 0:
                    continue
                dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)
            if len(dw_list) == 0:
                print("  No valid pairs."); continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            # photostim-group label per pair (for group-aware CV: leave whole targets out)
            pair_group = np.concatenate([np.full(len(dw_list[gi]), gi) for gi in range(len(dw_list))])
            n_pairs = len(Y_T)

            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx]); cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_weights[offset + qi, cl_arr[qi]] = 1.0 / len(cl_arr[qi])
                offset += n_nt

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping."); continue

            F_nan = F.copy(); F_nan[np.isnan(F_nan)] = 0
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
            n_epochs = len(EPOCH_ORDER)
            epoch_pre_act, epoch_post_act = {}, {}
            for ep in ['pre', 'go_cue']:
                t0, t1 = (ts_pre[0], ts_pre[-1]) if ep == 'pre' else (ts_go[0], ts_go[-1])
                t0_lag = max(0, min(t0 + lag_frames, n_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, n_frames - 1))
                epoch_pre_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)
                epoch_post_act[ep] = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)
            for ep in ['late', 'reward']:
                epoch_pre_act[ep] = np.zeros((n_neurons, trl))
                epoch_post_act[ep] = np.zeros((n_neurons, trl))
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    idx = get_indices_around_steps(tsta, rewards, pre=20, post=1); idx = idx[idx < n_frames]
                    if len(idx) > 0:
                        epoch_pre_act['late'][:, ti] = np.nanmean(F_nan[idx, :, ti], axis=0)
                        il = idx + lag_frames; il = il[(il >= 0) & (il < n_frames)]
                        if len(il) > 0:
                            epoch_post_act['late'][:, ti] = np.nanmean(F_nan[il, :, ti], axis=0)
                    idx = get_indices_around_steps(tsta, rewards, pre=1, post=10); idx = idx[idx < n_frames]
                    if len(idx) > 0:
                        epoch_pre_act['reward'][:, ti] = np.nanmean(F_nan[idx, :, ti], axis=0)
                        il = idx + lag_frames; il = il[(il >= 0) & (il < n_frames)]
                        if len(il) > 0:
                            epoch_post_act['reward'][:, ti] = np.nanmean(F_nan[il, :, ti], axis=0)

            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            # v3 per-epoch early-trial baselines (post & pre) + 20th-pctl gate
            baseline_post_mean_ep = {ep: np.nanmean(epoch_post_act[ep][:, baseline_trials_arr], axis=1)
                                     for ep in EPOCH_ORDER}
            baseline_pre_mean_ep = {ep: np.nanmean(epoch_pre_act[ep][:, baseline_trials_arr], axis=1)
                                    for ep in EPOCH_ORDER}
            pctl20_post_ep = {ep: np.percentile(epoch_post_act[ep], 20, axis=1) for ep in EPOCH_ORDER}
            baseline_pre_pair = {ep: cl_weights @ baseline_pre_mean_ep[ep] for ep in EPOCH_ORDER}

            cc_arrays = {mode: np.full((n_wins, n_pairs, n_epochs), np.nan) for mode in CC_MODES}
            win_hit = np.full(n_wins, np.nan); win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan); win_hit_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            for wi, ws in enumerate(win_starts):
                we = ws + WIN_SIZE; trial_idx = np.arange(ws, we)
                win_center[wi] = (ws + we) / 2.0
                win_hit[wi] = np.nanmean(hit[trial_idx]); win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx]); win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])
                for ei, ep in enumerate(EPOCH_ORDER):
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]     # (n_pairs, win_size)
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]
                    post_dev = post_act - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    pre_dev = pre_act - baseline_pre_pair[ep][:, np.newaxis]
                    # only fill the forms present in CC_MODES (so subsetting CC_MODES works)
                    if 'dot_prod' in cc_arrays:
                        cc_arrays['dot_prod'][wi, :, ei] = np.sum(pre_act * post_act, axis=1)
                    if 'dev2' in cc_arrays:
                        cc_arrays['dev2'][wi, :, ei] = np.sum(pre_act * post_dev, axis=1)
                    if 'pre_dev_only' in cc_arrays:
                        cc_arrays['pre_dev_only'][wi, :, ei] = np.sum(pre_dev * post_act, axis=1)
                    if 'pre_dev' in cc_arrays:
                        cc_arrays['pre_dev'][wi, :, ei] = np.sum(pre_dev * post_dev, axis=1)
                    if 'phi_prime_dev2' in cc_arrays:
                        gate = (post_act > pctl20_post_ep[ep][all_nt, np.newaxis]).astype(float)
                        cc_arrays['phi_prime_dev2'][wi, :, ei] = np.sum(pre_act * post_dev * gate, axis=1)

            for mode in CC_MODES:
                n_ep_mode = n_epochs
                cc_data = cc_arrays[mode]

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
                        hi_intercept[wi, ei] = coeffs[0]; hi_with_int[wi, ei] = coeffs[1]
                        hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

                # ---- NEW: joint MLR (partial coeff per window), ridge over lambdas ----
                # hi_mlr = standardized (corr units); hi_mlr_slope = raw-slope units (matches hi_with_int)
                hi_mlr, hi_mlr_slope = compute_mlr(cc_data, Y_T, n_wins, n_ep_mode, MLR_LAMBDAS)
                # group-aware CV to choose lambda (held-out R^2 predicting dW)
                mlr_cv_r2 = mlr_cv(cc_data, Y_T, pair_group, n_ep_mode, MLR_LAMBDAS)

                all_results[mode].append({
                    'mouse': mouse, 'session': session, 'n_pairs': n_pairs,
                    'n_trials': trl, 'n_windows': n_wins, 'lag_frames': lag_frames,
                    'lag_sec': lag_frames * dt_si, 'dt_si': dt_si, 'win_centers': win_center,
                    'hit_rate': np.nanmean(hit),
                    'hi_no_int': hi_no_int, 'hi_with_int': hi_with_int,
                    'hi_intercept': hi_intercept, 'hi_corr': hi_corr,
                    'hi_mlr': hi_mlr, 'hi_mlr_slope': hi_mlr_slope,
                    'mlr_cv_r2': mlr_cv_r2, 'mlr_lambdas': np.array(MLR_LAMBDAS),
                    'win_hit': win_hit, 'win_rpe': win_rpe, 'win_rt': win_rt,
                    'win_hit_rpe': win_hit_rpe, 'Y_T': Y_T,
                })
            print(f"  {n_wins} windows, {n_pairs} pairs")

        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc(); continue

for mode in CC_MODES:
    print(f"{mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2_mlr.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2_mlr.npy'),
                      allow_pickle=True).item()
CC_MODES = list(all_results.keys())
MLR_LAMBDAS = list(all_results[CC_MODES[0]][0]['mlr_lambdas'])
print(f"Loaded modes: {CC_MODES}; lambdas: {MLR_LAMBDAS}")

#%% ============================================================================
# CELL 6: Univariate HI  vs  MLR  — which tracks behavior better?
# Per session: spearman(estimator_over_windows, behavior_over_windows).
# Estimators: univariate slope (hi_with_int), univariate corr (hi_corr),
#             MLR partial slope for each ridge lambda (hi_mlr[...,li]).
# ============================================================================
beh_keys = [('win_rpe', 'RPE'), ('win_hit', 'Hit rate'),
            ('win_rt', 'RT'), ('win_hit_rpe', 'Hit RPE')]
# focus epoch: Pre (ei=0) for all v3 modes (all epoch-resolved: Pre/Go/Late/Reward)
FOCUS_EI = 0

est_labels = ['univar slope', 'univar corr'] + [f'MLR slope l={l:g}' for l in MLR_LAMBDAS]


def est_series(s, which):
    if which == 'univar slope':
        return s['hi_with_int'][:, FOCUS_EI]
    if which == 'univar corr':
        return s['hi_corr'][:, FOCUS_EI]
    li = est_labels.index(which) - 2
    return s['hi_mlr_slope'][:, FOCUS_EI, li]      # MLR in SLR-slope units


for mode in CC_MODES:
    results = all_results[mode]
    n_s = len(results)
    print(f"\n=== {mode}  (epoch idx {FOCUS_EI}, n={n_s} sessions) ===")
    print("  {:14s}".format('estimator') + "".join(f"{lab:>14s}" for _, lab in beh_keys))
    mat = np.full((len(est_labels), len(beh_keys)), np.nan)
    for ri, est in enumerate(est_labels):
        row = f"  {est:14s}"
        for bi, (bkey, blab) in enumerate(beh_keys):
            rs = []
            for s in results:
                hi = est_series(s, est); bv = s[bkey]
                ok = np.isfinite(hi) & np.isfinite(bv)
                if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                    rs.append(spearmanr(hi[ok], bv[ok])[0])
            rs = np.array(rs)
            if len(rs) >= 3:
                m = np.nanmean(rs)
                p = wilcoxon(rs)[1] if np.any(rs != 0) else np.nan
                mat[ri, bi] = m
                star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else ''
                row += f"{m:+.3f}{star:>3s}".rjust(14)
            else:
                row += f"{'--':>14s}"
        print(row)

    # heatmap for this mode
    vmax = max(0.05, np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(beh_keys), 0.5 + 0.5 * len(est_labels)))
    im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    for ri in range(len(est_labels)):
        for bi in range(len(beh_keys)):
            if np.isfinite(mat[ri, bi]):
                ax.text(bi, ri, f'{mat[ri, bi]:+.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(mat[ri, bi]) > 0.6 * vmax else 'k')
    ax.set_xticks(range(len(beh_keys))); ax.set_xticklabels([b for _, b in beh_keys], rotation=20, ha='right')
    ax.set_yticks(range(len(est_labels))); ax.set_yticklabels(est_labels)
    ax.set_title(f'{mode} (ei={FOCUS_EI}, n={n_s})\ncorr(estimator, behavior)', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label='mean Spearman rho')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f'mlr_vs_univar_{mode}.png'), dpi=150, bbox_inches='tight')
    plt.show()
print("\nDone. Compare the 'univar' rows to the 'MLR' rows for the RPE column.")

#%% ============================================================================
# CELL 7: behavior x epoch matrix (like v2 CELL 7), for SLR and each MLR lambda.
# One figure per mode; columns = estimators (SLR, MLR l=..), each a behavior x
# epoch heatmap of mean spearman(estimator_over_windows, behavior_over_windows).
# ============================================================================
beh_keys = [('win_rpe', 'RPE'), ('win_hit', 'Hit rate'),
            ('win_rt', 'RT'), ('win_hit_rpe', 'Hit RPE')]
EPOCH_LABELS_FULL = ['Pre', 'Go cue', 'Late', 'Reward']
estimators = [('SLR slope', 'slope', None)] + [(f'MLR slope l={l:g}', 'mlr', li)
                                               for li, l in enumerate(MLR_LAMBDAS)]


def _series(s, kind, li, ei):
    return s['hi_with_int'][:, ei] if kind == 'slope' else s['hi_mlr_slope'][:, ei, li]


def build_be_matrix(results, kind, li, n_ep):
    mat = np.full((len(beh_keys), n_ep), np.nan)
    pmat = np.full((len(beh_keys), n_ep), np.nan)
    for bi, (bkey, _) in enumerate(beh_keys):
        for ei in range(n_ep):
            rs = []
            for s in results:
                hi = _series(s, kind, li, ei); bv = s[bkey]
                ok = np.isfinite(hi) & np.isfinite(bv)
                if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                    rs.append(spearmanr(hi[ok], bv[ok])[0])
            rs = np.array(rs)
            if len(rs) >= 3:
                mat[bi, ei] = np.nanmean(rs)
                pmat[bi, ei] = wilcoxon(rs)[1] if np.any(rs != 0) else np.nan
    return mat, pmat


for mode in CC_MODES:
    results = all_results[mode]
    n_s = len(results)
    if n_s == 0:
        continue
    n_ep = results[0]['hi_with_int'].shape[1]
    ep_labels = EPOCH_LABELS_FULL if n_ep == 4 else ['Full trial']

    mats, pmats = [], []
    for _, kind, li in estimators:
        m, p = build_be_matrix(results, kind, li, n_ep)
        mats.append(m); pmats.append(p)
    # color scale referenced to the SLR panel so it stays readable even when the
    # OLS (lambda=0) MLR coefficients blow up; big MLR values just saturate (their
    # true value + stars are printed in the cell text regardless).
    vmax = max(0.05, np.nanmax(np.abs(mats[0])) * 1.2)

    fig, axes = plt.subplots(1, len(estimators),
                             figsize=(1.6 + 1.5 * len(estimators) * max(1, n_ep) / 4, 3.2),
                             squeeze=False)
    for col, (elabel, _, _) in enumerate(estimators):
        ax = axes[0, col]
        mat, pmat = mats[col], pmats[col]
        im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        for bi in range(len(beh_keys)):
            for ei in range(n_ep):
                if not np.isfinite(mat[bi, ei]):
                    continue
                p = pmat[bi, ei]
                star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else ''
                ax.text(ei, bi, f'{mat[bi, ei]:+.2f}' + (f'\n{star}' if star else ''),
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(mat[bi, ei]) > 0.6 * vmax else 'k')
        ax.set_xticks(range(n_ep)); ax.set_xticklabels(ep_labels, rotation=25, ha='right', fontsize=8)
        ax.set_yticks(range(len(beh_keys)))
        ax.set_yticklabels([b for _, b in beh_keys] if col == 0 else [''] * len(beh_keys), fontsize=8)
        ax.set_title(elabel, fontsize=9)
    fig.suptitle(f'{mode}  (n={n_s})  corr(estimator, behavior)', fontsize=10)
    fig.colorbar(im, ax=axes[0, :], shrink=0.7, label='mean Spearman rho')
    fig.savefig(os.path.join(RESULTS_DIR, f'mlr_epoch_matrix_{mode}.png'), dpi=150, bbox_inches='tight')
    plt.show()
print("\nBehavior x epoch matrices saved (SLR vs each MLR lambda).")

#%% ============================================================================
# CELL 8: Cross-validated lambda. Per mode (Pre epoch), aggregate the group-CV
# held-out R^2 across sessions, pick lambda* = argmax mean R^2, and show
# corr(coef, RPE) across lambda vs the SLR reference. The DATA chooses lambda.
#   lambda* small  -> joint structure genuinely predicts dW (MLR != SLR meaningfully)
#   lambda* large / R^2<=0 everywhere -> data can't support the joint fit; SLR is right
# ============================================================================
EI = 0  # Pre
lam = np.array(MLR_LAMBDAS)


def _corr_rpe(results, series_fn):
    rs = []
    for s in results:
        hi = series_fn(s); bv = s['win_rpe']
        ok = np.isfinite(hi) & np.isfinite(bv)
        if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
            rs.append(spearmanr(hi[ok], bv[ok])[0])
    rs = np.array(rs)
    p = wilcoxon(rs)[1] if len(rs) >= 3 and np.any(rs != 0) else np.nan
    return np.nanmean(rs) if len(rs) else np.nan, p


for mode in CC_MODES:
    results = all_results[mode]
    if len(results) == 0:
        continue
    R2 = np.array([s['mlr_cv_r2'][EI, :] for s in results])       # (n_sess, n_lambda)
    meanR2 = np.nanmean(R2, 0)
    if np.all(np.isnan(meanR2)):
        print(f"{mode}: CV R^2 all NaN, skipping"); continue
    li_star = int(np.nanargmax(meanR2)); lam_star = lam[li_star]
    slr_m, slr_p = _corr_rpe(results, lambda s: s['hi_with_int'][:, EI])

    print(f"\n=== {mode}  Pre epoch  (n={len(results)}) ===")
    print(f"  SLR slope: corr(HI,RPE)={slr_m:+.3f} (p={slr_p:.1e})")
    print("  {:>8s} {:>10s} {:>18s}".format('lambda', 'CV R^2', 'corr(coef,RPE)'))
    rpe_by_lam = []
    for li, L in enumerate(lam):
        m, p = _corr_rpe(results, lambda s, li=li: s['hi_mlr_slope'][:, EI, li])
        rpe_by_lam.append(m)
        star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else ''
        mark = '  <- lambda*' if li == li_star else ''
        print("  {:8.3g} {:+10.4f} {:+12.3f} {:3s}{}".format(L, meanR2[li], m, star, mark))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(lam))
    ax[0].plot(x, meanR2, 'o-'); ax[0].axvline(li_star, color='r', ls='--')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'{L:g}' for L in lam], rotation=45, fontsize=7)
    ax[0].set_xlabel('ridge lambda'); ax[0].set_ylabel('held-out CV R$^2$')
    ax[0].set_title(f'{mode}: CV R$^2$  ($\\lambda$*={lam_star:g})', fontsize=9)
    ax[1].plot(x, rpe_by_lam, 'o-', label='MLR'); ax[1].axhline(slr_m, color='g', ls='--', label='SLR')
    ax[1].axvline(li_star, color='r', ls='--', label='$\\lambda$*')
    ax[1].set_xticks(x); ax[1].set_xticklabels([f'{L:g}' for L in lam], rotation=45, fontsize=7)
    ax[1].set_xlabel('ridge lambda'); ax[1].set_ylabel('corr(coef, RPE)'); ax[1].legend(fontsize=8)
    ax[1].set_title('RPE tracking vs $\\lambda$', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f'mlr_cv_{mode}.png'), dpi=150, bbox_inches='tight')
    plt.show()
print("\nCV done. lambda* is data-chosen; read corr(coef,RPE) at lambda* against the SLR line.")
