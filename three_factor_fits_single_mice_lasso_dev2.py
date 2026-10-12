#%% ============================================================================
# CELL 1: Imports and Setup (run once)
# ============================================================================
"""
Dev2 variant of three_factor_fits_single_mice_lasso_2026claude.py.

Same regression framework, but CC is computed using the dev2 method:
  CC_dev2[i,j,t] = sum_timepoints F_i(t) * (F_j(t) - baseline_j)
where baseline_j is the mean activity of neuron j over the first N_BASELINE trials.

This matches the dev2 approach in sliding_window_temporal_offset.py.
"""
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import re
import time
from collections import OrderedDict
import traceback
import plotting_functions as pf
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration (edit as needed, run before session loop)
# ============================================================================

# TOGGLE: Choose analysis method
USE_TRIAL_LEVEL = True  # Set to False to use original binned approach

# Mice to analyze
mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]

# Analysis parameters
pairwise_mode = 'dev2'      # NEW: dev2 mode (baseline-subtracted dot product)
fit_type = 'pinv'           # pinv, ridge, lasso
alpha = 1e-3                # only used for ridge/lasso
num_bins = 10               # only used for binned approach
tau_elig = 10
N_BASELINE = 20             # number of early trials for baseline
shuffle = 0
plotting = 1
fitting = 1
correct_direct = True

# Epoch and behavior definitions
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_SPECS = [
    {"name": "hit", "source": "hit_bin"},
    {"name": "hit_rpe", "source": "hit_rpe_bin"},
    {"name": "10-rt", "source": lambda d: 10 - np.asarray(d["rt_bin"], dtype=float)},
    {"name": "rpe", "source": "rpe_bin"},
    {"name": "thr", "source": "thr_bin"},
    {"name": "2-factor", "source": lambda d: np.ones_like(np.asarray(d["hit_bin"], dtype=float))},
]

# Initialize storage
HI, RT, HIT, HIa, HIb, HIc, DOT, TRL, THR, RPE, FIT, GRP, RPE_FIT, DW = ([] for _ in range(14))
XALL, YALL, B, XPRE3, PVAL, BETA = ([] for _ in range(6))

print(f"Configuration set: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'}")
print(f"Analyzing mice: {mice}")
print(f"CC mode: dev2 (baseline from first {N_BASELINE} trials)")

#%% ============================================================================
# CELL 3: Main session loop
# ============================================================================
import csv, os
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meta_analysis_results')
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")


def compute_trial_level_cc_dev2(
    F, tsta, dt_si, reward_time, step_time=None, N_BASELINE=20,
):
    """
    Compute trial-level pairwise CC using the dev2 method:
      CC[i,j,t] = sum_timepoints F_i(timepoints) * (F_j(timepoints) - baseline_j)

    where baseline_j is the mean of F_j across the first N_BASELINE trials
    for each epoch separately.

    Returns
    -------
    CC_dict : dict
        Keys: 'pre', 'go_cue', 'late', 'reward'
        Values: ndarray (n_neurons, n_neurons, n_trials)
    """
    n_neurons = F.shape[1]
    n_trials = F.shape[2]

    F_nan = F.copy()
    F_nan[np.isnan(F_nan)] = 0

    # Time indices for fixed epochs
    ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
    ts_go = np.where((tsta > 0) & (tsta < 2))[0]

    # --- Collect epoch-averaged activity per neuron per trial ---
    # Shape: (n_neurons, n_trials) for each epoch
    epoch_act = {
        'pre': np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0) if len(ts_pre) > 0
               else np.zeros((n_neurons, n_trials)),
        'go_cue': np.nanmean(F_nan[ts_go[0]:ts_go[-1]+1, :, :], axis=0) if len(ts_go) > 0
                  else np.zeros((n_neurons, n_trials)),
        'late': np.zeros((n_neurons, n_trials)),
        'reward': np.zeros((n_neurons, n_trials)),
    }

    for ti in range(n_trials):
        rewards = reward_time[ti]
        if len(rewards) > 0:
            indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
            indices = indices[indices < F.shape[0]]
            if len(indices) > 0:
                epoch_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)

            indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
            indices = indices[indices < F.shape[0]]
            if len(indices) > 0:
                epoch_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)

    # --- Baseline: mean activity over first N_BASELINE trials per epoch ---
    baseline_trials = np.arange(min(N_BASELINE, n_trials))
    baseline = {}
    for ep in epoch_act:
        baseline[ep] = np.nanmean(epoch_act[ep][:, baseline_trials], axis=1)  # (n_neurons,)

    # --- Build CC_dict using full temporal dot product (dev2) ---
    CC_dict = {ep: np.zeros((n_neurons, n_neurons, n_trials)) for ep in epoch_act}

    for t in range(n_trials):
        # Pre epoch
        if len(ts_pre) > 0:
            F_pre = F_nan[ts_pre, :, t]  # (n_timepoints, n_neurons)
            # dev2: F_i(t) * (F_j(t) - baseline_j)
            F_dev = F_pre - baseline['pre'][np.newaxis, :]  # subtract baseline from post
            CC_dict['pre'][:, :, t] = F_pre.T @ F_dev

        # Go cue epoch
        if len(ts_go) > 0:
            F_go = F_nan[ts_go, :, t]
            F_dev = F_go - baseline['go_cue'][np.newaxis, :]
            CC_dict['go_cue'][:, :, t] = F_go.T @ F_dev

        # Late epoch
        if step_time is not None and len(step_time[t]) > 0:
            steps = step_time[t]
            indices = get_indices_around_steps(tsta, steps, pre=20, post=1)
            indices = indices[indices < F.shape[0]]
        elif len(reward_time[t]) > 0:
            indices = get_indices_around_steps(tsta, reward_time[t], pre=20, post=1)
            indices = indices[indices < F.shape[0]]
        else:
            indices = np.array([], dtype=int)
        if len(indices) > 0:
            F_late = F_nan[indices, :, t]
            F_dev = F_late - baseline['late'][np.newaxis, :]
            CC_dict['late'][:, :, t] = F_late.T @ F_dev

        # Reward epoch
        if len(reward_time[t]) > 0:
            indices = get_indices_around_steps(tsta, reward_time[t], pre=1, post=10)
            indices = indices[indices < F.shape[0]]
            if len(indices) > 0:
                F_rew = F_nan[indices, :, t]
                F_dev = F_rew - baseline['reward'][np.newaxis, :]
                CC_dict['reward'][:, :, t] = F_rew.T @ F_dev

    return CC_dict


for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) &
                           (list_of_dirs['Has data_main.npy']==True))[0]

    for sii in range(len(session_inds)):
        try:
            print(f"\n{'='*70}")
            print(f"Mouse {mi+1}/{len(mice)}: {mouse}, Session {sii+1}/{len(session_inds)}")
            print(f"Method: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'} | CC: dev2")
            print(f"{'='*70}")

            # ================================================================
            # Load data
            # ================================================================
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
                continue
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron',
                       'dt_si','step_time','reward_time','BCI_thresholds']

            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"Skipping — file not found.")
                continue

            # ================================================================
            # Preprocess
            # ================================================================
            BCI_thresholds = np.asarray(data["BCI_thresholds"], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i-1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            tsta = np.arange(0, 12, data['dt_si'])
            tsta = tsta - tsta[int(2/dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Behavioral variables
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0

            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)
            miss_rpe = compute_rpe((~hit).astype(float), baseline=0.0, tau=tau_elig, fill_value=1.0)

            # ================================================================
            # Feature construction
            # ================================================================

            if USE_TRIAL_LEVEL:
                print("Computing TRIAL-LEVEL dev2 features...")
                t0 = time.time()

                # Dev2 CC computation
                CC_dict = compute_trial_level_cc_dev2(
                    F=F,
                    tsta=tsta,
                    dt_si=dt_si,
                    reward_time=data['reward_time'],
                    step_time=data['step_time'],
                    N_BASELINE=N_BASELINE,
                )

                t_cc = time.time() - t0
                print(f"  CC computation (dev2): {t_cc:.2f}s")

                # Trial-level behavior with z-scoring
                behavior_dict_raw = {
                    'hit': hit.astype(float),
                    'hit_rpe': hit_rpe,
                    '10-rt': 10 - rt_filled,
                    'rpe': rt_rpe,
                    'thr': BCI_thresholds[1, :],
                    '2-factor': np.ones(trl),
                }

                behavior_dict = {}
                for name, values in behavior_dict_raw.items():
                    if name == '2-factor':
                        behavior_dict[name] = values
                    else:
                        mu, sigma = np.nanmean(values), np.nanstd(values)
                        if sigma == 0 or not np.isfinite(sigma):
                            sigma = 1.0
                        behavior_dict[name] = (values - mu) / sigma

                # Build features (uses same build_trial_level_features from BCI_data_helpers)
                t0 = time.time()
                X_T, Y_T, feature_names = build_trial_level_features(
                    CC_dict=CC_dict,
                    behavior_dict=behavior_dict,
                    stimDist=stimDist,
                    AMP=AMP,
                    dist_target_lt=10,
                    dist_nontarg_min=30,
                    dist_nontarg_max=1000,
                    amp0_thr=0.1,
                    amp1_thr=0.1,
                )
                t_feat = time.time() - t0

                # Z-score features by epoch
                print("\n  Z-scoring features by epoch...")
                for epoch in EPOCH_ORDER:
                    epoch_idx = [i for i, name in enumerate(feature_names) if name.startswith(epoch)]
                    if len(epoch_idx) > 0:
                        epoch_data = X_T[:, epoch_idx]
                        mu = epoch_data.mean(axis=0)
                        sigma = epoch_data.std(axis=0)
                        sigma[sigma == 0] = 1.0
                        X_T[:, epoch_idx] = (epoch_data - mu) / sigma
                        print(f"    {epoch}: rescaled {len(epoch_idx)} features")

                print(f"  Feature building: {t_feat:.2f}s")
                print(f"  Total: {t_cc + t_feat:.2f}s")
                print(f"  Features: {X_T.shape}, Pairs: {Y_T.shape[0]}")

            else:
                print("Computing BINNED dev2 features...")

                # For binned mode, compute epoch activity and subtract baseline
                F_nan = F.copy()
                F_nan[np.isnan(F_nan)] = 0
                n_neurons = F.shape[1]

                ts_pre_idx = np.where((tsta > -10) & (tsta < 0))[0]
                ts_go_idx = np.where((tsta > 0) & (tsta < 2))[0]

                kpre = np.nanmean(F_nan[ts_pre_idx[0]:ts_pre_idx[-1]+1, :, :], axis=0)
                k = np.nanmean(F_nan[ts_go_idx[0]:ts_go_idx[-1]+1, :, :], axis=0)
                kstep = np.zeros((n_neurons, trl))
                krewards = np.zeros((n_neurons, trl))

                for ti in range(trl):
                    rewards = data['reward_time'][ti]
                    if len(rewards) > 0:
                        indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                        indices = indices[indices < F.shape[0]]
                        if len(indices) > 0:
                            kstep[:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                        indices = indices[indices < F.shape[0]]
                        if len(indices) > 0:
                            krewards[:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)

                kpre[np.isnan(kpre)] = 0
                k[np.isnan(k)] = 0

                # Baseline: mean over first N_BASELINE trials per epoch
                bl_trials = np.arange(min(N_BASELINE, trl))
                bl_pre = np.nanmean(kpre[:, bl_trials], axis=1)
                bl_go = np.nanmean(k[:, bl_trials], axis=1)
                bl_late = np.nanmean(kstep[:, bl_trials], axis=1)
                bl_rew = np.nanmean(krewards[:, bl_trials], axis=1)

                # Subtract baseline: dev2 activity
                kpre_dev = kpre - bl_pre[:, np.newaxis]
                k_dev = k - bl_go[:, np.newaxis]
                kstep_dev = kstep - bl_late[:, np.newaxis]
                krewards_dev = krewards - bl_rew[:, np.newaxis]

                # Bin setup
                trial_bins = np.unique(np.linspace(0, trl, num_bins + 1).astype(int))
                nb = len(trial_bins) - 1

                cc = np.corrcoef(kstep)  # placeholder shape
                CCstep = np.full((n_neurons, n_neurons, nb), np.nan)
                CCrew = np.full((n_neurons, n_neurons, nb), np.nan)
                CCts = np.full((n_neurons, n_neurons, nb), np.nan)
                CCpre = np.full((n_neurons, n_neurons, nb), np.nan)

                # Compute dev2 pairwise CC per bin:
                # CC[i,j,bin] = activity_i (raw) @ deviation_j (baseline-subtracted)
                hit_bin = np.zeros(nb)
                rt_bin = np.zeros(nb)
                rpe_bin = np.zeros(nb)
                hit_rpe_bin = np.zeros(nb)
                miss_rpe_bin = np.zeros(nb)
                thr_bin = np.zeros(nb)
                miss_bin = np.zeros(nb)
                avg_dot_bin = np.zeros(nb)

                for bi in range(nb):
                    t0b, t1b = trial_bins[bi], trial_bins[bi+1]
                    trial_idx = np.arange(t0b, t1b)
                    if len(trial_idx) == 0:
                        continue

                    # Behaviors
                    hit_bin[bi] = np.nanmean(hit[trial_idx])
                    rt_bin[bi] = np.nanmean(rt_filled[trial_idx])
                    rpe_bin[bi] = np.nanmean(rt_rpe[trial_idx])
                    hit_rpe_bin[bi] = np.nanmean(hit_rpe[trial_idx])
                    miss_rpe_bin[bi] = np.nanmean(miss_rpe[trial_idx])
                    thr_bin[bi] = np.nanmean(BCI_thresholds[1, trial_idx])
                    miss_bin[bi] = np.nanmean((~hit[trial_idx]).astype(float))

                    # Dev2 CC: raw_pre @ dev_post for each epoch
                    pre_raw = kpre[:, trial_idx]  # (n_neurons, n_trials_bin)
                    pre_dev = kpre_dev[:, trial_idx]
                    CCpre[:, :, bi] = pre_raw @ pre_dev.T / len(trial_idx)

                    go_raw = k[:, trial_idx]
                    go_dev = k_dev[:, trial_idx]
                    CCts[:, :, bi] = go_raw @ go_dev.T / len(trial_idx)

                    late_raw = kstep[:, trial_idx]
                    late_dev = kstep_dev[:, trial_idx]
                    CCstep[:, :, bi] = late_raw @ late_dev.T / len(trial_idx)

                    rew_raw = krewards[:, trial_idx]
                    rew_dev = krewards_dev[:, trial_idx]
                    CCrew[:, :, bi] = rew_raw @ rew_dev.T / len(trial_idx)

                # Build pairwise features
                Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_stim, used_bins = \
                    build_pairwise_XY_per_bin(
                        CCstep, CCrew, CCts, CCpre,
                        stimDist, AMP,
                        nb=nb,
                        hit_bin=hit_bin,
                        dist_target_lt=10,
                        dist_nontarg_min=30,
                        dist_nontarg_max=1000,
                        amp0_thr=0.1,
                        amp1_thr=0.1
                    )

                # Build design matrix
                EPOCH_TO_XT = OrderedDict([
                    ("pre", Xpre_T),
                    ("go_cue", Xts_T),
                    ("late", Xstep_T),
                    ("reward", Xrew_T),
                ])

                binned_dict = {
                    "hit_bin": hit_bin,
                    "hit_rpe_bin": hit_rpe_bin,
                    "rt_bin": rt_bin,
                    "rpe_bin": rpe_bin,
                    "thr_bin": thr_bin,
                    "miss_bin": miss_bin,
                    "avg_dot_bin": avg_dot_bin,
                    "miss_rpe_bin": miss_rpe_bin,
                }

                trial_dict = {
                    "BCI_thresholds": np.asarray(data["BCI_thresholds"]).squeeze(),
                }

                ZSCORE_BEHAVIOR = [s["name"] for s in BEHAVIOR_SPECS if s["name"] != "2-factor"]
                zscore_cols_idx = [i for i, s in enumerate(BEHAVIOR_SPECS) if s["name"] in ZSCORE_BEHAVIOR]

                Bz, feature_names = build_behavior_matrix(
                    binned_dict=binned_dict,
                    trial_dict=trial_dict,
                    trial_bins=trial_bins,
                    used_bins=used_bins,
                    behavior_specs=BEHAVIOR_SPECS,
                    zscore_cols_idx=zscore_cols_idx,
                )

                X_T = build_design_matrix(EPOCH_TO_XT, Bz, EPOCH_ORDER)

                # Add nuisance regressor
                X_stim = np.asarray(X_stim, dtype=float).ravel()
                X_stim = np.nan_to_num(X_stim, nan=0.0)
                xs_mu, xs_sd = np.mean(X_stim), np.std(X_stim)
                if xs_sd == 0:
                    xs_sd = 1.0
                X_stim_z = (X_stim - xs_mu) / xs_sd

                if correct_direct:
                    X_T = np.hstack([X_T, X_stim_z[:, None]])

                print(f"  Features: {X_T.shape}, Pairs: {Y_T.shape[0]}")

            # ================================================================
            # Fitting (same for both methods)
            # ================================================================
            if fitting != 1:
                continue

            assert X_T.shape[0] == Y_T.shape[0], f"X/Y mismatch: {X_T.shape} vs {Y_T.shape}"

            n_epochs = len(EPOCH_ORDER)
            n_features_behav = len([s["name"] for s in BEHAVIOR_SPECS])
            n_base = n_epochs * n_features_behav

            if X_T.shape[1] > n_base:
                print(f"  WARNING: X has {X_T.shape[1]} features, expected {n_base}")
                print(f"  Removing extra columns (nuisance regressors)")
                X_T = X_T[:, :n_base]

            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            corr_train, p_train = [], []
            corr_test, p_test = [], []
            Y_test_all = np.array([])
            Y_test_pred_all = np.array([])
            beta_final = None

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_T, Y_T)):
                X_train, X_test = X_T[train_idx], X_T[test_idx]
                Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]

                assert X_train.shape[0] == Y_train.shape[0]
                assert X_test.shape[0] == Y_test.shape[0]

                mu, sigma = Y_train.mean(), Y_train.std()
                if sigma == 0 or not np.isfinite(sigma):
                    sigma = 1.0
                Y_train_z = (Y_train - mu) / sigma
                Y_test_z = (Y_test - mu) / sigma

                if fit_type == 'pinv':
                    beta = np.linalg.pinv(X_train) @ Y_train_z
                    Y_train_pred = X_train @ beta
                    Y_test_pred = X_test @ beta

                elif fit_type == 'ridge':
                    ridge = RidgeCV(alphas=np.logspace(-10, -4, 10), fit_intercept=True)
                    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', ridge)])
                    pipe.fit(X_train, Y_train_z)
                    Y_train_pred = pipe.predict(X_train)
                    Y_test_pred = pipe.predict(X_test)
                    beta = pipe.named_steps['ridge'].coef_

                elif fit_type == 'lasso':
                    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, tol=1e-4)
                    pipe = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
                    pipe.fit(X_train, Y_train_z)
                    Y_train_pred = pipe.predict(X_train)
                    Y_test_pred = pipe.predict(X_test)
                    beta = pipe.named_steps['lasso'].coef_

                else:
                    raise ValueError(f"Unknown fit_type: {fit_type}")

                if np.std(Y_train_pred) > 0:
                    r_train, pval_train = pearsonr(Y_train_pred, Y_train_z)
                else:
                    r_train, pval_train = 0.0, 1.0

                if np.std(Y_test_pred) > 0:
                    r_test, pval_test = pearsonr(Y_test_pred, Y_test_z)
                else:
                    r_test, pval_test = 0.0, 1.0

                corr_train.append(r_train)
                p_train.append(pval_train)
                corr_test.append(r_test)
                p_test.append(pval_test)

                Y_test_all = np.concatenate([Y_test_all, Y_test_z])
                Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])

                if fold_idx == 0:
                    beta_final = beta

            # Reshape beta for plotting
            if beta_final is not None:
                beta_reshaped = beta_final[:n_base].reshape(n_epochs, n_features_behav)
            else:
                beta_reshaped = None

            # Store results
            if beta_reshaped is not None:
                HI.append(beta_reshaped)
                BETA.append(beta_reshaped)

            RT.append(np.nanmean(rt))
            HIT.append(hit)
            FIT.append(np.mean(corr_test) if len(corr_test) > 0 else np.nan)
            GRP.append(stimDist.shape[1])
            ind = np.where(stimDist > 30)
            DW.append(np.nanmean(AMP[1][ind] - AMP[0][ind]) if len(ind[0]) > 0 else np.nan)
            XALL.append(X_T)
            YALL.append(Y_T)
            PVAL.append(np.exp(np.mean(np.log(p_test))) if len(p_test) > 0 else np.nan)

            # Plot
            if plotting == 1:
                print(f"\nResults:")
                print(f"  Train: r={np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
                print(f"  Test:  r={np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")

                plt.figure(figsize=(6, 8))
                plt.subplot(211)
                pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
                plt.xlabel(r'$Predicted \Delta W$')
                plt.ylabel(r'$\Delta W$')
                plt.title(f"{mouse} {session} (dev2) | hit={np.nanmean(hit)*100:.0f}% | r={np.mean(corr_test):.3f}")

                if beta_reshaped is not None:
                    plt.subplot(212)
                    behavior_names_plot = [s["name"] for s in BEHAVIOR_SPECS]
                    sns.heatmap(beta_reshaped, annot=True, fmt='.2f',
                               xticklabels=behavior_names_plot, yticklabels=EPOCH_ORDER,
                               cmap='coolwarm', center=0, vmin=-0.1, vmax=0.1)
                    plt.title(r'$\beta$ weights (dev2)')

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\n{'='*70}")
print(f"COMPLETE: Processed {len(BETA)} sessions")
print(f"Method: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'} | CC: dev2")
print(f"{'='*70}")

#%% ============================================================================
# CELL 4: Per-session summary
# ============================================================================
print("Summary (dev2):")
print(f"  Sessions: {len(FIT)}")
fits_valid = [f for f in FIT if np.isfinite(f)]
print(f"  Mean test r: {np.mean(fits_valid):.4f}")
print(f"  Median test r: {np.median(fits_valid):.4f}")
pvals_valid = [p for p in PVAL if np.isfinite(p)]
print(f"  Sessions with p<0.05: {sum(1 for p in pvals_valid if p < 0.05)}/{len(pvals_valid)}")

#%% ============================================================================
# CELL 5: Cross-session mean beta weights + Wilcoxon tests
# ============================================================================
from scipy.stats import wilcoxon

behavior_names_plot = [s["name"] for s in BEHAVIOR_SPECS]
n_epochs = len(EPOCH_ORDER)
n_beh = len(behavior_names_plot)

# Stack all beta matrices: (n_sessions, n_epochs, n_beh)
beta_stack = np.array(BETA)  # (n_sessions, n_epochs, n_beh)
n_sess = beta_stack.shape[0]

# Mean and p-value per cell
beta_mean = np.nanmean(beta_stack, axis=0)
beta_p = np.full((n_epochs, n_beh), np.nan)

for ei in range(n_epochs):
    for bi in range(n_beh):
        vals = beta_stack[:, ei, bi]
        v = vals[np.isfinite(vals)]
        if len(v) >= 3:
            try:
                _, beta_p[ei, bi] = wilcoxon(v)
            except Exception:
                beta_p[ei, bi] = 1.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A: Mean beta heatmap with significance
ax = axes[0]
vmax = np.nanmax(np.abs(beta_mean))
vmax = max(vmax, 0.01)
im = ax.imshow(beta_mean, cmap='coolwarm', vmin=-vmax, vmax=vmax,
               aspect='auto', interpolation='nearest')
for ei in range(n_epochs):
    for bi in range(n_beh):
        val = beta_mean[ei, bi]
        p = beta_p[ei, bi]
        if np.isnan(val):
            continue
        sig = ''
        if p < 0.001: sig = '***'
        elif p < 0.01: sig = '**'
        elif p < 0.05: sig = '*'
        txt = f'{val:+.3f}'
        if sig:
            txt += f'\n{sig}'
        ax.text(bi, ei, txt, ha='center', va='center',
                fontsize=9, fontweight='bold' if sig else 'normal')
ax.set_xticks(range(n_beh))
ax.set_xticklabels(behavior_names_plot, rotation=30, ha='right')
ax.set_yticks(range(n_epochs))
ax.set_yticklabels(EPOCH_ORDER)
plt.colorbar(im, ax=ax, shrink=0.8, label='Mean beta')
ax.set_title(f'Mean beta weights (dev2, n={n_sess})', fontweight='bold')

# Panel B: distribution of per-session test r
ax = axes[1]
fits_arr = np.array(FIT)
v = fits_arr[np.isfinite(fits_arr)]
jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(v))
ax.scatter(jitter, v, s=30, alpha=0.6, color='steelblue', edgecolors='k',
           linewidths=0.4, zorder=3)
m = np.mean(v)
sem = np.std(v, ddof=1) / np.sqrt(len(v))
ax.errorbar(0, m, yerr=sem, fmt='D', color='crimson', markersize=7,
            capsize=5, zorder=4, label=f'mean={m:.3f}')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_xlim(-0.5, 0.5)
ax.set_xticks([])
ax.set_ylabel('Test r (Pearson)')
ax.set_title('Per-session cross-validated fit', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Cross-session summary (dev2)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_dev2_summary.png'),
            dpi=200, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 6: Pooled lasso across all sessions
# ============================================================================
from sklearn.model_selection import ShuffleSplit

# Filter sessions with reasonable dW variance
aaa = [np.nanstd(y) for y in YALL]
ind = np.where(np.array(aaa) < 0.5)[0]
if len(ind) == 0:
    ind = np.arange(len(YALL))

X_concat = np.vstack([XALL[i] for i in ind])
Y_concat = np.concatenate([YALL[i] for i in ind])

mask = np.isfinite(Y_concat) & np.all(np.isfinite(X_concat), axis=1)
X_concat = X_concat[mask]
Y_concat = Y_concat[mask]

print(f"Pooled: {X_concat.shape[0]} pairs from {len(ind)} sessions")

# Train/test split
ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(ss.split(X_concat))

X_train, X_test = X_concat[train_idx], X_concat[test_idx]
Y_train, Y_test = Y_concat[train_idx], Y_concat[test_idx]

mu, sigma = Y_train.mean(), Y_train.std()
if sigma == 0:
    sigma = 1.0
Y_train_z = (Y_train - mu) / sigma
Y_test_z = (Y_test - mu) / sigma

# Fit pooled lasso
alpha_fast = 10**(-3.45)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=alpha_fast, fit_intercept=True, max_iter=5000, tol=1e-3))
])
model.fit(X_train, Y_train_z)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

r_tr, p_tr = pearsonr(Y_pred_train, Y_train_z) if np.std(Y_pred_train) > 0 else (np.nan, np.nan)
r_te, p_te = pearsonr(Y_pred_test, Y_test_z) if np.std(Y_pred_test) > 0 else (np.nan, np.nan)

beta_pooled = model.named_steps['lasso'].coef_
print(f"X shape: {X_concat.shape}")
print(f"nnz: {(beta_pooled != 0).sum()}/{beta_pooled.size}  alpha: {alpha_fast}")
print(f"Train r={r_tr:.3f} (p={p_tr:.2e}) | Test r={r_te:.3f} (p={p_te:.2e})")

# Reshape and plot
n_base = n_epochs * n_beh
beta_pooled_reshaped = beta_pooled[:n_base].reshape(n_epochs, n_beh)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
pf.mean_bin_plot(Y_pred_test, Y_test_z, 5, 1, 1, 'k')
ax = plt.gca()
ax.set_xlabel(r'Predicted (z-scored $\Delta W$)')
ax.set_ylabel(r'True (z-scored $\Delta W$)')
ax.set_title(f'Pooled dev2 | alpha={alpha_fast} | r={r_te:.3f}')

plt.subplot(122)
sns.heatmap(beta_pooled_reshaped, annot=True, fmt='.3f',
            xticklabels=behavior_names_plot, yticklabels=EPOCH_ORDER,
            cmap='coolwarm', center=0)
plt.title(r'Pooled $\beta$ weights (dev2)')
plt.xlabel('Behavioral feature')
plt.ylabel('Trial epoch')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_dev2_pooled.png'),
            dpi=200, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Sign consistency analysis
# ============================================================================
"""
For each beta coefficient (epoch × behavior), compute:
  - fraction of sessions with the same sign as the mean
  - sign-consistency ratio: |mean| / mean(|betas|)
    (1 = all same sign, 0 = perfectly split)
"""

beta_stack = np.array(BETA)  # (n_sessions, n_epochs, n_beh)
n_sess, n_epochs, n_beh = beta_stack.shape

# Sign consistency: fraction of sessions agreeing with mean sign
sign_frac = np.full((n_epochs, n_beh), np.nan)
sign_ratio = np.full((n_epochs, n_beh), np.nan)

for ei in range(n_epochs):
    for bi in range(n_beh):
        vals = beta_stack[:, ei, bi]
        v = vals[np.isfinite(vals)]
        if len(v) < 3:
            continue
        m = np.mean(v)
        if m > 0:
            sign_frac[ei, bi] = np.mean(v > 0)
        else:
            sign_frac[ei, bi] = np.mean(v < 0)
        # ratio: 1 means all same sign, 0 means cancellation
        denom = np.mean(np.abs(v))
        sign_ratio[ei, bi] = np.abs(m) / denom if denom > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Panel A: fraction of sessions with majority sign
ax = axes[0]
im = ax.imshow(sign_frac, cmap='RdYlGn', vmin=0.5, vmax=1.0,
               aspect='auto', interpolation='nearest')
for ei in range(n_epochs):
    for bi in range(n_beh):
        val = sign_frac[ei, bi]
        if np.isfinite(val):
            ax.text(bi, ei, f'{val:.2f}', ha='center', va='center', fontsize=9)
ax.set_xticks(range(n_beh))
ax.set_xticklabels(behavior_names_plot, rotation=30, ha='right')
ax.set_yticks(range(n_epochs))
ax.set_yticklabels(EPOCH_ORDER)
plt.colorbar(im, ax=ax, shrink=0.8, label='Fraction same sign')
ax.set_title(f'Sign consistency (dev2, n={n_sess})\nFraction of sessions with majority sign',
             fontweight='bold')

# Panel B: |mean| / mean(|beta|) ratio
ax = axes[1]
im2 = ax.imshow(sign_ratio, cmap='RdYlGn', vmin=0, vmax=1.0,
                aspect='auto', interpolation='nearest')
for ei in range(n_epochs):
    for bi in range(n_beh):
        val = sign_ratio[ei, bi]
        if np.isfinite(val):
            ax.text(bi, ei, f'{val:.2f}', ha='center', va='center', fontsize=9)
ax.set_xticks(range(n_beh))
ax.set_xticklabels(behavior_names_plot, rotation=30, ha='right')
ax.set_yticks(range(n_epochs))
ax.set_yticklabels(EPOCH_ORDER)
plt.colorbar(im2, ax=ax, shrink=0.8, label='|mean| / mean(|β|)')
ax.set_title(f'Sign ratio (dev2, n={n_sess})\n1=all same sign, 0=fully split',
             fontweight='bold')

fig.suptitle('Beta sign consistency across sessions (dev2)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_dev2_sign_consistency.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# Print summary
print("\nSign consistency summary (dev2):")
print(f"  Overall mean sign fraction: {np.nanmean(sign_frac):.3f}")
print(f"  Overall mean sign ratio:    {np.nanmean(sign_ratio):.3f}")
print(f"  Most consistent: ", end="")
best_idx = np.unravel_index(np.nanargmax(sign_frac), sign_frac.shape)
print(f"{EPOCH_ORDER[best_idx[0]]} × {behavior_names_plot[best_idx[1]]} "
      f"(frac={sign_frac[best_idx]:.2f}, ratio={sign_ratio[best_idx]:.2f})")
print(f"  Least consistent: ", end="")
worst_idx = np.unravel_index(np.nanargmin(sign_frac), sign_frac.shape)
print(f"{EPOCH_ORDER[worst_idx[0]]} × {behavior_names_plot[worst_idx[1]]} "
      f"(frac={sign_frac[worst_idx]:.2f}, ratio={sign_ratio[worst_idx]:.2f})")

# Save numeric results for comparison with dot_prod
np.save(os.path.join(RESULTS_DIR, 'three_factor_dev2_betas.npy'),
        {'beta_stack': beta_stack, 'sign_frac': sign_frac, 'sign_ratio': sign_ratio,
         'epochs': EPOCH_ORDER, 'behaviors': behavior_names_plot,
         'FIT': FIT, 'PVAL': PVAL}, allow_pickle=True)
