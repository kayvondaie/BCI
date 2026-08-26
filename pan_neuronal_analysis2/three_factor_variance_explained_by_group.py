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
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

# Environment toggle: 'local' or 'codeocean'
RUN_ENV = 'codeocean'

if RUN_ENV == 'codeocean':
    DATA_ROOT = r'/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal'
    RESULTS_DIR = '/root/capsule/results'
else:
    DATA_ROOT = r'//allen/aind/scratch/BCI/2p-raw'
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

session_counting._BASE_DIR_OVERRIDE = DATA_ROOT
list_of_dirs = session_counting.counter()

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
mice = ["BCI88","BCI93","BCI107"]
tau_elig = 10
N_BASELINE = 20

# Sliding window parameters
WIN_SIZE = 10
WIN_STEP = 5

# Fitting
fit_type = 'pinv'
n_cv_folds = 5

# Outlier classification
n_sd_threshold = 2

all_results_outlier = []
all_results_rest = []
print(f"Config: win={WIN_SIZE}, step={WIN_STEP}, {fit_type}, {n_cv_folds}-fold CV")
print(f"Outlier threshold: mean + {n_sd_threshold} SD of non-target amp")

#%% ============================================================================
# CELL 3: Main loop — runs separately for outlier and rest neurons
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
            folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
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
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Classify outlier neurons by non-target amp ----
            amp_ep0 = AMP[0]
            amp_masked = amp_ep0.copy()
            amp_masked[stimDist < 30] = np.nan
            mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

            mu = np.nanmean(mean_amp_nontarg)
            sd = np.nanstd(mean_amp_nontarg)
            outlier_threshold = mu + n_sd_threshold * sd
            is_outlier = mean_amp_nontarg > outlier_threshold
            n_outlier = np.sum(is_outlier)
            n_rest = np.sum(~is_outlier & np.isfinite(mean_amp_nontarg))
            print(f"  Outlier: {n_outlier}, Rest: {n_rest} (threshold={outlier_threshold:.4f})")

            # ---- Run analysis for each group ----
            for group_name, group_mask, results_list in [
                ('outlier', is_outlier, all_results_outlier),
                ('rest', ~is_outlier, all_results_rest),
            ]:
                # ---- Pair selection ----
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
                        (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000) &
                        group_mask
                    )[0]
                    if nontarg.size == 0:
                        continue
                    dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                    pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                    pair_nt_list.append(nontarg)

                if len(dw_list) == 0:
                    print(f"    {group_name}: No valid pairs.")
                    continue

                Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
                all_nt = np.concatenate(pair_nt_list)
                n_pairs = len(Y)

                cl_weights = np.zeros((n_pairs, n_neurons))
                offset = 0
                for gi_idx in range(len(dw_list)):
                    n_nt = len(dw_list[gi_idx])
                    cl_arr = pair_cl_list[gi_idx]
                    for qi in range(n_nt):
                        cl_neurons = cl_arr[qi]
                        cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                    offset += n_nt

                # ---- Pre-epoch activity ----
                F_nan = F.copy()
                F_nan[np.isnan(F_nan)] = 0
                ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
                epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)

                bl_trials = np.arange(min(N_BASELINE, trl))
                bl_mean = np.nanmean(epoch_act[:, bl_trials], axis=1)

                # ---- Sliding window CC ----
                r_pre = cl_weights @ epoch_act
                r_post_dev = epoch_act[all_nt, :] - bl_mean[all_nt, np.newaxis]
                cc_trial = r_pre * r_post_dev

                win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
                n_wins = len(win_starts)

                if n_wins < 3:
                    print(f"    {group_name}: Only {n_wins} windows, skipping.")
                    continue

                X = np.zeros((n_pairs, n_wins))
                for wi, ws in enumerate(win_starts):
                    X[:, wi] = np.sum(cc_trial[:, ws:ws+WIN_SIZE], axis=1)

                mu_x = X.mean(axis=0)
                sig_x = X.std(axis=0)
                sig_x[sig_x == 0] = 1.0
                X = (X - mu_x) / sig_x

                # ---- Cross-validated fitting ----
                cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                corr_test_folds = []
                corr_train_folds = []
                p_test_folds = []
                Y_test_all = []
                Y_pred_all = []
                beta_first = None

                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    Y_train, Y_test = Y[train_idx], Y[test_idx]

                    mu_y, sig_y = Y_train.mean(), Y_train.std()
                    if sig_y == 0 or not np.isfinite(sig_y):
                        sig_y = 1.0
                    Y_train_z = (Y_train - mu_y) / sig_y
                    Y_test_z = (Y_test - mu_y) / sig_y

                    if fit_type == 'pinv':
                        beta = np.linalg.pinv(X_train) @ Y_train_z
                    elif fit_type == 'ridge':
                        from sklearn.linear_model import RidgeCV
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import Pipeline
                        ridge = RidgeCV(alphas=np.logspace(-10, -4, 10),
                                        fit_intercept=True)
                        pipe = Pipeline([('scaler', StandardScaler()),
                                         ('ridge', ridge)])
                        pipe.fit(X_train, Y_train_z)
                        beta = pipe.named_steps['ridge'].coef_

                    Y_train_pred = X_train @ beta
                    Y_test_pred = X_test @ beta

                    if np.std(Y_train_pred) > 0:
                        r_tr, _ = pearsonr(Y_train_pred, Y_train_z)
                    else:
                        r_tr = 0.0

                    if np.std(Y_test_pred) > 0:
                        r_te, p_te = pearsonr(Y_test_pred, Y_test_z)
                    else:
                        r_te, p_te = 0.0, 1.0

                    corr_train_folds.append(r_tr)
                    corr_test_folds.append(r_te)
                    p_test_folds.append(p_te)
                    Y_test_all.append(Y_test_z)
                    Y_pred_all.append(Y_test_pred)

                    if fold_idx == 0:
                        beta_first = beta.copy()

                r_test_mean = np.mean(corr_test_folds)
                r_train_mean = np.mean(corr_train_folds)
                p_test_combined = np.exp(np.mean(np.log(
                    np.clip(p_test_folds, 1e-300, 1.0))))

                result = {
                    'mouse': mouse,
                    'session': session,
                    'group': group_name,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
                    'r_test': r_test_mean,
                    'r_train': r_train_mean,
                    'p_test': p_test_combined,
                    'r_test_folds': corr_test_folds,
                    'betas': beta_first,
                    'Y_test_all': np.concatenate(Y_test_all),
                    'Y_pred_all': np.concatenate(Y_pred_all),
                }
                results_list.append(result)

                sig = '*' if p_test_combined < 0.05 else ''
                print(f"    {group_name}: {n_pairs} pairs, {n_wins} wins | "
                      f"train r={r_train_mean:.3f}, test r={r_test_mean:.3f}, "
                      f"p={p_test_combined:.4f} {sig}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_results_outlier)} outlier sessions, "
      f"{len(all_results_rest)} rest sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'three_factor_ve_by_group.npy'),
        {'outlier': all_results_outlier, 'rest': all_results_rest},
        allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
_saved = np.load(
    os.path.join(RESULTS_DIR, 'three_factor_ve_by_group.npy'),
    allow_pickle=True).item()
all_results_outlier = _saved['outlier']
all_results_rest = _saved['rest']
print(f"Loaded {len(all_results_outlier)} outlier, {len(all_results_rest)} rest sessions")

#%% ============================================================================
# CELL 6: Summary figure — side by side comparison
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, (label, all_results, color) in enumerate([
    ('Rest', all_results_rest, '#2c3e50'),
    ('Outlier', all_results_outlier, '#c0392b'),
]):
    n_s = len(all_results)
    if n_s == 0:
        for ax in axes[row]:
            ax.set_visible(False)
        continue

    r_test = np.array([s['r_test'] for s in all_results])
    r_train = np.array([s['r_train'] for s in all_results])
    p_test = np.array([s['p_test'] for s in all_results])
    mice_arr = np.array([s['mouse'] for s in all_results])
    n_sig = np.sum(p_test < 0.05)

    # --- Panel A: Test r per session ---
    ax = axes[row, 0]
    mouse_list = sorted(set(mice_arr))
    cmap = plt.cm.Set2
    mouse_colors = {m: cmap(i) for i, m in enumerate(mouse_list)}

    for i in range(n_s):
        ec = 'k' if p_test[i] < 0.05 else 'none'
        lw = 1.5 if p_test[i] < 0.05 else 0
        ax.bar(i, r_test[i], color=mouse_colors[mice_arr[i]], alpha=0.7,
               edgecolor=ec, linewidth=lw)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axhline(np.median(r_test), color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Session')
    ax.set_ylabel('Test r (5-fold CV)')
    ax.set_title(f'{label}: {n_sig}/{n_s} p<0.05', fontweight='bold')
    if row == 0:
        for m in mouse_list:
            ax.scatter([], [], color=mouse_colors[m], label=m, s=40)
        ax.legend(loc='best', fontsize=7, ncol=2)

    # --- Panel B: Pooled binned prediction ---
    ax = axes[row, 1]
    Y_pred_pool = np.concatenate([s['Y_pred_all'] for s in all_results])
    Y_test_pool = np.concatenate([s['Y_test_all'] for s in all_results])

    n_bins_plot = 5
    edges = np.percentile(Y_pred_pool, np.linspace(0, 100, n_bins_plot + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins_plot):
        if bi < n_bins_plot - 1:
            mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool < edges[bi + 1])
        else:
            mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool <= edges[bi + 1])
        if np.sum(mask) < 3:
            continue
        bx.append(np.mean(Y_pred_pool[mask]))
        by.append(np.mean(Y_test_pool[mask]))
        be.append(np.std(Y_test_pool[mask]) / np.sqrt(np.sum(mask)))

    ax.errorbar(bx, by, yerr=be, fmt='o-', color=color, capsize=5,
                linewidth=2, markersize=7)
    ax.axhline(0, color='k', ls='-', alpha=0.2)
    r_pool, p_pool = pearsonr(Y_pred_pool, Y_test_pool)
    ax.set_xlabel('Predicted dW (z)')
    ax.set_ylabel('Actual dW (z)')
    ax.set_title(f'{label}: r={r_pool:.3f}, p={p_pool:.2e}', fontweight='bold')

    # --- Panel C: Train vs test r ---
    ax = axes[row, 2]
    ax.scatter(r_train, r_test, c=color, s=25, alpha=0.6, zorder=3)
    lim = max(np.max(np.abs(r_train)), np.max(np.abs(r_test))) * 1.1
    lim = max(lim, 0.1)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('Train r')
    ax.set_ylabel('Test r')
    ax.set_title(f'{label}: overfitting check', fontweight='bold')

    try:
        _, p_wilcox = wilcoxon(r_test)
    except Exception:
        p_wilcox = 1.0
    print(f"{label}: median r_test={np.median(r_test):.4f}, "
          f"mean={np.mean(r_test):.4f}, Wilcoxon p={p_wilcox:.4f}, "
          f"sig={n_sig}/{n_s}")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_variance_explained_by_group.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")
