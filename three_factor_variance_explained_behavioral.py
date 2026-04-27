#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Variant of three_factor_variance_explained.py.

Original: dW(pair) = sum_w beta_w * CC_w(pair)  — free betas fit by pinv
This:     dW(pair) = sum_w behavior_w * CC_w(pair) — betas replaced by
          behavioral variables (RPE, hit, RT, etc.). Zero free parameters.

For each behavioral variable, correlate the predicted dW with actual dW
across pairs. One Spearman r per session per variable.

Two CC modes: dot_prod and dev2.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

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
tau_elig = 10
N_BASELINE = 20

# Sliding window
WIN_SIZE = 10
WIN_STEP = 5

# Epochs to test
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]

# CC modes
CC_MODES = ['dot_prod', 'dev2']

# Behavioral variables to use as beta replacements
BEH_NAMES = ['RPE', 'hit', '10-rt', 'hit_RPE', 'ones']
BEH_LABELS = ['RPE', 'Hit rate', '10 - RT', 'Hit RPE', 'Ones (fitted HI)']

all_sessions = []
print(f"Config: win={WIN_SIZE}, step={WIN_STEP}")
print(f"CC modes: {CC_MODES}")
print(f"Behavioral variables: {BEH_NAMES}")
print(f"Epochs: {EPOCH_ORDER}")

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
import csv
from sklearn.model_selection import KFold

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
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Behavioral variables (per trial) ----
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

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
                    (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000)
                )[0]
                if nontarg.size == 0:
                    continue
                dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)

            if len(dw_list) == 0:
                print("  No valid pairs.")
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

            # ---- Epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]

            epoch_act = {}
            for ep in ['pre', 'go_cue']:
                if ep == 'pre':
                    t0, t1 = ts_pre[0], ts_pre[-1]
                else:
                    t0, t1 = ts_go[0], ts_go[-1]
                epoch_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)

            epoch_act['late'] = np.zeros((n_neurons, trl))
            epoch_act['reward'] = np.zeros((n_neurons, trl))
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)

            # Baseline for dev2
            bl_trials = np.arange(min(N_BASELINE, trl))

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 3:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # Per-window behavioral averages
            win_beh = {}
            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                win_beh.setdefault('RPE', []).append(np.nanmean(rt_rpe[trial_idx]))
                win_beh.setdefault('hit', []).append(np.nanmean(hit[trial_idx]))
                win_beh.setdefault('10-rt', []).append(np.nanmean(10 - rt_filled[trial_idx]))
                win_beh.setdefault('hit_RPE', []).append(np.nanmean(hit_rpe[trial_idx]))
                win_beh.setdefault('ones', []).append(1.0)

            for k in win_beh:
                win_beh[k] = np.array(win_beh[k])

            # ---- Compute CC per window and predict dW ----
            sess_result = {
                'mouse': mouse,
                'session': session,
                'n_pairs': n_pairs,
                'n_trials': trl,
                'n_windows': n_wins,
                'hit_rate': np.nanmean(hit),
            }

            for cc_mode in CC_MODES:
                for ep in EPOCH_ORDER:
                    act = epoch_act[ep]  # (n_neurons, trl)

                    # Pre-synaptic: cl-weighted
                    r_pre = cl_weights @ act  # (n_pairs, trl)
                    # Post-synaptic
                    r_post = act[all_nt, :]   # (n_pairs, trl)

                    if cc_mode == 'dev2':
                        bl_mean = np.nanmean(act[:, bl_trials], axis=1)
                        r_post = r_post - bl_mean[all_nt, np.newaxis]

                    # Per-trial CC: (n_pairs, trl)
                    cc_trial = r_pre * r_post

                    # Per-window CC: sum within each window -> (n_pairs, n_wins)
                    CC_win = np.zeros((n_pairs, n_wins))
                    for wi, ws in enumerate(win_starts):
                        CC_win[:, wi] = np.sum(cc_trial[:, ws:ws+WIN_SIZE], axis=1)

                    # For each behavioral variable, compute:
                    #   predicted_dW_ij = sum_w behavior_w * CC_w_ij
                    # Then correlate with actual dW
                    for bname in BEH_NAMES:
                        beh_w = win_beh[bname]  # (n_wins,)

                        # Zero-parameter prediction
                        pred_dw = CC_win @ beh_w  # (n_pairs,)

                        mask = np.isfinite(pred_dw) & np.isfinite(Y)
                        if np.sum(mask) > 5 and np.std(pred_dw[mask]) > 0:
                            r_val, p_val = spearmanr(pred_dw[mask], Y[mask])
                        else:
                            r_val, p_val = np.nan, np.nan

                        key = f"{cc_mode}_{ep}_{bname}"
                        sess_result[f'{key}_r'] = r_val
                        sess_result[f'{key}_p'] = p_val

                        # Store per-pair vectors for pooled analysis
                        sess_result[f'{key}_pred'] = pred_dw

                    # Also store the fitted-beta result for comparison
                    # dW = sum_w beta_w * CC_w, fit by pinv, 5-fold CV
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    r_test_folds = []
                    for train_idx, test_idx in cv.split(CC_win.T):
                        # train_idx/test_idx index into pairs
                        # Wait — KFold splits along axis 0 of CC_win.T which is windows
                        # We need to split across pairs, not windows
                        pass

                    # Redo: split across pairs
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    X = CC_win.copy()  # (n_pairs, n_wins)
                    mu_x = X.mean(axis=0)
                    sig_x = X.std(axis=0)
                    sig_x[sig_x == 0] = 1.0
                    X_z = (X - mu_x) / sig_x

                    r_test_folds = []
                    for train_idx, test_idx in cv.split(X_z):
                        X_tr, X_te = X_z[train_idx], X_z[test_idx]
                        Y_tr, Y_te = Y[train_idx], Y[test_idx]
                        mu_y, sig_y = Y_tr.mean(), Y_tr.std()
                        if sig_y == 0 or not np.isfinite(sig_y):
                            sig_y = 1.0
                        Y_tr_z = (Y_tr - mu_y) / sig_y
                        Y_te_z = (Y_te - mu_y) / sig_y
                        beta = np.linalg.pinv(X_tr) @ Y_tr_z
                        Y_pred = X_te @ beta
                        if np.std(Y_pred) > 0:
                            from scipy.stats import pearsonr
                            r_te, _ = pearsonr(Y_pred, Y_te_z)
                        else:
                            r_te = 0.0
                        r_test_folds.append(r_te)

                    key_fit = f"{cc_mode}_{ep}_fitted"
                    sess_result[f'{key_fit}_r'] = np.mean(r_test_folds)

            sess_result['Y'] = Y  # actual dW per pair
            all_sessions.append(sess_result)
            print(f"  {n_pairs} pairs, {n_wins} wins")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_sessions)} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'three_factor_variance_explained_behavioral.npy'),
        all_sessions, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_sessions = np.load(
    os.path.join(RESULTS_DIR, 'three_factor_variance_explained_behavioral.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_sessions)} sessions")

#%% ============================================================================
# CELL 6: Summary — strip plots per epoch × CC mode × behavioral variable
# ============================================================================
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
CC_MODES = ['dot_prod', 'dev2']
BEH_NAMES = ['RPE', 'hit', '10-rt', 'hit_RPE', 'ones']
BEH_LABELS = ['RPE', 'Hit', '10-RT', 'Hit RPE', 'Ones']

n_epochs = len(EPOCH_ORDER)
n_beh = len(BEH_NAMES)
n_modes = len(CC_MODES)

fig, axes = plt.subplots(n_epochs, n_beh * n_modes, figsize=(24, 12))

for ei, ep in enumerate(EPOCH_ORDER):
    for mi, cc_mode in enumerate(CC_MODES):
        for bi, (bname, blabel) in enumerate(zip(BEH_NAMES, BEH_LABELS)):
            ax_idx = mi * n_beh + bi
            ax = axes[ei, ax_idx]

            key = f"{cc_mode}_{ep}_{bname}_r"
            vals = np.array([s[key] for s in all_sessions])
            v = vals[np.isfinite(vals)]

            if len(v) >= 3:
                _, p = wilcoxon(v)
            else:
                p = np.nan

            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(v))
            color = 'steelblue' if cc_mode == 'dot_prod' else 'coral'
            ax.scatter(jitter, v, s=20, alpha=0.5, edgecolors='k',
                       linewidths=0.2, zorder=3, color=color)
            m = np.mean(v) if len(v) > 0 else 0
            sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar(0, m, yerr=sem, fmt='D', color='k', markersize=5,
                        capsize=3, zorder=4)
            ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Auto-scale
            if len(v) > 0:
                ypad = max(np.max(np.abs(v)) * 1.3, 0.01)
                ax.set_ylim(-ypad, ypad)

            sig = ''
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'

            ax.set_title(f'{cc_mode}\n{blabel} {sig}\np={p:.3f}',
                         fontsize=8)
            if ax_idx == 0:
                ax.set_ylabel(f'{ep}\nSpearman ρ')

fig.suptitle('dW = Σ_w behavior_w × CC_w  (zero free parameters)\n'
             'Spearman ρ(predicted dW, actual dW) per session',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_behavioral_summary.png'),
            dpi=200, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Compact heatmap — mean rho and significance across conditions
# ============================================================================
"""
Heatmap: rows = epochs, columns = behavioral variables.
Two side-by-side heatmaps for dot_prod and dev2.
Cell value = mean Spearman rho, stars for significance.
"""
fig, axes = plt.subplots(1, n_modes + 1, figsize=(16, 4),
                         gridspec_kw={'width_ratios': [1, 1, 0.3]})

for mi, cc_mode in enumerate(CC_MODES):
    ax = axes[mi]
    rho_mat = np.full((n_epochs, n_beh), np.nan)
    p_mat = np.full((n_epochs, n_beh), np.nan)

    for ei, ep in enumerate(EPOCH_ORDER):
        for bi, bname in enumerate(BEH_NAMES):
            key = f"{cc_mode}_{ep}_{bname}_r"
            vals = np.array([s[key] for s in all_sessions])
            v = vals[np.isfinite(vals)]
            rho_mat[ei, bi] = np.mean(v) if len(v) > 0 else np.nan
            if len(v) >= 3:
                _, p_mat[ei, bi] = wilcoxon(v)

    vmax = np.nanmax(np.abs(rho_mat))
    vmax = max(vmax, 0.005)
    im = ax.imshow(rho_mat, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    for ei in range(n_epochs):
        for bi in range(n_beh):
            val = rho_mat[ei, bi]
            p = p_mat[ei, bi]
            if np.isnan(val):
                continue
            sig = ''
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
            txt = f'{val:+.4f}'
            if sig:
                txt += f'\n{sig}'
            ax.text(bi, ei, txt, ha='center', va='center',
                    fontsize=8, fontweight='bold' if sig else 'normal')

    ax.set_xticks(range(n_beh))
    ax.set_xticklabels(BEH_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_epochs))
    ax.set_yticklabels(EPOCH_ORDER)
    ax.set_title(f'{cc_mode}', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean ρ')

# Also show the fitted-beta test r for comparison
ax = axes[2]
fitted_mat = np.full((n_epochs, n_modes), np.nan)
for mi, cc_mode in enumerate(CC_MODES):
    for ei, ep in enumerate(EPOCH_ORDER):
        key = f"{cc_mode}_{ep}_fitted_r"
        vals = np.array([s[key] for s in all_sessions])
        v = vals[np.isfinite(vals)]
        fitted_mat[ei, mi] = np.mean(v) if len(v) > 0 else np.nan

vmax_f = max(np.nanmax(np.abs(fitted_mat)), 0.005)
im2 = ax.imshow(fitted_mat, cmap='coolwarm', vmin=-vmax_f, vmax=vmax_f,
                aspect='auto', interpolation='nearest')
for ei in range(n_epochs):
    for mi in range(n_modes):
        val = fitted_mat[ei, mi]
        if np.isfinite(val):
            ax.text(mi, ei, f'{val:+.4f}', ha='center', va='center', fontsize=8)
ax.set_xticks(range(n_modes))
ax.set_xticklabels(CC_MODES, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(n_epochs))
ax.set_yticklabels(EPOCH_ORDER)
ax.set_title('Fitted β\n(CV test r)', fontweight='bold')
plt.colorbar(im2, ax=ax, shrink=0.8, label='Mean test r')

fig.suptitle('Behavioral weighting vs fitted betas\n'
             'dW = Σ_w behavior_w × CC_w',
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_behavioral_heatmap.png'),
            dpi=200, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Pooled analysis — dev2 pre epoch, RPE vs cumulative CC
# ============================================================================
"""
Focused test: pool all pairs across sessions.

Model A (three-factor): predicted_dW_ij = sum_w RPE_w * CC_w_ij  (0 params)
Model B (Hebbian only): predicted_dW_ij = sum_w CC_w_ij           (0 params)
Model C (single beta):  dW = beta * sum_w RPE_w * CC_w_ij         (1 param, CV)
Model D (two predictors): dW = b1 * X_RPE + b2 * X_cum            (2 params, CV)

CC = dev2, pre epoch.
"""
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# ---- Pool pairs across sessions ----
Y_pool = []
X_rpe_pool = []
X_cum_pool = []

for s in all_sessions:
    y = s['Y']
    x_rpe = s.get('dev2_pre_RPE_pred', None)
    x_cum = s.get('dev2_pre_ones_pred', None)
    if x_rpe is None or x_cum is None:
        continue
    mask = np.isfinite(y) & np.isfinite(x_rpe) & np.isfinite(x_cum)
    y_s, x_rpe_s, x_cum_s = y[mask], x_rpe[mask], x_cum[mask]

    # Z-score within session before pooling
    def _z(v):
        mu, sig = np.mean(v), np.std(v)
        return (v - mu) / sig if sig > 0 else v - mu
    Y_pool.append(_z(y_s))
    X_rpe_pool.append(_z(x_rpe_s))
    X_cum_pool.append(_z(x_cum_s))

Y_pool = np.concatenate(Y_pool)
X_rpe_pool = np.concatenate(X_rpe_pool)
X_cum_pool = np.concatenate(X_cum_pool)

n_total = len(Y_pool)
print(f"Pooled: {n_total} pairs from {len(all_sessions)} sessions")

# Already z-scored per session; use pooled values directly
Y_z = Y_pool
X_rpe_z = X_rpe_pool
X_cum_z = X_cum_pool

# ---- Model A: zero-parameter correlation (RPE-weighted) ----
r_A, p_A = spearmanr(X_rpe_pool, Y_pool)
print(f"Model A (RPE × CC):  Spearman r = {r_A:.4f}, p = {p_A:.2e}")

# ---- Model B: zero-parameter correlation (cumulative) ----
r_B, p_B = spearmanr(X_cum_pool, Y_pool)
print(f"Model B (Σ CC):      Spearman r = {r_B:.4f}, p = {p_B:.2e}")

# ---- Model C: single beta on RPE×CC, 5-fold CV ----
cv = KFold(n_splits=5, shuffle=True, random_state=42)
r_C_folds = []
Y_pred_C_all, Y_true_C_all = [], []

for train_idx, test_idx in cv.split(X_rpe_z):
    X_tr = X_rpe_z[train_idx, np.newaxis]
    X_te = X_rpe_z[test_idx, np.newaxis]
    Y_tr, Y_te = Y_z[train_idx], Y_z[test_idx]

    beta = np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0]
    Y_pred = X_te @ beta
    if np.std(Y_pred) > 0:
        r_fold, _ = pearsonr(Y_pred, Y_te)
    else:
        r_fold = 0.0
    r_C_folds.append(r_fold)
    Y_pred_C_all.append(Y_pred)
    Y_true_C_all.append(Y_te)

Y_pred_C_all = np.concatenate(Y_pred_C_all)
Y_true_C_all = np.concatenate(Y_true_C_all)
r_C = np.mean(r_C_folds)
print(f"Model C (β × RPE×CC): CV test r = {r_C:.4f}")

# ---- Model D: two predictors (RPE×CC + cumCC), 5-fold CV ----
X_both = np.column_stack([X_rpe_z, X_cum_z])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
r_D_folds = []
Y_pred_D_all, Y_true_D_all = [], []
beta_D_all = []

for train_idx, test_idx in cv.split(X_both):
    X_tr, X_te = X_both[train_idx], X_both[test_idx]
    Y_tr, Y_te = Y_z[train_idx], Y_z[test_idx]

    beta = np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0]
    beta_D_all.append(beta)
    Y_pred = X_te @ beta
    if np.std(Y_pred) > 0:
        r_fold, _ = pearsonr(Y_pred, Y_te)
    else:
        r_fold = 0.0
    r_D_folds.append(r_fold)
    Y_pred_D_all.append(Y_pred)
    Y_true_D_all.append(Y_te)

Y_pred_D_all = np.concatenate(Y_pred_D_all)
Y_true_D_all = np.concatenate(Y_true_D_all)
r_D = np.mean(r_D_folds)
beta_D_mean = np.mean(beta_D_all, axis=0)
print(f"Model D (b1*RPE×CC + b2*ΣCC): CV test r = {r_D:.4f}")
print(f"  β_RPE = {beta_D_mean[0]:.4f}, β_cum = {beta_D_mean[1]:.4f}")

# ---- Figure ----
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

def binned_plot(ax, x, y, n_bins=8, color='k'):
    """Percentile-binned mean ± SEM."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            mask = (x >= edges[bi]) & (x < edges[bi + 1])
        else:
            mask = (x >= edges[bi]) & (x <= edges[bi + 1])
        if np.sum(mask) < 3:
            continue
        bx.append(np.mean(x[mask]))
        by.append(np.mean(y[mask]))
        be.append(np.std(y[mask]) / np.sqrt(np.sum(mask)))
    ax.errorbar(bx, by, yerr=be, fmt='o-', color=color, capsize=4,
                linewidth=2, markersize=6)

# Panel A: RPE×CC vs dW (zero parameter)
ax = axes[0]
binned_plot(ax, X_rpe_pool, Y_pool, color='#2c3e50')
ax.axhline(0, color='k', ls='-', alpha=0.2)
ax.set_xlabel('Σ RPE × CC (dev2, pre)')
ax.set_ylabel('ΔW')
ax.set_title(f'Model A: RPE × CC\nρ = {r_A:.4f}, p = {p_A:.2e}',
             fontweight='bold')

# Panel B: Cumulative CC vs dW (zero parameter)
ax = axes[1]
binned_plot(ax, X_cum_pool, Y_pool, color='coral')
ax.axhline(0, color='k', ls='-', alpha=0.2)
ax.set_xlabel('Σ CC (dev2, pre)')
ax.set_ylabel('ΔW')
ax.set_title(f'Model B: Σ CC\nρ = {r_B:.4f}, p = {p_B:.2e}',
             fontweight='bold')

# Panel C: Single-beta predicted vs actual (CV)
ax = axes[2]
binned_plot(ax, Y_pred_C_all, Y_true_C_all, color='#2c3e50')
ax.axhline(0, color='k', ls='-', alpha=0.2)
ax.set_xlabel('Predicted ΔW (β × RPE×CC)')
ax.set_ylabel('Actual ΔW (z)')
ax.set_title(f'Model C: β × RPE×CC\nCV r = {r_C:.4f}',
             fontweight='bold')

# Panel D: Two-predictor model (CV)
ax = axes[3]
binned_plot(ax, Y_pred_D_all, Y_true_D_all, color='darkgreen')
ax.axhline(0, color='k', ls='-', alpha=0.2)
ax.set_xlabel('Predicted ΔW (b₁RPE×CC + b₂ΣCC)')
ax.set_ylabel('Actual ΔW (z)')
ax.set_title(f'Model D: b₁RPE×CC + b₂ΣCC\nCV r = {r_D:.4f}\n'
             f'β_RPE={beta_D_mean[0]:.3f}, β_cum={beta_D_mean[1]:.3f}',
             fontweight='bold')

fig.suptitle(f'Pooled three-factor test (dev2, pre epoch, n={n_total} pairs)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'three_factor_pooled_focused.png'),
            dpi=200, bbox_inches='tight')
plt.show()
