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
import plotting_functions as pf

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
PANEL_DIR = (r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written'
             r'\3-factor learning paper\claude code 032226'
             r'\meta_analysis_results\panels')
os.makedirs(PANEL_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',
})

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
tau_elig = 10
N_BASELINE = 20

WIN_SIZE = 10
WIN_STEP = 5

fit_type = 'pinv'
n_cv_folds = 5

results_3f = []  # 3-factor: fitted betas per window (CV)
results_2f = []  # 2-factor: weight = 1 (no fitting)

print(f"Config: win={WIN_SIZE}, step={WIN_STEP}, {fit_type}, {n_cv_folds}-fold CV")

#%% ============================================================================
# CELL 3: Main loop — compute both 2-factor and 3-factor per session
# ============================================================================

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

            # ---- Pre-epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)

            bl_trials = np.arange(min(N_BASELINE, trl))
            bl_mean = np.nanmean(epoch_act[:, bl_trials], axis=1)

            # ---- Per-trial CC (dev2, pre epoch) ----
            r_pre = cl_weights @ epoch_act
            r_post_dev = epoch_act[all_nt, :] - bl_mean[all_nt, np.newaxis]
            cc_trial = r_pre * r_post_dev  # (n_pairs, trl)

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 3:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # Design matrix: X = (n_pairs, n_wins)
            X = np.zeros((n_pairs, n_wins))
            for wi, ws in enumerate(win_starts):
                X[:, wi] = np.sum(cc_trial[:, ws:ws+WIN_SIZE], axis=1)

            # ============================================================
            # 2-FACTOR: total CC per pair, weight = 1, no fitting
            # ============================================================
            cc_total = np.sum(cc_trial, axis=1)  # (n_pairs,)
            # z-score for comparable r values
            mu_cc = np.mean(cc_total)
            sig_cc = np.std(cc_total)
            if sig_cc == 0:
                sig_cc = 1.0
            cc_total_z = (cc_total - mu_cc) / sig_cc
            mu_y = np.mean(Y)
            sig_y = np.std(Y)
            if sig_y == 0:
                sig_y = 1.0
            Y_z = (Y - mu_y) / sig_y

            r_2f, p_2f = pearsonr(cc_total_z, Y_z)
            results_2f.append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_pairs, 'n_trials': trl, 'n_windows': n_wins,
                'r_test': r_2f, 'p_test': p_2f,
                'Y_pred_all': cc_total_z, 'Y_test_all': Y_z,
            })

            # ============================================================
            # 3-FACTOR: fitted betas per window, cross-validated
            # ============================================================
            # Z-score X columns
            mu_x = X.mean(axis=0)
            sig_x = X.std(axis=0)
            sig_x[sig_x == 0] = 1.0
            X_z = (X - mu_x) / sig_x

            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            corr_test_folds = []
            corr_train_folds = []
            p_test_folds = []
            Y_test_all = []
            Y_pred_all = []

            for train_idx, test_idx in cv.split(X_z):
                X_train, X_test = X_z[train_idx], X_z[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                mu_yt, sig_yt = Y_train.mean(), Y_train.std()
                if sig_yt == 0 or not np.isfinite(sig_yt):
                    sig_yt = 1.0
                Y_train_z = (Y_train - mu_yt) / sig_yt
                Y_test_z = (Y_test - mu_yt) / sig_yt

                beta = np.linalg.pinv(X_train) @ Y_train_z
                Y_train_pred = X_train @ beta
                Y_test_pred = X_test @ beta

                r_tr = pearsonr(Y_train_pred, Y_train_z)[0] if np.std(Y_train_pred) > 0 else 0.0
                if np.std(Y_test_pred) > 0:
                    r_te, p_te = pearsonr(Y_test_pred, Y_test_z)
                else:
                    r_te, p_te = 0.0, 1.0

                corr_train_folds.append(r_tr)
                corr_test_folds.append(r_te)
                p_test_folds.append(p_te)
                Y_test_all.append(Y_test_z)
                Y_pred_all.append(Y_test_pred)

            r_test_mean = np.mean(corr_test_folds)
            r_train_mean = np.mean(corr_train_folds)

            # p-value from pooled out-of-fold predictions (not geometric mean of folds)
            Yt_cat = np.concatenate(Y_test_all)
            Yp_cat = np.concatenate(Y_pred_all)
            if np.std(Yp_cat) > 0:
                r_pooled, p_pooled = pearsonr(Yp_cat, Yt_cat)
            else:
                r_pooled, p_pooled = 0.0, 1.0

            results_3f.append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_pairs, 'n_trials': trl, 'n_windows': n_wins,
                'r_test': r_pooled, 'r_train': r_train_mean,
                'p_test': p_pooled,
                'Y_test_all': Yt_cat,
                'Y_pred_all': Yp_cat,
            })

            print(f"  {n_pairs} pairs, {n_wins} wins | "
                  f"2f r={r_2f:.3f} (p={p_2f:.4f}), "
                  f"3f test r={r_pooled:.3f} (p={p_pooled:.4f})")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(results_3f)} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'variance_explained_2f_3f.npy'),
        {'2f': results_2f, '3f': results_3f}, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
_loaded = np.load(os.path.join(RESULTS_DIR, 'variance_explained_2f_3f.npy'),
                  allow_pickle=True).item()
results_2f = _loaded['2f']
results_3f = _loaded['3f']
print(f"Loaded {len(results_3f)} sessions")

#%% ============================================================================
# CELL 6: Figure — 3-factor (fitted betas)
# ============================================================================
FIG_W_MM = 140
FIG_H_MM = 55
FIG_W = FIG_W_MM / 25.4
FIG_H = FIG_H_MM / 25.4

# Shared arrays used by Cells 6-9
n_s = len(results_3f)
mice_arr = np.array([s['mouse'] for s in results_3f])
mouse_list = sorted(set(mice_arr))
cmap = plt.cm.Set2
mouse_colors = {m: cmap(i) for i, m in enumerate(mouse_list)}

# Per-session r and p from spearmanr(Y_pred, Y_test)
from scipy.stats import spearmanr
r_test_2f = np.zeros(n_s)
p_test_2f = np.zeros(n_s)
r_test_3f = np.zeros(n_s)
p_test_3f = np.zeros(n_s)
for i in range(n_s):
    r_test_2f[i], p_test_2f[i] = spearmanr(results_2f[i]['Y_pred_all'],
                                             results_2f[i]['Y_test_all'])
    r_test_3f[i], p_test_3f[i] = spearmanr(results_3f[i]['Y_pred_all'],
                                             results_3f[i]['Y_test_all'])

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.85,
                                      'wspace': 0.40})

# --- Panel A: test r per session ---
ax = axes[0]
for i in range(n_s):
    ec = 'k' if p_test_3f[i] < 0.05 else 'none'
    lw = 1.0 if p_test_3f[i] < 0.05 else 0
    ax.bar(i, r_test_3f[i], color=mouse_colors[mice_arr[i]], alpha=0.7,
           edgecolor=ec, linewidth=lw)
ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.5)
ax.axhline(np.median(r_test_3f), color='k', ls='--', alpha=0.5, linewidth=0.5)
ax.set_xlabel('Session')
ax.set_ylabel('Test r (5-fold CV)')
n_sig = np.sum(p_test_3f < 0.05)
try:
    _, p_wilcox = wilcoxon(r_test_3f)
except Exception:
    p_wilcox = 1.0
ax.set_title(f'3-factor: dW ~ CC_bins\n{n_sig}/{n_s} sig, Wilcoxon p={p_wilcox:.4f}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: pooled predicted vs actual ---
ax = axes[1]
Y_pred_pool = np.concatenate([s['Y_pred_all'] for s in results_3f])
Y_test_pool = np.concatenate([s['Y_test_all'] for s in results_3f])
pf.mean_bin_plot(Y_pred_pool, Y_test_pool, col=5, color='k')
r_pool, p_pool = pearsonr(Y_pred_pool, Y_test_pool)
ax.set_xlabel('Predicted $\\Delta W$ (z)')
ax.set_ylabel('Actual $\\Delta W$ (z)')
ax.set_title(f'Pooled: r={r_pool:.3f}, p={p_pool:.2e}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname = 'three_factor_variance_explained'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 7: Figure — 2-factor (weight = 1, no fitting)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.85,
                                      'wspace': 0.40})

# --- Panel A: r per session ---
ax = axes[0]
for i in range(n_s):
    ec = 'k' if p_test_2f[i] < 0.05 else 'none'
    lw = 1.0 if p_test_2f[i] < 0.05 else 0
    ax.bar(i, r_test_2f[i], color=mouse_colors[mice_arr[i]], alpha=0.7,
           edgecolor=ec, linewidth=lw)
ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.5)
ax.axhline(np.median(r_test_2f), color='k', ls='--', alpha=0.5, linewidth=0.5)
ax.set_xlabel('Session')
ax.set_ylabel('r (total CC vs dW)')
n_sig_2f = np.sum(p_test_2f < 0.05)
try:
    _, p_wilcox_2f = wilcoxon(r_test_2f)
except Exception:
    p_wilcox_2f = 1.0
ax.set_title(f'2-factor: dW ~ total CC\n{n_sig_2f}/{n_s} sig, Wilcoxon p={p_wilcox_2f:.4f}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: pooled CC vs dW ---
ax = axes[1]
Y_pred_2f_pool = np.concatenate([s['Y_pred_all'] for s in results_2f])
Y_test_2f_pool = np.concatenate([s['Y_test_all'] for s in results_2f])
pf.mean_bin_plot(Y_pred_2f_pool, Y_test_2f_pool, col=5, color='k')
r_pool_2f, p_pool_2f = pearsonr(Y_pred_2f_pool, Y_test_2f_pool)
ax.set_xlabel('Total CC (z)')
ax.set_ylabel('$\\Delta W$ (z)')
ax.set_title(f'Pooled: r={r_pool_2f:.3f}, p={p_pool_2f:.2e}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname = 'two_factor_variance_explained'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 8: Figure — 2-factor vs 3-factor comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.85,
                                      'wspace': 0.40})

# --- Panel A: paired scatter, 2f vs 3f test r ---
ax = axes[0]
ax.scatter(r_test_2f, r_test_3f, s=20, color='k', alpha=0.6, edgecolor='none')
lims = [min(np.min(r_test_2f), np.min(r_test_3f)) - 0.02,
        max(np.max(r_test_2f), np.max(r_test_3f)) + 0.02]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
ax.set_xlabel('2-factor r (total CC)')
ax.set_ylabel('3-factor r (fitted betas, CV)')
n_above = np.sum(r_test_3f > r_test_2f)
try:
    _, p_paired = wilcoxon(r_test_3f - r_test_2f)
except Exception:
    p_paired = 1.0
ax.set_title(f'{n_above}/{n_s} 3f > 2f, Wilcoxon p={p_paired:.4f}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: bar plot of median r for each model ---
ax = axes[1]
medians = [np.median(r_test_2f), np.median(r_test_3f)]
sems = [np.std(r_test_2f) / np.sqrt(n_s), np.std(r_test_3f) / np.sqrt(n_s)]
colors = ['0.5', 'k']
ax.bar([0, 1], medians, yerr=sems, capsize=4, color=colors,
       edgecolor='k', linewidth=0.8, alpha=0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(['2-factor\n(weight=1)', '3-factor\n(fitted betas)'])
ax.set_ylabel('Median r')
ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.5)
ax.set_title('Model comparison')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname = 'two_vs_three_factor_comparison'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 9: Overlaid binned plots (z-scored predictions) + p-value barbell
# ============================================================================
FIG_W = 3
FIG_H = 1.5
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.85,
                                      'wspace': 0.40})

# --- Panel A: overlaid pooled binned plots, z-scored predictions ---
ax = axes[0]
Y_pred_2f_pool = np.concatenate([s['Y_pred_all'] for s in results_2f])
Y_test_2f_pool = np.concatenate([s['Y_test_all'] for s in results_2f])
Y_pred_3f_pool = np.concatenate([s['Y_pred_all'] for s in results_3f])
Y_test_3f_pool = np.concatenate([s['Y_test_all'] for s in results_3f])

# Z-score predictions so x-axes are on the same scale
def zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else x * 0

plt.sca(ax)
pf.mean_bin_plot(zscore(Y_pred_2f_pool), Y_test_2f_pool, col=5, color='0.5')
pf.mean_bin_plot(zscore(Y_pred_3f_pool), Y_test_3f_pool, col=5, color='k')

ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
ax.set_xlabel('Predicted $\\Delta W$ (z)')
ax.set_ylabel('Actual $\\Delta W$ (z)')
r_2f_pool, _ = pearsonr(Y_pred_2f_pool, Y_test_2f_pool)
r_3f_pool, _ = pearsonr(Y_pred_3f_pool, Y_test_3f_pool)
ax.set_title(f'2f r={r_2f_pool:.3f} (gray), 3f r={r_3f_pool:.3f} (black)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: barbell plot — 2f vs 3f on x-axis, -log10(p) on y ---
ax = axes[1]
lp_2f = -np.log10(np.clip(p_test_2f, 1e-300, 1.0))
lp_3f = -np.log10(np.clip(p_test_3f, 1e-300, 1.0))

rng = np.random.default_rng(42)
jitter = 0.08
x_2f = 0 + rng.uniform(-jitter, jitter, n_s)
x_3f = 1 + rng.uniform(-jitter, jitter, n_s)

for i in range(n_s):
    ax.plot([x_2f[i], x_3f[i]], [lp_2f[i], lp_3f[i]],
            '-', color='0.75', linewidth=0.5, zorder=1)
ax.scatter(x_2f, lp_2f, s=15, color='0.5', zorder=5, edgecolor='none')
ax.scatter(x_3f, lp_3f, s=15, color='k', zorder=5, edgecolor='none')
ax.axhline(-np.log10(0.05), color='k', ls=':', alpha=0.4, linewidth=0.5)

n_sig_2f = np.sum(p_test_2f < 0.05)
n_sig_3f = np.sum(p_test_3f < 0.05)
ax.set_xticks([0, 1])
ax.set_xticklabels(['2-factor', '3-factor'])
ax.set_ylabel('$-\\log_{10}(p)$')
ax.set_title(f'sig: 2f={n_sig_2f}/{n_s}, 3f={n_sig_3f}/{n_s}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fname = 'two_vs_three_factor_overlay'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved {fname}")
