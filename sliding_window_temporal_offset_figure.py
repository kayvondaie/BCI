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

WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20
OFFSET_SEC = 0

print("Setup complete!")

#%% ============================================================================
# CELL 2: Load saved results
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded modes: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 3: Compute within-session correlations
# ============================================================================
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Speed', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return -s['win_rt']  # flip sign: speed = -RT
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
# CELL 4: Coefficient matrices — behavior x epoch
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(4,4),
                         squeeze=False,  gridspec_kw={'left': 0.10, 'right': 0.95,
                                       'bottom': 0.20, 'top': 0.85,
                                       'wspace': 0.40})

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
        im = ax.imshow(mat_mean, cmap='bwr', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

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
                txt = ''
                if sig:
                    txt += f'\n{sig}'
                ax.text(ei, bi, txt, ha='center', va='center',
                        fontsize=7, fontweight='bold' if sig else 'normal')

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean rho')


lag_str = f"{OFFSET_SEC}s"

plt.tight_layout()
fig.savefig(os.path.join(PANEL_DIR, 'temporal_offset_matrices.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(PANEL_DIR, 'temporal_offset_matrices.svg'), bbox_inches='tight')
plt.show()
print("Coefficient matrices saved.")

#%% ============================================================================
# CELL 4b: Coefficient matrices — -log10(p) version (signed by mean rho)
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(4, 4),
                         squeeze=False, gridspec_kw={'left': 0.10, 'right': 0.95,
                                       'bottom': 0.20, 'top': 0.85,
                                       'wspace': 0.40})

for col, mode in enumerate(CC_MODES):
    n_s = len(all_results[mode])

    for row, (corr_arr, row_label) in enumerate([
        (corr_slope[mode], 'Slope'),
        (corr_intercept[mode], 'Intercept'),
    ]):
        ax = axes[row, col]
        mat_signed_logp = np.full((n_beh, n_epochs), np.nan)
        mat_p = np.full((n_beh, n_epochs), np.nan)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                vals = corr_arr[:, bi, ei]
                v = vals[np.isfinite(vals)]
                if len(v) < 3:
                    continue
                try:
                    _, p = wilcoxon(v)
                except Exception:
                    p = 1.0
                mat_p[bi, ei] = p
                sign = np.sign(np.mean(v))
                # Clamp p to avoid -log10(0)
                p_clamped = max(p, 1e-10)
                mat_signed_logp[bi, ei] = sign * (-np.log10(p_clamped))

        vmax = np.nanmax(np.abs(mat_signed_logp)) if np.any(np.isfinite(mat_signed_logp)) else 2.0
        vmax = max(vmax, 1.0)
        im = ax.imshow(mat_signed_logp, cmap='bwr', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        # Annotate with stars
        for bi in range(n_beh):
            for ei in range(n_epochs):
                p = mat_p[bi, ei]
                if np.isnan(p):
                    continue
                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                if sig:
                    ax.text(ei, bi, sig, ha='center', va='center',
                            fontsize=7, fontweight='bold')

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Signed $-\\log_{10}(p)$')

fig.savefig(os.path.join(PANEL_DIR, 'temporal_offset_matrices_logp.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(PANEL_DIR, 'temporal_offset_matrices_logp.svg'), bbox_inches='tight')
plt.show()
print("Signed -log10(p) matrices saved.")

#%% ============================================================================
# CELL 4c: Lasso MLR — which behavior-epoch combos jointly predict HI slope?
# ============================================================================
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Build pooled design matrix: each row = one window from one session
# Columns = behavior(b) x epoch(e) interaction features
# Target = HI slope for that window/epoch

# We'll predict HI slope in each epoch separately
mode_lasso = CC_MODES[0]  # use first mode (typically dev2_lag)
results_lasso = all_results[mode_lasso]

for ei_target in range(n_epochs):
    X_all = []
    Y_all = []

    for si, s in enumerate(results_lasso):
        n_win = s['hi_with_int'].shape[0]
        slope_ts = s['hi_with_int'][:, ei_target]

        # Build feature matrix for this session
        beh_mat = np.column_stack([get_beh(s, bn) for bn in beh_names])  # n_win x n_beh
        ok = np.all(np.isfinite(beh_mat), axis=1) & np.isfinite(slope_ts)
        if np.sum(ok) < 5:
            continue

        # Z-score within session
        beh_ok = beh_mat[ok]
        slope_ok = slope_ts[ok]
        beh_z = (beh_ok - np.mean(beh_ok, axis=0)) / np.clip(np.std(beh_ok, axis=0), 1e-10, None)
        slope_z = (slope_ok - np.mean(slope_ok)) / max(np.std(slope_ok), 1e-10)

        X_all.append(beh_z)
        Y_all.append(slope_z)

    X_all = np.vstack(X_all)
    Y_all = np.concatenate(Y_all)

    # Fit Lasso with CV
    lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 50), max_iter=10000)
    lasso.fit(X_all, Y_all)

    print(f"\n--- Epoch: {EPOCH_ORDER[ei_target]} ---")
    print(f"  Best alpha: {lasso.alpha_:.4f}")
    print(f"  R² (train): {lasso.score(X_all, Y_all):.4f}")
    for bi, bn in enumerate(beh_labels):
        coef = lasso.coef_[bi]
        marker = ' *' if abs(coef) > 0 else ''
        print(f"  {bn:15s}: {coef:+.4f}{marker}")

# Summary matrix: lasso coefficients for all epoch targets
lasso_coefs = np.zeros((n_beh, n_epochs))
lasso_r2 = np.zeros(n_epochs)

for ei_target in range(n_epochs):
    X_all = []
    Y_all = []
    for si, s in enumerate(results_lasso):
        n_win = s['hi_with_int'].shape[0]
        slope_ts = s['hi_with_int'][:, ei_target]
        beh_mat = np.column_stack([get_beh(s, bn) for bn in beh_names])
        ok = np.all(np.isfinite(beh_mat), axis=1) & np.isfinite(slope_ts)
        if np.sum(ok) < 5:
            continue
        beh_ok = beh_mat[ok]
        slope_ok = slope_ts[ok]
        beh_z = (beh_ok - np.mean(beh_ok, axis=0)) / np.clip(np.std(beh_ok, axis=0), 1e-10, None)
        slope_z = (slope_ok - np.mean(slope_ok)) / max(np.std(slope_ok), 1e-10)
        X_all.append(beh_z)
        Y_all.append(slope_z)

    X_all = np.vstack(X_all)
    Y_all = np.concatenate(Y_all)

    lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 50), max_iter=10000)
    lasso.fit(X_all, Y_all)
    lasso_coefs[:, ei_target] = lasso.coef_
    lasso_r2[ei_target] = lasso.score(X_all, Y_all)

# Plot lasso coefficient matrix
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5),
                       gridspec_kw={'left': 0.25, 'right': 0.85,
                                    'bottom': 0.25, 'top': 0.88})

vmax = np.max(np.abs(lasso_coefs)) if np.any(lasso_coefs != 0) else 0.1
vmax = max(vmax, 0.01)
im = ax.imshow(lasso_coefs, cmap='bwr', vmin=-vmax, vmax=vmax,
               aspect='auto', interpolation='nearest')

# Label nonzero coefficients
for bi in range(n_beh):
    for ei in range(n_epochs):
        c = lasso_coefs[bi, ei]
        if abs(c) > 0:
            ax.text(ei, bi, f'{c:+.3f}', ha='center', va='center',
                    fontsize=6, fontweight='bold')

ax.set_xticks(range(n_epochs))
ax.set_xticklabels([f'{e}\nR²={lasso_r2[ei]:.3f}' for ei, e in enumerate(epoch_labels)],
                   rotation=0, ha='center')
ax.set_yticks(range(n_beh))
ax.set_yticklabels(beh_labels)
ax.set_title('Lasso: behavior → HI slope')
plt.colorbar(im, ax=ax, shrink=0.8, label='Lasso coef')

fig.savefig(os.path.join(PANEL_DIR, 'lasso_behavior_hi_slope.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(PANEL_DIR, 'lasso_behavior_hi_slope.svg'), bbox_inches='tight')
plt.show()

# Save lasso results + alphas so we can inspect outside Spyder
lasso_results_save = {
    'lasso_coefs': lasso_coefs,
    'lasso_r2': lasso_r2,
    'lasso_alphas': np.array([0.0] * n_epochs),  # will be filled below
    'beh_labels': beh_labels,
    'epoch_labels': epoch_labels,
    'mode': mode_lasso,
}

# Re-fit to grab per-epoch alpha (reuses cached X_all/Y_all from last iteration only;
# redo all epochs to be safe)
_alphas = []
for ei_target in range(n_epochs):
    X_all = []
    Y_all = []
    for si, s in enumerate(results_lasso):
        slope_ts = s['hi_with_int'][:, ei_target]
        beh_mat = np.column_stack([get_beh(s, bn) for bn in beh_names])
        ok = np.all(np.isfinite(beh_mat), axis=1) & np.isfinite(slope_ts)
        if np.sum(ok) < 5:
            continue
        beh_ok = beh_mat[ok]
        slope_ok = slope_ts[ok]
        beh_z = (beh_ok - np.mean(beh_ok, axis=0)) / np.clip(np.std(beh_ok, axis=0), 1e-10, None)
        slope_z = (slope_ok - np.mean(slope_ok)) / max(np.std(slope_ok), 1e-10)
        X_all.append(beh_z)
        Y_all.append(slope_z)
    X_all = np.vstack(X_all)
    Y_all = np.concatenate(Y_all)
    _lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 50), max_iter=10000)
    _lasso.fit(X_all, Y_all)
    _alphas.append(_lasso.alpha_)

lasso_results_save['lasso_alphas'] = np.array(_alphas)
np.save(os.path.join(RESULTS_DIR, 'lasso_behavior_hi_slope.npy'), lasso_results_save)

# Write human-readable txt summary
txt_path = os.path.join(RESULTS_DIR, 'lasso_behavior_hi_slope.txt')
with open(txt_path, 'w') as f:
    f.write(f"Lasso MLR: behavioral predictors of HI slope\n")
    f.write(f"Mode: {mode_lasso}\n")
    f.write(f"{'='*60}\n\n")
    for ei in range(n_epochs):
        f.write(f"Target: HI slope in {EPOCH_ORDER[ei]} epoch\n")
        f.write(f"  Best alpha: {_alphas[ei]:.4f}\n")
        f.write(f"  R² (train): {lasso_r2[ei]:.4f}\n")
        f.write(f"  Coefficients:\n")
        for bi, bl in enumerate(beh_labels):
            c = lasso_coefs[bi, ei]
            marker = '  <-- nonzero' if abs(c) > 0 else ''
            f.write(f"    {bl:20s}: {c:+.6f}{marker}\n")
        f.write(f"\n")
    f.write(f"{'='*60}\n")
    f.write(f"Nonzero summary:\n")
    any_nonzero = False
    for bi, bl in enumerate(beh_labels):
        for ei, el in enumerate(epoch_labels):
            c = lasso_coefs[bi, ei]
            if abs(c) > 0:
                f.write(f"  {bl} x {el}: {c:+.6f}\n")
                any_nonzero = True
    if not any_nonzero:
        f.write("  (none — all coefficients shrunk to zero)\n")

print(f"Lasso results saved to {txt_path}")
print("Lasso coefficient matrix saved.")

#%% ============================================================================
# CELL 5: Binned scatter — RPE vs HI slope (dev2, pre epoch)
# ============================================================================
mode_plot = 'dev2_lag'
ei_plot = 0  # pre epoch

all_rpe_z = []
all_slope_z = []

for s in all_results[mode_plot]:
    rpe = s['win_rpe']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rpe) & np.isfinite(slope)
    if np.sum(ok) < 5:
        continue
    rpe_ok = rpe[ok]
    slope_ok = slope[ok]
    if np.std(rpe_ok) == 0 or np.std(slope_ok) == 0:
        continue
    all_rpe_z.append((rpe_ok - np.mean(rpe_ok)) / np.std(rpe_ok))
    all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))

all_rpe_z = np.concatenate(all_rpe_z)
all_slope_z = np.concatenate(all_slope_z)

# Find session ranked by RPE-slope correlation
RANK = 3

all_corrs = []
for si, s in enumerate(all_results[mode_plot]):
    rpe = s['win_rpe']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rpe) & np.isfinite(slope)
    if np.sum(ok) >= 5 and np.std(slope[ok]) > 0 and np.std(rpe[ok]) > 0:
        r, _ = spearmanr(rpe[ok], slope[ok])
        all_corrs.append((r, si))
    else:
        all_corrs.append((np.nan, si))

all_corrs.sort(key=lambda x: -x[0] if np.isfinite(x[0]) else np.inf)
best_corr, best_idx = all_corrs[RANK - 1]
best_s = all_results[mode_plot][best_idx]

FIG_W_MM = 140
FIG_H_MM = 55
FIG_W = FIG_W_MM / 25.4
FIG_H = FIG_H_MM / 25.4

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.85,
                                      'wspace': 0.40})

# Left: binned scatter
ax = axes[0]
plt.sca(ax)
pf.mean_bin_plot(all_rpe_z, all_slope_z, col=3, color='k')
ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.5)
ax.axvline(0, color='k', ls='--', alpha=0.3, linewidth=0.5)
ax.set_xlabel('RPE (within-session z)')
ax.set_ylabel('HI slope (within-session z)')
ax.set_title(f'dev2 pre epoch\nn={len(all_results[mode_plot])} sessions')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: time series for best session
ax2 = axes[1]
wc = best_s['win_centers']
rpe_ts = best_s['win_rpe']
slope_ts = best_s['hi_with_int'][:, ei_plot]

C_3RD = '#ea580c'
ax2.plot(wc, (rpe_ts - np.nanmean(rpe_ts)) / np.nanstd(rpe_ts),
         'o-', color=C_3RD, linewidth=1, markersize=3, label='RPE')
ax2.plot(wc, (slope_ts - np.nanmean(slope_ts)) / np.nanstd(slope_ts),
         'o-', color='k', linewidth=1, markersize=3, label='HI slope')
ax2.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.5)
ax2.set_xlabel('Trial (window center)')
ax2.set_ylabel('z-score')
ax2.legend(loc='best', frameon=False)
ax2.set_title(f'{best_s["mouse"]} {best_s["session"]}, rho={best_corr:.3f}')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.savefig(os.path.join(PANEL_DIR, 'rpe_vs_hi_slope.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, 'rpe_vs_hi_slope.svg'))
plt.show()
print("RPE vs HI slope saved.")

#%% ============================================================================
# CELL 6: Single-session dW vs CC scatter, split by RPE (re-computes CC)
# ============================================================================
ex_mouse = best_s['mouse']
ex_session = best_s['session']
ex_lag_sec = best_s['lag_sec']
ei_plot_10 = 0

print(f"Re-computing CC for {ex_mouse} {ex_session}, lag={ex_lag_sec}s ...")

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
epoch_pre = np.nanmean(F_nan[t0e:t1e+1, :, :], axis=0)
epoch_post = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

baseline_trials_arr = np.arange(min(N_BASELINE, trl))
bl_post_mean = np.nanmean(epoch_post[:, baseline_trials_arr], axis=1)

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

# Split windows by RPE percentile
hi_rpe = rpe_per_win >= np.percentile(rpe_per_win, 90)
lo_rpe = rpe_per_win < np.percentile(rpe_per_win, 10)

cc_hi = np.nanmean(cc_per_win[hi_rpe, :], axis=0)
cc_lo = np.nanmean(cc_per_win[lo_rpe, :], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True,
                         gridspec_kw={'left': 0.10, 'right': 0.95,
                                      'bottom': 0.20, 'top': 0.82,
                                      'wspace': 0.25})

for ax, cc, label, color in [
    (axes[0], cc_hi, f'High RPE ({np.sum(hi_rpe)} wins)', '#e74c3c'),
    (axes[1], cc_lo, f'Low RPE ({np.sum(lo_rpe)} wins)', '#3498db'),
]:
    ok = np.isfinite(cc) & np.isfinite(Y_T)
    cc_ok = cc[ok]
    dw_ok = Y_T[ok]

    if len(cc_ok) < 5:
        continue

    plt.sca(ax)
    pf.mean_bin_plot(cc_ok, dw_ok, col=5, color=color)

    if np.std(cc_ok) > 0:
        A = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A, dw_ok, rcond=None)[0]
        r, p = spearmanr(cc_ok, dw_ok)
        ax.set_title(f'{label}\nslope={coeffs[1]:.4f}, r={r:.3f}')

    ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
    ax.axvline(0, color='k', ls='--', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('CC (dev2, pre epoch)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel(r'$\Delta W$')

fig.suptitle(f'{ex_mouse} {ex_session} — dW vs CC split by RPE')
fig.savefig(os.path.join(PANEL_DIR, 'dw_vs_cc_rpe_split.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, 'dw_vs_cc_rpe_split.svg'))
plt.show()
print("dW vs CC RPE split saved.")
