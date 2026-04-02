#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LassoCV
import plotting_functions as pf

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

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']
n_epochs = len(EPOCH_ORDER)

beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

print("Setup complete!")

#%% ============================================================================
# CELL 2: Load saved results
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Per-session lasso — predict HI slope from behavioral variables
# ============================================================================
# For each session and each epoch, fit lasso: HI_slope ~ beh_vars
# Then summarize which predictors survive across sessions

mode_lasso = CC_MODES[0]
results_lasso = all_results[mode_lasso]
n_sessions = len(results_lasso)

# Storage: which coefficients are nonzero, and their values
coef_nonzero = np.zeros((n_sessions, n_beh, n_epochs))  # 1 if nonzero
coef_values = np.full((n_sessions, n_beh, n_epochs), np.nan)
coef_sign = np.full((n_sessions, n_beh, n_epochs), np.nan)  # +1 or -1
session_r2 = np.full((n_sessions, n_epochs), np.nan)
session_alpha = np.full((n_sessions, n_epochs), np.nan)
session_n_windows = np.zeros(n_sessions, dtype=int)

MIN_WINDOWS = 8  # need enough data points for 4-predictor lasso

for si, s in enumerate(results_lasso):
    for ei in range(n_epochs):
        slope_ts = s['hi_with_int'][:, ei]
        beh_mat = np.column_stack([get_beh(s, bn) for bn in beh_names])
        ok = np.all(np.isfinite(beh_mat), axis=1) & np.isfinite(slope_ts)
        n_ok = np.sum(ok)
        session_n_windows[si] = n_ok

        if n_ok < MIN_WINDOWS:
            continue

        X = beh_mat[ok]
        Y = slope_ts[ok]

        # Z-score within session
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1e-10
        X = (X - np.mean(X, axis=0)) / X_std
        Y_std = np.std(Y)
        if Y_std == 0:
            continue
        Y = (Y - np.mean(Y)) / Y_std

        # Fit lasso with CV (use 3-fold for small n)
        n_cv = min(5, max(2, n_ok // 3))
        lasso = LassoCV(cv=n_cv, alphas=np.logspace(-3, 1, 50), max_iter=10000)
        lasso.fit(X, Y)

        coef_values[si, :, ei] = lasso.coef_
        coef_nonzero[si, :, ei] = (np.abs(lasso.coef_) > 0).astype(float)
        coef_sign[si, :, ei] = np.sign(lasso.coef_)
        session_r2[si, ei] = lasso.score(X, Y)
        session_alpha[si, ei] = lasso.alpha_

    mouse = s.get('mouse', '?')
    sess = s.get('session', '?')
    n_nz = int(np.nansum(coef_nonzero[si, :, :]))
    print(f"  {si:2d} {mouse} {sess}: {session_n_windows[si]} windows, "
          f"{n_nz}/16 nonzero coefs")

print(f"\nDone. {n_sessions} sessions processed.")

#%% ============================================================================
# CELL 4: Summary — fraction of sessions where each predictor survives
# ============================================================================
# Count fraction nonzero (among sessions with enough data)
valid_mask = np.isfinite(session_r2)  # n_sessions x n_epochs

frac_nonzero = np.full((n_beh, n_epochs), np.nan)
frac_positive = np.full((n_beh, n_epochs), np.nan)
mean_coef = np.full((n_beh, n_epochs), np.nan)
p_wilcoxon = np.full((n_beh, n_epochs), np.nan)

for bi in range(n_beh):
    for ei in range(n_epochs):
        valid = valid_mask[:, ei]
        n_valid = np.sum(valid)
        if n_valid < 3:
            continue
        nz = coef_nonzero[valid, bi, ei]
        frac_nonzero[bi, ei] = np.mean(nz)

        # Among nonzero sessions, fraction positive
        vals = coef_values[valid, bi, ei]
        nz_mask = np.abs(vals) > 0
        if np.sum(nz_mask) > 0:
            frac_positive[bi, ei] = np.mean(vals[nz_mask] > 0)

        # Mean coefficient across all valid sessions
        mean_coef[bi, ei] = np.mean(vals)

        # Wilcoxon on nonzero coefficients
        nz_vals = vals[nz_mask]
        if len(nz_vals) >= 3:
            try:
                _, p = wilcoxon(nz_vals)
                p_wilcoxon[bi, ei] = p
            except Exception:
                pass

# --- Figure 1: Fraction of sessions with nonzero coefficient ---
fig, axes = plt.subplots(1, 2, figsize=(5, 2.5),
                         gridspec_kw={'left': 0.18, 'right': 0.92,
                                      'bottom': 0.25, 'top': 0.85,
                                      'wspace': 0.50})

# Left: fraction nonzero
ax = axes[0]
im = ax.imshow(frac_nonzero, cmap='Oranges', vmin=0, vmax=1,
               aspect='auto', interpolation='nearest')
for bi in range(n_beh):
    for ei in range(n_epochs):
        v = frac_nonzero[bi, ei]
        if np.isfinite(v):
            ax.text(ei, bi, f'{v:.0%}', ha='center', va='center', fontsize=7)
ax.set_xticks(range(n_epochs))
ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
ax.set_yticks(range(n_beh))
ax.set_yticklabels(beh_labels)
ax.set_title('Fraction sessions\nwith nonzero coef')
plt.colorbar(im, ax=ax, shrink=0.8)

# Right: mean coefficient (signed)
ax = axes[1]
vmax = np.nanmax(np.abs(mean_coef)) if np.any(np.isfinite(mean_coef)) else 0.1
vmax = max(vmax, 0.01)
im = ax.imshow(mean_coef, cmap='bwr', vmin=-vmax, vmax=vmax,
               aspect='auto', interpolation='nearest')
for bi in range(n_beh):
    for ei in range(n_epochs):
        v = mean_coef[bi, ei]
        p = p_wilcoxon[bi, ei]
        if np.isnan(v):
            continue
        sig = ''
        if np.isfinite(p):
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
        txt = f'{v:+.3f}'
        if sig:
            txt += f'\n{sig}'
        ax.text(ei, bi, txt, ha='center', va='center', fontsize=6)
ax.set_xticks(range(n_epochs))
ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
ax.set_yticks(range(n_beh))
ax.set_yticklabels(beh_labels)
ax.set_title('Mean lasso coef\n(across sessions)')
plt.colorbar(im, ax=ax, shrink=0.8, label='Coef')

fig.savefig(os.path.join(PANEL_DIR, 'lasso_per_session_summary.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(PANEL_DIR, 'lasso_per_session_summary.svg'), bbox_inches='tight')
plt.show()

# --- Figure 2: R² distribution per epoch ---
fig, ax = plt.subplots(1, 1, figsize=(3, 2),
                       gridspec_kw={'left': 0.20, 'right': 0.90,
                                    'bottom': 0.25, 'top': 0.88})
r2_valid = []
for ei in range(n_epochs):
    vals = session_r2[:, ei]
    r2_valid.append(vals[np.isfinite(vals)])

parts = ax.violinplot(r2_valid, positions=range(n_epochs), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('0.7')
    pc.set_alpha(0.6)
ax.set_xticks(range(n_epochs))
ax.set_xticklabels(epoch_labels)
ax.set_ylabel('R² (lasso, per session)')
ax.set_title(f'Per-session lasso fit quality (n={n_sessions})')
ax.axhline(0, color='k', ls='--', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(PANEL_DIR, 'lasso_per_session_r2.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(PANEL_DIR, 'lasso_per_session_r2.svg'), bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 5: Save results as .txt
# ============================================================================
txt_path = os.path.join(RESULTS_DIR, 'lasso_per_session_summary.txt')
with open(txt_path, 'w') as f:
    f.write(f"Per-session Lasso: behavioral predictors of HI slope\n")
    f.write(f"Mode: {mode_lasso}\n")
    f.write(f"Sessions: {n_sessions}, MIN_WINDOWS: {MIN_WINDOWS}\n")
    f.write(f"{'='*70}\n\n")

    f.write("Fraction of sessions with nonzero coefficient:\n")
    f.write(f"{'':20s}  {'  '.join(f'{e:>10s}' for e in epoch_labels)}\n")
    for bi, bl in enumerate(beh_labels):
        row = '  '.join(f'{frac_nonzero[bi,ei]:10.1%}' if np.isfinite(frac_nonzero[bi,ei])
                        else f'{"N/A":>10s}' for ei in range(n_epochs))
        f.write(f'{bl:20s}  {row}\n')

    f.write(f"\nMean lasso coefficient (across all valid sessions):\n")
    f.write(f"{'':20s}  {'  '.join(f'{e:>10s}' for e in epoch_labels)}\n")
    for bi, bl in enumerate(beh_labels):
        row = '  '.join(f'{mean_coef[bi,ei]:+10.4f}' if np.isfinite(mean_coef[bi,ei])
                        else f'{"N/A":>10s}' for ei in range(n_epochs))
        f.write(f'{bl:20s}  {row}\n')

    f.write(f"\nWilcoxon p-value (nonzero coefs != 0):\n")
    f.write(f"{'':20s}  {'  '.join(f'{e:>10s}' for e in epoch_labels)}\n")
    for bi, bl in enumerate(beh_labels):
        row = '  '.join(f'{p_wilcoxon[bi,ei]:10.4f}' if np.isfinite(p_wilcoxon[bi,ei])
                        else f'{"N/A":>10s}' for ei in range(n_epochs))
        f.write(f'{bl:20s}  {row}\n')

    f.write(f"\nFraction positive (among nonzero):\n")
    f.write(f"{'':20s}  {'  '.join(f'{e:>10s}' for e in epoch_labels)}\n")
    for bi, bl in enumerate(beh_labels):
        row = '  '.join(f'{frac_positive[bi,ei]:10.1%}' if np.isfinite(frac_positive[bi,ei])
                        else f'{"N/A":>10s}' for ei in range(n_epochs))
        f.write(f'{bl:20s}  {row}\n')

    f.write(f"\nR² per epoch (median [min, max]):\n")
    for ei, el in enumerate(epoch_labels):
        vals = session_r2[:, ei]
        v = vals[np.isfinite(vals)]
        if len(v) > 0:
            f.write(f"  {el:10s}: median={np.median(v):.4f}  "
                    f"[{np.min(v):.4f}, {np.max(v):.4f}]  "
                    f"mean={np.mean(v):.4f}\n")

print(f"Results saved to {txt_path}")
