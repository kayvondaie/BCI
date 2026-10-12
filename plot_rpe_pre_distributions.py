"""
Plot the per-session Spearman rho distributions (RPE × Pre epoch, Slope)
for dot_prod vs dev2, to illustrate why dev2 is significant but dot_prod is not.

Loads the same saved results as sliding_window_temporal_offset.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

# ── paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()

CC_MODES = list(all_results.keys())

# ── recompute per-session rho (RPE × Pre, Slope) ────────────────────────
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']

BI = beh_names.index('RPE')   # 1
EI = EPOCH_ORDER.index('pre') # 0

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

rho_by_mode = {}
for mode in CC_MODES:
    results = all_results[mode]
    rhos = []
    for s in results:
        bvar = get_beh(s, 'RPE')
        if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
            rhos.append(np.nan)
            continue
        slope = s['hi_with_int'][:, EI]
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
            r, _ = spearmanr(bvar[ok], slope[ok])
            rhos.append(r)
        else:
            rhos.append(np.nan)
    rho_by_mode[mode] = np.array(rhos)

# ── plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

for ax, mode in zip(axes, CC_MODES):
    vals = rho_by_mode[mode]
    v = vals[np.isfinite(vals)]
    n = len(v)

    # strip plot (jittered dots)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=n)
    ax.scatter(jitter, v, s=30, alpha=0.6, color='steelblue', edgecolors='k',
               linewidths=0.4, zorder=3)

    # mean ± SEM
    m = np.mean(v)
    sem = np.std(v, ddof=1) / np.sqrt(n)
    ax.errorbar(0, m, yerr=sem, fmt='D', color='crimson', markersize=7,
                capsize=5, zorder=4, label=f'mean={m:+.3f}')

    # Wilcoxon p-value
    try:
        _, p = wilcoxon(v)
    except Exception:
        p = 1.0
    sig = ''
    if p < 0.001: sig = '***'
    elif p < 0.01: sig = '**'
    elif p < 0.05: sig = '*'

    ax.set_title(f'{mode}\np={p:.4f} {sig}', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.legend(fontsize=8, loc='upper right')

axes[0].set_ylabel('Per-session Spearman rho\n(RPE vs Slope)', fontsize=10)
fig.suptitle('RPE × Pre epoch (Slope): per-session distributions',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rpe_pre_slope_distributions.png'),
            dpi=200, bbox_inches='tight')
plt.show()
