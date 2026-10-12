"""
Analyze why some sessions flip from negative RPE-slope correlation (dot_prod)
to positive (dev2).

dev2_CC = dot_prod_CC - pre_sum * baseline_post_mean, so:
  dev2_slope ≈ dot_prod_slope + correction_slope

The correction_slope captures how the baseline subtraction reshapes the
CC-vs-dW regression per window. If correction_slope correlates with RPE,
the baseline term itself carries RPE-dependent information.

We decompose each session's dev2 rho into:
  1. rho(RPE, dot_prod_slope)       — raw coactivity
  2. rho(RPE, correction_slope)     — baseline correction contribution
  3. rho(RPE, dev2_slope)           — the combined signal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()

EI = 0  # pre epoch

# ── Per-session: compute rho for dot_prod, dev2, and the correction ─────
results_dp = all_results['dot_prod_lag']
results_d2 = all_results['dev2_lag']
n_sess = len(results_dp)

rho_dp = np.full(n_sess, np.nan)
rho_d2 = np.full(n_sess, np.nan)
rho_corr = np.full(n_sess, np.nan)  # rho(RPE, correction_slope)
labels = []

for si in range(n_sess):
    s_dp = results_dp[si]
    s_d2 = results_d2[si]
    labels.append(f"{s_dp['mouse']}_{s_dp['session']}")

    rpe = s_dp['win_rpe']
    slope_dp = s_dp['hi_with_int'][:, EI]
    slope_d2 = s_d2['hi_with_int'][:, EI]

    # correction_slope: what the baseline subtraction adds
    correction = slope_d2 - slope_dp

    ok = np.isfinite(rpe) & np.isfinite(slope_dp) & np.isfinite(slope_d2)
    if np.sum(ok) < 5:
        continue
    if np.std(rpe[ok]) == 0:
        continue

    if np.std(slope_dp[ok]) > 0:
        rho_dp[si], _ = spearmanr(rpe[ok], slope_dp[ok])
    if np.std(slope_d2[ok]) > 0:
        rho_d2[si], _ = spearmanr(rpe[ok], slope_d2[ok])
    if np.std(correction[ok]) > 0:
        rho_corr[si], _ = spearmanr(rpe[ok], correction[ok])

valid = np.isfinite(rho_dp) & np.isfinite(rho_d2) & np.isfinite(rho_corr)

# ── Classify sessions ───────────────────────────────────────────────────
flippers = valid & (rho_dp < 0) & (rho_d2 > 0)
same_pos = valid & (rho_dp > 0) & (rho_d2 > 0)
same_neg = valid & (rho_dp < 0) & (rho_d2 < 0)
reverse  = valid & (rho_dp > 0) & (rho_d2 < 0)

print(f"Total valid sessions: {np.sum(valid)}")
print(f"  Flippers (dp<0 -> d2>0):  {np.sum(flippers)}")
print(f"  Both positive:            {np.sum(same_pos)}")
print(f"  Both negative:            {np.sum(same_neg)}")
print(f"  Reverse (dp>0 -> d2<0):   {np.sum(reverse)}")

# ── Figure 1: dev2 rho vs dot_prod rho, colored by correction rho ───────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: scatter of dev2 vs dot_prod rho
ax = axes[0]
ax.scatter(rho_dp[same_pos], rho_d2[same_pos], c='grey', s=40, alpha=0.6,
           label=f'both >0 (n={np.sum(same_pos)})', zorder=2)
ax.scatter(rho_dp[same_neg], rho_d2[same_neg], c='steelblue', s=40, alpha=0.6,
           label=f'both <0 (n={np.sum(same_neg)})', zorder=2)
ax.scatter(rho_dp[flippers], rho_d2[flippers], c='crimson', s=60, alpha=0.8,
           edgecolors='k', linewidths=0.5,
           label=f'flippers (n={np.sum(flippers)})', zorder=3)
ax.scatter(rho_dp[reverse], rho_d2[reverse], c='orange', s=40, alpha=0.6,
           label=f'reverse (n={np.sum(reverse)})', zorder=2)
lims = [-0.8, 0.9]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.axhline(0, color='k', alpha=0.2)
ax.axvline(0, color='k', alpha=0.2)
ax.set_xlabel('rho(RPE, dot_prod slope)')
ax.set_ylabel('rho(RPE, dev2 slope)')
ax.set_title('Per-session RPE correlation:\ndev2 vs dot_prod', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Panel B: correction rho for each group
ax = axes[1]
groups = [
    ('Flippers', rho_corr[flippers], 'crimson'),
    ('Both >0', rho_corr[same_pos], 'grey'),
    ('Both <0', rho_corr[same_neg], 'steelblue'),
]
for gi, (label, vals, color) in enumerate(groups):
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        continue
    jitter = np.random.default_rng(gi).uniform(-0.15, 0.15, size=len(v))
    ax.scatter(np.full(len(v), gi) + jitter, v, s=30, alpha=0.6,
               color=color, edgecolors='k', linewidths=0.3, zorder=3)
    m = np.mean(v)
    sem = np.std(v, ddof=1) / np.sqrt(len(v))
    ax.errorbar(gi, m, yerr=sem, fmt='D', color='k', markersize=7,
                capsize=5, zorder=4)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=10)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_ylabel('rho(RPE, correction slope)')
ax.set_title('Baseline correction term\nvs RPE, by group', fontweight='bold')

# Panel C: decomposition — paired dot_prod rho vs correction rho, flippers only
ax = axes[2]
v_dp = rho_dp[valid]
v_corr = rho_corr[valid]
sc = ax.scatter(v_dp, v_corr, c=rho_d2[valid], cmap='coolwarm',
                vmin=-0.5, vmax=0.5, s=50, edgecolors='k', linewidths=0.3,
                zorder=3)
ax.axhline(0, color='k', alpha=0.2)
ax.axvline(0, color='k', alpha=0.2)
ax.set_xlabel('rho(RPE, dot_prod slope)')
ax.set_ylabel('rho(RPE, correction slope)')
ax.set_title('Decomposition\n(color = dev2 rho)', fontweight='bold')
plt.colorbar(sc, ax=ax, shrink=0.8, label='rho(RPE, dev2 slope)')

fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rpe_flip_analysis.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# ── Population stats ────────────────────────────────────────────────────
print("\n--- Population-level Wilcoxon tests on correction rho ---")
v = rho_corr[valid]
_, p = wilcoxon(v[np.isfinite(v)])
print(f"  All sessions: mean={np.nanmean(v):+.3f}, p={p:.4f} (n={np.sum(np.isfinite(v))})")

for label, mask in [('Flippers', flippers), ('Both >0', same_pos), ('Both <0', same_neg)]:
    v = rho_corr[mask]
    v = v[np.isfinite(v)]
    if len(v) >= 3:
        _, p = wilcoxon(v)
        print(f"  {label:12s}: mean={np.mean(v):+.3f}, p={p:.4f} (n={len(v)})")

#%% ============================================================================
# Figure 2: Session-level deviation metric vs flip tendency
# Loads v2 results which contain sess_mean_dev and sess_abs_dev
# ============================================================================
v2_path = os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2.npy')
if not os.path.exists(v2_path):
    print(f"\nv2 results not found at {v2_path} — run sliding_window_temporal_offset_v2.py first.")
else:
    v2_results = np.load(v2_path, allow_pickle=True).item()
    # Use any mode from v2 to get the deviation metrics (same across modes)
    v2_mode = list(v2_results.keys())[0]
    v2_sessions = v2_results[v2_mode]

    # Build lookup: (mouse, session) -> deviation metrics
    dev_lookup = {}
    for s in v2_sessions:
        dev_lookup[(s['mouse'], s['session'])] = {
            'sess_mean_dev': s.get('sess_mean_dev', np.nan),
            'sess_abs_dev': s.get('sess_abs_dev', np.nan),
        }

    # Match to original sessions
    sess_mean_dev = np.full(n_sess, np.nan)
    sess_abs_dev = np.full(n_sess, np.nan)
    for si in range(n_sess):
        key = (results_dp[si]['mouse'], results_dp[si]['session'])
        if key in dev_lookup:
            sess_mean_dev[si] = dev_lookup[key]['sess_mean_dev']
            sess_abs_dev[si] = dev_lookup[key]['sess_abs_dev']

    matched = valid & np.isfinite(sess_abs_dev)
    print(f"\nMatched {np.sum(matched)}/{np.sum(valid)} sessions with v2 deviation metrics")

    # delta_rho: how much the baseline correction changed the RPE correlation
    delta_rho = rho_d2 - rho_dp

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: sess_abs_dev by group
    ax = axes[0]
    groups = [
        ('Flippers', flippers & matched, 'crimson'),
        ('Both >0', same_pos & matched, 'grey'),
        ('Both <0', same_neg & matched, 'steelblue'),
    ]
    for gi, (label, mask, color) in enumerate(groups):
        v = sess_abs_dev[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        jitter = np.random.default_rng(gi).uniform(-0.15, 0.15, size=len(v))
        ax.scatter(np.full(len(v), gi) + jitter, v, s=30, alpha=0.6,
                   color=color, edgecolors='k', linewidths=0.3, zorder=3)
        m = np.mean(v)
        sem = np.std(v, ddof=1) / np.sqrt(len(v))
        ax.errorbar(gi, m, yerr=sem, fmt='D', color='k', markersize=7,
                    capsize=5, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel('Mean |deviation from baseline|\n(pre epoch, nontarget neurons)')
    ax.set_title('Baseline drift magnitude\nby group', fontweight='bold')

    # Panel B: sess_abs_dev vs delta_rho (rho_d2 - rho_dp)
    ax = axes[1]
    for label, mask, color in groups:
        m = mask & matched
        ax.scatter(sess_abs_dev[m], delta_rho[m], c=color, s=40, alpha=0.6,
                   edgecolors='k', linewidths=0.3, label=label, zorder=3)
    ax.axhline(0, color='k', alpha=0.2)
    ax.set_xlabel('Mean |deviation from baseline|')
    ax.set_ylabel('delta rho (dev2 - dot_prod)')
    ax.set_title('Baseline drift vs\ncorrection benefit', fontweight='bold')
    ax.legend(fontsize=8)
    # Correlation across all matched sessions
    x = sess_abs_dev[matched]
    y = delta_rho[matched]
    ok_xy = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok_xy) >= 5:
        r, p = spearmanr(x[ok_xy], y[ok_xy])
        ax.text(0.05, 0.95, f'rho={r:+.3f}\np={p:.4f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel C: sess_mean_dev (signed) vs delta_rho
    ax = axes[2]
    for label, mask, color in groups:
        m = mask & matched
        ax.scatter(sess_mean_dev[m], delta_rho[m], c=color, s=40, alpha=0.6,
                   edgecolors='k', linewidths=0.3, label=label, zorder=3)
    ax.axhline(0, color='k', alpha=0.2)
    ax.axvline(0, color='k', alpha=0.2, linestyle='--')
    ax.set_xlabel('Mean signed deviation from baseline')
    ax.set_ylabel('delta rho (dev2 - dot_prod)')
    ax.set_title('Drift direction vs\ncorrection benefit', fontweight='bold')
    ax.legend(fontsize=8)
    x = sess_mean_dev[matched]
    ok_xy = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok_xy) >= 5:
        r, p = spearmanr(x[ok_xy], y[ok_xy])
        ax.text(0.05, 0.95, f'rho={r:+.3f}\np={p:.4f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Session-level baseline deviation vs flip tendency',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rpe_flip_deviation_analysis.png'),
                dpi=200, bbox_inches='tight')
    plt.show()
    print("Deviation analysis figure saved.")

#%% ============================================================================
# Figure 3: Per-pair decomposition — does corr(correction_CC, dW) differ
# between flippers and non-flippers?
#
# For each session, we have per-pair CC arrays (n_wins x n_pairs) and dW (n_pairs).
# We compute, per session:
#   - corr(raw_CC_mean, dW)        across pairs
#   - corr(correction_CC_mean, dW) across pairs
#   - corr(dev2_CC_mean, dW)       across pairs
# where *_CC_mean is the time-averaged CC per pair (mean across windows).
# ============================================================================
v2_path = os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2.npy')
if not os.path.exists(v2_path):
    print(f"\nv2 results not found — run sliding_window_temporal_offset_v2.py first.")
else:
    from scipy.stats import pearsonr
    v2_results = np.load(v2_path, allow_pickle=True).item()
    v2_mode = list(v2_results.keys())[0]
    v2_sessions = v2_results[v2_mode]

    # Build lookup: (mouse, session) -> per-pair data
    pair_lookup = {}
    for s in v2_sessions:
        key = (s['mouse'], s['session'])
        if 'raw_cc_pre' not in s:
            continue
        pair_lookup[key] = {
            'raw_cc_pre': s['raw_cc_pre'],       # (n_wins, n_pairs)
            'dev2_cc_pre': s['dev2_cc_pre'],      # (n_wins, n_pairs)
            'Y_T': s['Y_T'],                      # (n_pairs,)
        }

    # Per-session: correlation of time-averaged CC with dW, across pairs
    r_raw_dw = np.full(n_sess, np.nan)
    r_corr_dw = np.full(n_sess, np.nan)
    r_dev2_dw = np.full(n_sess, np.nan)

    for si in range(n_sess):
        key = (results_dp[si]['mouse'], results_dp[si]['session'])
        if key not in pair_lookup:
            continue
        d = pair_lookup[key]
        Y_T_sess = d['Y_T']
        # Mean CC across windows for each pair
        raw_mean = np.nanmean(d['raw_cc_pre'], axis=0)     # (n_pairs,)
        dev2_mean = np.nanmean(d['dev2_cc_pre'], axis=0)   # (n_pairs,)
        corr_mean = dev2_mean - raw_mean                    # correction term

        ok = (np.isfinite(raw_mean) & np.isfinite(dev2_mean)
              & np.isfinite(Y_T_sess))
        if np.sum(ok) < 10:
            continue
        if np.std(Y_T_sess[ok]) == 0:
            continue
        if np.std(raw_mean[ok]) > 0:
            r_raw_dw[si], _ = pearsonr(raw_mean[ok], Y_T_sess[ok])
        if np.std(corr_mean[ok]) > 0:
            r_corr_dw[si], _ = pearsonr(corr_mean[ok], Y_T_sess[ok])
        if np.std(dev2_mean[ok]) > 0:
            r_dev2_dw[si], _ = pearsonr(dev2_mean[ok], Y_T_sess[ok])

    has_pairs = np.isfinite(r_raw_dw) & np.isfinite(r_corr_dw) & valid
    print(f"\nPer-pair analysis: {np.sum(has_pairs)} sessions with data")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: corr(correction_CC, dW) by group
    ax = axes[0]
    groups = [
        ('Flippers', flippers & has_pairs, 'crimson'),
        ('Both >0', same_pos & has_pairs, 'grey'),
        ('Both <0', same_neg & has_pairs, 'steelblue'),
    ]
    for gi, (label, mask, color) in enumerate(groups):
        v = r_corr_dw[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        jitter = np.random.default_rng(gi).uniform(-0.15, 0.15, size=len(v))
        ax.scatter(np.full(len(v), gi) + jitter, v, s=30, alpha=0.6,
                   color=color, edgecolors='k', linewidths=0.3, zorder=3)
        m = np.mean(v)
        sem = np.std(v, ddof=1) / np.sqrt(len(v))
        ax.errorbar(gi, m, yerr=sem, fmt='D', color='k', markersize=7,
                    capsize=5, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Pearson r(correction_CC, dW)\nacross pairs')
    ax.set_title('Correction-plasticity alignment\nby group', fontweight='bold')

    # Panel B: corr(raw_CC, dW) vs corr(correction_CC, dW)
    ax = axes[1]
    for label, mask, color in groups:
        ax.scatter(r_raw_dw[mask], r_corr_dw[mask], c=color, s=40, alpha=0.6,
                   edgecolors='k', linewidths=0.3, label=label, zorder=3)
    ax.axhline(0, color='k', alpha=0.2)
    ax.axvline(0, color='k', alpha=0.2)
    ax.set_xlabel('r(raw_CC, dW) across pairs')
    ax.set_ylabel('r(correction_CC, dW) across pairs')
    ax.set_title('Raw vs correction\nalignment with plasticity', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C: corr(correction_CC, dW) vs rho(RPE, correction_slope)
    # Asks: do sessions where the correction aligns with dW across pairs
    #       also show RPE modulation of the correction?
    ax = axes[2]
    for label, mask, color in groups:
        ax.scatter(r_corr_dw[mask], rho_corr[mask], c=color, s=40, alpha=0.6,
                   edgecolors='k', linewidths=0.3, label=label, zorder=3)
    ax.axhline(0, color='k', alpha=0.2)
    ax.axvline(0, color='k', alpha=0.2)
    ax.set_xlabel('r(correction_CC, dW) across pairs')
    ax.set_ylabel('rho(RPE, correction_slope)\nacross windows')
    ax.set_title('Pair-level alignment vs\nwindow-level RPE modulation', fontweight='bold')
    ax.legend(fontsize=8)
    x = r_corr_dw[has_pairs]
    y = rho_corr[has_pairs]
    ok_xy = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok_xy) >= 5:
        r, p = spearmanr(x[ok_xy], y[ok_xy])
        ax.text(0.05, 0.95, f'rho={r:+.3f}\np={p:.4f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Per-pair decomposition: correction alignment with plasticity',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rpe_flip_pair_analysis.png'),
                dpi=200, bbox_inches='tight')
    plt.show()

    # Stats
    print("\n--- Per-pair correction-dW alignment (Wilcoxon) ---")
    for label, mask in [('All', has_pairs), ('Flippers', flippers & has_pairs),
                        ('Both >0', same_pos & has_pairs), ('Both <0', same_neg & has_pairs)]:
        v = r_corr_dw[mask]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            _, p = wilcoxon(v)
            print(f"  {label:12s}: mean r={np.mean(v):+.3f}, p={p:.4f} (n={len(v)})")
    print("Per-pair analysis figure saved.")
