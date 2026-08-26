"""
Direct Term 2 analysis: compare actual ΔW to ground-truth Term 1 and Term 2
contributions.

Since we have access to the model internals, we can accumulate the actual
weight updates attributable to Term 1 (local) and Term 2 (semi-local)
separately. Then we ask:

  1. How much of ΔW is explained by Term 1 vs Term 2?
  2. After regressing out Term 1's contribution, does the residual
     correlate with Term 2's contribution?

For the LOCAL rule:
  - dW_from_local IS the total ΔW (no Term 2)
  - The residual after removing Term 1 should be ~zero
  - Correlation of residual with the *theoretical* Term 2 prediction
    (computed post-hoc) should be low

For the SEMI-LOCAL rule:
  - dW_from_local + dW_from_semi = total ΔW
  - The residual after removing Term 1 should correlate with dW_from_semi
  - This correlation should be higher than for the local rule

The key metric: partial correlation of ΔW with the Term 2 predictor
after controlling for Term 1. This should be high for semi-local, ~0 for local.

Usage: Run in Spyder.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model")

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

from bci_model import DEFAULTS, train

# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────
N_RUNS = 10
RULES = ["local", "semi_local"]
SAVE_DIR = r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\results"

SPARSITY = 1.0
SEMI_LOCAL_SCALE = 1.0
ETA = 1e-2


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
def analyze_direct(net, W_initial, neuron_means_all):
    """
    Analyze the direct Term 1 and Term 2 contributions to ΔW.

    For semi-local: uses net.dW_from_local and net.dW_from_semi (ground truth).
    For local: uses net.dW_from_local as ground truth, and computes a
               post-hoc Term 2 predictor from W_initial and activity.

    Returns dict with correlations and partial correlations.
    """
    N = net.p["N"]
    mask = ~np.eye(N, dtype=bool)
    if hasattr(net, 'W_mask'):
        mask = mask & (net.W_mask > 0)

    dW_total = (net.W - W_initial)[mask]
    dW_local = net.dW_from_local[mask]

    # For semi-local, we have ground truth Term 2
    has_semi = hasattr(net, 'dW_from_semi')
    if has_semi:
        dW_semi = net.dW_from_semi[mask]
    else:
        dW_semi = np.zeros_like(dW_total)

    # Compute post-hoc Term 2 predictor from activity and W_initial
    # This is what the 2nd-order CC approximates:
    # For each pair (i,j): sum_t r_j(t) * Σ_k W_ik * (r_k(t) - r̄_k)
    n_trials = neuron_means_all.shape[0]
    act = neuron_means_all.T  # (N, n_trials)
    bl_mean = np.mean(act[:, :10], axis=1)
    pop_dev = act - bl_mean[:, None]

    post_idx, pre_idx = np.where(mask)

    # Post-hoc Term 2 predictor (from activity + initial W)
    ds = W_initial @ pop_dev  # (N, n_trials)
    r_pre = act[pre_idx, :]
    ds_post = ds[post_idx, :]
    posthoc_term2 = np.sum(r_pre * ds_post, axis=1)  # (n_pairs,)

    n_pairs = len(dW_total)

    result = {}

    # --- Correlations ---
    # r(ΔW, Term1)
    r_dw_t1, p_dw_t1 = pearsonr(dW_total, dW_local)
    result['r_dw_term1'] = r_dw_t1

    # r(ΔW, ground-truth Term2)
    if has_semi and np.std(dW_semi) > 1e-15:
        r_dw_t2, _ = pearsonr(dW_total, dW_semi)
    else:
        r_dw_t2 = 0.0
    result['r_dw_term2_gt'] = r_dw_t2

    # r(ΔW, post-hoc Term2 predictor)
    if np.std(posthoc_term2) > 1e-15:
        r_dw_ph, _ = pearsonr(dW_total, posthoc_term2)
    else:
        r_dw_ph = 0.0
    result['r_dw_term2_posthoc'] = r_dw_ph

    # --- Partial correlations ---
    # Regress out Term 1 from ΔW → residual, then correlate with Term 2
    def partial_corr(Y, X_remove, X_test):
        """Correlation of Y with X_test after regressing out X_remove."""
        if np.std(X_remove) < 1e-15 or np.std(X_test) < 1e-15:
            return 0.0, 1.0
        # Residualize Y
        b = np.dot(Y - Y.mean(), X_remove - X_remove.mean()) / np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean())
        Y_resid = Y - (Y.mean() + b * (X_remove - X_remove.mean()))
        # Residualize X_test
        b2 = np.dot(X_test - X_test.mean(), X_remove - X_remove.mean()) / np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean())
        X_resid = X_test - (X_test.mean() + b2 * (X_remove - X_remove.mean()))
        if np.std(Y_resid) < 1e-15 or np.std(X_resid) < 1e-15:
            return 0.0, 1.0
        return pearsonr(Y_resid, X_resid)

    # Partial: ΔW ~ ground-truth Term2 | Term1
    if has_semi and np.std(dW_semi) > 1e-15:
        r_partial_gt, p_partial_gt = partial_corr(dW_total, dW_local, dW_semi)
    else:
        r_partial_gt, p_partial_gt = 0.0, 1.0
    result['r_partial_term2_gt'] = r_partial_gt
    result['p_partial_term2_gt'] = p_partial_gt

    # Partial: ΔW ~ post-hoc Term2 predictor | Term1
    r_partial_ph, p_partial_ph = partial_corr(dW_total, dW_local, posthoc_term2)
    result['r_partial_term2_posthoc'] = r_partial_ph
    result['p_partial_term2_posthoc'] = p_partial_ph

    # Fraction of ΔW variance from each term
    if np.var(dW_total) > 1e-30:
        result['frac_var_term1'] = np.var(dW_local) / np.var(dW_total)
        result['frac_var_term2'] = np.var(dW_semi) / np.var(dW_total) if has_semi else 0.0
    else:
        result['frac_var_term1'] = 0.0
        result['frac_var_term2'] = 0.0

    result['n_pairs'] = n_pairs

    return result


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
all_results = {rule: [] for rule in RULES}

for rule in RULES:
    print(f"\n{'='*60}")
    print(f"  Rule: {rule}")
    print(f"{'='*60}")

    for run in range(N_RUNS):
        print(f"\n--- Run {run+1}/{N_RUNS} ---")
        p = dict(DEFAULTS)
        p["learning_rule"] = rule
        p["semi_local_scale"] = SEMI_LOCAL_SCALE
        p["eta"] = ETA
        p["sparsity"] = SPARSITY
        p["seed"] = None

        (net, trial_logs, W_initial, W_snapshots,
         neuron_means_all, neuron_maxs_all, coact_session,
         coact_rpe_session) = train(p)

        result = analyze_direct(net, W_initial, neuron_means_all)
        all_results[rule].append(result)

        print(f"  r(ΔW,T1)={result['r_dw_term1']:.3f}  "
              f"r(ΔW,T2_gt)={result['r_dw_term2_gt']:.3f}  "
              f"partial(T2_gt|T1)={result['r_partial_term2_gt']:.3f}  "
              f"partial(T2_ph|T1)={result['r_partial_term2_posthoc']:.3f}")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

metrics = ['r_dw_term1', 'r_dw_term2_gt', 'r_partial_term2_gt', 'r_partial_term2_posthoc']
metric_labels = {
    'r_dw_term1': 'r(ΔW, Term1)',
    'r_dw_term2_gt': 'r(ΔW, Term2 ground-truth)',
    'r_partial_term2_gt': 'partial r(ΔW, Term2 GT | Term1)',
    'r_partial_term2_posthoc': 'partial r(ΔW, Term2 post-hoc | Term1)',
}

summary = {}
for rule in RULES:
    summary[rule] = {}
    print(f"\n{rule}:")
    for m in metrics:
        vals = [r[m] for r in all_results[rule]]
        mean_v, std_v = np.mean(vals), np.std(vals)
        sem_v = std_v / np.sqrt(len(vals))
        summary[rule][m] = {'mean': mean_v, 'std': std_v, 'sem': sem_v, 'values': vals}
        print(f"  {metric_labels[m]:45s}: {mean_v:.4f} ± {sem_v:.4f}")

# ──────────────────────────────────────────────────────────────────────
# Save JSON
# ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

json_out = {}
for rule in RULES:
    json_out[rule] = {}
    for m in metrics:
        d = summary[rule][m]
        json_out[rule][m] = {
            'mean': float(d['mean']), 'std': float(d['std']),
            'values': [float(v) for v in d['values']],
        }
json_path = os.path.join(SAVE_DIR, "direct_term2_analysis.json")
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"\nSaved → {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────
colors = {"local": "steelblue", "semi_local": "tomato"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Direct Term 2 analysis (sparsity={SPARSITY}, scale={SEMI_LOCAL_SCALE}, eta={ETA})\n"
             f"N_runs={N_RUNS}", fontsize=12)

# --- Panel A: Correlations with ΔW ---
ax = axes[0]
plot_metrics = ['r_dw_term1', 'r_dw_term2_gt']
plot_labels = ['r(ΔW, T1)', 'r(ΔW, T2 GT)']
x = np.arange(len(plot_metrics))
width = 0.35

for ri, rule in enumerate(RULES):
    means = [summary[rule][m]['mean'] for m in plot_metrics]
    sems = [summary[rule][m]['sem'] for m in plot_metrics]
    offset = (ri - 0.5) * width
    ax.bar(x + offset, means, width * 0.9, yerr=sems, capsize=4,
           color=colors[rule], alpha=0.8, edgecolor='k', linewidth=0.8, label=rule)

ax.set_xticks(x)
ax.set_xticklabels(plot_labels, fontsize=9)
ax.set_ylabel("Pearson r")
ax.set_title("Correlations with ΔW", fontweight='bold')
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.legend(fontsize=10)

# --- Panel B: Partial correlations (the key metric) ---
ax = axes[1]
partial_metrics = ['r_partial_term2_gt', 'r_partial_term2_posthoc']
partial_labels = ['Term2 GT | Term1', 'Term2 post-hoc | Term1']
x2 = np.arange(len(partial_metrics))

for ri, rule in enumerate(RULES):
    means = [summary[rule][m]['mean'] for m in partial_metrics]
    sems = [summary[rule][m]['sem'] for m in partial_metrics]
    offset = (ri - 0.5) * width
    ax.bar(x2 + offset, means, width * 0.9, yerr=sems, capsize=4,
           color=colors[rule], alpha=0.8, edgecolor='k', linewidth=0.8, label=rule)

    # Overlay dots
    for mi, m in enumerate(partial_metrics):
        vals = summary[rule][m]['values']
        jitter = np.random.default_rng(ri).uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), x2[mi] + offset) + jitter, vals,
                   color=colors[rule], s=20, alpha=0.5, edgecolor='none', zorder=3)

ax.set_xticks(x2)
ax.set_xticklabels(partial_labels, fontsize=9)
ax.set_ylabel("Partial r")
ax.set_title("Partial correlation\n(ΔW ~ Term2 | Term1)", fontweight='bold')
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.legend(fontsize=10)

# --- Panel C: Paired comparison of post-hoc partial r ---
ax = axes[2]
local_vals = summary['local']['r_partial_term2_posthoc']['values']
semi_vals = summary['semi_local']['r_partial_term2_posthoc']['values']
n = min(len(local_vals), len(semi_vals))

for v1, v2 in zip(local_vals[:n], semi_vals[:n]):
    ax.plot([0, 1], [v1, v2], color='gray', alpha=0.4, lw=0.8)
ax.scatter(np.zeros(n), local_vals[:n], color=colors['local'], s=40, zorder=3)
ax.scatter(np.ones(n), semi_vals[:n], color=colors['semi_local'], s=40, zorder=3)
ax.scatter([0], [np.mean(local_vals[:n])], color=colors['local'],
           s=120, marker='D', edgecolor='k', zorder=5)
ax.scatter([1], [np.mean(semi_vals[:n])], color=colors['semi_local'],
           s=120, marker='D', edgecolor='k', zorder=5)
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(RULES)
ax.set_ylabel("Partial r (Term2 post-hoc | Term1)")
ax.set_title("Paired: post-hoc partial r", fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "direct_term2_analysis.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved figure → {fig_path}")
