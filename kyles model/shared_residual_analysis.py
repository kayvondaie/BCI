"""
Shared residual analysis: distinguishing local vs non-local learning.

Key idea: Under the semi-local rule, Term 2 contributes a component to ΔW_ij
that depends on the postsynaptic neuron i (Σ_k W_ik * (h_k - h̄_k)) but is
shared across all presynaptic neurons j. Under local learning, no such shared
component exists.

Method:
  1. Regress ΔW on the local predictor (1st-order CC) → residuals
  2. Group residuals by postsynaptic neuron
  3. Compute variance of post-neuron mean residuals
  4. Compare to shuffled null (permuted post labels)

The ratio (real variance / shuffle variance) should be >1 for semi-local
and ~1 for local.

Usage: Run in Spyder.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model")

import numpy as np
import matplotlib.pyplot as plt
import json

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

N_BASELINE = 10
N_SHUFFLES = 200


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
def analyze_shared_residuals(neuron_means_all, W_initial, W_final, N, W_mask=None):
    """
    Compute post-neuron residual clustering metric.

    1. Build local predictor: session-summed 1st-order CC per synapse
    2. Regress ΔW on local predictor → residuals
    3. Group residuals by post neuron, compute mean per group
    4. Metric: variance of group means vs shuffle null

    Returns dict with variance_ratio, real_var, shuffle_var_median.
    """
    n_trials = neuron_means_all.shape[0]
    dW = W_final - W_initial

    if W_mask is not None:
        mask = (W_mask > 0) & (~np.eye(N, dtype=bool))
    else:
        mask = ~np.eye(N, dtype=bool)

    post_idx, pre_idx = np.where(mask)
    n_pairs = len(post_idx)
    Y = dW[mask]

    # Activity: (N, n_trials)
    act = neuron_means_all.T
    bl_mean = np.mean(act[:, :N_BASELINE], axis=1)
    pop_dev = act - bl_mean[:, None]

    # Local predictor: sum over trials of r_j(t) * (r_i(t) - r̄_i)
    r_pre = act[pre_idx, :]            # (n_pairs, n_trials)
    r_post_dev = pop_dev[post_idx, :]  # (n_pairs, n_trials)
    local_predictor = np.sum(r_pre * r_post_dev, axis=1)  # (n_pairs,)

    # Regress ΔW on local predictor → residuals
    # Simple OLS: Y = a + b*X + residual
    X = local_predictor
    X_mean = X.mean()
    Y_mean = Y.mean()
    b = np.sum((X - X_mean) * (Y - Y_mean)) / (np.sum((X - X_mean)**2) + 1e-30)
    a = Y_mean - b * X_mean
    Y_pred = a + b * X
    residuals = Y - Y_pred

    # Group residuals by post neuron
    unique_post = np.unique(post_idx)
    post_mean_resid = np.zeros(len(unique_post))
    for gi, pi in enumerate(unique_post):
        sel = post_idx == pi
        post_mean_resid[gi] = np.mean(residuals[sel])

    real_var = np.var(post_mean_resid)

    # Shuffle null: permute post-neuron labels
    rng = np.random.default_rng(42)
    shuffle_vars = np.zeros(N_SHUFFLES)
    for shi in range(N_SHUFFLES):
        shuf_post = rng.permutation(post_idx)
        shuf_means = np.zeros(len(unique_post))
        for gi, pi in enumerate(unique_post):
            sel = shuf_post == pi
            if np.any(sel):
                shuf_means[gi] = np.mean(residuals[sel])
        shuffle_vars[shi] = np.var(shuf_means)

    shuffle_median = np.median(shuffle_vars)
    variance_ratio = real_var / shuffle_median if shuffle_median > 0 else np.nan

    # Also compute R² of the local regression
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - Y_mean)**2)
    r2_local = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'variance_ratio': variance_ratio,
        'real_var': real_var,
        'shuffle_median': shuffle_median,
        'shuffle_vars': shuffle_vars,
        'r2_local': r2_local,
        'n_pairs': n_pairs,
    }


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

        result = analyze_shared_residuals(
            neuron_means_all, W_initial, net.W, p["N"], W_mask=net.W_mask)
        all_results[rule].append(result)

        print(f"  variance_ratio={result['variance_ratio']:.3f}  "
              f"real={result['real_var']:.6f}  "
              f"shuffle={result['shuffle_median']:.6f}  "
              f"R²_local={result['r2_local']:.3f}")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY: Post-neuron residual clustering")
print(f"{'='*60}")

summary = {}
for rule in RULES:
    ratios = [r['variance_ratio'] for r in all_results[rule]]
    m, s = np.mean(ratios), np.std(ratios)
    sem = s / np.sqrt(len(ratios))
    summary[rule] = {
        'mean_ratio': m, 'std_ratio': s, 'sem_ratio': sem,
        'ratios': ratios,
        'r2_locals': [r['r2_local'] for r in all_results[rule]],
    }
    print(f"  {rule:12s}: variance_ratio = {m:.3f} ± {sem:.3f}  (n={len(ratios)})")

# ──────────────────────────────────────────────────────────────────────
# Save JSON
# ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

json_out = {}
for rule in RULES:
    json_out[rule] = {
        'mean_ratio': float(summary[rule]['mean_ratio']),
        'std_ratio': float(summary[rule]['std_ratio']),
        'ratios': [float(v) for v in summary[rule]['ratios']],
        'r2_locals': [float(v) for v in summary[rule]['r2_locals']],
    }
json_path = os.path.join(SAVE_DIR, "shared_residual_analysis.json")
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"\nSaved → {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"Shared residual analysis (sparsity={SPARSITY}, scale={SEMI_LOCAL_SCALE}, eta={ETA})\n"
             f"N_runs={N_RUNS}, N_shuffles={N_SHUFFLES}",
             fontsize=12)

# --- Panel A: Variance ratio by rule ---
ax = axes[0]
colors = {"local": "steelblue", "semi_local": "tomato"}

for ri, rule in enumerate(RULES):
    ratios = summary[rule]['ratios']
    m = summary[rule]['mean_ratio']
    sem = summary[rule]['sem_ratio']
    ax.bar(ri, m, yerr=sem, capsize=6, color=colors[rule],
           alpha=0.8, edgecolor='k', linewidth=1, label=rule)
    # Overlay individual dots
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(ratios))
    ax.scatter(np.full(len(ratios), ri) + jitter, ratios,
               color=colors[rule], s=30, alpha=0.6, edgecolor='none', zorder=3)

ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='null (ratio=1)')
ax.set_xticks(range(len(RULES)))
ax.set_xticklabels(RULES)
ax.set_ylabel("Variance ratio\n(real / shuffle)")
ax.set_title("Post-neuron residual clustering", fontweight='bold')
ax.legend(fontsize=9)

# --- Panel B: Paired dots (local vs semi-local) ---
ax = axes[1]
local_ratios = summary['local']['ratios']
semi_ratios = summary['semi_local']['ratios']
n = min(len(local_ratios), len(semi_ratios))

for v1, v2 in zip(local_ratios[:n], semi_ratios[:n]):
    ax.plot([0, 1], [v1, v2], color='gray', alpha=0.4, lw=0.8)
ax.scatter(np.zeros(n), local_ratios[:n], color=colors['local'], s=40, zorder=3)
ax.scatter(np.ones(n), semi_ratios[:n], color=colors['semi_local'], s=40, zorder=3)
ax.scatter([0], [np.mean(local_ratios[:n])], color=colors['local'],
           s=120, marker='D', edgecolor='k', zorder=5)
ax.scatter([1], [np.mean(semi_ratios[:n])], color=colors['semi_local'],
           s=120, marker='D', edgecolor='k', zorder=5)
ax.axhline(1.0, color='k', ls='--', alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(RULES)
ax.set_ylabel("Variance ratio")
ax.set_title("Paired comparison", fontweight='bold')

# --- Panel C: Example shuffle distribution for one run ---
ax = axes[2]
# Show last run of each rule
for rule in RULES:
    last = all_results[rule][-1]
    ax.hist(last['shuffle_vars'], bins=30, alpha=0.5, color=colors[rule],
            label=f"{rule} shuffle", density=True)
    ax.axvline(last['real_var'], color=colors[rule], ls='-', lw=2,
               label=f"{rule} real")
ax.set_xlabel("Variance of post-neuron mean residuals")
ax.set_ylabel("Density")
ax.set_title("Example: real vs shuffle distribution\n(last run)", fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "shared_residual_analysis.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved figure → {fig_path}")
