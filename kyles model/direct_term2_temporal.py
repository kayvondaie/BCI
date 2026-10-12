"""
Direct Term 2 analysis with temporal resolution.

Same as direct_term2_analysis.py but the post-hoc Term 2 predictor is
computed at timestep resolution (matching ~60 Hz imaging) rather than
trial-averaged rates.

Post-hoc Term 2 predictor (per synapse i,j):
  Σ_t RPE(t) * r_j(t) * [Σ_k W_ik * (r_k(t) - r̄_k)]

This includes:
  - Frame-by-frame activity (not trial averages)
  - RPE weighting at each timestep
  - Deviation from running baseline

Usage: Run in Spyder.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model")

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

from bci_model import DEFAULTS, train, phi_prime

# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────
N_RUNS = 10
RULES = ["local", "semi_local"]
SAVE_DIR = r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\results"

SPARSITY = DEFAULTS["sparsity"]
SEMI_LOCAL_SCALE = DEFAULTS["semi_local_scale"]
ETA = DEFAULTS["eta"]
ACTIVATION = DEFAULTS["activation"]

# Downsample factor: model runs at 100 Hz (dt=0.01), imaging at 60 Hz
# Use every frame (DS=1) or downsample to ~60 Hz (DS=2 → 50 Hz)
DS = 1  # keep full resolution for now


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────
def compute_posthoc_term2_temporal(rates_all_trials, rpe_all_trials,
                                   W_initial, N, p, h_bar_all_trials=None,
                                   W_mask=None, ds=1):
    """
    Compute the post-hoc Term 2 predictor using timestep-level activity,
    including φ'(h) and eligibility trace accumulation.

    Uses h_bar traces from training if provided, matching the model's
    exponential running average baseline exactly.

    Returns (N, N) matrix of predictor values.
    """
    n_trials = len(rates_all_trials)
    dt_tau_elig = p["dt"] / p["tau_elig"]

    posthoc = np.zeros((N, N))

    for trial_idx in range(n_trials):
        rates = rates_all_trials[trial_idx]
        rpe = rpe_all_trials[trial_idx]
        n_steps = rates.shape[0]

        # Baseline: use h_bar from training (matches model's tau_bl)
        if h_bar_all_trials is not None:
            baseline = h_bar_all_trials[trial_idx]  # (n_steps, N)
        else:
            bl = np.mean(rates[:10, :], axis=0)
            baseline = np.tile(bl, (n_steps, 1))

        act = p.get("activation", "tanh")
        dphi = phi_prime(rates, act)
        dev = rates - baseline
        weighted_dev = dev * dphi
        bp = weighted_dev @ W_initial  # (n_steps, N)

        # Accumulate eligibility trace and apply RPE
        E_bar = np.zeros((N, N))
        for t in range(1, n_steps):
            # Instantaneous Term 2 eligibility
            signal_i = bp[t, :] * dphi[t-1, :]  # (N,)
            E_inst = signal_i[:, None] * rates[t-1, None, :]  # (N, N)

            # Leaky accumulation
            E_bar = (1 - dt_tau_elig) * E_bar + dt_tau_elig * E_inst

            # Weight update contribution
            posthoc += rpe[t] * E_bar

    return posthoc


def compute_posthoc_term1_temporal(rates_all_trials, rpe_all_trials,
                                    N, p, h_bar_all_trials=None,
                                    W_mask=None, ds=1):
    """
    Compute the post-hoc Term 1 predictor using timestep-level activity,
    including φ'(h) and eligibility trace accumulation.

    Returns (N, N) matrix.
    """
    n_trials = len(rates_all_trials)
    dt_tau_elig = p["dt"] / p["tau_elig"]

    posthoc = np.zeros((N, N))

    for trial_idx in range(n_trials):
        rates = rates_all_trials[trial_idx]
        rpe = rpe_all_trials[trial_idx]
        n_steps = rates.shape[0]

        # Baseline: use h_bar from training (matches model's tau_bl)
        if h_bar_all_trials is not None:
            baseline = h_bar_all_trials[trial_idx]  # (n_steps, N)
        else:
            bl = np.mean(rates[:10, :], axis=0)
            baseline = np.tile(bl, (n_steps, 1))

        act = p.get("activation", "tanh")
        dphi = phi_prime(rates, act)
        dev = rates - baseline

        E_bar = np.zeros((N, N))
        for t in range(1, n_steps):
            signal_i = dev[t, :] * dphi[t, :]  # (N,)
            E_inst = signal_i[:, None] * rates[t-1, None, :]  # (N, N)

            E_bar = (1 - dt_tau_elig) * E_bar + dt_tau_elig * E_inst

            posthoc += rpe[t] * E_bar

    return posthoc


def compute_posthoc_term2_h(h_all_trials, h_bar_all_trials, rates_all_trials,
                            rpe_all_trials, W_initial, N, p):
    """
    Post-hoc Term 2 using ground-truth h and h_bar (not observable).
    This is a diagnostic to check if h vs r is the bottleneck.
    """
    n_trials = len(rates_all_trials)
    dt_tau_elig = p["dt"] / p["tau_elig"]

    posthoc = np.zeros((N, N))

    for trial_idx in range(n_trials):
        rates = rates_all_trials[trial_idx]
        rpe = rpe_all_trials[trial_idx]
        h = h_all_trials[trial_idx]
        h_bar = h_bar_all_trials[trial_idx]
        n_steps = rates.shape[0]

        # Use actual h for φ'
        act = p.get("activation", "tanh")
        dphi = phi_prime(h, act)  # (n_steps, N)
        delta_h = h - h_bar  # (n_steps, N)

        # Backprop signal using h-space deviation
        weighted_dev = delta_h * dphi  # c_k=1
        bp = weighted_dev @ W_initial  # (n_steps, N)

        E_bar = np.zeros((N, N))
        for t in range(1, n_steps):
            dphi_prev = phi_prime(h[t-1, :], act)
            signal_i = bp[t, :] * dphi_prev
            E_inst = signal_i[:, None] * rates[t-1, None, :]
            E_bar = (1 - dt_tau_elig) * E_bar + dt_tau_elig * E_inst
            posthoc += rpe[t] * E_bar

    return posthoc


def compute_posthoc_term1_h(h_all_trials, h_bar_all_trials, rates_all_trials,
                             rpe_all_trials, N, p):
    """Post-hoc Term 1 using ground-truth h and h_bar."""
    n_trials = len(rates_all_trials)
    dt_tau_elig = p["dt"] / p["tau_elig"]

    posthoc = np.zeros((N, N))

    for trial_idx in range(n_trials):
        rates = rates_all_trials[trial_idx]
        rpe = rpe_all_trials[trial_idx]
        h = h_all_trials[trial_idx]
        h_bar = h_bar_all_trials[trial_idx]
        n_steps = rates.shape[0]

        act = p.get("activation", "tanh")
        dphi = phi_prime(h, act)
        delta_h = h - h_bar

        E_bar = np.zeros((N, N))
        for t in range(1, n_steps):
            signal_i = delta_h[t, :] * dphi[t, :]
            E_inst = signal_i[:, None] * rates[t-1, None, :]
            E_bar = (1 - dt_tau_elig) * E_bar + dt_tau_elig * E_inst
            posthoc += rpe[t] * E_bar

    return posthoc


def analyze_direct_temporal(net, W_initial, neuron_means_all,
                             rates_all_trials, rpe_all_trials,
                             h_all_trials=None, h_bar_all_trials=None):
    """Run the full analysis with temporal post-hoc predictors."""
    N = net.p["N"]
    mask = ~np.eye(N, dtype=bool)
    if hasattr(net, 'W_mask'):
        mask = mask & (net.W_mask > 0)

    dW_total = (net.W - W_initial)[mask]
    dW_local = net.dW_from_local[mask]

    has_semi = hasattr(net, 'dW_from_semi')
    dW_semi = net.dW_from_semi[mask] if has_semi else np.zeros_like(dW_total)

    # Post-hoc predictors (temporal resolution with eligibility trace)
    ph_t2 = compute_posthoc_term2_temporal(
        rates_all_trials, rpe_all_trials, W_initial, N, net.p,
        h_bar_all_trials=h_bar_all_trials,
        W_mask=getattr(net, 'W_mask', None), ds=DS)
    ph_t1 = compute_posthoc_term1_temporal(
        rates_all_trials, rpe_all_trials, N, net.p,
        h_bar_all_trials=h_bar_all_trials,
        W_mask=getattr(net, 'W_mask', None), ds=DS)

    ph_t2_flat = ph_t2[mask]
    ph_t1_flat = ph_t1[mask]

    # Also compute trial-averaged post-hoc for comparison
    act = neuron_means_all.T
    bl_mean = np.mean(act[:, :10], axis=1)
    pop_dev = act - bl_mean[:, None]
    post_idx, pre_idx = np.where(mask)
    ds_avg = W_initial @ pop_dev
    ph_t2_avg = np.sum(act[pre_idx, :] * ds_avg[post_idx, :], axis=1)

    def partial_corr(Y, X_remove, X_test):
        if np.std(X_remove) < 1e-15 or np.std(X_test) < 1e-15:
            return 0.0, 1.0
        b = np.dot(Y - Y.mean(), X_remove - X_remove.mean()) / (np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean()) + 1e-30)
        Y_resid = Y - (Y.mean() + b * (X_remove - X_remove.mean()))
        b2 = np.dot(X_test - X_test.mean(), X_remove - X_remove.mean()) / (np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean()) + 1e-30)
        X_resid = X_test - (X_test.mean() + b2 * (X_remove - X_remove.mean()))
        if np.std(Y_resid) < 1e-15 or np.std(X_resid) < 1e-15:
            return 0.0, 1.0
        return pearsonr(Y_resid, X_resid)

    result = {}

    # Ground-truth correlations
    result['r_dw_term1'] = pearsonr(dW_total, dW_local)[0] if np.std(dW_local) > 1e-15 else 0.0
    result['r_dw_term2_gt'] = pearsonr(dW_total, dW_semi)[0] if (has_semi and np.std(dW_semi) > 1e-15) else 0.0

    # Partial: ground-truth Term2 | Term1
    if has_semi and np.std(dW_semi) > 1e-15:
        result['r_partial_gt'], _ = partial_corr(dW_total, dW_local, dW_semi)
    else:
        result['r_partial_gt'] = 0.0

    # Partial: temporal post-hoc Term2 | Term1 (using temporal post-hoc T1 as control)
    result['r_partial_temporal'], _ = partial_corr(dW_total, ph_t1_flat, ph_t2_flat)

    # Partial: trial-averaged post-hoc (for comparison)
    r_pre_avg = act[pre_idx, :]
    r_post_dev_avg = pop_dev[post_idx, :]
    ph_t1_avg = np.sum(r_pre_avg * r_post_dev_avg, axis=1)
    result['r_partial_trial_avg'], _ = partial_corr(dW_total, ph_t1_avg, ph_t2_avg)

    # Also: partial using ground-truth Term1 as control, temporal post-hoc Term2
    result['r_partial_temporal_vs_gt_t1'], _ = partial_corr(dW_total, dW_local, ph_t2_flat)

    # Raw correlations of post-hoc predictors
    result['r_dw_ph_t1_temporal'] = pearsonr(dW_total, ph_t1_flat)[0] if np.std(ph_t1_flat) > 1e-15 else 0.0
    result['r_dw_ph_t2_temporal'] = pearsonr(dW_total, ph_t2_flat)[0] if np.std(ph_t2_flat) > 1e-15 else 0.0

    # Diagnostic: h-based predictor (uses ground-truth h, h_bar — not observable)
    if h_all_trials is not None:
        ph_t2_h = compute_posthoc_term2_h(
            h_all_trials, h_bar_all_trials, rates_all_trials,
            rpe_all_trials, W_initial, N, net.p)
        ph_t1_h = compute_posthoc_term1_h(
            h_all_trials, h_bar_all_trials, rates_all_trials,
            rpe_all_trials, N, net.p)
        ph_t2_h_flat = ph_t2_h[mask]
        ph_t1_h_flat = ph_t1_h[mask]
        result['r_partial_h_based'], _ = partial_corr(dW_total, ph_t1_h_flat, ph_t2_h_flat)
    else:
        result['r_partial_h_based'] = np.nan

    # ── Simplified predictors: just raw coactivity, no baselines/φ'/elig ──
    # Local:     Σ_t RPE(t) * r_i(t) * r_j(t-1)
    # Non-local: Σ_t RPE(t) * [Σ_k W_ki * r_k(t)] * r_j(t-1)
    simple_local = np.zeros((N, N))
    simple_nonlocal = np.zeros((N, N))
    for trial_idx in range(len(rates_all_trials)):
        rates = rates_all_trials[trial_idx]
        rpe = rpe_all_trials[trial_idx]
        n_steps = rates.shape[0]
        for t in range(1, n_steps):
            r_post = rates[t, :]        # r_i(t)
            r_pre = rates[t-1, :]       # r_j(t-1)
            bp = r_post @ W_initial     # Σ_k W_ki * r_k(t), shape (N,)
            simple_local += rpe[t] * r_post[:, None] * r_pre[None, :]
            simple_nonlocal += rpe[t] * bp[:, None] * r_pre[None, :]

    simple_local_flat = simple_local[mask]
    simple_nonlocal_flat = simple_nonlocal[mask]
    result['r_partial_simple'], _ = partial_corr(dW_total, simple_local_flat, simple_nonlocal_flat)

    return result


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
all_results = {rule: [] for rule in RULES}

for rule in RULES:
    print(f"\n{'='*60}")
    print(f"  Rule: {rule}")
    print(f"{'='*60}")

    run = 0
    while len(all_results[rule]) < N_RUNS:
        run += 1
        print(f"\n--- Run {len(all_results[rule])+1}/{N_RUNS} (attempt {run}) ---")
        try:
            p = dict(DEFAULTS)
            p["learning_rule"] = rule
            p["semi_local_scale"] = SEMI_LOCAL_SCALE
            p["eta"] = ETA
            p["sparsity"] = SPARSITY
            p["activation"] = ACTIVATION
            p["seed"] = None

            (net, trial_logs, W_initial, W_snapshots,
             neuron_means_all, neuron_maxs_all, coact_session,
             coact_rpe_session, rates_all_trials, rpe_all_trials,
             h_all_trials, h_bar_all_trials) = train(p)

            # Check for NaNs in weights
            if np.any(np.isnan(net.W)):
                print("  ** NaN detected in weights, skipping run **")
                continue

            result = analyze_direct_temporal(net, W_initial, neuron_means_all,
                                              rates_all_trials, rpe_all_trials,
                                              h_all_trials, h_bar_all_trials)
            all_results[rule].append(result)

            print(f"  GT partial={result['r_partial_gt']:.3f}  "
                  f"h-based={result['r_partial_h_based']:.3f}  "
                  f"r-based={result['r_partial_temporal']:.3f}  "
                  f"simple={result['r_partial_simple']:.3f}  "
                  f"trial-avg={result['r_partial_trial_avg']:.3f}")
        except Exception as e:
            print(f"  ** Run failed ({e}), skipping **")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

metrics = ['r_dw_term1', 'r_dw_term2_gt', 'r_partial_gt',
           'r_partial_h_based', 'r_partial_temporal', 'r_partial_simple',
           'r_partial_trial_avg', 'r_partial_temporal_vs_gt_t1']
metric_labels = {
    'r_dw_term1': 'r(ΔW, Term1 GT)',
    'r_dw_term2_gt': 'r(ΔW, Term2 GT)',
    'r_partial_gt': 'partial(T2 GT | T1 GT)',
    'r_partial_h_based': 'partial(T2 h-based | T1 h-based)  [uses GT h,h_bar]',
    'r_partial_temporal': 'partial(T2 r-based | T1 r-based)  [observable]',
    'r_partial_simple': 'partial(T2 simple | T1 simple)  [RPE*Wr*r]',
    'r_partial_trial_avg': 'partial(T2 trial-avg | T1 trial-avg)',
    'r_partial_temporal_vs_gt_t1': 'partial(T2 r-based temporal | T1 GT)',
}

summary = {}
for rule in RULES:
    summary[rule] = {}
    print(f"\n{rule}:")
    for m in metrics:
        vals = [r[m] for r in all_results[rule]]
        mean_v = np.mean(vals)
        sem_v = np.std(vals) / np.sqrt(len(vals))
        summary[rule][m] = {'mean': mean_v, 'sem': sem_v, 'values': vals}
        print(f"  {metric_labels[m]:55s}: {mean_v:.4f} ± {sem_v:.4f}")

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
            'mean': float(d['mean']), 'sem': float(d['sem']),
            'values': [float(v) for v in d['values']],
        }
json_path = os.path.join(SAVE_DIR, "direct_term2_temporal.json")
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"\nSaved → {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────
colors = {"local": "steelblue", "semi_local": "tomato"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(f"Direct Term 2 analysis: temporal vs trial-averaged post-hoc predictors\n"
             f"(sparsity={SPARSITY}, scale={SEMI_LOCAL_SCALE}, eta={ETA}, act={ACTIVATION}, N_runs={N_RUNS})",
             fontsize=12)

# --- Panel A: All partial correlations ---
ax = axes[0]
partial_metrics = ['r_partial_gt', 'r_partial_h_based', 'r_partial_temporal', 'r_partial_simple', 'r_partial_trial_avg']
partial_labels = ['GT\n(T2|T1)', 'h-based\n(GT h,h_bar)', 'r-based\n(observable)', 'simple\n(RPE*Wr*r)', 'trial-avg']
x = np.arange(len(partial_metrics))
width = 0.35

for ri, rule in enumerate(RULES):
    means = [summary[rule][m]['mean'] for m in partial_metrics]
    sems = [summary[rule][m]['sem'] for m in partial_metrics]
    offset = (ri - 0.5) * width
    ax.bar(x + offset, means, width * 0.9, yerr=sems, capsize=4,
           color=colors[rule], alpha=0.8, edgecolor='k', linewidth=0.8, label=rule)

ax.set_xticks(x)
ax.set_xticklabels(partial_labels, fontsize=9)
ax.set_ylabel("Partial r (ΔW ~ T2 | T1)")
ax.set_title("Partial correlations", fontweight='bold')
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.legend(fontsize=10)

# --- Panel B: Paired comparison — temporal post-hoc ---
ax = axes[1]
local_vals = summary['local']['r_partial_temporal']['values']
semi_vals = summary['semi_local']['r_partial_temporal']['values']
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
ax.set_ylabel("Partial r (temporal post-hoc)")
ax.set_title("Paired: temporal post-hoc T2|T1", fontweight='bold')

# --- Panel C: Temporal vs trial-avg comparison ---
ax = axes[2]
for ri, rule in enumerate(RULES):
    temporal = summary[rule]['r_partial_temporal']['values']
    trial_avg = summary[rule]['r_partial_trial_avg']['values']
    n_r = min(len(temporal), len(trial_avg))
    color = colors[rule]
    for v1, v2 in zip(trial_avg[:n_r], temporal[:n_r]):
        ax.plot([0, 1], [v1, v2], color=color, alpha=0.3, lw=0.8)
    jitter = (ri - 0.5) * 0.05
    ax.scatter(np.zeros(n_r) + jitter, trial_avg[:n_r], color=color, s=30, alpha=0.7, zorder=3)
    ax.scatter(np.ones(n_r) + jitter, temporal[:n_r], color=color, s=30, alpha=0.7, zorder=3)
    ax.scatter([0 + jitter], [np.mean(trial_avg[:n_r])], color=color,
               s=100, marker='D', edgecolor='k', zorder=5, label=rule)
    ax.scatter([1 + jitter], [np.mean(temporal[:n_r])], color=color,
               s=100, marker='D', edgecolor='k', zorder=5)

ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Trial-averaged", "Temporal (frame-by-frame)"])
ax.set_ylabel("Partial r (T2 | T1)")
ax.set_title("Trial-avg vs temporal resolution", fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "direct_term2_temporal.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved figure → {fig_path}")
