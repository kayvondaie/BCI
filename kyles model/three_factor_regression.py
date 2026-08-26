"""
Three-factor regression analysis for BCI model.

Mirrors the analysis in three_factor_variance_explained_2nd_order.py,
adapted for simulated data where W is known exactly.

For each pair (i,j) with i=postsynaptic, j=presynaptic:
  - 1st order CC:  r_j(t) * (r_i(t) - r̄_i)
  - 2nd order CC:  r_j(t) * Σ_k W_{ik} * (r_k(t) - r̄_k)   [outgoing from post]
  - combined:      both 1st + 2nd order features
  - shuffled:      2nd order with permuted W rows

The 3rd factor (HI / RPE) is treated as a free parameter:
  - Sliding windows across trials give features with varying effective RPE
  - Cross-validated regression fits the mapping from CC features to ΔW
  - The regression coefficients per window implicitly capture the RPE modulation

Usage:
    Run in Spyder. Trains both local and semi_local rules, then compares.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

from bci_model import DEFAULTS, train

# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────
N_RUNS = 10
SAVE_DIR = r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\results"

WIN_SIZE = 10       # trials per window
WIN_STEP = 5        # stride
N_CV_FOLDS = 5
N_SHUFFLES = 50     # connectivity shuffles for null distribution
N_BASELINE = 10     # trials for baseline mean

CC_MODES = ['1st_order', '2nd_order', 'combined', 'shuffled']

# Sparse connectivity (fraction of non-zero connections)
SPARSITY = 0.1

# Learning rate sweep to find what works with sparse connectivity
ETA_VALUES = [2e-2]

# Compare local vs semi_local (scale=1.0) across learning rates
CONDITIONS = []  # (rule, scale, eta)
for eta in ETA_VALUES:
    CONDITIONS.append(("local", 1.0, eta))
    CONDITIONS.append(("semi_local", 1.0, eta))

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def zscore_cols(X):
    """Z-score each column of X."""
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig == 0] = 1.0
    return (X - mu) / sig


def run_cv(X_mode, Y_mode, n_cv_folds=5):
    """Cross-validated regression: X -> Y using pinv.
    Returns test r (Spearman), train r, p-value, pooled predictions, and R²."""
    n = len(Y_mode)
    if n < n_cv_folds * 2:
        return np.nan, np.nan, 1.0, np.array([]), np.array([]), np.nan

    cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    Y_test_all, Y_pred_all, r_train_folds = [], [], []

    for train_idx, test_idx in cv.split(X_mode):
        X_train, X_test = X_mode[train_idx], X_mode[test_idx]
        Y_train, Y_test = Y_mode[train_idx], Y_mode[test_idx]

        mu_y, sig_y = Y_train.mean(), Y_train.std()
        if sig_y == 0 or not np.isfinite(sig_y):
            sig_y = 1.0
        Y_train_z = (Y_train - mu_y) / sig_y
        Y_test_z = (Y_test - mu_y) / sig_y

        beta = np.linalg.pinv(X_train) @ Y_train_z
        Y_train_pred = X_train @ beta
        Y_test_pred = X_test @ beta

        r_tr = pearsonr(Y_train_pred, Y_train_z)[0] if np.std(Y_train_pred) > 0 else 0.0
        r_train_folds.append(r_tr)
        Y_test_all.append(Y_test_z)
        Y_pred_all.append(Y_test_pred)

    Yt = np.concatenate(Y_test_all)
    Yp = np.concatenate(Y_pred_all)
    if np.std(Yp) > 0:
        r_test, p_test = spearmanr(Yp, Yt)
    else:
        r_test, p_test = 0.0, 1.0
    r_train = np.mean(r_train_folds)

    # R² on pooled out-of-fold predictions
    ss_res = np.sum((Yt - Yp) ** 2)
    ss_tot = np.sum((Yt - Yt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return r_test, r_train, p_test, Yt, Yp, r2


def analyze_run(neuron_means_all, W_initial, W_final, N, W_mask=None):
    """
    Run the three-factor regression analysis on one training run.

    neuron_means_all: (n_trials, N) mean firing rate per neuron per trial
    W_initial, W_final: (N, N) weight matrices
    N: number of neurons
    W_mask: (N, N) structural connectivity mask (1=exists, 0=absent).
            If provided, only pairs with existing connections are analyzed.

    Returns dict with r_test per CC_MODE.
    """
    n_trials = neuron_means_all.shape[0]

    # ΔW for all off-diagonal pairs
    dW = W_final - W_initial

    if W_mask is not None:
        # Only analyze pairs that have structural connections
        mask = (W_mask > 0) & (~np.eye(N, dtype=bool))
    else:
        mask = ~np.eye(N, dtype=bool)

    # Pairs: (i, j) where i=post, j=pre
    post_idx, pre_idx = np.where(mask)
    n_pairs = len(post_idx)
    Y = dW[mask]  # (n_pairs,)
    print(f"  Analyzing {n_pairs} pairs (of {N*(N-1)} possible)")

    # Activity matrix: (N, n_trials)
    act = neuron_means_all.T  # (N, n_trials)

    # Baseline: mean over first N_BASELINE trials
    bl_mean = np.mean(act[:, :N_BASELINE], axis=1)  # (N,)

    # Deviation from baseline
    pop_dev = act - bl_mean[:, None]  # (N, n_trials)

    # Per-pair, per-trial CC
    # 1st order: r_j(t) * (r_i(t) - r̄_i)
    r_pre = act[pre_idx, :]           # (n_pairs, n_trials)
    r_post_dev = pop_dev[post_idx, :] # (n_pairs, n_trials)
    cc_1st = r_pre * r_post_dev       # (n_pairs, n_trials)

    # 2nd order: r_j(t) * Σ_k W_{ik} * (r_k(t) - r̄_k)
    # For each postsynaptic neuron i, compute downstream sum:
    #   ds_i(t) = Σ_k W_{ik} * (r_k(t) - r̄_k)
    # Then cc_2nd[pair, t] = r_j(t) * ds_i(t)
    downstream_sum = W_initial @ pop_dev  # (N, n_trials): ds[i,t] = Σ_k W_ik * dev_k(t)
    ds_per_pair = downstream_sum[post_idx, :]  # (n_pairs, n_trials)
    cc_2nd = r_pre * ds_per_pair               # (n_pairs, n_trials)

    # Sliding windows
    win_starts = np.arange(0, n_trials - WIN_SIZE + 1, WIN_STEP)
    n_wins = len(win_starts)

    if n_wins < 3:
        print(f"  Only {n_wins} windows, skipping.")
        return None

    # Build X matrices
    X_1st = np.zeros((n_pairs, n_wins))
    X_2nd = np.zeros((n_pairs, n_wins))
    for wi, ws in enumerate(win_starts):
        X_1st[:, wi] = np.sum(cc_1st[:, ws:ws+WIN_SIZE], axis=1)
        X_2nd[:, wi] = np.sum(cc_2nd[:, ws:ws+WIN_SIZE], axis=1)

    X_comb = np.hstack([X_1st, X_2nd])

    # Z-score features
    X_dict = {
        '1st_order': zscore_cols(X_1st),
        '2nd_order': zscore_cols(X_2nd),
        'combined':  zscore_cols(X_comb),
    }

    # Shuffled control: permute W rows used for downstream sum
    rng = np.random.default_rng(42)
    shuf_r_tests = np.zeros(N_SHUFFLES)
    for shi in range(N_SHUFFLES):
        perm = rng.permutation(N)
        W_shuf = W_initial[perm, :]  # permute post neuron's outgoing weights
        ds_shuf = W_shuf @ pop_dev
        ds_shuf_pairs = ds_shuf[post_idx, :]
        cc_shuf = r_pre * ds_shuf_pairs
        X_sh = np.zeros((n_pairs, n_wins))
        for wi, ws in enumerate(win_starts):
            X_sh[:, wi] = np.sum(cc_shuf[:, ws:ws+WIN_SIZE], axis=1)
        X_sh_z = zscore_cols(X_sh)
        r_te, _, _, _, _, _ = run_cv(X_sh_z, Y, N_CV_FOLDS)
        shuf_r_tests[shi] = r_te

    # Run CV for each mode
    result = {}
    for mode in ['1st_order', '2nd_order', 'combined']:
        r_test, r_train, p_test, Yt, Yp, r2 = run_cv(X_dict[mode], Y, N_CV_FOLDS)
        result[mode] = {
            'r_test': r_test, 'r_train': r_train, 'p_test': p_test, 'r2': r2,
        }

    # Partial R²: unique variance contributed by each predictor set
    # partial_r2_2nd = R²_combined - R²_1st   (what 2nd order adds beyond 1st)
    # partial_r2_1st = R²_combined - R²_2nd   (what 1st order adds beyond 2nd)
    r2_1st = result['1st_order']['r2']
    r2_2nd = result['2nd_order']['r2']
    r2_comb = result['combined']['r2']

    result['partial_r2_2nd'] = r2_comb - r2_1st if (np.isfinite(r2_comb) and np.isfinite(r2_1st)) else np.nan
    result['partial_r2_1st'] = r2_comb - r2_2nd if (np.isfinite(r2_comb) and np.isfinite(r2_2nd)) else np.nan

    # Shuffled: use median and compute p-value
    real_2nd_r = result['2nd_order']['r_test']
    shuf_pval = np.mean(shuf_r_tests >= real_2nd_r) if np.isfinite(real_2nd_r) else 1.0
    result['shuffled'] = {
        'r_test': np.median(shuf_r_tests),
        'r_train': np.nan,
        'p_test': shuf_pval,
        'shuf_distribution': shuf_r_tests,
        'r2': np.nan,
    }

    return result


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
# Condition labels for display/storage
def cond_label(rule, scale, eta):
    if rule == "local":
        return f"local_eta{eta}"
    return f"semi_local_{scale}_eta{eta}"

COND_LABELS = [cond_label(r, s, e) for r, s, e in CONDITIONS]

all_results = {cl: {mode: [] for mode in CC_MODES} for cl in COND_LABELS}
all_r2 = {cl: {mode: [] for mode in CC_MODES} for cl in COND_LABELS}
all_partial_r2 = {cl: {'partial_r2_1st': [], 'partial_r2_2nd': []} for cl in COND_LABELS}

for (rule, scale, eta), cl in zip(CONDITIONS, COND_LABELS):
    print(f"\n{'='*60}")
    print(f"  Condition: {cl}  (rule={rule}, scale={scale}, eta={eta})")
    print(f"{'='*60}")

    for run in range(N_RUNS):
        print(f"\n--- Run {run+1}/{N_RUNS} ---")
        p = dict(DEFAULTS)
        p["learning_rule"] = rule
        p["semi_local_scale"] = scale
        p["eta"] = eta
        p["sparsity"] = SPARSITY
        p["seed"] = None

        (net, trial_logs, W_initial, W_snapshots,
         neuron_means_all, neuron_maxs_all, coact_session,
         coact_rpe_session) = train(p)

        result = analyze_run(neuron_means_all, W_initial, net.W, p["N"],
                             W_mask=net.W_mask)

        if result is None:
            print("  Skipped (too few windows)")
            continue

        for mode in CC_MODES:
            all_results[cl][mode].append(result[mode]['r_test'])
            all_r2[cl][mode].append(result[mode]['r2'])

        all_partial_r2[cl]['partial_r2_1st'].append(result['partial_r2_1st'])
        all_partial_r2[cl]['partial_r2_2nd'].append(result['partial_r2_2nd'])

        parts = []
        for mode in CC_MODES:
            r = result[mode]['r_test']
            parts.append(f"{mode}={r:.3f}")
        pr2_1 = result['partial_r2_1st']
        pr2_2 = result['partial_r2_2nd']
        parts.append(f"pR2_1st={pr2_1:.4f}")
        parts.append(f"pR2_2nd={pr2_2:.4f}")
        print(f"  " + ", ".join(parts))

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY (cross-validated test r, Spearman)")
print(f"{'='*60}")

summary = {}
for cl in COND_LABELS:
    summary[cl] = {}
    print(f"\n{cl}:")
    for mode in CC_MODES:
        vals = all_results[cl][mode]
        r2_vals = all_r2[cl][mode]
        if len(vals) > 0:
            m, s = np.mean(vals), np.std(vals)
            m_r2, s_r2 = np.mean(r2_vals), np.std(r2_vals)
            summary[cl][mode] = {"mean": m, "std": s, "values": vals,
                                   "r2_mean": m_r2, "r2_std": s_r2, "r2_values": r2_vals}
            print(f"  {mode:12s}: r={m:.4f}±{s:.4f}  R²={m_r2:.4f}±{s_r2:.4f}  (n={len(vals)})")
        else:
            summary[cl][mode] = {"mean": np.nan, "std": np.nan, "values": [],
                                   "r2_mean": np.nan, "r2_std": np.nan, "r2_values": []}
            print(f"  {mode:12s}: no data")

    # Partial R²
    pr2_1 = all_partial_r2[cl]['partial_r2_1st']
    pr2_2 = all_partial_r2[cl]['partial_r2_2nd']
    summary[cl]['partial_r2_1st'] = {"mean": np.mean(pr2_1), "std": np.std(pr2_1), "values": pr2_1}
    summary[cl]['partial_r2_2nd'] = {"mean": np.mean(pr2_2), "std": np.std(pr2_2), "values": pr2_2}
    print(f"  {'partial_r2_1st':12s}: {np.mean(pr2_1):.4f} ± {np.std(pr2_1):.4f}")
    print(f"  {'partial_r2_2nd':12s}: {np.mean(pr2_2):.4f} ± {np.std(pr2_2):.4f}")
    print(f"  --> Unique 2nd-order variance beyond 1st: {np.mean(pr2_2):.4f}")

# ──────────────────────────────────────────────────────────────────────
# Save JSON
# ──────────────────────────────────────────────────────────────────────
import json
os.makedirs(SAVE_DIR, exist_ok=True)

json_summary = {}
for cl in COND_LABELS:
    json_summary[cl] = {}
    for mode in CC_MODES:
        d = summary[cl][mode]
        json_summary[cl][mode] = {
            "mean": float(d["mean"]) if np.isfinite(d["mean"]) else None,
            "std": float(d["std"]) if np.isfinite(d["std"]) else None,
            "values": [float(v) for v in d["values"]],
            "r2_mean": float(d["r2_mean"]) if np.isfinite(d["r2_mean"]) else None,
            "r2_std": float(d["r2_std"]) if np.isfinite(d["r2_std"]) else None,
            "r2_values": [float(v) for v in d["r2_values"]],
        }
    for pkey in ['partial_r2_1st', 'partial_r2_2nd']:
        d = summary[cl][pkey]
        json_summary[cl][pkey] = {
            "mean": float(d["mean"]) if np.isfinite(d["mean"]) else None,
            "std": float(d["std"]) if np.isfinite(d["std"]) else None,
            "values": [float(v) for v in d["values"]],
        }
json_path = os.path.join(SAVE_DIR, "three_factor_regression.json")
with open(json_path, "w") as f:
    json.dump(json_summary, f, indent=2)
print(f"\nSaved summary → {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────
cond_colors = {}
cmap = plt.cm.RdYlBu_r
for ci, cl in enumerate(COND_LABELS):
    if cl == "local":
        cond_colors[cl] = "steelblue"
    else:
        # gradient from light to dark red for increasing scale
        frac = (ci) / max(len(COND_LABELS) - 1, 1)
        cond_colors[cl] = cmap(0.55 + 0.4 * frac)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Three-factor regression: eta sweep (sparsity={SPARSITY}, scale=1.0)\n"
             f"(N_runs={N_RUNS}, {N_CV_FOLDS}-fold CV, win={WIN_SIZE}, step={WIN_STEP})",
             fontsize=13)

# --- Panel A: Bar plot of R² per mode, per condition ---
ax = axes[0]
x = np.arange(len(CC_MODES))
n_cond = len(COND_LABELS)
width = 0.8 / n_cond

for ci, cl in enumerate(COND_LABELS):
    r2_means = []
    r2_sems = []
    for m in CC_MODES:
        r2_vals = summary[cl][m].get("r2_values", [])
        if len(r2_vals) > 0 and not all(np.isnan(r2_vals)):
            r2_clean = [v for v in r2_vals if np.isfinite(v)]
            r2_means.append(np.mean(r2_clean) if r2_clean else 0)
            r2_sems.append(np.std(r2_clean) / np.sqrt(len(r2_clean)) if len(r2_clean) > 1 else 0)
        else:
            r2_means.append(0)
            r2_sems.append(0)
    offset = (ci - (n_cond - 1) / 2) * width
    ax.bar(x + offset, r2_means, width * 0.9, yerr=r2_sems, capsize=3,
           label=cl, color=cond_colors[cl], alpha=0.85, edgecolor='k', linewidth=0.7)

ax.set_xticks(x)
ax.set_xticklabels(['1st order', '2nd order', 'Combined', 'Shuffled'], fontsize=9)
ax.set_ylabel("R² (CV)")
ax.set_title("Cross-validated R²", fontweight='bold')
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.legend(fontsize=8, loc='upper right')

# --- Panel B: Partial R² of 2nd order (unique non-local variance) ---
ax = axes[1]
x_cond = np.arange(n_cond)
pr2_2_means = []
pr2_2_sems = []
pr2_2_all = []
for cl in COND_LABELS:
    vals = summary[cl]['partial_r2_2nd']['values']
    pr2_2_all.append(vals)
    pr2_2_means.append(np.mean(vals) if len(vals) > 0 else 0)
    pr2_2_sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

bars = ax.bar(x_cond, pr2_2_means, yerr=pr2_2_sems, capsize=5,
              color=[cond_colors[cl] for cl in COND_LABELS],
              edgecolor='k', linewidth=1, alpha=0.85)

# Overlay dots
for ci, (cl, vals) in enumerate(zip(COND_LABELS, pr2_2_all)):
    ax.scatter(np.full(len(vals), ci) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)),
               vals, color=cond_colors[cl], s=25, alpha=0.6, edgecolor='none', zorder=3)

ax.set_xticks(x_cond)
ax.set_xticklabels(COND_LABELS, fontsize=8, rotation=15, ha='right')
ax.set_ylabel("Partial R²")
ax.set_title("Unique 2nd-order variance\n(ΔR² = combined - 1st order)", fontweight='bold')
ax.axhline(0, color='k', ls='-', alpha=0.3)

# --- Panel C: Partial R² 1st vs 2nd, per condition ---
ax = axes[2]
x_pr = np.arange(2)
width_pr = 0.8 / n_cond

for ci, cl in enumerate(COND_LABELS):
    pr1 = summary[cl]['partial_r2_1st']['values']
    pr2 = summary[cl]['partial_r2_2nd']['values']
    means_pr = [np.mean(pr1) if pr1 else 0, np.mean(pr2) if pr2 else 0]
    sems_pr = [np.std(pr1)/np.sqrt(len(pr1)) if len(pr1) > 1 else 0,
               np.std(pr2)/np.sqrt(len(pr2)) if len(pr2) > 1 else 0]
    offset = (ci - (n_cond - 1) / 2) * width_pr
    ax.bar(x_pr + offset, means_pr, width_pr * 0.9, yerr=sems_pr, capsize=3,
           label=cl, color=cond_colors[cl], alpha=0.85, edgecolor='k', linewidth=0.7)

ax.set_xticks(x_pr)
ax.set_xticklabels(["Unique 1st order\n(comb - 2nd)",
                     "Unique 2nd order\n(comb - 1st)"], fontsize=9)
ax.set_ylabel("Partial R²")
ax.set_title("Unique variance comparison", fontweight='bold')
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "three_factor_regression.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved figure → {fig_path}")
