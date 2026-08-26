"""
Batch comparison of local vs semi_local learning rules.

Runs 10 initializations of each rule, collects:
  - r(ΔW, local predictor)
  - r(ΔW, non-local predictor)
  - partial r(ΔW, non-local | local)
  - ΔR² from adding non-local predictor

Produces summary statistics and a comparison figure.
"""

import sys
import os
sys.path.insert(0, r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model")

import numpy as np
import matplotlib.pyplot as plt
import json

from bci_model import DEFAULTS, BCINetwork, BCITask, phi, train

# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────
N_RUNS = 10
RULES = ["local", "semi_local"]
SAVE_DIR = r"C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\kyles model\results"

# ──────────────────────────────────────────────────────────────────────
# Analysis helper
# ──────────────────────────────────────────────────────────────────────
def compute_plasticity_correlations(W_initial, W_final, coact_rpe_session, N):
    """Compute local, non-local, and partial correlations with ΔW.
    Uses RPE-weighted coactivity to match the learning rule."""
    dW = W_final - W_initial
    C = coact_rpe_session  # (N, N): RPE-weighted C_{kj}

    # Predictors
    P_local = C                    # P_local[i,j] = C[i,j]
    P_nonlocal = W_initial.T @ C   # P_nonlocal[i,j] = sum_k W_initial[k,i] * C[k,j]

    # Flatten, exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    dw = dW[mask]
    pl = P_local[mask]
    pnl = P_nonlocal[mask]

    # Raw correlations
    r_local = np.corrcoef(dw, pl)[0, 1]
    r_nonlocal = np.corrcoef(dw, pnl)[0, 1]

    # R² local only
    coef_l = np.polyfit(pl, dw, 1)
    pred_l = np.polyval(coef_l, pl)
    ss_tot = np.sum((dw - dw.mean()) ** 2)
    r2_local = 1 - np.sum((dw - pred_l) ** 2) / ss_tot

    # R² full model (local + nonlocal)
    X = np.column_stack([pl, pnl, np.ones(len(pl))])
    beta, _, _, _ = np.linalg.lstsq(X, dw, rcond=None)
    pred_full = X @ beta
    r2_full = 1 - np.sum((dw - pred_full) ** 2) / ss_tot
    delta_r2 = r2_full - r2_local

    # Partial correlation: regress local out of both ΔW and P_nonlocal
    resid_dw = dw - np.polyval(np.polyfit(pl, dw, 1), pl)
    resid_pnl = pnl - np.polyval(np.polyfit(pl, pnl, 1), pl)
    r_partial = np.corrcoef(resid_dw, resid_pnl)[0, 1]

    return {
        "r_local": r_local,
        "r_nonlocal": r_nonlocal,
        "r_partial": r_partial,
        "r2_local": r2_local,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
    }


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
results = {rule: [] for rule in RULES}

for rule in RULES:
    print(f"\n{'='*60}")
    print(f"  Rule: {rule}")
    print(f"{'='*60}")
    for run in range(N_RUNS):
        print(f"\n--- Run {run+1}/{N_RUNS} ---")
        p = dict(DEFAULTS)
        p["learning_rule"] = rule
        p["seed"] = None  # different each run

        (net, trial_logs, W_initial, W_snapshots,
         neuron_means_all, neuron_maxs_all, coact_session,
         coact_rpe_session) = train(p)

        stats = compute_plasticity_correlations(
            W_initial, net.W, coact_rpe_session, p["N"])
        stats["run"] = run
        results[rule].append(stats)

        print(f"  r_local={stats['r_local']:.4f}  "
              f"r_nonlocal={stats['r_nonlocal']:.4f}  "
              f"r_partial={stats['r_partial']:.4f}  "
              f"ΔR²={stats['delta_r2']:.4f}")

# ──────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

summary = {}
for rule in RULES:
    d = results[rule]
    summary[rule] = {}
    for key in ["r_local", "r_nonlocal", "r_partial", "r2_local", "delta_r2"]:
        vals = [s[key] for s in d]
        summary[rule][key] = {"mean": np.mean(vals), "std": np.std(vals),
                              "values": vals}
    print(f"\n{rule}:")
    print(f"  r(ΔW, local):      {summary[rule]['r_local']['mean']:.4f} "
          f"± {summary[rule]['r_local']['std']:.4f}")
    print(f"  r(ΔW, non-local):  {summary[rule]['r_nonlocal']['mean']:.4f} "
          f"± {summary[rule]['r_nonlocal']['std']:.4f}")
    print(f"  partial r:         {summary[rule]['r_partial']['mean']:.4f} "
          f"± {summary[rule]['r_partial']['std']:.4f}")
    print(f"  R² (local only):   {summary[rule]['r2_local']['mean']:.4f} "
          f"± {summary[rule]['r2_local']['std']:.4f}")
    print(f"  ΔR² (non-local):   {summary[rule]['delta_r2']['mean']:.4f} "
          f"± {summary[rule]['delta_r2']['std']:.4f}")

# ──────────────────────────────────────────────────────────────────────
# Save summary JSON
# ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
json_summary = {}
for rule in RULES:
    json_summary[rule] = {}
    for key in ["r_local", "r_nonlocal", "r_partial", "r2_local", "delta_r2"]:
        json_summary[rule][key] = {
            "mean": float(summary[rule][key]["mean"]),
            "std": float(summary[rule][key]["std"]),
            "values": [float(v) for v in summary[rule][key]["values"]],
        }
json_path = os.path.join(SAVE_DIR, "batch_comparison.json")
with open(json_path, "w") as f:
    json.dump(json_summary, f, indent=2)
print(f"\nSaved summary → {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────
metrics = ["r_local", "r_nonlocal", "r_partial", "delta_r2"]
labels = ["r(ΔW, local)", "r(ΔW, non-local)", "partial r\n(non-local | local)", "ΔR²"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Local vs Semi-local: plasticity predictor comparison "
             f"(N={N_RUNS} runs)", fontsize=13)

for ax, metric, label in zip(axes, metrics, labels):
    local_vals = [s[metric] for s in results["local"]]
    semi_vals = [s[metric] for s in results["semi_local"]]

    # Paired dots + lines
    for lv, sv in zip(local_vals, semi_vals):
        ax.plot([0, 1], [lv, sv], color="0.7", lw=0.8)
    ax.scatter(np.zeros(N_RUNS), local_vals, color="steelblue", s=40,
               zorder=3, label="local")
    ax.scatter(np.ones(N_RUNS), semi_vals, color="tomato", s=40,
               zorder=3, label="semi_local")

    # Means
    ax.plot([0, 1], [np.mean(local_vals), np.mean(semi_vals)],
            "k-", lw=2, zorder=4)
    ax.scatter([0, 1], [np.mean(local_vals), np.mean(semi_vals)],
               color="k", s=80, zorder=5, marker="D")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["local", "semi_local"])
    ax.set_ylabel(label)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    if metric == "r_local":
        ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, "batch_comparison.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved figure → {fig_path}")
