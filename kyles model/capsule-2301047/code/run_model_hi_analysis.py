#%% ============================================================================
# CELL 1: Imports and paths
# ============================================================================
"""
Drive Kyle's toy BCI model + Hebbian-index analysis from a plain cell-based
script (Spyder). Goal of this first pass: confirm that both eligibility forms

    'hebb'      = r_pre * r_post              (RAW)
    'dpost_pre' = r_pre * (r_post - <r_post>) (FLUCTUATION)

produce a time-dependent Hebbian index (HI) that correlates with the model's
true RPE -- reproducing the model-side result before we start digging in
(e.g. adding the mean-drive term <r_post>*r_pre).

Prereq: run build_kyle_pipeline.py once to generate kyle_pipeline.py.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import bci_analysis                 # his analysis module (compute_local_hebbian_indexes)
import kyle_pipeline as kp          # generated: default_toy_params, train_task

plt.rcParams.update({'font.size': 12})
print("Imports OK")

#%% ============================================================================
# CELL 2: Build params and train one session of Kyle's model
# ============================================================================
SEED = 0
params = kp.default_toy_params(seed=SEED, verbose=True)
task_params, train_params, net_params = params

# Kyle's notebook helpers read task_params / train_params / net_params as GLOBALS
# (notebook-style scoping). Inject them into the pipeline module namespace so
# those references resolve. Same dict objects params holds, so in-place mutations
# during training stay consistent between the globals and train_task's locals.
kp.task_params, kp.train_params, kp.net_params = params

# --- DEVIATION (flagged): skip cached-probe tuning for CN selection -----------
# His CN selection normally estimates neuron tunings via a probe session driven
# by a cached reference trace (test_bci_activity_seed22222.pkl), then restricts
# the CN to low-tuned neurons (matched to experiment). That file isn't in the
# capsule. Returning None tunings makes find_another_cn pick the CN by activity
# percentile only (its activity_high path), skipping the low-tuning constraint.
# Reversible: drop these two lines and place the real file at C:\scratch\ for
# bit-faithful CN selection.
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})

# Ask training to retain the arrays the HI analysis reads. If train_task errors
# on an unknown name here, trim the list; if compute_local_hebbian_indexes later
# KeyErrors on a missing array, add its name here.
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward',
               'total_rpes', 'loss_steps', 'act_fn_p_pre_act_vals']

# train_task returns (params, train_outputs_all, net, task, task_ps)
_pout, train_outputs_all, net, task, _task_ps = kp.train_task(
    params, output_vars=output_vars, verbose=True)
train_outputs = train_outputs_all[0]
print("\nTrained. train_outputs keys:")
print(" ", sorted(train_outputs.keys()))

#%% ============================================================================
# CELL 3: Hebbian index vs true RPE for three eligibility forms
# ============================================================================
# 'true'      : model's actual stored eligibility trace (ground-truth rule)
# 'hebb'      : r_pre * r_post              (RAW)
# 'dpost_pre' : r_pre * (r_post - <r_post>) (FLUCTUATION)
elig_types = ('true', 'hebb', 'dpost_pre')
N_DIV = 20

(rpes_divs, div_slopes, div_slopes_full_delta,
 full_slopes, _elig_divs, _mlr) = bci_analysis.compute_local_hebbian_indexes(
    train_outputs, params, n_divisions=N_DIV, elig_types=elig_types)

# NOTE on which HI to use:
#   div_slopes            = HI in each division fit to the dW *within that division*
#   div_slopes_full_delta = HI in each division fit to the *whole-session* dW
# The data analysis regresses windowed coactivity against the fixed photostim dW
# (whole-session change), so div_slopes_full_delta is the data-matched quantity.
# We report both so we can see the model's "within-window" version too.
def corr_with_rpe(hi):
    ok = np.isfinite(hi) & np.isfinite(rpes_divs)
    r_s, p_s = spearmanr(hi[ok], rpes_divs[ok])
    r_p, p_p = pearsonr(hi[ok], rpes_divs[ok])
    return r_s, p_s, r_p, p_p

print(f"\nHI(division) vs true RPE   [n_div={N_DIV}]")
for target_name, HI in [('full-session dW', div_slopes_full_delta),
                        ('within-division dW', div_slopes)]:
    print(f"\n  target = {target_name}")
    for i, et in enumerate(elig_types):
        r_s, p_s, r_p, p_p = corr_with_rpe(HI[i])
        print(f"    {et:10s}: spearman={r_s:+.3f} (p={p_s:.2g})   "
              f"pearson={r_p:+.3f} (p={p_p:.2g})")

#%% ============================================================================
# CELL 4: Plot HI(t) vs RPE(t) per eligibility form (whole-session-dW target)
# ============================================================================
rpe_z = (rpes_divs - np.nanmean(rpes_divs)) / np.nanstd(rpes_divs)

fig, axes = plt.subplots(len(elig_types), 1, figsize=(7, 2.3 * len(elig_types)),
                         sharex=True)
for i, (et, ax) in enumerate(zip(elig_types, axes)):
    hi = div_slopes_full_delta[i]
    ok = np.isfinite(hi) & np.isfinite(rpes_divs)
    hi_z = (hi - np.nanmean(hi[ok])) / np.nanstd(hi[ok])
    r_s, _ = spearmanr(hi[ok], rpes_divs[ok])
    ax.plot(rpe_z, 'k-', marker='.', label='RPE (z)')
    ax.plot(hi_z, color='tab:orange', marker='.', label=f'HI [{et}] (z)')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_ylabel('z')
    ax.set_title(f"{et}:  spearman(HI, RPE) = {r_s:+.3f}", fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
axes[-1].set_xlabel('division')
fig.suptitle(f'Model HI vs true RPE  (seed={SEED}, n_div={N_DIV})', y=1.01)
plt.tight_layout()
plt.show()
