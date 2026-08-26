#%% ============================================================================
# Diagnostic: are the model's deviation forms REALLY ~0.996 correlated, and why?
#   - exact correlations (not rounded)
#   - is it exactly 1.0 (would signal identical arrays / bug)?
#   - dimensionality of the activity fluctuations (participation ratio)
#   - mean vs fluctuation magnitude, and coherence of Dr across neurons
# Low-dimensional activity -> outer-product eligibilities collapse -> high corr.
import os, sys
import numpy as np
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

ELIG = ('hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEED = 0

params = kp.default_toy_params(seed=SEED, verbose=False)
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]
_rd, _ds, _dsf, _fs, elig_divs, _mlr = kp.compute_local_hebbian_indexes(
    to, task_hist, params, hi_params, return_elig_divs=True)

n_form, n_div, n_post, n_pre = elig_divs.shape
offdiag = ~np.eye(n_post, n_pre, dtype=bool)
E = np.stack([elig_divs[k].sum(axis=0)[offdiag] for k in range(n_form)], axis=0)
C = np.corrcoef(E)
print("EXACT eligibility correlations (5 dp), order:", ELIG)
np.set_printoptions(precision=5, suppress=True)
print(C)
print("\nmax off-diagonal =", np.max(C[~np.eye(4, dtype=bool)]),
      " (exactly 1.0 would mean identical arrays)")
print("are dpost_pre & dpost_dpre the same array?",
      np.array_equal(E[1], E[2]))

# ---- dimensionality of the activity fluctuations ----
act = np.asarray(to['output'], float)            # (n_steps, n_neurons)
act = act[np.all(np.isfinite(act), axis=1)]
mu = act.mean(0); fluc = act - mu
cov = np.cov(fluc.T)
ev = np.linalg.eigvalsh(cov); ev = ev[ev > 0]
PR = (ev.sum() ** 2) / np.sum(ev ** 2)           # participation ratio
print("\nActivity: n_neurons={}, n_steps={}".format(act.shape[1], act.shape[0]))
print("  participation ratio of fluctuations = {:.2f}  (of {} neurons)".format(PR, act.shape[1]))
print("  top-1 / top-3 variance fraction = {:.2f} / {:.2f}".format(
    ev[-1] / ev.sum(), ev[-3:].sum() / ev.sum()))
print("  mean|mu| = {:.3f},  mean std(fluc) = {:.3f},  ratio = {:.2f}".format(
    np.mean(np.abs(mu)), np.mean(fluc.std(0)), np.mean(np.abs(mu)) / np.mean(fluc.std(0))))

# coherence of Dr across neuron pairs (how 1-D are the fluctuations)
cc = np.corrcoef(fluc.T)
offd = cc[~np.eye(cc.shape[0], dtype=bool)]
print("  mean pairwise corr of neuron fluctuations = {:.3f}".format(np.nanmean(offd)))
