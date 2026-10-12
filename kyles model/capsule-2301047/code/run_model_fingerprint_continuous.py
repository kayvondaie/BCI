#%% ============================================================================
# The real portability test: fingerprint weighted by a CONTINUOUS BEHAVIORAL
# signal (the model's analog of continuous lickport), not the privileged true RPE
# or the too-coarse per-trial dSpeed.
#   spout_steps      = per-timestep forward spout/lickport movement
#   spout_rpe_steps  = its prediction error (continuous behavioral RPE)
# Native fine regime (timestep activity, timestep running-mean baseline).
# If S_post >> S_pre with the continuous spout signal -> ports to data via lickport.
import os, sys
import numpy as np
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

SEED = 0
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']

params = kp.default_toy_params(seed=SEED, verbose=False)
task_params, train_params, net_params = params
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]; task_hist = task.hists[0]

out = np.asarray(to['output'], float)
n_steps, n_neu = out.shape
loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
Wv = np.asarray(to['W_rec_vals'], float)
dW = Wv[loss_step_divs[-1]] - Wv[0]
off = ~np.eye(n_neu, dtype=bool)

# true RPE (per loss-step -> raw) for reference
total_rpes = np.asarray(to['total_rpes'], float)
loss_steps_arr = np.asarray(to['loss_steps'], int)
L = min(len(total_rpes), len(loss_steps_arr))
true_raw = np.zeros(n_steps); prev = 0
for k in range(L):
    e = min(n_steps, loss_steps_arr[k] + 1); true_raw[prev:e] = total_rpes[k]; prev = e

# CONTINUOUS behavioral signals (per raw step) -- the lickport analog
rpe_est = kp.get_rpe_estimates(task_hist)


def fit_len(x):
    x = np.asarray(x, float)
    return x[:n_steps] if len(x) >= n_steps else np.concatenate([x, np.zeros(n_steps - len(x))])


# running-mean baseline (native fine regime)
n_bl = int(train_params.get('n_window_baseline', 20))
mean_act = np.zeros_like(out); prev = np.zeros(n_neu)
for t in range(n_steps):
    prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
dev = out - mean_act
m = out.mean(0)
M = np.outer(m, m)
rng = np.random.default_rng(0)


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def betas(w):
    q = (w[:, None] * dev).sum(0)
    I = (w[:, None] * dev).T @ dev
    X = np.column_stack([zc(I[off]), zc(np.outer(q, m)[off]), zc(np.outer(m, q)[off]), zc(M[off])])
    y = zc(dW[off])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum(y ** 2)
    return b, r2


WEIGHTS = {
    'true RPE (ref)':      true_raw,
    'spout_rpe_steps':     fit_len(rpe_est.get('spout_rpe_steps', np.zeros(n_steps))),
    'spout_steps':         fit_len(rpe_est.get('spout_steps', np.zeros(n_steps))),
    'hits_rpe_steps':      fit_len(rpe_est.get('hits_rpe_steps', np.zeros(n_steps))),
    'shuffled':            true_raw[rng.permutation(n_steps)],
}
print("Fingerprint with CONTINUOUS behavioral weightings (seed {}):".format(SEED))
print("  {:18s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s}".format('weight', 'I', 'S_post', 'S_pre', 'M', 'R2'))
for wn, w in WEIGHTS.items():
    if np.std(w) == 0:
        print("  {:18s}  (all zero, skipped)".format(wn)); continue
    b, r2 = betas(w)
    print("  {:18s} {:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f} {:7.3f}".format(wn, b[0], b[1], b[2], b[3], r2))
print("\nPorts to data via continuous lickport if spout_rpe_steps gives S_post > S_pre.")
