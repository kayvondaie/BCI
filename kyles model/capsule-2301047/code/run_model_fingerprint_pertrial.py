#%% ============================================================================
# Fair data-portability test of the fingerprint: MODEL in the DATA'S convention.
#   per-trial activity, trailing-20-TRIAL baseline, per-trial RPE weighting.
# Here per-trial RPE is FINER than the 20-trial baseline (unlike the timestep-
# baseline test), so the covariance identity should break -> S_post recoverable
# even with a behavioral proxy. Regress dW ~ [I, S_post, S_pre, M].
import os, sys
import numpy as np
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

SEED = 0
N_BASELINE = 20
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

# per-trial activity
ts = np.asarray(task_hist['trial_starts'], int)
tb = list(ts) + [n_steps]
A = np.array([np.nanmean(out[tb[k]:tb[k + 1]], axis=0) for k in range(len(ts)) if tb[k + 1] > tb[k]])
n_trials = A.shape[0]


def trailing(A, w):
    bl = np.empty_like(A)
    for t in range(A.shape[0]):
        bl[t] = A[max(0, t - w + 1):t + 1].mean(0)
    return bl


dev = A - trailing(A, N_BASELINE)                 # per-trial deviation, 20-trial baseline
m = A.mean(0)
M = np.outer(m, m)

# per-trial RPE weightings
total_rpes = np.asarray(to['total_rpes'], float)
loss_steps_arr = np.asarray(to['loss_steps'], int)
Lk = min(len(total_rpes), len(loss_steps_arr))
true_raw = np.zeros(n_steps); prev = 0
for k in range(Lk):
    e = min(n_steps, loss_steps_arr[k] + 1); true_raw[prev:e] = total_rpes[k]; prev = e
true_pt = np.array([true_raw[tb[k]:tb[k + 1]].sum() for k in range(n_trials)])
rpe_est = kp.get_rpe_estimates(task_hist)
dspeed = np.asarray(rpe_est['speed_rpe'], float)[:n_trials]
hits = np.asarray(rpe_est['hits'], float)[:n_trials]
rng = np.random.default_rng(0)


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def betas(w):
    w = w[:n_trials]
    q = (w[:, None] * dev).sum(0)
    I = (w[:, None] * dev).T @ dev
    X = np.column_stack([zc(I[off]), zc(np.outer(q, m)[off]), zc(np.outer(m, q)[off]), zc(M[off])])
    y = zc(dW[off])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum(y ** 2)
    return b, r2


WEIGHTS = {
    'true RPE (per-trial)': true_pt,
    'dSpeed':               dspeed,
    'hits':                 hits,
    'shuffled':             true_pt[rng.permutation(n_trials)],
}
print("MODEL in DATA convention (per-trial, trailing-20-trial baseline), seed {}:".format(SEED))
print("  {:20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s}".format('weight', 'I', 'S_post', 'S_pre', 'M', 'R2'))
for wn, w in WEIGHTS.items():
    b, r2 = betas(w)
    print("  {:20s} {:+8.3f} {:+8.3f} {:+8.3f} {:+8.3f} {:7.3f}".format(wn, b[0], b[1], b[2], b[3], r2))
print("\nPorts to data if dSpeed (per-trial) gives S_post > S_pre here.")
