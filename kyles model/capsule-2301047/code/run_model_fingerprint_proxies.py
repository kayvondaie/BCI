#%% ============================================================================
# Does the eligibility fingerprint survive with a BEHAVIORAL RPE proxy (dSpeed),
# instead of the model's privileged internal RPE? (The data only has proxies.)
# Same regression dW ~ [I, S_post, S_pre, M], but weight the eligibility by:
#   true RPE (per loss-step) | dSpeed (per-trial) | hits (per-trial) | shuffled.
# If S_post >> S_pre still holds with dSpeed, the method ports to data.
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

# true RPE: per loss-step -> raw steps
total_rpes = np.asarray(to['total_rpes'], float)
loss_steps_arr = np.asarray(to['loss_steps'], int)
L = min(len(total_rpes), len(loss_steps_arr))
total_rpes, loss_steps_arr = total_rpes[:L], loss_steps_arr[:L]


def expand_loss(rpe_ls):
    r = np.zeros(n_steps); prev = 0
    for k in range(L):
        e = min(n_steps, loss_steps_arr[k] + 1)
        r[prev:e] = rpe_ls[k]; prev = e
        if prev >= n_steps:
            break
    if prev < n_steps and L:
        r[prev:] = rpe_ls[-1]
    return r


# behavioral proxies: per trial -> raw steps
rpe_est = kp.get_rpe_estimates(task_hist)
ts = np.asarray(task_hist['trial_starts'], int)
tb = list(ts) + [n_steps]


def expand_trial(vals):
    w = np.zeros(n_steps)
    for k in range(len(ts)):
        w[tb[k]:tb[k + 1]] = vals[k] if k < len(vals) else 0.0
    return w


n_bl = int(train_params.get('n_window_baseline', 20))
mean_act = np.zeros_like(out); prev = np.zeros(n_neu)
for t in range(n_steps):
    prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
dev = out - mean_act
m = out.mean(0)
M = np.outer(m, m)


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


rng = np.random.default_rng(0)
# true RPE coarsened to per-trial (sum of loss-step RPE within each trial) -> isolates RESOLUTION
true_raw = expand_loss(total_rpes)
true_per_trial = np.array([true_raw[tb[k]:tb[k + 1]].mean() for k in range(len(ts))])
WEIGHTS = {
    'true RPE':          expand_loss(total_rpes),
    'true RPE per-trial': expand_trial(true_per_trial),
    'dSpeed':            expand_trial(np.asarray(rpe_est['speed_rpe'], float)),
    'hits':              expand_trial(np.asarray(rpe_est['hits'], float)),
    'shuffled':          expand_loss(total_rpes[rng.permutation(L)]),
}
print("Fingerprint with different RPE weightings (seed {}):".format(SEED))
print("  {:10s} {:>9s} {:>9s} {:>9s} {:>9s} {:>7s}".format('weight', 'I', 'S_post', 'S_pre', 'M', 'R2'))
for wn, w in WEIGHTS.items():
    b, r2 = betas(w)
    print("  {:10s} {:+9.3f} {:+9.3f} {:+9.3f} {:+9.3f} {:7.3f}".format(wn, b[0], b[1], b[2], b[3], r2))
print("\nPortable if dSpeed still gives S_post >> S_pre.")
