#%% ============================================================================
# 15-seed validation of the fingerprint driven by a CONTINUOUS BEHAVIORAL signal
# (spout_rpe_steps = per-timestep lickport-movement prediction error; the data
# analog is continuous lickport). Native fine regime. Shuffle null.
# Expect S_post(spout) reliably > 0 and > S_pre, collapsing under shuffle.
import os, sys
import numpy as np
from scipy.stats import wilcoxon
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

SEEDS = list(range(15))
N_SHUF = 5
BLOCKS = ['I', 'S_post', 'S_pre', 'M']
WEIGHT_KEY = 'spout_rpe_steps'
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
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
    rpe_est = kp.get_rpe_estimates(task_hist)
    w0 = np.asarray(rpe_est[WEIGHT_KEY], float)
    w0 = w0[:n_steps] if len(w0) >= n_steps else np.concatenate([w0, np.zeros(n_steps - len(w0))])

    n_bl = int(train_params.get('n_window_baseline', 20))
    mean_act = np.zeros_like(out); prev = np.zeros(n_neu)
    for t in range(n_steps):
        prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
    dev = out - mean_act
    m = out.mean(0); M = np.outer(m, m); y = zc(dW[off])

    def betas(w):
        q = (w[:, None] * dev).sum(0)
        I = (w[:, None] * dev).T @ dev
        X = np.column_stack([zc(I[off]), zc(np.outer(q, m)[off]), zc(np.outer(m, q)[off]), zc(M[off])])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b

    b_true = betas(w0)
    rng = np.random.default_rng(seed)
    b_shuf = np.mean([betas(w0[rng.permutation(n_steps)]) for _ in range(N_SHUF)], axis=0)
    return b_true, b_shuf


BT, BS = [], []
for s in SEEDS:
    try:
        bt, bs = run_seed(s); BT.append(bt); BS.append(bs)
        print("seed {:2d}: S_post spout={:+.3f} shuf={:+.3f} | S_pre spout={:+.3f}".format(s, bt[1], bs[1], bt[2]))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
BT = np.array(BT); BS = np.array(BS)
np.save(os.path.join(OUT, 'model_fingerprint_continuous_multiseed.npy'), {'true': BT, 'shuf': BS}, allow_pickle=True)


def star(v):
    v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


print("\nContinuous spout signal across {} seeds (standardized beta):".format(len(BT)))
print("  {:10s} {:>18s} {:>18s}".format('block', 'spout_rpe_steps', 'shuffled'))
for i, b in enumerate(BLOCKS):
    print("  {:10s} {:+.3f}+/-{:.3f} {:>4s}   {:+.3f}+/-{:.3f} {:>4s}".format(
        b, BT[:, i].mean(), BT[:, i].std() / np.sqrt(len(BT)), star(BT[:, i]),
        BS[:, i].mean(), BS[:, i].std() / np.sqrt(len(BS)), star(BS[:, i])))
d = BT[:, 1] - BS[:, 1]
dps = BT[:, 1] - BT[:, 2]
print("\n  S_post(spout) - S_post(shuffle): {:+.3f} +/- {:.3f}  {}".format(d.mean(), d.std() / np.sqrt(len(d)), star(d)))
print("  S_post - S_pre (spout):          {:+.3f} +/- {:.3f}  {}".format(dps.mean(), dps.std() / np.sqrt(len(dps)), star(dps)))

fig = plt.figure(figsize=(4.6, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0 / fw, 0.95 / fh, 3.3 / fw, 1.8 / fh])
x = np.arange(4)
for j, (B, c, lab) in enumerate([(BT, '#1baf7a', 'spout signal'), (BS, '#b0b0b0', 'shuffled')]):
    mn = B.mean(0); se = B.std(0) / np.sqrt(len(B))
    ax.bar(x + (j - 0.5) * 0.4, mn, 0.38, yerr=se, color=c, capsize=2, label=lab)
    for i in range(4):
        ax.scatter(np.full(len(B), x[i] + (j - 0.5) * 0.4) + np.random.uniform(-0.06, 0.06, len(B)),
                   B[:, i], s=6, color='k', alpha=0.3, zorder=3)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['I', 'S_post\n(POST dev)', 'S_pre\n(PRE dev)', 'M'], fontsize=7)
ax.set_ylabel('standardized $\\beta$')
ax.set_title('Fingerprint via continuous behavioral signal (n=15)', fontsize=8)
ax.legend(fontsize=7, frameon=False, loc='best')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_fingerprint_continuous_multiseed.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
