#%% ============================================================================
# Which eligibility does the model's dW reflect: the PRODUCT (pre*dpost) or a
# MARGINAL (pre / dpost / post)? Four structurally-distinct candidates (a product
# vs three marginals), fit jointly so they can be told apart.
#
# Computed at RAW-STEP resolution (out = per-step activity, dev = out - running-mean
# baseline), so the UNIFORM and true-RPE-weighted versions share the exact same
# substrate and differ ONLY by the per-step weight w(t):
#   pre_x_dpost[i,j] = sum_t w(t) * dev_post_i(t) * out_pre_j(t)   (product; model's rule)
#   pre[i,j]         = sum_t w(t) * out_pre_j(t)                   (pre marginal; j only)
#   dpost[i,j]       = sum_t w(t) * dev_post_i(t)                  (post-dev marginal; i only)
#   post[i,j]        = sum_t w(t) * out_post_i(t)                  (post raw marginal; i only)
# Target: dW = W_rec(end) - W_rec(0), off-diagonal pairs.
# Question: does RPE-weighting promote the PRODUCT over the (mean-drive) pre marginal?
import os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay

if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

TERMS = ['pre_x_dpost', 'pre', 'dpost', 'post']
TERM_LABEL = ['$r_{pre}\\cdot\\Delta r_{post}$', '$r_{pre}$', '$\\Delta r_{post}$', '$r_{post}$']
WEIGHTINGS = ['uniform', 'true_rpe']
SEEDS = list(range(15))
output_vars = ['W_rec_vals', 'output', 'reward', 'total_rpes', 'loss_steps']


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    task_params, train_params, net_params = params
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]

    out = np.asarray(to['output'], float)
    n_steps, N = out.shape

    # running-mean baseline (native fine regime) -> dev = out - baseline
    n_bl = int(train_params.get('n_window_baseline', 200))
    mean_act = np.zeros_like(out); prev = np.zeros(N)
    for t in range(n_steps):
        prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
    dev = out - mean_act

    # true RPE (per loss-step -> per raw step)
    total_rpes = np.asarray(to['total_rpes'], float)
    loss_steps_arr = np.asarray(to['loss_steps'], int)
    L = min(len(total_rpes), len(loss_steps_arr))
    true_raw = np.zeros(n_steps); p = 0
    for k in range(L):
        e = min(n_steps, loss_steps_arr[k] + 1); true_raw[p:e] = total_rpes[k]; p = e

    Wv = np.asarray(to['W_rec_vals'], float)
    dW = Wv[-1] - Wv[0]
    off = ~np.eye(N, dtype=bool)
    y = dW[off]

    W = {'uniform': np.ones(n_steps), 'true_rpe': true_raw}
    res = {}
    for wn, w in W.items():
        qw = (w[:, None] * out).sum(0)                     # sum_t w*out  (pre & post-raw marginal)
        qd = (w[:, None] * dev).sum(0)                     # sum_t w*dev  (dpost marginal)
        E = {
            'pre_x_dpost': (w[:, None] * dev).T @ out,     # [i,j] = sum_t w*dev_i*out_j
            'pre':   np.tile(qw[None, :], (N, 1)),         # [i,j] = qw[j]
            'dpost': np.tile(qd[:, None], (1, N)),         # [i,j] = qd[i]
            'post':  np.tile(qw[:, None], (1, N)),         # [i,j] = qw[i]
        }
        cols = {t: E[t][off] for t in TERMS}
        uni = {t: spearmanr(cols[t], y)[0] for t in TERMS}
        X = np.column_stack([_z(cols[t]) for t in TERMS])
        beta, *_ = np.linalg.lstsq(X, _z(y), rcond=None)
        res[wn] = {'uni': uni, 'beta': dict(zip(TERMS, beta))}
    return res


allres = []
for s in SEEDS:
    try:
        r = run_seed(s); allres.append(r)
        print("seed {} | uniform beta pre={:+.3f} prod={:+.3f} | true_rpe beta pre={:+.3f} prod={:+.3f}".format(
            s, r['uniform']['beta']['pre'], r['uniform']['beta']['pre_x_dpost'],
            r['true_rpe']['beta']['pre'], r['true_rpe']['beta']['pre_x_dpost']))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))

np.save(os.path.join(OUT, 'model_product_vs_marginals.npy'),
        {'res': allres, 'terms': TERMS, 'weightings': WEIGHTINGS}, allow_pickle=True)


def _star(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    p = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else 'ns'


n = len(allres)
for wn in WEIGHTINGS:
    print("\nMODEL [{}]: which eligibility explains dW? (n={} seeds)".format(wn, n))
    print("  {:14s} {:>18s} {:>18s}".format('term', 'univariate rho', 'joint beta'))
    for t, lab in zip(TERMS, TERM_LABEL):
        u = np.array([d[wn]['uni'][t] for d in allres])
        b = np.array([d[wn]['beta'][t] for d in allres])
        print("  {:14s} {:+.3f}+/-{:.3f} {:3s}   {:+.3f}+/-{:.3f} {:3s}".format(
            t, u.mean(), u.std() / n**.5, _star(u), b.mean(), b.std() / n**.5, _star(b)))

# figure: joint betas, uniform vs true_rpe
fig = plt.figure(figsize=(4.8, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0 / fw, 0.75 / fh, 3.4 / fw, 1.9 / fh])
x = np.arange(len(TERMS))
for j, (wn, c) in enumerate([('uniform', '#999'), ('true_rpe', '#c0392b')]):
    B = np.array([[d[wn]['beta'][t] for t in TERMS] for d in allres])
    ax.bar(x + (j - 0.5) * 0.4, B.mean(0), 0.38, yerr=B.std(0) / n**.5,
           color=c, capsize=2, label=wn)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(TERM_LABEL, fontsize=7)
ax.set_ylabel('joint $\\beta$ with $\\Delta W$')
ax.set_title('MODEL: product vs marginals, uniform vs RPE (n={})'.format(n))
ax.legend(frameon=False, fontsize=7)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_product_vs_marginals.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
