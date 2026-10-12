#%% ============================================================================
# pre x dpost model (true rule = r_pre*phi'*dr_post). Same recovery test, but
# compare TEMPORAL RESOLUTION: recover RPE from eligibility at
#   binsize=1  (one column per loss step, the fine version)  vs
#   binsize=5  (average the eligibility over every 5 loss steps).
# Grouped bars show how coarsening resolution changes each candidate's recovery.
import os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 8, 'svg.fonttype': 'none'})
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp
from net_helpers import accumulate_decay
if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)

output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes',
               'loss_steps', 'act_fn_p_pre_act_vals']
SEED = 0
BINSIZES = [1, 5]


def build_trace(E_step, loss_idx, n_win, mode='ema', n_spl=5):
    n = E_step.shape[1]; cols = []
    tot = np.zeros((n, n)); g = 1 - 1.0 / n_win
    ls = set(loss_idx.tolist())
    for t in range(E_step.shape[0]):
        tot = (g * tot + (1 - g) * E_step[t]) if mode == 'ema' else (tot + E_step[t] / n_spl)
        if t in ls:
            cols.append(tot.ravel().copy())
            if mode != 'ema':
                tot = np.zeros((n, n))
    return np.array(cols).T


def recover(A, dW, rpe, eta, bs):
    m = A.shape[1]; nb = m // bs
    if nb < 3:
        return np.nan
    Ab = A[:, :nb * bs].reshape(A.shape[0], nb, bs).mean(2)     # avg elig over bin
    rb = rpe[:nb * bs].reshape(nb, bs).mean(1)                   # avg RPE over bin
    c = np.linalg.lstsq(Ab, dW, rcond=None)[0]; rhat = c / eta
    ok = np.isfinite(rhat) & np.isfinite(rb)
    return spearmanr(rhat[ok], rb[ok])[0] if ok.sum() > 3 else np.nan


params = kp.default_toy_params(seed=SEED, verbose=False)
task_params, train_params, net_params = params
kp.task_params, kp.train_params, kp.net_params = params
kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
_po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
to = toa[0]
eta = train_params['eta']; n_be = int(train_params['n_window_baseline'])
n_el = int(train_params.get('n_window_elig', 40)); n_spl = int(train_params['n_steps_per_loss'])

rr = np.asarray(to['output'], float)
phi = np.asarray(to['act_fn_p_pre_act_vals'], float)
T, n = rr.shape
loss_idx = np.asarray(to['loss_steps'], int)
rpe = np.asarray(to['total_rpes'], float)
Wv = np.asarray(to['W_rec_vals'], float)
dW = (Wv[-1] - Wv[0]).ravel()
L = min(len(loss_idx), len(rpe), len(Wv))

ema = np.zeros_like(rr); prev = np.zeros(n)
for t in range(T):
    prev = accumulate_decay(prev, rr[t], n_window=n_be); ema[t] = prev
dev_ema = rr - ema
pre_prev = np.vstack([np.zeros((1, n)), rr[:-1]])


def E_of(post_fac, pre_fac):
    return post_fac[:, :, None] * pre_fac[:, None, :]


CANDS = {
    'true (dev*phi, prev, ema)': (dev_ema * phi, pre_prev, 'ema'),
    'no phi':                    (dev_ema,       pre_prev, 'ema'),
    'no trace':                  (dev_ema * phi, pre_prev, 'wipe'),
    'WRONG raw':                 (rr * phi,      pre_prev, 'ema'),
    'WRONG both-dev':            (dev_ema * phi, dev_ema,  'ema'),
    'WRONG pre-dev':             (rr * phi,      dev_ema,  'ema'),
}
short = [
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(true)",
    r"$r_{pre}\Delta r_{post}$" + "\n(no $\phi'$)",
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(no trace)",
    r"$r_{pre}\phi' r_{post}$" + "\n(wrong)",
    r"$\Delta r_{pre}\phi'\Delta r_{post}$" + "\n(wrong)",
    r"$\Delta r_{pre}\phi' r_{post}$" + "\n(wrong)",
]

res = {bs: [] for bs in BINSIZES}
print("{:30s}".format("candidate") + "".join("  bs={}".format(bs) for bs in BINSIZES))
for name, (postf, pref, mode) in CANDS.items():
    A = build_trace(E_of(postf, pref), loss_idx[:L], n_el, mode, n_spl)
    row = "  {:28s}".format(name)
    for bs in BINSIZES:
        c = recover(A[:, :L], dW, rpe[:L], eta, bs)
        res[bs].append(c); row += "  {:+.3f}".format(c)
    print(row)

fig = plt.figure(figsize=(8.6, 4.0)); fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.05 / fw, 1.45 / fh, 7.3 / fw, 2.05 / fh])
x = np.arange(len(short)); w = 0.38
shades = {1: '#3b6fb0', 5: '#c0762e'}
for j, bs in enumerate(BINSIZES):
    ax.bar(x + (j - 0.5) * w, res[bs], w, color=shades[bs], label='binsize {} (avg {} steps)'.format(bs, bs))
    for xi, v in zip(x, res[bs]):
        ax.text(xi + (j - 0.5) * w, v + 0.02, '{:.2f}'.format(v), ha='center', fontsize=7.5)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel('corr( HI (MLR) , true RPE )', fontsize=9); ax.set_ylim(min(0, min(min(res[1]), min(res[5])) - 0.05), 1.1)
ax.set_title("pre x dpost model: recovery at fine (binsize 1) vs coarse (binsize 5) resolution", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc='upper right')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_test_binned.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
