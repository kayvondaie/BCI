#%% ============================================================================
# CONTROL: train the network with the RAW rule (r_pre * phi' * r_post, no post
# deviation) via adjust_type='3factor_rawpost', then run the same recovery test.
# If the analysis is unbiased, the RAW form should now recover RPE (~1) and the
# DEVIATED form (r_pre*phi'*dr_post) -- the true rule in the other experiment --
# should now FAIL. Anchor = recorded eligibility (should still be ~1.0).
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


params = kp.default_toy_params(seed=SEED, verbose=False)
task_params, train_params, net_params = params
train_params['adjust_type'] = '3factor_rawpost'          # <-- RAW post rule
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
    'true RAW (raw*phi, prev, ema-trace)':  (rr * phi,      pre_prev, 'ema'),
    'no phi (raw, prev, ema-trace)':        (rr,            pre_prev, 'ema'),
    'no trace (raw*phi, per-loss wipe)':    (rr * phi,      pre_prev, 'wipe'),
    'WRONG post-dev (dev*phi, prev)':       (dev_ema * phi, pre_prev, 'ema'),
    'WRONG both-dev (dev*phi, dev-pre)':    (dev_ema * phi, dev_ema,  'ema'),
    'WRONG pre-dev (raw*phi, dev-pre)':     (rr * phi,      dev_ema,  'ema'),
}
short = [
    r"$r_{pre}\phi' r_{post}$" + "\n(true rule = RAW)",
    r"$r_{pre} r_{post}$" + "\n(no $\phi'$)",
    r"$r_{pre}\phi' r_{post}$" + "\n(no trace)",
    r"$r_{pre}\phi'\Delta r_{post}$" + "\n(wrong: post-dev)",
    r"$\Delta r_{pre}\phi'\Delta r_{post}$" + "\n(wrong)",
    r"$\Delta r_{pre}\phi' r_{post}$" + "\n(wrong)",
]

Erec = np.asarray(to['W_rec_elg_vals'], float)[:L].reshape(L, -1).T
c_rec, *_ = np.linalg.lstsq(Erec, dW, rcond=None)
print("anchor (recorded raw-elig) recovery corr:", round(spearmanr(c_rec / eta, rpe[:L])[0], 4))
print("learning check: total_rpe std =", round(float(np.nanstd(rpe)), 4))

print("\n{:38s} {:>10s}".format("analysis eligibility", "recov corr"))
corrs, kinds = [], []
for name, (postf, pref, mode) in CANDS.items():
    A = build_trace(E_of(postf, pref), loss_idx[:L], n_el, mode, n_spl)
    m = min(A.shape[1], L); A = A[:, :m]; rp = rpe[:m]
    c, *_ = np.linalg.lstsq(A, dW, rcond=None); rhat = c / eta
    ok = np.isfinite(rhat) & np.isfinite(rp)
    corr = spearmanr(rhat[ok], rp[ok])[0] if ok.sum() > 3 else np.nan
    print("  {:36s} {:+10.3f}".format(name, corr))
    corrs.append(corr)
    kinds.append('true' if name.startswith('true') else
                 'wrong' if name.startswith('WRONG') else 'ablate')

col = {'true': '#2c7d3f', 'ablate': '#7f7f7f', 'wrong': '#b0392b'}
fig = plt.figure(figsize=(8.4, 4.0)); fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.05 / fw, 1.45 / fh, 7.1 / fw, 2.05 / fh])
x = np.arange(len(corrs))
ax.bar(x, corrs, color=[col[k] for k in kinds], width=0.72)
for xi, v in zip(x, corrs):
    ax.text(xi, v + 0.025, '{:.2f}'.format(v), ha='center', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel('corr( HI (MLR) , true RPE )', fontsize=9); ax.set_ylim(min(0, min(corrs) - 0.05), 1.1)
ax.set_title("CONTROL: network trained with the RAW rule ($r_{pre}\\phi' r_{post}$)\n"
             "does recovery follow the generative rule? (green=true=RAW, red=wrong incl. post-dev)",
             fontsize=8.5)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'model_test_rawpost.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
