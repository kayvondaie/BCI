#%% ============================================================================
# Full eligibility-form identification from dW + activity.
# Every form is a distinct combination of 4 building blocks (RPE-weighted):
#   both-dev = I ;  post-dev = I + S_post ;  pre-dev = I + S_pre ;
#   raw      = I + S_post + S_pre + Q*M
# with  I = sum_t RPE*dpost_i*dpre_j ,  S_post = q_i*m_j ,  S_pre = m_i*q_j ,
#       M = m_i*m_j ,  q_i = sum_t RPE*dact_i ,  m = mean act.
# Regress dW ~ [I, S_post, S_pre, M]; the coefficient fingerprint names the form.
# Shuffled-RPE control: the RPE-dependent blocks (I,S_post,S_pre) should collapse,
# M (RPE-independent) should hold -> proves the ID rides on RPE timing.
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
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

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

total_rpes = np.asarray(to['total_rpes'], float)
loss_steps_arr = np.asarray(to['loss_steps'], int)
L = min(len(total_rpes), len(loss_steps_arr))
total_rpes, loss_steps_arr = total_rpes[:L], loss_steps_arr[:L]


def expand(rpe_ls):
    r = np.zeros(n_steps); prev = 0
    for k in range(L):
        e = min(n_steps, loss_steps_arr[k] + 1)
        r[prev:e] = rpe_ls[k]; prev = e
        if prev >= n_steps:
            break
    if prev < n_steps and L:
        r[prev:] = rpe_ls[-1]
    return r


n_bl = int(train_params.get('n_window_baseline', 20))
mean_act = np.zeros_like(out); prev = np.zeros(n_neu)
for t in range(n_steps):
    prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
dev = out - mean_act
m = out.mean(0)
M = np.outer(m, m)
BLOCKS = ['I', 'S_post', 'S_pre', 'M']


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def regress(rpe_raw):
    q = (rpe_raw[:, None] * dev).sum(0)
    I = (rpe_raw[:, None] * dev).T @ dev
    S_post = np.outer(q, m)
    S_pre = np.outer(m, q)
    X = np.column_stack([zc(I[off]), zc(S_post[off]), zc(S_pre[off]), zc(M[off])])
    y = zc(dW[off])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum(y ** 2)
    return beta, r2


rng = np.random.default_rng(0)
rpe_true = expand(total_rpes)
rpe_shuf = expand(total_rpes[rng.permutation(L)])
bt, r2t = regress(rpe_true)
bs, r2s = regress(rpe_shuf)

print("Fingerprint regression  dW ~ [I, S_post, S_pre, M]  (standardized beta, seed {}):".format(SEED))
print("  {:12s} {:>10s} {:>10s}".format('block', 'true RPE', 'shuffled'))
for i, b in enumerate(BLOCKS):
    print("  {:12s} {:+10.3f} {:+10.3f}".format(b, bt[i], bs[i]))
print("  {:12s} {:10.3f} {:10.3f}".format('R2', r2t, r2s))
print("\nModel rule expectation: I and S_post present, S_pre ~ 0, M ~ 0  -> post-deviated (pre raw).")

fig = plt.figure(figsize=(4.4, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0 / fw, 0.95 / fh, 3.1 / fw, 1.8 / fh])
x = np.arange(4)
ax.bar(x - 0.2, bt, 0.38, color='#1baf7a', label='true RPE (R$^2$={:.2f})'.format(r2t))
ax.bar(x + 0.2, bs, 0.38, color='#b0b0b0', label='shuffled RPE (R$^2$={:.2f})'.format(r2s))
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['I\n(interaction)', 'S_post\n(POST dev)', 'S_pre\n(PRE dev)', 'M\n(mean-mean)'], fontsize=7)
ax.set_ylabel('standardized $\\beta$')
ax.set_title('Eligibility fingerprint: $\\Delta W$ ~ I + S$_{post}$ + S$_{pre}$ + M', fontsize=8)
ax.legend(fontsize=7, frameon=False, loc='best')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_form_identification.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
