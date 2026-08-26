#%% ============================================================================
# Surgical test: which factor does the eligibility deviate (post or pre), and is
# it RPE-driven? For a POST-deviated rule, dW = interaction + S_post, where
#   S_post(i,j) = q_i * m_j,   q_i = sum_t RPE(t)*dpost_i(t),  m_j = mean pre_j
# A PRE-deviated rule would instead contain S_pre(i,j) = m_i * q_j.
# Regress dW on [interaction, S_post, S_pre]; the true factor's term wins, the
# other ~0, and everything collapses under shuffled RPE (timing matters).
# Sanity: reconstruct dW from the model's REAL eligibility trace * RPE (should ~match).
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

# RPE expanded to raw-step resolution
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


# running-mean baseline (model convention) -> deviation
n_bl = int(train_params.get('n_window_baseline', 20))
mean_act = np.zeros_like(out); prev = np.zeros(n_neu)
for t in range(n_steps):
    prev = accumulate_decay(prev, out[t], n_window=n_bl); mean_act[t] = prev
dev = out - mean_act
m = out.mean(0)                                  # per-neuron mean activity


def build(rpe_raw):
    q = (rpe_raw[:, None] * dev).sum(0)          # per-neuron RPE-weighted fluctuation
    I = (rpe_raw[:, None] * dev).T @ dev         # interaction (i=post,j=pre)
    S_post = np.outer(q, m)                       # q_post * m_pre   (POST deviated)
    S_pre = np.outer(m, q)                        # m_post * q_pre   (PRE deviated)
    return I, S_post, S_pre


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def regress(rpe_raw):
    I, Sp, Sq = build(rpe_raw)
    X = np.column_stack([zc(I[off]), zc(Sp[off]), zc(Sq[off])])
    y = zc(dW[off])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)
    # partial correlations (each term's unique contribution)
    return beta, r2


rng = np.random.default_rng(0)
rpe_true = expand(total_rpes)
rpe_shuf = expand(total_rpes[rng.permutation(L)])

# sanity: reconstruct dW from the REAL eligibility trace * RPE
W_elg = np.asarray(to['W_rec_elg_vals'], float)          # eligibility trace, per loss step
Lk = min(L, W_elg.shape[0])
elig_recon = np.zeros((n_neu, n_neu))
for k in range(Lk):
    elig_recon += total_rpes[k] * W_elg[k]               # RPE-weighted eligibility, per loss step
sanity = np.corrcoef(elig_recon[off], dW[off])[0, 1]

print("Sanity: corr( RPE-weighted true-elig-trace , dW ) = {:+.3f}".format(sanity))
for name, rpe in [('true RPE', rpe_true), ('shuffled RPE', rpe_shuf)]:
    beta, r2 = regress(rpe)
    print("\n{}:  R2={:.3f}".format(name, r2))
    print("  beta[interaction] = {:+.3f}".format(beta[0]))
    print("  beta[S_post  (q_post*m_pre, POST-deviated)] = {:+.3f}".format(beta[1]))
    print("  beta[S_pre   (m_post*q_pre, PRE-deviated)]  = {:+.3f}".format(beta[2]))

# figure: bars of the three betas, true vs shuffled
bt, r2t = regress(rpe_true); bs, r2s = regress(rpe_shuf)
fig = plt.figure(figsize=(4.2, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.0 / fw, 0.95 / fh, 2.9 / fw, 1.8 / fh])
x = np.arange(3)
ax.bar(x - 0.2, bt, 0.38, color='#1baf7a', label='true RPE (R$^2$={:.2f})'.format(r2t))
ax.bar(x + 0.2, bs, 0.38, color='#b0b0b0', label='shuffled RPE (R$^2$={:.2f})'.format(r2s))
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['interaction', 'S_post\n(POST dev)', 'S_pre\n(PRE dev)'], fontsize=7)
ax.set_ylabel('standardized regression $\\beta$')
ax.set_title('Which factor is deviated? (dW ~ interaction + S_post + S_pre)', fontsize=7.5)
ax.legend(fontsize=7, frameon=False)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_asymmetry_test.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
