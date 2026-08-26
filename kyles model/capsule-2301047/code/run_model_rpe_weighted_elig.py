#%% ============================================================================
# NEW analysis: identify the eligibility FORM and the RPE the model actually uses,
# by reconstructing dW with the RPE-WEIGHTED eligibility (per timestep):
#   R_form(i,j) = sum_t  w(t) * f_form(pre_j(t)) * g_form(post_i(t))
#   score(form, w) = corr( R_form , dW )  across pairs.
# The per-timestep weighting w(t) breaks the covariance identity, so the forms
# separate (unlike the uniform HI analysis). Sweep form x weighting.
#   - true form (r_pre * dev(r_post)) with TRUE RPE should reconstruct dW best.
#   - "uniform" column = the old HI analysis -> forms collapse (control).
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

out = np.asarray(to['output'], float)                       # (n_steps, n_neurons)
n_steps, n_neu = out.shape
# total_rpes is per LOSS step; expand to raw-step resolution (how the rule applies it)
total_rpes = np.asarray(to['total_rpes'], float)
loss_steps_arr = np.asarray(to['loss_steps'], int)
L = min(len(total_rpes), len(loss_steps_arr))
total_rpes, loss_steps_arr = total_rpes[:L], loss_steps_arr[:L]


def expand_to_raw(rpe_ls):
    r = np.zeros(n_steps); prev = 0
    for k in range(L):
        e = min(n_steps, loss_steps_arr[k] + 1)
        r[prev:e] = rpe_ls[k]; prev = e
        if prev >= n_steps:
            break
    if prev < n_steps and L:
        r[prev:] = rpe_ls[-1]
    return r


loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
Wv = to['W_rec_vals']
dW = (Wv[loss_step_divs[-1]] - Wv[0])
off = ~np.eye(n_neu, dtype=bool)
dWv = dW[off]

# running-mean baseline (model's own convention: n_window_baseline)
n_bl = int(train_params.get('n_window_baseline', 20))
mean_act = np.zeros_like(out)
prev = np.zeros(n_neu)
for t in range(n_steps):
    prev = accumulate_decay(prev, out[t], n_window=n_bl)
    mean_act[t] = prev
dev = out - mean_act                                        # deviation from running mean

# candidate forms: (post-factor g, pre-factor f)
FORMS = {
    'raw  r_pre r_post':      (out, out),
    'r_pre dr_post (TRUE)':   (dev, out),
    'dr_pre dr_post':         (dev, dev),
    'dr_pre r_post':          (out, dev),
}
# candidate weightings w(t)
rng = np.random.default_rng(0)
rpe_raw = expand_to_raw(total_rpes)
rpe_shuf = expand_to_raw(total_rpes[rng.permutation(L)])
WEIGHTS = {
    'true RPE':   rpe_raw,
    'uniform':    np.ones(n_steps),
    'shuffledRPE': rpe_shuf,
    '|RPE| only':  np.abs(rpe_raw),
}


def R_of(g_post, f_pre, w):
    # R(i,j) = sum_t w(t) g_post(t,i) f_pre(t,j)
    return ((w[:, None] * g_post).T @ f_pre)


def score(g, f, w):
    R = R_of(g, f, w)[off]
    if np.std(R) == 0:
        return np.nan
    return np.corrcoef(R, dWv)[0, 1]


mat = np.full((len(FORMS), len(WEIGHTS)), np.nan)
for fi, (fn, (g, f)) in enumerate(FORMS.items()):
    for wi, (wn, w) in enumerate(WEIGHTS.items()):
        mat[fi, wi] = score(g, f, w)

print("Reconstruction of dW: corr(R_form_weight, dW) across pairs (seed {})".format(SEED))
print("  {:22s}".format('form \\ weight') + "".join("{:>13s}".format(w) for w in WEIGHTS))
for fi, fn in enumerate(FORMS):
    print("  {:22s}".format(fn) + "".join("{:+13.3f}".format(mat[fi, wi]) for wi in range(len(WEIGHTS))))

vmax = np.nanmax(np.abs(mat))
fig = plt.figure(figsize=(4.8, 3.2))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.9 / fw, 1.15 / fh, 2.4 / fw, 1.7 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for fi in range(len(FORMS)):
    for wi in range(len(WEIGHTS)):
        if np.isnan(mat[fi, wi]):
            continue
        ax.text(wi, fi, '{:+.2f}'.format(mat[fi, wi]), ha='center', va='center',
                fontsize=7.5, color='white' if abs(mat[fi, wi]) > 0.6 * vmax else 'k')
ax.set_xticks(range(len(WEIGHTS))); ax.set_xticklabels(list(WEIGHTS.keys()), rotation=30, ha='right', fontsize=7)
ax.set_yticks(range(len(FORMS))); ax.set_yticklabels(list(FORMS.keys()), fontsize=7)
ax.set_title('Reconstruct $\\Delta W$ with RPE-weighted eligibility\ncorr(R, $\\Delta W$) across pairs', fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04); cb.set_label('corr with $\\Delta W$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_rpe_weighted_elig.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
