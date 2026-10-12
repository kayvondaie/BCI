#%% ============================================================================
# Why does dr_pre * r_post predict dW's interaction as well as the model's form?
# Decompose each deviation eligibility into the shared pure interaction plus a
# "half-mean" term, and test the INCREMENTAL value of each half over interaction.
#   r_pre  dr_post = interaction + post_half   (post_half = r_bar_pre  * dr_post)
#   dr_pre r_post  = interaction + pre_half    (pre_half  = dr_pre     * r_bar_post)
#   dr_pre dr_post = interaction
# Built by subtraction from Kyle's elig_divs:
#   interaction = elig[dpost_dpre]
#   post_half   = elig[dpost_pre]  - elig[dpost_dpre]
#   pre_half    = elig[post_dpre]  - elig[dpost_dpre]
# dW double-centered (interaction test). 5-fold CV across pairs, 15 seeds.
import os, sys
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import bci_analysis
import kyle_pipeline as kp

if not hasattr(np, 'astype'):
    np.astype = lambda x, dtype, copy=True, device=None: np.asarray(x).astype(dtype, copy=copy)
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

ELIG = ('dpost_pre', 'dpost_dpre', 'post_dpre')
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))
N_CV = 5
CONDS = ['interaction', 'post_half', 'pre_half', 'int+post_half', 'int+pre_half', 'int+both']


def double_center(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def cv_test_r(X, Y):
    ok = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 2 * N_CV or np.std(Y) == 0:
        return np.nan
    sd = X.std(0); sd[sd == 0] = 1.0
    Xz = (X - X.mean(0)) / sd
    cv = KFold(n_splits=N_CV, shuffle=True, random_state=42)
    rs = []
    for tr, te in cv.split(Xz):
        muy, sdy = Y[tr].mean(), Y[tr].std()
        sdy = sdy if (sdy > 0 and np.isfinite(sdy)) else 1.0
        Ytr, Yte = (Y[tr] - muy) / sdy, (Y[te] - muy) / sdy
        beta = np.linalg.pinv(Xz[tr]) @ Ytr
        pred = Xz[te] @ beta
        rs.append(pearsonr(pred, Yte)[0] if np.std(pred) > 0 else 0.0)
    return np.mean(rs)


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]; task_hist = task.hists[0]
    _rd, _ds, _dsf, _fs, elig_divs, _mlr = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    Wv = to['W_rec_vals']
    Y = double_center(Wv[loss_step_divs[-1]] - Wv[0]).flatten()
    n_div = elig_divs.shape[1]
    i_dp, i_dd, i_pd = ELIG.index('dpost_pre'), ELIG.index('dpost_dpre'), ELIG.index('post_dpre')
    inter = elig_divs[i_dd].reshape(n_div, -1).T                       # (n_pairs, n_div)
    post_half = (elig_divs[i_dp] - elig_divs[i_dd]).reshape(n_div, -1).T
    pre_half = (elig_divs[i_pd] - elig_divs[i_dd]).reshape(n_div, -1).T
    return {
        'interaction': cv_test_r(inter, Y),
        'post_half': cv_test_r(post_half, Y),
        'pre_half': cv_test_r(pre_half, Y),
        'int+post_half': cv_test_r(np.hstack([inter, post_half]), Y),
        'int+pre_half': cv_test_r(np.hstack([inter, pre_half]), Y),
        'int+both': cv_test_r(np.hstack([inter, post_half, pre_half]), Y),
    }


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        r = allres[-1]
        print("seed {:2d}: int={:+.3f}  +post={:+.3f}  +pre={:+.3f}".format(
            s, r['interaction'], r['int+post_half'], r['int+pre_half']))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_halfmean_decomp.npy'), allres, allow_pickle=True)


def arr(k):
    v = np.array([r[k] for r in allres], float); return v[np.isfinite(v)]


print("\nHALF-MEAN DECOMPOSITION (model, dW centered, n={} seeds):".format(len(allres)))
for c in CONDS:
    v = arr(c)
    print("  {:16s} test r = {:+.3f} +/- {:.3f}".format(c, np.mean(v), np.std(v) / np.sqrt(len(v))))

# incremental value of each half over pure interaction (paired across seeds)
base = np.array([r['interaction'] for r in allres])
for half, lab in [('int+post_half', 'post-half (model form)'), ('int+pre_half', 'pre-half (dr_pre r_post)')]:
    d = np.array([r[half] for r in allres]) - base
    p = wilcoxon(d)[1] if np.any(d != 0) else np.nan
    print("  incremental over interaction: {:22s} d={:+.3f} +/- {:.3f}  Wilcoxon p={:.4f}".format(
        lab, np.mean(d), np.std(d) / np.sqrt(len(d)), p))

# figure
fig = plt.figure(figsize=(4.4, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([0.95 / fw, 1.15 / fh, 3.3 / fw, 1.6 / fh])
m = [np.mean(arr(c)) for c in CONDS]; s = [np.std(arr(c)) / np.sqrt(len(arr(c))) for c in CONDS]
cols = ['#2c3e50', '#7aa6c2', '#c2907a', '#1f6f3f', '#8c3a2f', '#555555']
ax.bar(range(len(CONDS)), m, yerr=s, color=cols, capsize=2)
ax.axhline(np.mean(arr('interaction')), color='k', ls='--', lw=0.8, alpha=0.6)
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS, rotation=30, ha='right', fontsize=7)
ax.set_ylabel('CV test r (dW centered)')
ax.set_title('Does deviating pre add anything beyond the interaction?')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_halfmean_decomp.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
