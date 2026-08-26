#%% ============================================================================
# MODEL MLR variance-explained, regressing out the NON-INTERACTING MEAN TERMS.
# The raw eligibility wins the plain MLR only because dW has additive main-effect
# structure (per-post and per-pre "how much this neuron changes regardless of its
# partner"). Remove that by double-centering (subtract row means + col means +
# grand mean) so the fit is a fair test of the genuine pre-post INTERACTION.
# Three conditions per eligibility form (5-fold CV test r, predicted vs actual dW):
#   Y_orig  : original dW                (main effects present)
#   Y_dc    : dW double-centered         (non-interacting means regressed out of target)
#   XY_dc   : dW AND each elig window double-centered (interaction vs interaction)
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

ELIG = ('true', 'hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')
ELIG_LABEL = ['true', '$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
SHORT = ['true', 'r_pre r_post', 'r_pre dr_post', 'dr_pre dr_post', 'dr_pre r_post']
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))
N_CV = 5


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
    dW = Wv[loss_step_divs[-1]] - Wv[0]                  # (n_post, n_pre)
    n_post, n_pre = dW.shape
    n_div = elig_divs.shape[1]
    Y_orig = dW.flatten()
    Y_dc = double_center(dW).flatten()
    out = {}
    for k, form in enumerate(ELIG):
        E = elig_divs[k]                                  # (n_div, n_post, n_pre)
        X = E.reshape(n_div, -1).T                        # (n_pairs, n_div)
        X_dc = np.stack([double_center(E[d]).flatten() for d in range(n_div)], axis=1)
        out[(form, 'Y_orig')] = cv_test_r(X, Y_orig)
        out[(form, 'Y_dc')] = cv_test_r(X, Y_dc)
        out[(form, 'XY_dc')] = cv_test_r(X_dc, Y_dc)
    return out


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        r = allres[-1]
        print("seed {:2d}: Y_dc: ".format(s) + "  ".join(
            "{}={:+.3f}".format(sh, r[(f, 'Y_dc')]) for f, sh in zip(ELIG, SHORT)))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_mlr_centered.npy'), allres, allow_pickle=True)

CONDS = ['Y_orig', 'Y_dc', 'XY_dc']
CONDLAB = {'Y_orig': 'dW original', 'Y_dc': 'dW centered', 'XY_dc': 'dW & elig centered'}


def agg(form, cond):
    v = np.array([r[(form, cond)] for r in allres], float)
    v = v[np.isfinite(v)]
    return (np.mean(v), np.std(v) / np.sqrt(len(v))) if len(v) else (np.nan, np.nan)


print("\nMODEL MLR test r (n={} seeds), regressing out non-interacting means:".format(len(allres)))
print("  {:16s} {:>14s} {:>14s} {:>16s}".format("eligibility", *[CONDLAB[c] for c in CONDS]))
for f, sh in zip(ELIG, SHORT):
    print("  {:16s}".format(sh) + "".join("   {:+.3f}+/-{:.3f}".format(*agg(f, c)) for c in CONDS))

# figure: grouped bars, one group per form, one bar per condition
fig = plt.figure(figsize=(5.2, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([0.95 / fw, 0.95 / fh, 3.9 / fw, 1.8 / fh])
x = np.arange(len(ELIG))
colors = {'Y_orig': '#b0b0b0', 'Y_dc': '#2c3e50', 'XY_dc': '#8c6d31'}
for j, c in enumerate(CONDS):
    m = [agg(f, c)[0] for f in ELIG]; s = [agg(f, c)[1] for f in ELIG]
    ax.bar(x + (j - 1) * 0.27, m, 0.25, yerr=s, color=colors[c], capsize=2, label=CONDLAB[c])
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(ELIG_LABEL, fontsize=7)
ax.set_ylabel('CV test r (pred vs actual $\\Delta W$)')
ax.set_title('MODEL MLR: regressing out non-interacting means (n=15)')
ax.legend(loc='upper right', frameon=False, fontsize=7)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_mlr_centered.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
