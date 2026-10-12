#%% ============================================================================
# MODEL variance-explained (MLR) test, per eligibility form.
# Mirrors the data three_factor_variance_explained.py exactly:
#   dW(pair) = sum_w beta_w * elig_w(pair), 5-fold CV across pairs, report the
#   held-out Pearson r between predicted and actual dW.
# Run for each eligibility form, INCLUDING the true eligibility as the ceiling.
# The question: does the MLR actually explain plasticity out-of-sample, and how
# well for each eligibility? (corr(HI,Outcome) assumes it does; this checks it.)
#   two_factor = single full-session slope (HI held constant, the null).
#   three_factor = one beta per division (time-varying HI, the MLR).
import os, sys
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
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
ELIG_LABEL = ['true\n(ceiling)', '$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))
N_CV = 5


def cv_test_r(X, Y):
    """5-fold CV across pairs; z-score X cols globally, Y within-fold; pinv fit."""
    ok = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 2 * N_CV or np.std(Y) == 0:
        return np.nan
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
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
    to = toa[0]
    task_hist = task.hists[0]
    _rd, _ds, _dsf, _fs, elig_divs, _mlr = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    Wv = to['W_rec_vals']
    Y = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()          # actual dW per pair
    n_div = elig_divs.shape[1]
    out = {}
    for k, form in enumerate(ELIG):
        X = elig_divs[k].reshape(n_div, -1).T                # (n_pairs, n_div)
        out[(form, '3f')] = cv_test_r(X, Y)                  # MLR (time-varying HI)
        out[(form, '2f')] = cv_test_r(X.sum(1, keepdims=True), Y)  # constant HI (null)
    return out


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        r = allres[-1]
        print("seed {:2d}: ".format(s) + "  ".join(
            "{}={:+.3f}".format(f, r[(f, '3f')]) for f in ELIG))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_mlr_variance_explained.npy'), allres, allow_pickle=True)


def agg(form, kind):
    v = np.array([r[(form, kind)] for r in allres], float)
    v = v[np.isfinite(v)]
    return (np.mean(v), np.std(v) / np.sqrt(len(v))) if len(v) else (np.nan, np.nan)


print("\nMODEL MLR variance-explained (CV test r, n={} seeds):".format(len(allres)))
print("  {:24s} {:>16s} {:>16s}".format("eligibility", "3-factor (MLR)", "2-factor (const)"))
for f, lab in zip(ELIG, ['true', 'r_pre r_post', 'r_pre dr_post', 'dr_pre dr_post', 'dr_pre r_post']):
    m3, s3 = agg(f, '3f'); m2, s2 = agg(f, '2f')
    print("  {:24s} {:+.3f} +/- {:.3f}   {:+.3f} +/- {:.3f}".format(lab, m3, s3, m2, s2))

# figure: 3-factor MLR test r per eligibility (+ 2-factor)
fig = plt.figure(figsize=(4.6, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([0.95 / fw, 0.95 / fh, 3.3 / fw, 1.8 / fh])
x = np.arange(len(ELIG))
m3 = [agg(f, '3f')[0] for f in ELIG]; s3 = [agg(f, '3f')[1] for f in ELIG]
m2 = [agg(f, '2f')[0] for f in ELIG]; s2 = [agg(f, '2f')[1] for f in ELIG]
ax.bar(x - 0.2, m3, 0.38, yerr=s3, color='#2c3e50', capsize=2, label='3-factor (MLR)')
ax.bar(x + 0.2, m2, 0.38, yerr=s2, color='#b0b0b0', capsize=2, label='2-factor (const HI)')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(ELIG_LABEL, fontsize=7)
ax.set_ylabel('CV test r (predicted vs actual $\\Delta W$)')
ax.set_title('MODEL: does the MLR predict $\\Delta W$? (n=15)')
ax.legend(loc='best', frameon=False)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_mlr_variance_explained.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
