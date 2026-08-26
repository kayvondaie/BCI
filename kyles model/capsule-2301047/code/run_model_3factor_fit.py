#%% ============================================================================
# Unified 3-factor fit (model): parametrize HI(w) as a weighted sum of outcome
# variables, so the two analyses become one regression.
#   dW(pair) = sum_w HI(w) * elig_w(pair),  HI(w) = sum_k a_k * O_k(w)
#            = sum_k a_k * [ sum_w O_k(w) * elig_w(pair) ]  =  sum_k a_k * Z_k(pair)
# Outcomes O_k: const (=> 2-factor full-session elig), true RPE, hits, dSpeed, hit x RPE.
# The fitted a_k are the outcome sensitivities (corr(HI,outcome) side); the CV test r
# is the variance-explained side. Per eligibility form. dW double-centered so the raw
# form can't win on non-interacting mean structure (the honest interaction test).
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
OUTC = ['const', 'true', 'hits', 'speed_rpe', 'hits_rpe']
OUTC_LABEL = ['const (2-factor)', 'true RPE', 'hits', '$\\Delta$Speed', 'hit$\\times$RPE']
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true', 'hits', 'speed_rpe', 'hits_rpe')}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))
N_CV = 5


def double_center(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def zc(v):
    v = np.asarray(v, float).copy()
    m = np.isfinite(v)
    if m.sum() < 2 or np.std(v[m]) == 0:
        return np.zeros_like(v)
    v[~m] = np.nanmean(v[m])
    return (v - v.mean()) / v.std()


def cv_fit(X, Y):
    """5-fold CV across pairs. Returns mean test r and full-data weights (z-scored)."""
    ok = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 2 * N_CV or np.std(Y) == 0:
        return np.nan, np.full(X.shape[1], np.nan)
    sd = X.std(0); sd[sd == 0] = 1.0
    Xz = (X - X.mean(0)) / sd
    Yz_full = (Y - Y.mean()) / (Y.std() if Y.std() > 0 else 1.0)
    cv = KFold(n_splits=N_CV, shuffle=True, random_state=42)
    rs = []
    for tr, te in cv.split(Xz):
        muy, sdy = Y[tr].mean(), Y[tr].std()
        sdy = sdy if (sdy > 0 and np.isfinite(sdy)) else 1.0
        Ytr, Yte = (Y[tr] - muy) / sdy, (Y[te] - muy) / sdy
        beta = np.linalg.pinv(Xz[tr]) @ Ytr
        pred = Xz[te] @ beta
        rs.append(pearsonr(pred, Yte)[0] if np.std(pred) > 0 else 0.0)
    beta_full = np.linalg.pinv(Xz) @ Yz_full
    return np.mean(rs), beta_full


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]; task_hist = task.hists[0]
    rpes_divs, _ds, _dsf, _fs, elig_divs, _mlr = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params, return_elig_divs=True)
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    Wv = to['W_rec_vals']
    dW = Wv[loss_step_divs[-1]] - Wv[0]
    n_div = elig_divs.shape[1]
    Y_orig = dW.flatten()
    Y_dc = double_center(dW).flatten()
    # outcome design over divisions: const + z-scored outcomes  (n_div x K)
    cols = [np.ones(n_div)]
    for o in OUTC[1:]:
        cols.append(zc(rpes_divs[o])[:n_div] if o in rpes_divs and rpes_divs[o] is not None
                    else zc(rpes_divs['true'])[:n_div])
    O = np.stack(cols, axis=1)                            # (n_div, K)
    out = {}
    for k, form in enumerate(ELIG):
        E = elig_divs[k].reshape(n_div, -1)              # (n_div, n_pairs)
        Z = E.T @ O                                      # (n_pairs, K) outcome-weighted elig
        r_o, _ = cv_fit(Z, Y_orig)
        r_c, beta_c = cv_fit(Z, Y_dc)
        out[(form, 'r_orig')] = r_o
        out[(form, 'r_dc')] = r_c
        out[(form, 'w_dc')] = beta_c
    return out


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        r = allres[-1]
        print("seed {:2d}: r_dc: ".format(s) + "  ".join(
            "{}={:+.3f}".format(sh, r[(f, 'r_dc')]) for f, sh in zip(ELIG, SHORT)))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_3factor_fit.npy'), allres, allow_pickle=True)

# free-MLR ceiling (dW centered) from the earlier run, if available
ceil = {}
_cp = os.path.join(OUT, 'model_mlr_centered.npy')
if os.path.exists(_cp):
    _c = list(np.load(_cp, allow_pickle=True))
    for f in ELIG:
        v = np.array([d[(f, 'Y_dc')] for d in _c], float); v = v[np.isfinite(v)]
        ceil[f] = np.mean(v) if len(v) else np.nan


def agg(form, key):
    v = np.array([r[(form, key)] for r in allres], float)
    v = v[np.isfinite(v)]
    return (np.mean(v), np.std(v) / np.sqrt(len(v))) if len(v) else (np.nan, np.nan)


print("\n3-FACTOR FIT (model, dW centered), CV test r  [free-MLR ceiling in brackets]:")
for f, sh in zip(ELIG, SHORT):
    m, s = agg(f, 'r_dc')
    print("  {:16s} {:+.3f} +/- {:.3f}   [{:+.3f}]".format(sh, m, s, ceil.get(f, np.nan)))

print("\nFitted outcome weights (mean over seeds, dW centered; Wilcoxon p vs 0):")
print("  {:16s}".format('elig') + "".join("{:>16s}".format(l) for l in OUTC_LABEL))
W = {}
for fi, (f, sh) in enumerate(zip(ELIG, SHORT)):
    wmat = np.array([r[(f, 'w_dc')] for r in allres], float)   # (seeds, K)
    W[f] = wmat
    row = "  {:16s}".format(sh)
    for ki in range(len(OUTC)):
        col = wmat[:, ki]; col = col[np.isfinite(col)]
        p = wilcoxon(col)[1] if len(col) >= 2 and np.any(col != 0) else np.nan
        star = '*' if (np.isfinite(p) and p < 0.05) else ' '
        row += "{:+.2f}{}         ".format(np.mean(col), star)[:16]
    print(row)

# figure: test r per form (3-factor fit vs free-MLR ceiling) + weight heatmap
fig = plt.figure(figsize=(6.6, 3.0))
fw, fh = fig.get_size_inches()
axb = fig.add_axes([0.9 / fw, 0.95 / fh, 2.7 / fw, 1.8 / fh])
x = np.arange(len(ELIG))
m = [agg(f, 'r_dc')[0] for f in ELIG]; s = [agg(f, 'r_dc')[1] for f in ELIG]
axb.bar(x - 0.2, m, 0.38, yerr=s, color='#2c3e50', capsize=2, label='3-factor fit')
axb.bar(x + 0.2, [ceil.get(f, np.nan) for f in ELIG], 0.38, color='#c0c0c0', label='free-MLR ceiling')
axb.axhline(0, color='k', lw=0.8)
axb.set_xticks(x); axb.set_xticklabels(ELIG_LABEL, fontsize=7, rotation=20, ha='right')
axb.set_ylabel('CV test r (dW centered)')
axb.set_title('3-factor fit vs free-MLR ceiling')
axb.legend(loc='upper right', frameon=False, fontsize=7)

axh = fig.add_axes([4.5 / fw, 0.95 / fh, 1.7 / fw, 1.8 / fh])
wmean = np.array([[np.nanmean(W[f][:, ki]) for ki in range(len(OUTC))] for f in ELIG])
vmax = np.nanmax(np.abs(wmean))
im = axh.imshow(wmean, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for fi in range(len(ELIG)):
    for ki in range(len(OUTC)):
        axh.text(ki, fi, '{:+.2f}'.format(wmean[fi, ki]), ha='center', va='center', fontsize=6)
axh.set_xticks(range(len(OUTC))); axh.set_xticklabels(OUTC_LABEL, rotation=40, ha='right', fontsize=6)
axh.set_yticks(range(len(ELIG))); axh.set_yticklabels(SHORT, fontsize=6)
axh.set_title('fitted outcome weights', fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_3factor_fit.' + ext), dpi=200, bbox_inches='tight')
print("\nSaved to", OUT)
