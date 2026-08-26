#%% ============================================================================
# Model HI vs behavior x eligibility 4x4, but with the DATA convention for the
# eligibility: per-trial activity, first-20-trial fixed baseline (NOT Kyle's
# running-EMA). HI(div) = slope of whole-session dW vs the per-division eligibility
# (summed over the division's trials), then corr with behavior across divisions.
# 15 seeds. Companion to run_model_behavior_elig_matrix.py (which used running-EMA).
import os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon, linregress
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

FORMS = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre']
FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
BEH = ['hits', 'speed_rpe', 'speed', 'hits_rpe']
BEH_LABEL = ['Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']
NPD = 5                                                       # trials per division
N_BASELINE = 20
hi_params = {'div_mode': 'trials', 'n_trials_per_div': NPD, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',) + tuple(BEH)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]; task_hist = task.hists[0]
    rpes_divs, _ds, _dsf, _fs, _e, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params)
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    n_div = loss_step_divs.shape[0] - 1
    Wv = to['W_rec_vals']
    dW = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()

    # per-trial activity (mean over each trial's steps), first-20 baseline
    out = np.asarray(to['output'], float)
    ts = np.asarray(task_hist['trial_starts'], int)
    bounds = list(ts) + [out.shape[0]]
    A = np.array([np.nanmean(out[bounds[k]:bounds[k + 1]], axis=0)
                  for k in range(len(ts)) if bounds[k + 1] > bounds[k]])
    D = A - A[:min(N_BASELINE, A.shape[0])].mean(0)

    HI = {f: np.full(n_div, np.nan) for f in FORMS}
    for d in range(n_div):
        s, e = d * NPD, (d + 1) * NPD
        if e > A.shape[0]:
            break
        Ad, Dd = A[s:e], D[s:e]
        forms = {'hebb': Ad.T @ Ad, 'dpost_pre': Dd.T @ Ad,
                 'dpost_dpre': Dd.T @ Dd, 'post_dpre': Ad.T @ Dd}
        for f, E in forms.items():
            ef = E.flatten()
            if np.std(ef) > 0:
                HI[f][d] = linregress(ef, dW).slope

    res = {}
    for f in FORMS:
        for b in BEH:
            bv = np.asarray(rpes_divs.get(b), float) if rpes_divs.get(b) is not None else None
            r = np.nan
            if bv is not None:
                h = HI[f][:len(bv)]; bvv = bv[:len(h)]
                ok = np.isfinite(h) & np.isfinite(bvv)
                if ok.sum() >= 3 and np.std(h[ok]) > 0 and np.std(bvv[ok]) > 0:
                    r = spearmanr(h[ok], bvv[ok])[0]
            res[(f, b)] = r
    return res


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s)); print("seed {} done".format(s))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_behavior_elig_datmatched.npy'), allres, allow_pickle=True)

mat = np.full((len(BEH), len(FORMS)), np.nan)
matp = np.full((len(BEH), len(FORMS)), np.nan)
for bi, b in enumerate(BEH):
    for fi, f in enumerate(FORMS):
        v = np.array([r[(f, b)] for r in allres], float); v = v[np.isfinite(v)]
        if len(v):
            mat[bi, fi] = np.mean(v)
            matp[bi, fi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan

vmax = max(0.2, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(4.4, 3.4))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.35 / fw, 1.05 / fh, 2.4 / fw, 2.0 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi in range(len(BEH)):
    for fi in range(len(FORMS)):
        if np.isnan(mat[bi, fi]):
            continue
        p = matp[bi, fi]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(fi, bi, '{:+.2f}\n{}'.format(mat[bi, fi], st), ha='center', va='center',
                fontsize=7, fontweight='bold' if st else 'normal')
ax.set_xticks(range(len(FORMS))); ax.set_xticklabels(FORM_LABEL, fontsize=7)
ax.set_yticks(range(len(BEH))); ax.set_yticklabels(BEH_LABEL)
ax.set_title('MODEL, data-matched elig (first-20, per-trial), n=15')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_4x4_datmatched.' + ext), dpi=200, bbox_inches='tight')

print("\nMODEL (data-matched elig) HI vs behavior x form (mean rho):")
for bi, b in enumerate(BEH):
    print("  {:14s}".format(BEH_LABEL[bi].split(' ')[0]) + "".join("{:+8.2f}".format(mat[bi, fi]) for fi in range(len(FORMS))))
print("\nSaved to", OUT)
