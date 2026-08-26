#%% ============================================================================
# Validation: under the data-matched convention (per-trial activity, trailing-20
# baseline, 10-trial sliding windows step 5), does the correct eligibility
# (r_pre*dr_post) HI still track RPE -- while the forms stay distinguishable?
# Correlate HI(window) with behavior(window), INCLUDING the model's TRUE internal
# RPE (what generated dW). 15 seeds; aggregate mean rho + Wilcoxon.
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
BEH = ['true', 'hits', 'speed_rpe', 'speed', 'hits_rpe']
BEH_LABEL = ['RPE (true)', 'Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']
WIN_SIZE, WIN_STEP, N_BASELINE = 10, 5, 20
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ('hebb',), 'rpe_types': ('true',)}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))


def trailing_baseline(A, w):
    bl = np.empty_like(A)
    for t in range(A.shape[0]):
        bl[t] = A[max(0, t - w + 1):t + 1].mean(0)
    return bl


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]; task_hist = task.hists[0]

    rpe_est = kp.get_rpe_estimates(task_hist)
    ts = np.asarray(task_hist['trial_starts'], int)
    out = np.asarray(to['output'], float)
    total_rpes = np.asarray(to['total_rpes'], float)
    bounds = list(ts) + [out.shape[0]]
    A = np.array([np.nanmean(out[bounds[k]:bounds[k + 1]], axis=0)
                  for k in range(len(ts)) if bounds[k + 1] > bounds[k]])
    # per-trial true RPE = sum of total_rpes over the trial's steps
    true_t = np.array([np.nansum(total_rpes[bounds[k]:bounds[k + 1]])
                       for k in range(len(ts)) if bounds[k + 1] > bounds[k]])
    beh_trial = {'true': true_t}
    for b in ('hits', 'speed_rpe', 'speed', 'hits_rpe'):
        beh_trial[b] = np.asarray(rpe_est[b], float)

    D = A - trailing_baseline(A, N_BASELINE)
    n_trials = min([A.shape[0]] + [len(beh_trial[b]) for b in BEH])
    loss_step_divs = kp.get_div_idxs(task_hist, to['loss_steps'], hi_params)
    Wv = to['W_rec_vals']
    dW = (Wv[loss_step_divs[-1]] - Wv[0]).flatten()

    win_starts = list(range(0, n_trials - WIN_SIZE + 1, WIN_STEP))
    HI = {f: [] for f in FORMS}
    BW = {b: [] for b in BEH}
    for ws in win_starts:
        tr = slice(ws, ws + WIN_SIZE)
        Ad, Dd = A[tr], D[tr]
        forms = {'hebb': Ad.T @ Ad, 'dpost_pre': Dd.T @ Ad, 'dpost_dpre': Dd.T @ Dd, 'post_dpre': Ad.T @ Dd}
        for f, E in forms.items():
            ef = E.flatten()
            HI[f].append(linregress(ef, dW).slope if np.std(ef) > 0 else np.nan)
        for b in BEH:
            BW[b].append(np.nanmean(beh_trial[b][tr]))
    res = {}
    for f in FORMS:
        h = np.asarray(HI[f], float)
        for b in BEH:
            bv = np.asarray(BW[b], float)
            ok = np.isfinite(h) & np.isfinite(bv)
            res[(f, b)] = spearmanr(h[ok], bv[ok])[0] if (ok.sum() >= 4 and np.std(h[ok]) > 0 and np.std(bv[ok]) > 0) else np.nan
    return res


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s)); print("seed {} done".format(s))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_behavior_elig_windowed_multiseed.npy'), allres, allow_pickle=True)

mat = np.full((len(BEH), len(FORMS)), np.nan); matp = np.full((len(BEH), len(FORMS)), np.nan)
for bi, b in enumerate(BEH):
    for fi, f in enumerate(FORMS):
        v = np.array([r[(f, b)] for r in allres], float); v = v[np.isfinite(v)]
        if len(v):
            mat[bi, fi] = np.mean(v)
            matp[bi, fi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan

vmax = max(0.2, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(4.6, 3.8))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.5 / fw, 1.05 / fh, 2.5 / fw, 2.4 / fh])
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
ax.set_title('MODEL, data conv (trailing-20, 10-trial win), n=15')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04); cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_4x4_windowed_multiseed.' + ext), dpi=200, bbox_inches='tight')

print("\nMODEL (data conv, windowed) corr(HI, outcome), mean rho over 15 seeds:")
print("  {:16s}".format('outcome') + "".join("{:>11s}".format(l) for l in ['r r', 'r dr', 'dr dr', 'dr r']))
for bi, b in enumerate(BEH):
    print("  {:16s}".format(BEH_LABEL[bi].split(' (')[0]) + "".join("{:+11.2f}".format(mat[bi, fi]) for fi in range(len(FORMS))))
print("\nSaved to", OUT)
