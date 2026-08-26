#%% ============================================================================
# MODEL 4x4 matrix: HI(division) correlated with each behavioral variable,
# for each of the 4 eligibility forms (2x2 of raw/deviation on pre & post).
#   columns (elig): hebb=r_pre r_post, dpost_pre=r_pre dr_post,
#                   dpost_dpre=dr_pre dr_post, post_dpre=dr_pre r_post
#   rows (behavior): hits, speed_rpe(=dSpeed/RPE), speed, hits_rpe
# HI = div_slopes_full_delta (HI vs whole-session dW, data-matched). 15 seeds.
import os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
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

ELIG = ('hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre')
ELIG_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
BEH = ('hits', 'speed_rpe', 'speed', 'hits_rpe')
BEH_LABEL = ['Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']

hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true',) + BEH}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]
    task_hist = task.hists[0]
    rpes_divs, _dl, ds_full, _fs, _e, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params)
    res = {}
    for fi, form in enumerate(ELIG):
        hi = np.asarray(ds_full[fi], float)
        for beh in BEH:
            bv = rpes_divs.get(beh)
            r = np.nan
            if bv is not None:
                bv = np.asarray(bv, float)
                ok = np.isfinite(hi) & np.isfinite(bv)
                if ok.sum() >= 3 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                    r = spearmanr(hi[ok], bv[ok])[0]
            res[(form, beh)] = r
    return res


allres = []
for s in SEEDS:
    try:
        allres.append(run_seed(s))
        print("seed {} done".format(s))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_behavior_elig_matrix.npy'), allres, allow_pickle=True)

# aggregate mean corr + Wilcoxon p per (behavior, elig)
mat = np.full((len(BEH), len(ELIG)), np.nan)
matp = np.full((len(BEH), len(ELIG)), np.nan)
for bi, beh in enumerate(BEH):
    for fi, form in enumerate(ELIG):
        v = np.array([r[(form, beh)] for r in allres], float)
        v = v[np.isfinite(v)]
        if len(v):
            mat[bi, fi] = np.mean(v)
            matp[bi, fi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan

vmax = max(0.2, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(4.4, 3.4))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.35 / fw, 1.05 / fh, 2.4 / fw, 2.0 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi in range(len(BEH)):
    for fi in range(len(ELIG)):
        if np.isnan(mat[bi, fi]):
            continue
        p = matp[bi, fi]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(fi, bi, '{:+.2f}\n{}'.format(mat[bi, fi], st), ha='center', va='center',
                fontsize=7, fontweight='bold' if st else 'normal')
ax.set_xticks(range(len(ELIG))); ax.set_xticklabels(ELIG_LABEL, fontsize=7)
ax.set_yticks(range(len(BEH))); ax.set_yticklabels(BEH_LABEL)
ax.set_title('MODEL: HI vs behavior x eligibility (n=15)')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_4x4.' + ext), dpi=200, bbox_inches='tight')

print("\nMODEL HI vs behavior x eligibility (mean rho):")
print("  {:16s}".format('behavior') + ''.join('{:>14s}'.format(l.replace('$', '').replace('\\', '')) for l in ELIG_LABEL))
for bi, beh in enumerate(BEH):
    print("  {:16s}".format(BEH_LABEL[bi].split(' ')[0]) + ''.join('{:+14.3f}'.format(mat[bi, fi]) for fi in range(len(ELIG))))
print("\nSaved to", OUT)
