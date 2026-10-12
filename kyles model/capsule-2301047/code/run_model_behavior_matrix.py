#%% ============================================================================
# Model HI x behavioral-variable matrix, for RAW and FLUCTUATION eligibility.
# ============================================================================
# Analog of the data behavior x epoch matrix. For each eligibility form
# (true / hebb=raw r_pre*r_post / dpost_pre=fluct r_pre*(r_post-avg)) correlate
# HI(division) with each of the model's behavioral variables (its analogs of
# hit rate / speed / dSpeed(RPE) / hit x RPE). 15 seeds.
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
                     'ytick.labelsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

ELIG = ('true', 'hebb', 'dpost_pre')
ELIG_LABEL = {'true': 'true elig', 'hebb': 'raw\n$r_{pre}r_{post}$',
              'dpost_pre': 'fluct\n$r_{pre}(r_{post}{-}\\overline{r})$'}
RPE_TYPES = ('true', 'hits', 'hits_rpe', 'speed', 'speed_rpe')
BEH_ORDER = ['hits', 'speed_rpe', 'speed', 'hits_rpe', 'true']
BEH_LABEL = {'hits': 'Hit rate', 'speed_rpe': '$\\Delta$Speed (RPE)',
             'speed': 'Speed', 'hits_rpe': 'Hit $\\times$ RPE', 'true': 'RPE (true)'}
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': RPE_TYPES}
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
        hi = ds_full[fi]
        for beh in RPE_TYPES:
            bv = rpes_divs.get(beh)
            r = np.nan
            if bv is not None:
                bv = np.asarray(bv, dtype=float)
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

np.save(os.path.join(OUT, 'model_behavior_matrix.npy'), allres, allow_pickle=True)

# aggregate: mean corr + Wilcoxon p, per (behavior, form)
mat = np.full((len(BEH_ORDER), len(ELIG)), np.nan)
matp = np.full((len(BEH_ORDER), len(ELIG)), np.nan)
for bi, beh in enumerate(BEH_ORDER):
    for fi, form in enumerate(ELIG):
        v = np.array([r[(form, beh)] for r in allres], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            mat[bi, fi] = np.mean(v)
            matp[bi, fi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan

vmax = max(0.3, np.nanmax(np.abs(mat)))
fig = plt.figure(figsize=(3.8, 3.4))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([1.2 / fw, 1.0 / fh, 2.0 / fw, 2.1 / fh])
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
for bi in range(len(BEH_ORDER)):
    for fi in range(len(ELIG)):
        if np.isnan(mat[bi, fi]):
            continue
        p = matp[bi, fi]
        st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
        ax.text(fi, bi, '{:+.2f}\n{}'.format(mat[bi, fi], st), ha='center', va='center',
                fontsize=7, fontweight='bold' if st else 'normal')
ax.set_xticks(range(len(ELIG))); ax.set_xticklabels([ELIG_LABEL[f] for f in ELIG])
ax.set_yticks(range(len(BEH_ORDER))); ax.set_yticklabels([BEH_LABEL[b] for b in BEH_ORDER])
ax.set_title('MODEL: HI vs behavior, by eligibility form (n=15)')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
cb.set_label('mean Spearman $\\rho$')
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_behavior_matrix.' + ext), dpi=200, bbox_inches='tight')

print("\nMODEL HI-vs-behavior (mean rho):")
print("  {:14s} {:>8s} {:>8s} {:>10s}".format("behavior", "true", "raw", "fluct"))
for bi, beh in enumerate(BEH_ORDER):
    print("  {:14s} {:+8.2f} {:+8.2f} {:+10.2f}".format(BEH_LABEL[beh].split(' ')[0],
          mat[bi, 0], mat[bi, 1], mat[bi, 2]))
print("\nSaved to", OUT)
