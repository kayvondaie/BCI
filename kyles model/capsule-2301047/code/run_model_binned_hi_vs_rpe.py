#%% ============================================================================
# Model version of the sliding-window CELL 9 binned plot (HI vs RPE).
# Recipe copied from sliding_window_temporal_offset_v2.py CELL 9:
#   per "session" (here: per seed) z-score HI(division) and behavior(division),
#   pool the z-scored pairs over all seeds, bin by behavior percentile (n_bins),
#   plot mean HI-z vs mean behavior-z with SEM.
# 2x2: columns = eligibility form (raw r_pre*r_post, fluct r_pre*(r_post-avg));
#      rows    = RPE type (dSpeed = behavioral/data-matched, true internal RPE).
# The dSpeed row is the data-matched test; the true-RPE row is "the puzzle".
import os, sys
import numpy as np
from scipy.stats import spearmanr
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

ELIG = ('true', 'hebb', 'dpost_pre')            # dpost_pre = fluct, hebb = raw
hi_params = {'div_mode': 'trials', 'n_trials_per_div': 5, 'n_divisions': 20,
             'elig_types': ELIG, 'rpe_types': ('true', 'speed_rpe')}
output_vars = ['W_rec_vals', 'W_rec_elg_vals', 'output', 'reward', 'total_rpes', 'loss_steps']
SEEDS = list(range(15))
n_bins = 3


def run_seed(seed):
    params = kp.default_toy_params(seed=seed, verbose=False)
    kp.task_params, kp.train_params, kp.net_params = params
    kp.compute_estimated_tunings_our_bci = lambda *a, **k: (None, {})
    _po, toa, _net, task, _tp = kp.train_task(params, output_vars=output_vars, verbose=False)
    to = toa[0]
    task_hist = task.hists[0]
    rpes_divs, _dl, ds_full, _fs, _e, _m = kp.compute_local_hebbian_indexes(
        to, task_hist, params, hi_params)
    i_h, i_d = ELIG.index('hebb'), ELIG.index('dpost_pre')
    return {'hi_raw': np.asarray(ds_full[i_h], float),
            'hi_fluct': np.asarray(ds_full[i_d], float),
            'rpe_true': np.asarray(rpes_divs['true'], float),
            'rpe_beh': np.asarray(rpes_divs['speed_rpe'], float)}


seed_data = []
for s in SEEDS:
    try:
        seed_data.append(run_seed(s))
        print("seed {} done".format(s))
    except Exception as e:
        print("seed {} FAILED: {}".format(s, e))
np.save(os.path.join(OUT, 'model_binned_hi_vs_rpe.npy'), seed_data, allow_pickle=True)

#%% ============================================================================
# Binned figure (identical z-score/pool/bin recipe as data CELL 9)
# ============================================================================
seed_data = list(np.load(os.path.join(OUT, 'model_binned_hi_vs_rpe.npy'),
                         allow_pickle=True))
COLS = [('hi_raw', 'raw\n$r_{pre}r_{post}$'),
        ('hi_fluct', 'fluct\n$r_{pre}(r_{post}{-}\\overline{r})$')]
ROWS = [('rpe_beh', '$\\Delta$Speed\n(behavioral RPE)', '#eb6834'),
        ('rpe_true', 'true RPE\n(internal)', '#888888')]


def pooled_binned(hi_key, rpe_key):
    beh_z, slope_z = [], []
    for s in seed_data:
        bvar = s[rpe_key]
        slope = s[hi_key]
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) < 5:
            continue
        b, sl = bvar[ok], slope[ok]
        if np.std(b) == 0 or np.std(sl) == 0:
            continue
        beh_z.append((b - np.mean(b)) / np.std(b))
        slope_z.append((sl - np.mean(sl)) / np.std(sl))
    if not beh_z:
        return None
    beh_z = np.concatenate(beh_z)
    slope_z = np.concatenate(slope_z)
    edges = np.percentile(beh_z, np.linspace(0, 100, n_bins + 1))
    bc, bm, bs = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            m = (beh_z >= edges[bi]) & (beh_z < edges[bi + 1])
        else:
            m = (beh_z >= edges[bi]) & (beh_z <= edges[bi + 1])
        if np.sum(m) < 3:
            continue
        bc.append(np.mean(beh_z[m]))
        bm.append(np.mean(slope_z[m]))
        bs.append(np.std(slope_z[m]) / np.sqrt(np.sum(m)))
    rho = spearmanr(beh_z, slope_z)[0]
    return np.array(bc), np.array(bm), np.array(bs), rho


# axes sized in inches (paper style)
AXW, AXH = 1.5, 1.3
L0, GAPX = 0.8, 0.95
B0, GAPY = 0.62, 0.85
fig_w = L0 + AXW + GAPX + AXW + 0.25
fig_h = B0 + AXH + GAPY + AXH + 0.55
fig = plt.figure(figsize=(fig_w, fig_h))


def ax_at(li, bi):
    return fig.add_axes([li / fig_w, bi / fig_h, AXW / fig_w, AXH / fig_h])


for ri, (rpe_key, rpe_lab, clr) in enumerate(ROWS):
    bi_in = B0 + (len(ROWS) - 1 - ri) * (AXH + GAPY)
    for ci, (hi_key, hi_lab) in enumerate(COLS):
        li_in = L0 + ci * (AXW + GAPX)
        ax = ax_at(li_in, bi_in)
        res = pooled_binned(hi_key, rpe_key)
        if res is not None:
            bc, bm, bs, rho = res
            ax.errorbar(bc, bm, yerr=bs, fmt='o-', color=clr, capsize=4,
                        linewidth=1.5, markersize=5)
            ax.text(0.04, 0.92, r'$\rho$={:+.2f}'.format(rho), transform=ax.transAxes,
                    ha='left', va='top', fontsize=8)
        ax.axhline(0, color='k', ls='-', alpha=0.3, lw=0.8)
        ax.axvline(0, color='k', ls='--', alpha=0.3, lw=0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel('{} (within-seed z)'.format(rpe_lab.split(chr(10))[0]))
        if ci == 0:
            ax.set_ylabel('HI (within-seed z)')
        if ri == 0:
            ax.set_title(hi_lab, fontweight='bold')

fig.suptitle('MODEL: binned HI vs RPE (n={} seeds, {} bins)'.format(len(seed_data), n_bins),
             y=0.99)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_model_binned_hi_vs_rpe.' + ext),
                dpi=200, bbox_inches='tight')

print("\nMODEL binned HI vs RPE, pooled Spearman rho:")
for rpe_key, rpe_lab, _ in ROWS:
    for hi_key, hi_lab in COLS:
        r = pooled_binned(hi_key, rpe_key)
        print("  {:10s} x {:10s}  rho={:+.3f}".format(
            rpe_key, hi_key, np.nan if r is None else r[3]))
print("\nSaved to", OUT)
