#%% ============================================================================
# Binned HI-vs-RPE figures (data vs model), using the SAME z-score/pool/bin
# recipe as sliding_window_temporal_offset.py CELL 9.
#   per "session" (data) / "seed" (model): z-score HI(division) and RPE(division),
#   pool z-scored pairs, bin by RPE percentile (n_bins), plot mean HI-z vs RPE-z.
# DATA side uses the old dev2_lag / dot_prod_lag modes (pre epoch), NOT the
# full-trial-baseline modes.  MODEL rule is left as-is (hebb=raw, dpost_pre=fluct).
# No retraining: model per-division vectors are cached in model_binned_hi_vs_rpe.npy.
import os
import numpy as np
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'

OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'
DATA_NPY = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\meta_analysis_results\sliding_window_temporal_offset.npy'
MODEL_NPY = os.path.join(OUT, 'model_binned_hi_vs_rpe.npy')

data = np.load(DATA_NPY, allow_pickle=True).item()          # {dot_prod_lag, dev2_lag}
model = list(np.load(MODEL_NPY, allow_pickle=True))          # list of per-seed dicts

C_DATA, C_MODEL = '#2c7fb8', '#eb6834'
n_bins = 3
EI_PRE = 0                                                   # pre epoch (as in CELL 9)


def pooled_binned(pairs):
    """pairs: list of (behavior_vec, hi_vec). Returns bin centers, means, sems, rho."""
    bz, sz = [], []
    for bvar, slope in pairs:
        bvar = np.asarray(bvar, float)
        slope = np.asarray(slope, float)
        ok = np.isfinite(bvar) & np.isfinite(slope)
        if np.sum(ok) < 5:
            continue
        b, sl = bvar[ok], slope[ok]
        if np.std(b) == 0 or np.std(sl) == 0:
            continue
        bz.append((b - np.mean(b)) / np.std(b))
        sz.append((sl - np.mean(sl)) / np.std(sl))
    if not bz:
        return None
    bz = np.concatenate(bz)
    sz = np.concatenate(sz)
    edges = np.percentile(bz, np.linspace(0, 100, n_bins + 1))
    bc, bm, bs = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            m = (bz >= edges[bi]) & (bz < edges[bi + 1])
        else:
            m = (bz >= edges[bi]) & (bz <= edges[bi + 1])
        if np.sum(m) < 3:
            continue
        bc.append(np.mean(bz[m]))
        bm.append(np.mean(sz[m]))
        bs.append(np.std(sz[m]) / np.sqrt(np.sum(m)))
    return np.array(bc), np.array(bm), np.array(bs), spearmanr(bz, sz)[0]


def data_pairs(mode):
    return [(s['win_rpe'], s['hi_with_int'][:, EI_PRE]) for s in data[mode]]


def model_pairs(hi_key, rpe_key):
    return [(s[rpe_key], s[hi_key]) for s in model]


def grid_fig(rows, cols, fname, suptitle):
    """rows: list of (row_label, color, get_pairs_fn(col_key), xlabel, group_id).
    y-axis range is shared within each group_id (e.g. all data, all model),
    but NOT across groups."""
    AXW, AXH = 1.5, 1.3
    L0, GAPX = 0.85, 0.95
    B0, GAPY = 0.62, 0.95
    fw = L0 + AXW + GAPX + AXW + 0.25
    fh = B0 + len(rows) * AXH + (len(rows) - 1) * GAPY + 0.55
    fig = plt.figure(figsize=(fw, fh))

    # pass 1: compute each binned result once; track max |mean|+sem per group
    cache, gmax = {}, {}
    for ri, (rlab, clr, getp, xlab, grp) in enumerate(rows):
        for ci, (ckey, ctitle) in enumerate(cols):
            res = pooled_binned(getp(ckey))
            cache[(ri, ci)] = res
            if res is not None:
                bc, bm, bs, rho = res
                gmax[grp] = max(gmax.get(grp, 0.0), np.max(np.abs(bm) + bs))
    gylim = {grp: max(0.2, np.ceil(mx * 1.15 / 0.1) * 0.1) for grp, mx in gmax.items()}

    # pass 2: draw
    for ri, (rlab, clr, getp, xlab, grp) in enumerate(rows):
        bi_in = B0 + (len(rows) - 1 - ri) * (AXH + GAPY)
        for ci, (ckey, ctitle) in enumerate(cols):
            ax = fig.add_axes([(L0 + ci * (AXW + GAPX)) / fw, bi_in / fh, AXW / fw, AXH / fh])
            res = cache[(ri, ci)]
            if res is not None:
                bc, bm, bs, rho = res
                ax.errorbar(bc, bm, yerr=bs, fmt='o-', color=clr, capsize=4,
                            linewidth=1.5, markersize=5)
                ax.text(0.04, 0.93, r'$\rho$={:+.2f}'.format(rho), transform=ax.transAxes,
                        ha='left', va='top', fontsize=8)
            ax.axhline(0, color='k', ls='-', alpha=0.3, lw=0.8)
            ax.axvline(0, color='k', ls='--', alpha=0.3, lw=0.8)
            ax.set_ylim(-gylim[grp], gylim[grp])
            ax.set_xlabel(xlab)
            if ci == 0:
                ax.set_ylabel('{}\nHI (within-z)'.format(rlab))
            if ri == 0:
                ax.set_title(ctitle, fontweight='bold')
    fig.suptitle(suptitle, y=0.99)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, fname + '.' + ext), dpi=200, bbox_inches='tight')
    return fig


RAW_T = 'raw\n$r_{pre}r_{post}$'
FLUCT_T = 'fluct\n$r_{pre}(r_{post}{-}\\overline{r})$'

# ---- FIG A: data vs model, both against the behavioral RPE ----------------
# data cols keyed by mode; model cols keyed by hi field. Different x-getters per
# row, so pass a row-specific getter.
colsA = [('raw', RAW_T), ('fluct', FLUCT_T)]
data_mode = {'raw': 'dot_prod_lag', 'fluct': 'dev2_lag'}
model_hi = {'raw': 'hi_raw', 'fluct': 'hi_fluct'}
rowsA = [
    ('DATA (n={})'.format(len(data['dev2_lag'])), C_DATA,
     lambda ck: data_pairs(data_mode[ck]), 'RPE ($\\Delta$Speed), z', 'data'),
    ('MODEL (n={})'.format(len(model)), C_MODEL,
     lambda ck: model_pairs(model_hi[ck], 'rpe_beh'), '$\\Delta$Speed (behavioral), z', 'model'),
]
grid_fig(rowsA, colsA, 'talk_fig_binned_data_vs_model',
         'Binned HI vs RPE: data (dev2 / dot-prod) and model agree')

# ---- FIG B: model only, behavioral RPE vs the privileged true RPE ----------
rowsB = [
    ('$\\Delta$Speed\n(behavioral)', C_MODEL,
     lambda ck: model_pairs(model_hi[ck], 'rpe_beh'), '$\\Delta$Speed (behavioral), z', 'model'),
    ('true RPE\n(internal)', '#888888',
     lambda ck: model_pairs(model_hi[ck], 'rpe_true'), 'true RPE (internal), z', 'model'),
]
grid_fig(rowsB, colsA, 'talk_fig_model_binned_true_vs_beh',
         'MODEL binned HI: only fluct tracks behavioral RPE; true RPE is the puzzle')

# ---- console summary -------------------------------------------------------
print("Pooled Spearman rho (binned):")
for ck, ct in colsA:
    rd = pooled_binned(data_pairs(data_mode[ck]))
    rmb = pooled_binned(model_pairs(model_hi[ck], 'rpe_beh'))
    rmt = pooled_binned(model_pairs(model_hi[ck], 'rpe_true'))
    print("  {:6s}  data(dev2/dotprod)={:+.3f}   model vs dSpeed={:+.3f}   model vs trueRPE={:+.3f}".format(
        ck, rd[3], rmb[3], rmt[3]))
print("\nSaved to", OUT)
