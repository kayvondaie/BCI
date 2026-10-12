#%% ============================================================================
# Side-by-side 4x4: HI vs behavior (rows) x eligibility form (cols), data & model.
#   cols (user order): r_pre r_post | r_pre dr_post | dr_pre dr_post | dr_pre r_post
#   rows: Hit rate | dSpeed (RPE) | Speed/RT | Hit x RPE
# The model has no epochs; the data has 4. Two versions are produced:
#   (1) epochs AVERAGED together
#   (2) the single epoch whose data matrix best matches the model matrix
#       (across-cell correlation; all 4 epoch similarities are printed)
# DATA = sliding_window_four_elig.npy (dev2 early-trial baseline).
# MODEL = model_behavior_elig_matrix.npy (hebb/dpost_pre/dpost_dpre/post_dpre).
import os
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'
DATA_NPY = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\meta_analysis_results\sliding_window_four_elig.npy'
MODEL_NPY = os.path.join(OUT, 'model_behavior_elig_matrix.npy')

COLS = [('dot_prod_lag', 'hebb', '$r_{pre}r_{post}$'),
        ('dev2_lag', 'dpost_pre', '$r_{pre}\\Delta r_{post}$'),
        ('dpost_dpre_lag', 'dpost_dpre', '$\\Delta r_{pre}\\Delta r_{post}$'),
        ('post_dpre_lag', 'post_dpre', '$\\Delta r_{pre}r_{post}$')]
# (data key, model key, label, data_sign) -- data_sign flips RT so higher=faster,
# matching the model's speed (Spearman: sign only, magnitude unchanged).
ROWS = [('win_hit', 'hits', 'Hit rate', 1),
        ('win_rpe', 'speed_rpe', '$\\Delta$Speed (RPE)', 1),
        ('win_rt', 'speed', 'Speed', -1),
        ('win_hit_rpe', 'hits_rpe', 'Hit $\\times$ RPE', 1)]
EPOCH_NAMES = ['Pre', 'Go cue', 'Late', 'Reward']

model = list(np.load(MODEL_NPY, allow_pickle=True))
data = np.load(DATA_NPY, allow_pickle=True).item() if os.path.exists(DATA_NPY) else None


def model_mat():
    m = np.full((len(ROWS), len(COLS)), np.nan)
    p = np.full((len(ROWS), len(COLS)), np.nan)
    for ri, (_, mbeh, _, _) in enumerate(ROWS):
        for ci, (_, mform, _) in enumerate(COLS):
            v = np.array([r[(mform, mbeh)] for r in model], float)
            v = v[np.isfinite(v)]
            if len(v):
                m[ri, ci] = np.mean(v)
                p[ri, ci] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan
    return m, p


def data_mat(epoch_sel):
    """epoch_sel: 'avg' (mean over epochs) or int epoch index."""
    m = np.full((len(ROWS), len(COLS)), np.nan)
    p = np.full((len(ROWS), len(COLS)), np.nan)
    for ri, (dbeh, _, _, dsign) in enumerate(ROWS):
        for ci, (dmode, _, _) in enumerate(COLS):
            vv = []
            for s in data[dmode]:
                hi = np.nanmean(s['hi_with_int'], axis=1) if epoch_sel == 'avg' \
                    else s['hi_with_int'][:, epoch_sel]
                bv = dsign * s[dbeh]
                ok = np.isfinite(hi) & np.isfinite(bv)
                if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                    vv.append(spearmanr(hi[ok], bv[ok])[0])
            vv = np.array(vv)
            if len(vv):
                m[ri, ci] = np.mean(vv)
                p[ri, ci] = wilcoxon(vv)[1] if len(vv) >= 2 and np.any(vv != 0) else np.nan
    return m, p


def similarity(a, b):
    fa, fb = a.flatten(), b.flatten()
    ok = np.isfinite(fa) & np.isfinite(fb)
    return np.corrcoef(fa[ok], fb[ok])[0, 1] if ok.sum() >= 3 else np.nan


def draw(ax, mat, matp, title, vmax):
    im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    for ri in range(len(ROWS)):
        for ci in range(len(COLS)):
            if np.isnan(mat[ri, ci]):
                continue
            pp = matp[ri, ci]
            st = '***' if pp < 1e-3 else '**' if pp < 1e-2 else '*' if pp < 5e-2 else ''
            ax.text(ci, ri, '{:+.2f}\n{}'.format(mat[ri, ci], st), ha='center', va='center',
                    fontsize=7, fontweight='bold' if st else 'normal')
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[2] for c in COLS], fontsize=7)
    ax.set_title(title)
    return im


def make_fig(dm, dp, mm, mp, fname, dtitle):
    # each panel normalized to its OWN max |rho|, with its own colorbar
    vmax_d = max(0.05, np.nanmax(np.abs(dm)))
    vmax_m = max(0.05, np.nanmax(np.abs(mm)))
    AXW, AXH, CBW, CBPAD, GAP, L0, B0 = 2.0, 1.9, 0.12, 0.10, 0.85, 1.35, 0.95
    x_axd = L0
    x_cbd = x_axd + AXW + CBPAD
    x_axm = x_cbd + CBW + GAP
    x_cbm = x_axm + AXW + CBPAD
    fw = x_cbm + CBW + 0.25
    fh = B0 + AXH + 0.6
    fig = plt.figure(figsize=(fw, fh))

    axd = fig.add_axes([x_axd / fw, B0 / fh, AXW / fw, AXH / fh])
    imd = draw(axd, dm, dp, dtitle, vmax_d)
    axd.set_yticks(range(len(ROWS))); axd.set_yticklabels([r[2] for r in ROWS])
    cbd = fig.colorbar(imd, cax=fig.add_axes([x_cbd / fw, B0 / fh, CBW / fw, AXH / fh]))
    cbd.set_label('$\\rho$ (data scale)')

    axm = fig.add_axes([x_axm / fw, B0 / fh, AXW / fw, AXH / fh])
    imm = draw(axm, mm, mp, 'MODEL (n=15)', vmax_m)
    axm.set_yticks(range(len(ROWS))); axm.set_yticklabels([])
    cbm = fig.colorbar(imm, cax=fig.add_axes([x_cbm / fw, B0 / fh, CBW / fw, AXH / fh]))
    cbm.set_label('$\\rho$ (model scale)')

    fig.suptitle('HI vs behavior x eligibility form  (each panel normalized independently)',
                 y=0.99, fontsize=8)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, fname + '.' + ext), dpi=200, bbox_inches='tight')


mm, mp = model_mat()
print("MODEL matrix (rows=behavior, cols=elig):\n", np.round(mm, 2))

if data is None:
    print("\nDATA npy not found yet:\n  ", DATA_NPY,
          "\n  -> run sliding_window_four_elig.py (through the save cell) first.")
else:
    # version 1: epochs averaged
    dm_avg, dp_avg = data_mat('avg')
    make_fig(dm_avg, dp_avg, mm, mp, 'talk_fig_4x4_data_vs_model_epochavg',
             'DATA (n=44, epochs averaged)')

    # Pre epoch only
    dm_pre, dp_pre = data_mat(0)
    make_fig(dm_pre, dp_pre, mm, mp, 'talk_fig_4x4_data_vs_model_pre',
             'DATA (n=44, Pre epoch)')
    print("\nDATA matrix (Pre epoch):\n", np.round(dm_pre, 2))

    # version 2: best-matching epoch
    sims = [similarity(data_mat(ei)[0], mm) for ei in range(len(EPOCH_NAMES))]
    best = int(np.nanargmax(sims))
    dm_be, dp_be = data_mat(best)
    make_fig(dm_be, dp_be, mm, mp, 'talk_fig_4x4_data_vs_model_bestepoch',
             'DATA (n=44, {} epoch)'.format(EPOCH_NAMES[best]))

    print("\nEpoch-vs-model across-cell similarity (Pearson r):")
    for ei, nm in enumerate(EPOCH_NAMES):
        mark = '  <-- best' if ei == best else ''
        print("  {:8s} r={:+.3f}{}".format(nm, sims[ei], mark))
    print("\nDATA matrix (epochs averaged):\n", np.round(dm_avg, 2))
    print("\nDATA matrix ({} epoch):\n".format(EPOCH_NAMES[best]), np.round(dm_be, 2))

print("\nSaved to", OUT)
