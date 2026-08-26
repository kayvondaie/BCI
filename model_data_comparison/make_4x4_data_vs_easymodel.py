#%% ============================================================================
# Side-by-side 4x4: DATA (epochs averaged) vs EASIER-TASK MODEL (thr=0.5), whose
# hit rate now matches the data (~0.88). Each panel normalized independently.
# Saved as a NEW figure (does not touch the baseline-model side-by-sides).
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
MODEL_NPY = os.path.join(OUT, 'model_behavior_elig_matrix_easy.npy')   # <-- easier-task model

COLS = [('dot_prod_lag', 'hebb', '$r_{pre}r_{post}$'),
        ('dev2_lag', 'dpost_pre', '$r_{pre}\\Delta r_{post}$'),
        ('dpost_dpre_lag', 'dpost_dpre', '$\\Delta r_{pre}\\Delta r_{post}$'),
        ('post_dpre_lag', 'post_dpre', '$\\Delta r_{pre}r_{post}$')]
ROWS = [('win_hit', 'hits', 'Hit rate', 1),
        ('win_rpe', 'speed_rpe', '$\\Delta$Speed (RPE)', 1),
        ('win_rt', 'speed', 'Speed', -1),
        ('win_hit_rpe', 'hits_rpe', 'Hit $\\times$ RPE', 1)]

model = list(np.load(MODEL_NPY, allow_pickle=True))
data = np.load(DATA_NPY, allow_pickle=True).item()


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


def data_mat_avg():
    m = np.full((len(ROWS), len(COLS)), np.nan)
    p = np.full((len(ROWS), len(COLS)), np.nan)
    for ri, (dbeh, _, _, dsign) in enumerate(ROWS):
        for ci, (dmode, _, _) in enumerate(COLS):
            vv = []
            for s in data[dmode]:
                hi = np.nanmean(s['hi_with_int'], axis=1)
                bv = dsign * s[dbeh]
                ok = np.isfinite(hi) & np.isfinite(bv)
                if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                    vv.append(spearmanr(hi[ok], bv[ok])[0])
            vv = np.array(vv)
            if len(vv):
                m[ri, ci] = np.mean(vv)
                p[ri, ci] = wilcoxon(vv)[1] if len(vv) >= 2 and np.any(vv != 0) else np.nan
    return m, p


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


dm, dp = data_mat_avg()
mm, mp = model_mat()
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
imd = draw(axd, dm, dp, 'DATA (n=44, epochs averaged)', vmax_d)
axd.set_yticks(range(len(ROWS))); axd.set_yticklabels([r[2] for r in ROWS])
cbd = fig.colorbar(imd, cax=fig.add_axes([x_cbd / fw, B0 / fh, CBW / fw, AXH / fh]))
cbd.set_label('$\\rho$ (data scale)')

axm = fig.add_axes([x_axm / fw, B0 / fh, AXW / fw, AXH / fh])
imm = draw(axm, mm, mp, 'MODEL (easier task, thr=0.5, n=15)', vmax_m)
axm.set_yticks(range(len(ROWS))); axm.set_yticklabels([])
cbm = fig.colorbar(imm, cax=fig.add_axes([x_cbm / fw, B0 / fh, CBW / fw, AXH / fh]))
cbm.set_label('$\\rho$ (model scale)')

fig.suptitle('HI vs behavior x eligibility  (data epoch-avg vs easier-task model; each panel normalized)',
             y=0.99, fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_4x4_data_vs_easymodel_epochavg.' + ext),
                dpi=200, bbox_inches='tight')

print("DATA (epochs avg):\n", np.round(dm, 2))
print("\nMODEL (easier task):\n", np.round(mm, 2))
print("\nSaved talk_fig_4x4_data_vs_easymodel_epochavg to", OUT)
