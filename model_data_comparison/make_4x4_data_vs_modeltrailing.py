#%% ============================================================================
# HI vs behavior x eligibility, DATA vs MODEL, both with the data convention
# (per-trial, trailing-20 baseline). Data (epochs-averaged) typed from the run;
# model recomputed in run_model_behavior_elig_trailing.py. Each panel normalized
# independently. Replaces the earlier data|model figure whose model panel used
# Kyle's running-EMA eligibility (the generative one, which inflates RPE tracking).
import os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

FORMS = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre']
FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
BEH = ['hits', 'speed_rpe', 'speed', 'hits_rpe']
BEH_LABEL = ['Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']

# DATA epochs-averaged (from the data run; Speed row uses -RT). No stars stored here.
D_data = np.array([[-0.01, 0.09, 0.01, 0.06],
                   [-0.04, 0.08, -0.04, 0.02],
                   [-0.00, 0.06, 0.01, 0.09],
                   [-0.03, 0.06, -0.02, 0.02]])
Dp_data = np.full_like(D_data, np.nan)

mres = list(np.load(os.path.join(OUT, 'model_behavior_elig_trailing.npy'), allow_pickle=True))
D_mod = np.full((4, 4), np.nan); Dp_mod = np.full((4, 4), np.nan)
for bi, b in enumerate(BEH):
    for fi, f in enumerate(FORMS):
        v = np.array([r[(f, b)] for r in mres], float); v = v[np.isfinite(v)]
        if len(v):
            D_mod[bi, fi] = np.mean(v)
            Dp_mod[bi, fi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan


def draw(ax, M, P, title, ylabels):
    vmax = max(0.05, np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    for bi in range(4):
        for fi in range(4):
            if np.isnan(M[bi, fi]):
                continue
            st = ''
            if not np.isnan(P[bi, fi]):
                p = P[bi, fi]
                st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
            ax.text(fi, bi, '{:+.2f}\n{}'.format(M[bi, fi], st), ha='center', va='center',
                    fontsize=7, fontweight='bold' if st else 'normal')
    ax.set_xticks(range(4)); ax.set_xticklabels(FORM_LABEL, fontsize=6.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(BEH_LABEL if ylabels else [''] * 4)
    ax.set_title(title, fontsize=8)
    return im


AXW, AXH, CBW, CBPAD, GAP, L0, B0 = 2.0, 1.9, 0.12, 0.1, 0.9, 1.4, 0.95
x_axd, x_cbd = L0, L0 + AXW + CBPAD
x_axm = x_cbd + CBW + GAP
x_cbm = x_axm + AXW + CBPAD
fw = x_cbm + CBW + 0.25
fh = B0 + AXH + 0.6
fig = plt.figure(figsize=(fw, fh))
axd = fig.add_axes([x_axd / fw, B0 / fh, AXW / fw, AXH / fh])
imd = draw(axd, D_data, Dp_data, 'DATA (n=44, epochs avg)', True)
fig.colorbar(imd, cax=fig.add_axes([x_cbd / fw, B0 / fh, CBW / fw, AXH / fh])).set_label('$\\rho$ (data)')
axm = fig.add_axes([x_axm / fw, B0 / fh, AXW / fw, AXH / fh])
imm = draw(axm, D_mod, Dp_mod, 'MODEL (n=15, trailing-20)', False)
fig.colorbar(imm, cax=fig.add_axes([x_cbm / fw, B0 / fh, CBW / fw, AXH / fh])).set_label('$\\rho$ (model)')
fig.suptitle('HI vs behavior x eligibility, data convention (trailing-20 baseline); each panel normalized',
             y=0.99, fontsize=7.5)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_4x4_data_vs_modeltrailing.' + ext), dpi=200, bbox_inches='tight')
print("MODEL trailing matrix:\n", np.round(D_mod, 2))
print("Saved to", OUT)
