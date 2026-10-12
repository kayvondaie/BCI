#%% ============================================================================
# Side-by-side normalized heatmaps: corr(HI, behavior) x eligibility form,
# DATA (per-form HI-vs-behavior, n=44, pre epoch, from sliding_window_four_elig)
# vs MODEL (n=15, from model_behavior_elig_matrix, EMA convention).
# Each panel normalized to its own scale so the *shape* is comparable.
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

FORM_LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
              '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']
BEH_LABEL = ['Hit rate', '$\\Delta$Speed (RPE)', 'Speed', 'Hit $\\times$ RPE']
DATA_MODES = ['dot_prod_lag', 'dev2_lag', 'dpost_dpre_lag', 'post_dpre_lag']
DATA_BEH = [('win_hit', 1), ('win_rpe', 1), ('win_rt', -1), ('win_hit_rpe', 1)]
MODEL_FORMS = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre']
MODEL_BEH = ['hits', 'speed_rpe', 'speed', 'hits_rpe']
EI_PRE = 0


def _dedup(sessions):
    seen, out = set(), []
    for s in sessions:
        k = (s.get('mouse'), s.get('session'))
        if k not in seen:
            seen.add(k); out.append(s)
    return out


# ---- data matrix ----
data = np.load(DATA_NPY, allow_pickle=True).item()
data = {m: _dedup(data[m]) for m in DATA_MODES}
n_data = len(data[DATA_MODES[0]])
D = np.full((4, 4), np.nan); Dp = np.full((4, 4), np.nan)
for bi, (bkey, bsign) in enumerate(DATA_BEH):
    for mi, m in enumerate(DATA_MODES):
        rs = []
        for s in data[m]:
            hi = s['hi_with_int'][:, EI_PRE]; bv = bsign * np.asarray(s[bkey], float)
            ok = np.isfinite(hi) & np.isfinite(bv)
            if ok.sum() >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
                rs.append(spearmanr(hi[ok], bv[ok])[0])
        rs = np.array(rs)
        D[bi, mi] = rs.mean()
        Dp[bi, mi] = wilcoxon(rs)[1] if len(rs) >= 2 and np.any(rs != 0) else np.nan

# ---- model matrix ----
model = list(np.load(MODEL_NPY, allow_pickle=True))
n_model = len(model)
Mmat = np.full((4, 4), np.nan); Mp = np.full((4, 4), np.nan)
for bi, b in enumerate(MODEL_BEH):
    for mi, f in enumerate(MODEL_FORMS):
        v = np.array([r[(f, b)] for r in model], float); v = v[np.isfinite(v)]
        Mmat[bi, mi] = v.mean()
        Mp[bi, mi] = wilcoxon(v)[1] if len(v) >= 2 and np.any(v != 0) else np.nan


def draw(ax, Mm, Pp, title, ylabels):
    vmax = max(0.05, np.nanmax(np.abs(Mm)))
    im = ax.imshow(Mm, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(4):
        for j in range(4):
            p = Pp[i, j]
            st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''
            ax.text(j, i, '{:+.2f}\n{}'.format(Mm[i, j], st), ha='center', va='center',
                    fontsize=7, fontweight='bold' if st else 'normal',
                    color='white' if abs(Mm[i, j]) > 0.6 * vmax else 'k')
    ax.set_xticks(range(4)); ax.set_xticklabels(FORM_LABEL, fontsize=6.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(BEH_LABEL if ylabels else [''] * 4)
    ax.set_title(title, fontsize=8)
    return im


AXW, AXH, CBW, CBPAD, GAP, L0, B0 = 2.0, 1.9, 0.12, 0.1, 0.95, 1.4, 0.95
x_axd, x_cbd = L0, L0 + AXW + CBPAD
x_axm = x_cbd + CBW + GAP
x_cbm = x_axm + AXW + CBPAD
fw = x_cbm + CBW + 0.25
fh = B0 + AXH + 0.6
fig = plt.figure(figsize=(fw, fh))
axd = fig.add_axes([x_axd / fw, B0 / fh, AXW / fw, AXH / fh])
imd = draw(axd, D, Dp, 'DATA (n={})'.format(n_data), True)
fig.colorbar(imd, cax=fig.add_axes([x_cbd / fw, B0 / fh, CBW / fw, AXH / fh])).set_label('$\\rho$ (data)')
axm = fig.add_axes([x_axm / fw, B0 / fh, AXW / fw, AXH / fh])
imm = draw(axm, Mmat, Mp, 'MODEL (n={})'.format(n_model), False)
fig.colorbar(imm, cax=fig.add_axes([x_cbm / fw, B0 / fh, CBW / fw, AXH / fh])).set_label('$\\rho$ (model)')
fig.suptitle('corr(HI, behavior) x eligibility form  (each panel normalized independently)', y=0.99, fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_form_behavior_sidebyside.' + ext), dpi=200, bbox_inches='tight')
print("DATA (n={}):\n".format(n_data), np.round(D, 2))
print("MODEL (n={}):\n".format(n_model), np.round(Mmat, 2))
print("Saved to", OUT)
