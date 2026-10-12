#%% ============================================================================
# Same equal-footing metric (centered-dW MLR test r), but plotted RELATIVE to the
# pure Hebbian-coincidence floor (Dr_pre*Dr_post), which every deviation form
# shares. Subtracting that shared floor exposes the real (small but consistent)
# separation the absolute bars hid. Two versions: with / without 'true'.
# Post-processes model_mlr_centered.npy (key 'Y_dc'); no retraining.
import os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

res = list(np.load(os.path.join(OUT, 'model_mlr_centered.npy'), allow_pickle=True))
FLOOR = 'dpost_dpre'                       # pure interaction = shared floor
MODEL_FORM = 'dpost_pre'
LABEL = {'hebb': '$r_{pre}r_{post}$', 'dpost_pre': '$r_{pre}\\Delta r_{post}$\n(model rule)',
         'post_dpre': '$\\Delta r_{pre}r_{post}$', 'true': 'true'}


def rel(form):
    d = np.array([r[(form, 'Y_dc')] - r[(FLOOR, 'Y_dc')] for r in res], float)
    return d[np.isfinite(d)]


def star(d):
    p = wilcoxon(d)[1] if np.any(d != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


def make(forms, fname):
    fig = plt.figure(figsize=(0.72 * len(forms) + 1.5, 3.1))
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([1.15 / fw, 1.0 / fh, (0.72 * len(forms) + 0.15) / fw, 1.8 / fh])
    tops = [np.mean(rel(f)) + np.std(rel(f)) / np.sqrt(len(rel(f))) for f in forms]
    bots = [np.mean(rel(f)) - np.std(rel(f)) / np.sqrt(len(rel(f))) for f in forms]
    for i, f in enumerate(forms):
        d = rel(f); m, se = np.mean(d), np.std(d) / np.sqrt(len(d))
        col = '#1baf7a' if f == MODEL_FORM else ('#c77' if m < 0 else '#9a9a95')
        ax.bar(i, m, 0.66, yerr=se, color=col, capsize=3, zorder=2)
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.12, 0.12, len(d)), d,
                   s=8, color='k', alpha=0.3, zorder=3)
        va = 'bottom' if m >= 0 else 'top'
        off = se + 0.006 if m >= 0 else -(se + 0.006)
        ax.text(i, m + off, star(d), ha='center', va=va, fontsize=7.5, color='#555')
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xticks(range(len(forms))); ax.set_xticklabels([LABEL[f] for f in forms], fontsize=7)
    ax.set_ylabel('CV test $r$ relative to\npure Hebbian coincidence')
    ax.set_title('vs pure interaction $\\Delta r_{pre}\\Delta r_{post}$ (dashed 0)', fontsize=8)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylim(min(min(bots), 0) - 0.035, max(max(tops), 0) + 0.02)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, fname + '.' + ext), dpi=200, bbox_inches='tight')


make(['hebb', 'dpost_pre', 'post_dpre', 'true'], 'talk_fig_elig_vs_floor_withtrue')
make(['hebb', 'dpost_pre', 'post_dpre'], 'talk_fig_elig_vs_floor_notrue')

print("Test r RELATIVE to pure interaction floor (Dr_pre*Dr_post), mean +/- sem, Wilcoxon vs 0:")
for f in ['hebb', 'dpost_pre', 'post_dpre', 'true']:
    d = rel(f)
    print("  {:12s} {:+.3f} +/- {:.3f}  {}".format(f, np.mean(d), np.std(d) / np.sqrt(len(d)), star(d)))
print("\nSaved both versions to", OUT)
