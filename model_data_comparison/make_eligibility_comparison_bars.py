#%% ============================================================================
# Equal-footing comparison of eligibility forms: identical MLR procedure for each
# (dW = sum_w beta_w * elig_w, 5-fold CV across pairs), on the double-centered dW
# (non-interacting means removed so raw can't win on that artifact).
# Two versions: with and without the 'true' eligibility.
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
MODEL_FORM = 'dpost_pre'

# display order; 'true' appended last so it is easy to drop
ORDER = ['hebb', 'dpost_pre', 'dpost_dpre', 'post_dpre', 'true']
LABEL = {'hebb': '$r_{pre}r_{post}$', 'dpost_pre': '$r_{pre}\\Delta r_{post}$\n(model rule)',
         'dpost_dpre': '$\\Delta r_{pre}\\Delta r_{post}$', 'post_dpre': '$\\Delta r_{pre}r_{post}$',
         'true': 'true'}


def vals(form):
    v = np.array([r[(form, 'Y_dc')] for r in res], float)
    return v[np.isfinite(v)]


def make(forms, fname, title):
    fig = plt.figure(figsize=(0.62 * len(forms) + 1.4, 3.0))
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([1.05 / fw, 0.95 / fh, (0.62 * len(forms) + 0.1) / fw, 1.85 / fh])
    mform = vals(MODEL_FORM)
    for i, f in enumerate(forms):
        v = vals(f)
        m, se = np.mean(v), np.std(v) / np.sqrt(len(v))
        col = '#1baf7a' if f == MODEL_FORM else '#9a9a95'
        ax.bar(i, m, 0.66, yerr=se, color=col, capsize=3, zorder=2)
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.13, 0.13, len(v)), v,
                   s=8, color='k', alpha=0.3, zorder=3)
        # paired difference vs the model form (is this form distinguishable from the winner?)
        if f != MODEL_FORM and len(v) == len(mform):
            d = mform - v
            p = wilcoxon(d)[1] if np.any(d != 0) else np.nan
            st = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'
            ax.text(i, m + se + 0.008, st, ha='center', va='bottom', fontsize=7.5, color='#555')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(forms))); ax.set_xticklabels([LABEL[f] for f in forms], fontsize=7)
    ax.set_ylabel('CV test $r$  ($\\Delta W$ centered)')
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, max(np.mean(vals(f)) for f in forms) * 1.28)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, fname + '.' + ext), dpi=200, bbox_inches='tight')


make(ORDER, 'talk_fig_elig_comparison_withtrue',
     'Eligibility comparison (equal footing)\nstars = diff vs model rule')
make([f for f in ORDER if f != 'true'], 'talk_fig_elig_comparison_notrue',
     'Eligibility comparison (equal footing)\nstars = diff vs model rule')

print("Equal-footing MLR test r (dW centered), mean +/- sem:")
for f in ORDER:
    v = vals(f)
    print("  {:12s} {:+.3f} +/- {:.3f}".format(f, np.mean(v), np.std(v) / np.sqrt(len(v))))
mf = vals(MODEL_FORM)
print("\nPaired diff (model rule - form), Wilcoxon p:")
for f in ORDER:
    if f == MODEL_FORM:
        continue
    d = mf - vals(f)
    print("  vs {:12s} d={:+.3f}  p={:.4f}".format(f, np.mean(d), wilcoxon(d)[1]))
print("\nSaved both versions to", OUT)
