#%% ============================================================================
# Summary bar: the metric that shows the correct eligibility form is recovered.
# For each way of adding a deviation on top of the pure Hebbian coincidence
# (Dr_pre*Dr_post), how much unique CV test-r does it add to predicting the
# (double-centered, interaction-only) dW?  The model's rule deviates the POST
# factor -> r_pre*Dr_post.  If recovered, "deviate post" >> "deviate pre".
# Post-processes model_halfmean_decomp.npy (no retraining).
import os
import numpy as np
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

res = list(np.load(os.path.join(OUT, 'model_halfmean_decomp.npy'), allow_pickle=True))
base = np.array([r['interaction'] for r in res], float)
post = np.array([r['int+post_half'] for r in res], float) - base   # deviate post (model's rule)
pre = np.array([r['int+pre_half'] for r in res], float) - base     # deviate pre

vals = [post, pre]
labels = ['deviate post\n$r_{pre}\\Delta r_{post}$\n(model rule)', 'deviate pre\n$\\Delta r_{pre}r_{post}$']
colors = ['#1baf7a', '#888780']


def star(v):
    p = wilcoxon(v)[1] if np.any(v != 0) else np.nan
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


fig = plt.figure(figsize=(3.0, 3.0))
fw, fh = fig.get_size_inches()
ax = fig.add_axes([0.95 / fw, 0.75 / fh, 1.7 / fw, 1.95 / fh])
x = np.arange(2)
for i, (v, c) in enumerate(zip(vals, colors)):
    m, se = np.mean(v), np.std(v) / np.sqrt(len(v))
    ax.bar(i, m, 0.62, yerr=se, color=c, capsize=3, zorder=2)
    ax.scatter(np.full(len(v), i) + np.random.uniform(-0.12, 0.12, len(v)), v,
               s=10, color='k', alpha=0.35, zorder=3)
    ax.text(i, m + se + 0.006, star(v), ha='center', va='bottom', fontsize=9)
# paired seed lines to show consistency
for j in range(len(post)):
    ax.plot([0, 1], [post[j], pre[j]], color='0.6', lw=0.5, alpha=0.35, zorder=1)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('unique CV test $r$ added over\nHebbian coincidence floor')
ax.set_title('Recovered eligibility\ndeviates the POST factor', fontsize=8)
ax.set_ylim(-0.01, max(np.max(post), np.max(pre)) * 1.18)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_recovery_summary.' + ext), dpi=200, bbox_inches='tight')

print("deviate post: {:+.3f} +/- {:.3f}  {}".format(np.mean(post), np.std(post) / np.sqrt(len(post)), star(post)))
print("deviate pre : {:+.3f} +/- {:.3f}  {}".format(np.mean(pre), np.std(pre) / np.sqrt(len(pre)), star(pre)))
print("post > pre in {}/{} seeds".format(int(np.sum(post > pre)), len(post)))
print("Saved talk_fig_recovery_summary to", OUT)
