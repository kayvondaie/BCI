#%% ============================================================================
# Eligibility-form similarity, corrected for baseline convention.
# The model's ~0.996 was largely a RUNNING-EMA baseline artifact. Under the data's
# fixed early-trial baseline the model forms separate (~0.79) but still less than
# the data (~0.16). Three panels: model(EMA) -> model(data-matched) -> data.
# Matrices typed from the diagnostic runs (seed 0 model; n=44 data).
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 8})
mpl.rcParams['svg.fonttype'] = 'none'
OUT = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\model_data_comparison'

LABEL = ['$r_{pre}r_{post}$', '$r_{pre}\\Delta r_{post}$',
         '$\\Delta r_{pre}\\Delta r_{post}$', '$\\Delta r_{pre}r_{post}$']

C_ema = np.array([[1.00, 0.341, 0.287, 0.341],
                  [0.341, 1.00, 0.996, 0.995],
                  [0.287, 0.996, 1.00, 0.996],
                  [0.341, 0.995, 0.996, 1.00]])
C_matched = np.array([[1.00, 0.757, 0.597, 0.757],
                      [0.757, 1.00, 0.831, 0.709],
                      [0.597, 0.831, 1.00, 0.831],
                      [0.757, 0.709, 0.831, 1.00]])
C_data = np.array([[1.00, 0.20, 0.51, 0.18],
                   [0.20, 1.00, 0.16, 0.06],
                   [0.51, 0.16, 1.00, 0.16],
                   [0.18, 0.06, 0.16, 1.00]])
PANELS = [(C_ema, 'MODEL\n(running-EMA baseline)'),
          (C_matched, 'MODEL\n(early-trial baseline,\ndata-matched)'),
          (C_data, 'DATA (n=44)\n(early-trial baseline)')]

AXW, AXH, GAP, L0, B0 = 1.7, 1.7, 0.5, 1.5, 1.55
fw = L0 + 3 * AXW + 2 * GAP + 0.85
fh = B0 + AXH + 0.75
fig = plt.figure(figsize=(fw, fh))
im = None
for p, (C, title) in enumerate(PANELS):
    ax = fig.add_axes([(L0 + p * (AXW + GAP)) / fw, B0 / fh, AXW / fw, AXH / fh])
    im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '{:.2f}'.format(C[i, j]), ha='center', va='center',
                    fontsize=7, color='white' if abs(C[i, j]) > 0.6 else 'k')
    ax.set_xticks(range(4)); ax.set_xticklabels(LABEL, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(LABEL if p == 0 else [''] * 4, fontsize=6)
    ax.set_title(title, fontsize=7.5)
cax = fig.add_axes([(fw - 0.7) / fw, B0 / fh, 0.1 / fw, AXH / fh])
cb = fig.colorbar(im, cax=cax); cb.set_label('Pearson r')
fig.suptitle('Eligibility-form similarity: the ~1.0 collapse is a running-EMA baseline artifact;\n'
             'data-matched, model forms still more correlated (0.79) than data (0.16)',
             y=1.0, fontsize=7.5)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_elig_correlation_3panel.' + ext), dpi=200, bbox_inches='tight')

print("deviation-form mean off-diagonal:")
for C, name in [(C_ema, 'model EMA'), (C_matched, 'model matched'), (C_data, 'data')]:
    print("  {:16s} {:.3f}".format(name, np.mean([C[1, 2], C[1, 3], C[2, 3]])))
print("Saved to", OUT)
