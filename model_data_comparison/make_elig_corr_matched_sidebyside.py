#%% ============================================================================
# Eligibility-form similarity, MODEL vs DATA, under the SAME (data) convention:
# per-trial activity, first-20-trial baseline, summed over trials.
# Model 4x4 from model_elig_corr_datamatched.npy['first20']; data 4x4 typed from
# the data run (mean per-session corr, n=44). This replaces the earlier
# apples-to-oranges figure (which used the model's running-EMA convention).
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
C_model = np.load(os.path.join(OUT, 'model_elig_corr_datamatched.npy'), allow_pickle=True).item()['first20']
C_data = np.array([[1.00, 0.20, 0.51, 0.18],
                   [0.20, 1.00, 0.16, 0.06],
                   [0.51, 0.16, 1.00, 0.16],
                   [0.18, 0.06, 0.16, 1.00]])

AXW, AXH, GAP, L0, B0 = 1.85, 1.85, 1.05, 1.55, 1.35
fw = L0 + 2 * AXW + GAP + 0.85
fh = B0 + AXH + 0.55
fig = plt.figure(figsize=(fw, fh))


def panel(x_in, C, title, ylabels):
    ax = fig.add_axes([x_in / fw, B0 / fh, AXW / fw, AXH / fh])
    im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '{:.2f}'.format(C[i, j]), ha='center', va='center',
                    fontsize=7.5, color='white' if abs(C[i, j]) > 0.6 else 'k')
    ax.set_xticks(range(4)); ax.set_xticklabels(LABEL, fontsize=6.5, rotation=40, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(LABEL if ylabels else [''] * 4, fontsize=6.5)
    ax.set_title(title, fontsize=8)
    return im


panel(L0, C_model, 'MODEL (1 run)', True)
im = panel(L0 + AXW + GAP, C_data, 'DATA (n=44)', False)
cax = fig.add_axes([(fw - 0.7) / fw, B0 / fh, 0.12 / fw, AXH / fh])
cb = fig.colorbar(im, cax=cax); cb.set_label('Pearson r')
fig.suptitle('Eligibility-form similarity (same convention: per-trial, first-20 baseline)\n'
             'model deviation forms ~0.83 correlated; data ~0.13',
             y=1.0, fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_elig_corr_matched_sidebyside.' + ext), dpi=200, bbox_inches='tight')
print("model deviation-form mean off-diag = {:.3f}".format(np.mean([C_model[1, 2], C_model[1, 3], C_model[2, 3]])))
print("data  deviation-form mean off-diag = {:.3f}".format(np.mean([C_data[1, 2], C_data[1, 3], C_data[2, 3]])))
print("Saved to", OUT)
