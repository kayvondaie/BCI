#%% ============================================================================
# Eligibility-form similarity: MODEL vs DATA, side by side.
# In the model the three deviation forms are ~identical (degenerate); in the data
# they are nearly orthogonal (distinct). Model 4x4 from model_elig_correlation.npy;
# data 4x4 typed from the data run (mean per-session corr, n=44).
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

C_model = np.load(os.path.join(OUT, 'model_elig_correlation.npy'), allow_pickle=True)
C_data = np.array([[1.00, 0.20, 0.51, 0.18],
                   [0.20, 1.00, 0.16, 0.06],
                   [0.51, 0.16, 1.00, 0.16],
                   [0.18, 0.06, 0.16, 1.00]])

AXW, AXH, GAP, L0, B0 = 1.85, 1.85, 1.05, 1.55, 1.35
fw = L0 + 2 * AXW + GAP + 0.85
fh = B0 + AXH + 0.6
fig = plt.figure(figsize=(fw, fh))


def panel(x_in, C, title, ylabels):
    ax = fig.add_axes([x_in / fw, B0 / fh, AXW / fw, AXH / fh])
    im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '{:.2f}'.format(C[i, j]), ha='center', va='center',
                    fontsize=7.5, color='white' if abs(C[i, j]) > 0.6 else 'k')
    ax.set_xticks(range(4)); ax.set_xticklabels(LABEL, fontsize=6.5, rotation=40, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(LABEL if ylabels else [''] * 4, fontsize=6.5)
    ax.set_title(title, fontsize=8)
    return im


panel(L0, C_model, 'MODEL (1 run)', True)
im = panel(L0 + AXW + GAP, C_data, 'DATA (n=44)', False)
cax = fig.add_axes([(fw - 0.7) / fw, B0 / fh, 0.12 / fw, AXH / fh])
cb = fig.colorbar(im, cax=cax); cb.set_label('Pearson r')
fig.suptitle('Eligibility-form similarity: deviation forms are degenerate in the model, distinct in the data',
             y=0.99, fontsize=8)
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, 'talk_fig_elig_correlation_sidebyside.' + ext), dpi=200, bbox_inches='tight')

print("MODEL:\n", np.round(C_model, 2))
print("DATA:\n", np.round(C_data, 2))
print("\nDeviation-form off-diagonals (r_pre.dr_post, dr_pre.dr_post, dr_pre.r_post):")
print("  model: {:.2f} {:.2f} {:.2f}".format(C_model[1, 2], C_model[1, 3], C_model[2, 3]))
print("  data : {:.2f} {:.2f} {:.2f}".format(C_data[1, 2], C_data[1, 3], C_data[2, 3]))
print("\nSaved to", OUT)
