import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- Data ---
control_slopes = np.array([0.0033257098944879356, 0.005539725259982027,
                            0.006591869959161078, 0.011642493952169305,
                            0.003078741858589683, 0.003324817585913344,
                            0.0034852386629479933, 0.0021846757608199124])

ko_slopes = np.array([-0.002615613540260325, -0.0006896597412609817,
                      -0.0009372726183045325, 0.00013200382724731135,
                       0.007131773143919907, -0.00019404873976661247,
                       0.004636665165005843, 0.009258616651441119,
                       0.01154682747320794, 0.0013185022887969471])

ko_rg = np.array([1.2488356078141951, 0.42893818156602875, 0.9370774418508617,
                  4.284051306955634, 1.4575533208318316, 0.9707068865498043,
                  0.9707068865498043, 4.005872131056588,
                  7.200405131498857, 8.229683986553505])


# --- Grouping ---
thr = 2.0
ko_low_slopes = ko_slopes[ko_rg < thr]
ko_ctrl_slopes = ko_slopes[ko_rg > thr]
control_slopes = np.concatenate([control_slopes,ko_ctrl_slopes])
groups = [control_slopes, ko_low_slopes]
labels = ["Control", "Knockout"]

# --- Plot ---
fig, ax = plt.subplots(figsize=(4,4))

for i, vals in enumerate(groups):
    # jitter x slightly for clarity
    x = (i+1) + 0.1*np.random.randn(len(vals))
    ax.scatter(x, vals, color='k', s=60, alpha=0.8, zorder=3)
    ax.hlines(np.median(vals), i+0.8, i+1.2, color='red', lw=2)

ax.set_xticks([1,2])
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Slope")
ax.axhline(0, color='gray', linestyle='--', lw=0.8)

plt.tight_layout()
plt.show()

# --- Stats ---
stat, pval = mannwhitneyu(groups[0], groups[1], alternative='greater')
print(f"Mann–Whitney U test p = {pval:.3g}")


