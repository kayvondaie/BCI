# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:47:14 2025
@author: kayvon.daie
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# === Load data ===
ops = np.load(os.path.join(folder, 'suite2p_spont/plane0/ops.npy'), allow_pickle=True).tolist()
stat = np.load(os.path.join(folder, 'suite2p_spont/plane0/stat.npy'), allow_pickle=True).tolist()
iscell = np.load(os.path.join(folder, 'suite2p_spont/plane0/iscell.npy'), allow_pickle=True)

cell_inds = np.where(iscell[:, 0] == 1)[0]

# === Time axis ===
dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])

df = data['spont']
T = np.arange(df.shape[1]) * dt_si    # time in seconds

# === Figure layout ===
fig, axes = plt.subplots(3, 1, figsize=(9, 9),
                         gridspec_kw={'height_ratios': [2, 3]},
                         sharex=False)
plt.subplots_adjust(hspace=0.25)

# ============================
# 1️⃣ Top: FOV image + ROI outlines + 100 µm scale bar
# ============================
ax = axes[0]
img = ops['meanImg']

ax.imshow(img, cmap='gray', vmin=0, vmax=np.percentile(img, 99))
for ci in cell_inds:
    ypix = stat[ci]['ypix']
    xpix = stat[ci]['xpix']
    ax.scatter(xpix, ypix, s=0.2, color='r')

# Scale bar (100 µm)
umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
barLength = 100 / umPerPix
y_pos = img.shape[0] - int(img.shape[0] * 0.05)
x_pos = img.shape[1] - int(img.shape[1] * 0.15)

ax.hlines(y=y_pos, xmin=x_pos, xmax=x_pos + barLength, colors='r', linewidth=3)
ax.text(x_pos + barLength / 2, y_pos - 10, '100 µm', color='r',
        ha='center', va='bottom', fontsize=10, weight='bold')
ax.axis('off')
ax.set_title(os.path.basename(folder))

# ============================
# 2️⃣ Bottom: stacked ΔF/F traces (first 10 cells)
# ============================
ax2 = axes[2]
n_cells = min(10, df.shape[0])
colors = plt.cm.turbo(np.linspace(0, 1, n_cells))[:, :3]

# Determine an offset that avoids overlap
offset = 1.5 * np.nanstd(df[:n_cells, :])

for i in range(n_cells):
    ax2.plot(T, df[i, :] + i * offset, color=colors[i], linewidth=1)
    ax2.text(T[-1] + 0.01*T[-1], df[i, -1] + i * offset, f'{i+1}', 
             va='center', ha='left', fontsize=8, color=colors[i])

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ΔF/F (stacked)')
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title('First 10 cells (ΔF/F, color-coded)')
ax2.margins(x=0)

plt.show()
