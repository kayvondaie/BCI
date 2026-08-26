"""
Compute favg for BCI116 / 012726 using ALL cells (not just iscell==1).
Reuses create_photostim_Fstim from data_dict_create_module_bruker.
"""
import sys
sys.path.insert(0, r'C:\Users\kayvon.daie\Documents\claude_code\bonsai')

import numpy as np
import os
import data_dict_create_module_bruker as ddc

folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/012726/pophys/'

# Load all cells (do NOT filter by iscell) — same path pattern as ddc
iscell = np.load(folder + r'/suite2p_photostim_single/plane0/iscell.npy', allow_pickle=True)
stat = np.load(folder + r'/suite2p_photostim_single/plane0/stat.npy', allow_pickle=True)
Ftrace = np.load(folder + r'/suite2p_photostim_single/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_photostim_single/plane0/ops.npy', allow_pickle=True).tolist()
siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist()

cells = np.arange(iscell.shape[0])
Ftrace = Ftrace[cells, :]
stat = stat[cells]

print(f'Loaded {len(cells)} cells (all, no iscell filter)')
print(f'Ftrace shape: {Ftrace.shape}')

Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw, stim_params = \
    ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat, 0)
#%%
# Average response across far cells (>200 um from target) for each target group
import matplotlib.pyplot as plt

dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
t = np.arange(favg.shape[0]) * dt_si

# Per-cell baseline: mean of first 8 frames averaged across all trials, shape (cells,)
F0 = np.nanmean(np.nanmean(Fstim_raw[:8, :, :], axis=0), axis=-1)  # (cells,)

# Baseline-subtract each (cell, target) trace, then divide by F0
favg_raw_bs = favg_raw - np.nanmean(favg_raw[:8, :, :], axis=0, keepdims=True)
favg_dff = favg_raw_bs 

dim_thresh = np.nanpercentile(F0, 20)
dim_mask = F0 <= dim_thresh
not_cell_mask = iscell[:, 0] == 0

far_avg = np.full((favg.shape[0], favg.shape[2]), np.nan)
for gi in range(favg.shape[2]):
    far_cells = np.where((stimDist[:, gi] > 200) & dim_mask)[0]
    if len(far_cells) > 0:
        far_avg[:, gi] = np.nanmean(favg_dff[:, far_cells, gi], axis=1)

plt.figure(figsize=(6, 4))
plt.plot(t[:60], np.nanmean(far_avg[:60], axis=1), 'k', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Mean response (far cells, >200 um)')
plt.title(f'BCI116 012726 — {favg.shape[2]} target groups')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print(f'favg shape: {favg.shape}')
print(f'Fstim shape: {Fstim.shape}')

