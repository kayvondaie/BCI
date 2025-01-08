# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 08:17:08 2025

@author: kayvon.daie
"""

direct_cells = []
indirect_cells = []
for epoch_i in range(2):
    if epoch_i == 0:
        stimDist = data['photostim']['stimDist']
        favg_raw = data['photostim']['favg_raw']
    else:
        stimDist = data['photostim2']['stimDist']
        favg_raw = data['photostim2']['favg_raw']
    favg = favg_raw*0
    for i in range(favg.shape[1]):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])
    favg[4:6, :, :] = np.nan
    favg = np.apply_along_axis(lambda m: np.interp(np.arange(len(m)), np.where(~np.isnan(m))[0], m[~np.isnan(m)]), axis=0, arr=favg)
    amp = np.nanmean(favg[6:14, :, :], axis=0) - np.nanmean(favg[0:4, :, :], axis=0)
    
    N = favg.shape[2]
    N = 10
    # Create a figure and axes for a 5x5 grid
    fig, axes = plt.subplots(N,N, figsize=(10, 10))
    stim_cells = np.argmin(stimDist, axis=0)
    # Loop through each row and column to customize subplots
    direct = []
    indirect = []
    for gi in range(N):
        for ci in range(N):
            # Access the subplot at [i, j]
            ax = axes[gi, ci]
            if ci == gi:
                col = 'r'
                direct.append(favg[0:20,stim_cells[ci],gi])
            else:
                col = 'k'
                indirect.append(favg[0:20,stim_cells[ci],gi])
            ax.plot(favg[0:20,stim_cells[ci],gi],color = col)
            if ci != gi:
                ax.set_ylim(-.1,.1)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    direct = np.asarray(direct)
    indirect = np.asarray(indirect)
    direct_cells.append(np.nanmean(direct,axis=0))
    indirect_cells.append(np.nanmean(indirect,axis=0))

#%%
indirect = np.asarray(indirect_cells)
direct = np.asarray(direct_cells)
plt.subplot(211);
plt.plot(direct.T)
plt.subplot(212);
plt.plot(indirect.T)





