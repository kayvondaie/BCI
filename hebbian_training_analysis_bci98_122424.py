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
    offset = 0
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
                direct.append(favg[0:20,stim_cells[ci+offset],gi+offset])
            else:
                col = 'k'
                indirect.append(favg[0:20,stim_cells[ci+offset],gi+offset])
            ax.plot(favg[0:20,stim_cells[ci+offset],gi+offset],color = col)
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
#%%

amp_by_trl = []
for gi in range(20):
    Fstim = data['photostim']['Fstim'];
    seq = data['photostim']['seq']-1
    
    Fstim2 = data['photostim2']['Fstim'];
    seq2 = data['photostim2']['seq']-1
    
    ind = np.where(seq == gi)[0]
    ind2 = np.where(seq2 == gi)[0]
    k = Fstim[:,:,ind]
    k2 = Fstim2[:,:,ind2]
    a=np.nanmean(k[6:10,stim_cells[gi],:],axis=0)
    a2=np.nanmean(k2[6:10,stim_cells[gi],:],axis=0)
    ind = np.where(a2>.5)[0]
    plt.plot(np.nanmean(k[:,stim_cells[gi],:],axis=1))
    plt.plot(np.nanmean(k2[:,stim_cells[gi],ind],axis=1))
    amp_by_trl.append(a)
    #plt.plot(amp_by_trl[gi])
    plt.show()
max_length = max(len(arr) for arr in amp_by_trl)
plt.show()

# Pad each array with NaNs to make them the same length
padded_amp_by_trl = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in amp_by_trl])
plt.plot(np.nanmean(padded_amp_by_trl,axis=1))

#%%
amp_by_trl = []
round1 = []
round2 = []
favg_raw = data['photostim']['favg_raw']
favg_new = favg_raw * 0
favg_new2 = favg_raw * 0
for gi in range(20):
    # Load data for Fstim and sequences
    Fstim = data['photostim']['Fstim']
    seq = data['photostim']['seq'] - 1
    
    Fstim2 = data['photostim2']['Fstim']
    seq2 = data['photostim2']['seq'] - 1
    
    # Get indices for the current group
    ind = np.where(seq == gi)[0]
    ind2 = np.where(seq2 == gi)[0]
    
    # Subset data based on the indices
    k = Fstim[:, :, ind]
    k2 = Fstim2[:, :, ind2]
    
    # Calculate the metrics for a and a2
    a = np.nanmean(k[6:10, stim_cells[gi], :], axis=0) - np.nanmean(k[0:4, stim_cells[gi], :], axis=0)
    a2 = np.nanmean(k2[6:10, stim_cells[gi], :], axis=0) - np.nanmean(k2[0:4, stim_cells[gi], :], axis=0)
    
    # Generalized trial selection: remove the largest responses in `a`
   
    
    # Match the number of trials in `a` to those in `a2`
    ind_a = np.argsort(a)[:-6]  # Remove the largest responses from `a`
    ind_a2 = np.argsort(a2)[6:]
    
    # Plot the average responses
    round1.append(np.nanmean(k[0:25, stim_cells[gi], ind_a], axis=1))
    round2.append(np.nanmean(k2[0:25, stim_cells[gi], ind_a2], axis=1))
    favg_new[:,:,gi] = np.nanmean(k[:,:,ind_a],axis=2)
    favg_new2[:,:,gi] = np.nanmean(k2[:,:,ind_a2],axis=2)
    # plt.plot(np.nanmean(k[:, stim_cells[gi], ind_a], axis=1), label="a (down-sampled)")
    # plt.plot(np.nanmean(k2[:, stim_cells[gi], ind_a2], axis=1), label="a2")
    
    # plt.legend()
    # plt.title(f"Group {gi}")
    # plt.show()
    
    # Append the down-sampled data for further analysis
    amp_by_trl.append(a[ind_a])
favg_new[4:6, :, :] = np.nan
favg_new = np.apply_along_axis(
    lambda m: (
        np.interp(
            np.arange(len(m)),
            np.where(~np.isnan(m))[0],
            m[~np.isnan(m)]
        ) if np.any(~np.isnan(m)) else m  # Return m as-is if all values are NaN
    ),
    axis=0,
    arr=favg_new
)

favg_new2[4:6, :, :] = np.nan
favg_new2 = np.apply_along_axis(
    lambda m: (
        np.interp(
            np.arange(len(m)),
            np.where(~np.isnan(m))[0],
            m[~np.isnan(m)]
        ) if np.any(~np.isnan(m)) else m  # Return m as-is if all values are NaN
    ),
    axis=0,
    arr=favg_new2
)

#plt.plot(np.nanmean(np.asarray(round1),axis=0))
#plt.plot(np.nanmean(np.asarray(round2),axis=0))
direct = []
indirect = []
direct2 = []
indirect2 = []
offset=0
for gi in range(N):
    for ci in range(N):
        # Access the subplot at [i, j]
        ax = axes[gi, ci]
        if ci == gi:
            col = 'r'
            direct.append(favg_new[0:25,stim_cells[ci+offset],gi+offset])
            direct2.append(favg_new2[0:25,stim_cells[ci+offset],gi+offset])
        else:
            col = 'k'
            indirect.append(favg_new[0:25,stim_cells[ci+offset],gi+offset])
            indirect2.append(favg_new2[0:25,stim_cells[ci+offset],gi+offset])
        ax.plot(favg[0:25,stim_cells[ci+offset],gi+offset],color = col)
        if ci != gi:
            ax.set_ylim(-.1,.1)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
direct = np.asarray(direct)
indirect = np.asarray(indirect)
plt.subplot(211)
plt.plot(np.nanmean(direct,axis=0))
plt.plot(np.nanmean(direct2,axis=0))
plt.subplot(212)
plt.plot(np.nanmean(indirect,axis=0))
plt.plot(np.nanmean(indirect2,axis=0))
#%%
fig = plt.figure(figsize=(10, 5))
amp = np.nanmean(favg_new[6:14, :, :], axis=0) - np.nanmean(favg_new[0:4, :, :], axis=0)
amp2 = np.nanmean(favg_new2[6:14, :, :], axis=0) - np.nanmean(favg_new2[0:4, :, :], axis=0)
scl = .15
offset = 0
w1 = amp[stim_cells[0+offset:10+offset],0+offset:10+offset]
w2 = amp2[stim_cells[0+offset:10+offset],0+offset:10+offset]
plt.subplot(121)
plt.imshow(amp[stim_cells[0+offset:10+offset],0+offset:10+offset],vmin=-scl,vmax=scl,cmap='seismic', aspect='auto')
plt.xlabel('Pre')
plt.ylabel('Post')
plt.subplot(122)
plt.imshow(amp2[stim_cells[0+offset:10+offset],0+offset:10+offset],vmin=-scl,vmax=scl,cmap='seismic', aspect='auto')
plt.xlabel('Pre')
plt.tight_layout()

#%%
Dw = w2 - w1
plt.bar(['Upper Triangle', 'Lower Triangle'], [np.mean(Dw[np.triu_indices(10, k=1)]), np.mean(Dw[np.tril_indices(10, k=-1)])], color=['blue', 'red'])
