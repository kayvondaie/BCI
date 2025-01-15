# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:11:41 2025

@author: kayvon.daie
"""
#%%
direct_cells = []
indirect_cells = []
AMP = []
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
for epoch_i in range(2):
    if epoch_i == 0:
        stimDist = data['photostim']['stimDist']*umPerPix 

        favg_raw = data['photostim']['favg_raw']
    else:
        stimDist = data['photostim2']['stimDist']*umPerPix 
        favg_raw = data['photostim2']['favg_raw']
    favg = favg_raw*0
    for i in range(favg.shape[1]):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])
    favg[18:23, :, :] = np.nan
    
    favg = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
        m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
    ),
    axis=0,
    arr=favg
    )

    
    
    amp = np.nanmean(favg[25:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
    AMP.append(amp)
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
                direct.append(favg[10:40,stim_cells[ci+offset],gi+offset])
            else:
                col = 'k'
                indirect.append(favg[10:40,stim_cells[ci+offset],gi+offset])
            ax.plot(favg[10:40,stim_cells[ci+offset],gi+offset],color = col)
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
stimDist = data['photostim']['stimDist']*umPerPix
plt.figure(figsize=(8,4))  # Set figure size to 10x10 inches
df = data['df_closedloop']
cc = np.corrcoef(df)
F = data['F']
ko = np.nanmean(F[120:200,:,0:10],axis=0)
k = np.nanmean(F[120:240,:,0:],axis=0)
cc = np.corrcoef(k)
cco = np.corrcoef(ko)
ei = 1;
X = []
Y = []
Yo = []
for gi in range(stimDist.shape[1]):
    cl = np.where((stimDist[:,gi]<30) & (AMP[ei][:,gi]> .3))[0]
    #plt.plot(favg[0:80,cl,gi])
    
    x = np.nanmean(cc[cl,:],axis=0)    
    nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<1000))
    y = AMP[1][nontarg,gi]
    yo = AMP[0][nontarg,gi]
    #plt.scatter(x[nontarg],amp[nontarg,gi])
    X.append(x[nontarg])
    Y.append(y)
    Yo.append(yo)

X = np.concatenate(X)
Y = np.concatenate(Y,axis=1)
Yo = np.concatenate(Yo,axis=1)
plt.subplot(221)
pf.mean_bin_plot(X,Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('Before learning')

plt.subplot(223)
pf.mean_bin_plot(X,Y,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('After learning')

plt.subplot(122)
pf.mean_bin_plot(X,Y-Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()
plt.title(data['mouse'] + ' ' + data['session'])
#%%



