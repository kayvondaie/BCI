# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:50:06 2025

@author: kayvon.daie
"""


mouse = 'BCI103'
session = '012225'
folder = folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
#%%
data = np.load(folder + 'data_main.npy',allow_pickle=True)
data['photostim'] = np.load(folder + 'data_photostim.npy',allow_pickle=True)
data['photostim2'] = np.load(folder + 'data_photostim2.npy',allow_pickle=True)
#%%
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
    favg[18:27, :, :] = np.nan
    
    favg = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
        m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
    ),
    axis=0,
    arr=favg
    )

    amp = np.nanmean(favg[26:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
    AMP.append(amp)
    plt.plot(np.nanmean(np.nanmean(favg[0:40,:,:],axis=2),axis=1))
#%%
F = data['F']
trl = F.shape[2]
tsta = np.arange(0,12,data['dt_si'])
tsta=tsta-tsta[int(2/dt_si)]
k = np.zeros((F.shape[1],trl))
for ti in range(trl):
    steps = data['step_time'][ti]
    indices = np.searchsorted(tsta, steps)
    indices = np.sort(np.concatenate((indices,indices-1,indices-2,indices-3)))
    indices = indices[indices<350]
    k[:,ti] = np.nanmean(F[indices,:,ti],axis=0)
k[np.isnan(k)==1]=0
ccn = np.corrcoef(k[:,17:])
cco = np.corrcoef(k[:,0:17])
cc = np.corrcoef(k[:,:])



trial_bins = np.arange(0,F.shape[2],10)
CC = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
for i in range(len(trial_bins)-1):
    ind = np.arange(trial_bins[i],trial_bins[i+1])
    CC[:,:,i] = np.corrcoef(k[:,ind])

import plotting_functions as pf


XX = []
for i in range(CC.shape[2]):
    X = []
    X2 = []
    Y = []
    Yo = []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:,gi]<15) & (AMP[0][:,gi]> .1) * ((AMP[1][:,gi]> .1)))[0]
        #plt.plot(favg[0:80,cl,gi])
        if len(cl)>0:
            x = np.nanmean(CC[cl,:,i],axis=0)
                
            nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<10000))
            y = AMP[1][nontarg,gi]
            yo = AMP[0][nontarg,gi]
            Y.append(y)
            Yo.append(yo)
            X.append(x[nontarg])
    
    
    
    X = np.concatenate(X)
    Y = np.concatenate(Y,axis=1)
    Yo = np.concatenate(Yo,axis=1)
    XX.append(X)

X = np.asarray(XX)
X[np.isnan(X)==1] = 0
Y[np.isnan(Y)==1] = 0
Yo[np.isnan(Yo)==1] = 0
X_T = X.T  # Shape: (82045, 13)
Y_T = Y.T.ravel() - Yo.T.ravel() # Shape: (82045,) — ravel to make it 1D

X_T = X.T  # (82045, 13)
Y_T = Y.T  # (82045, 1)

# Compute pseudoinverse solution
beta = np.linalg.pinv(X_T) @ Y_T  # (13, 1)

Y_pred = np.dot(beta.T,X_T.T)


pf.mean_bin_plot(Y_pred,Y,5,1,1,'k')
#%%
Yo = np.concatenate(Yo,axis=1)
plt.subplot(231)
pf.mean_bin_plot(X,Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('Before learning')

plt.subplot(234)
pf.mean_bin_plot(X,Y,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('After learning')

plt.subplot(132)
pf.mean_bin_plot(X,Y-Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()

