# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:55:28 2024

@author: kayvon.daie
"""
folders = []
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI93/091124/')
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI88/081924/')
folder = folders[1]
data = ddc.load_data_dict(folder)
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
#%%
favg_raw = data['photostim']['favg_raw']
favg = np.zeros(favg_raw.shape)
stimDist = data['photostim']['stimDist']
bl = np.percentile(Ftrace,50,axis=1)
N = stimDist.shape[0]
for i in range(N):
    favg[:,i] = (favg_raw[:,i] - np.nanmean(favg_raw[0:3,i]))

amp = np.nanmean(favg[11:15,:,:],axis = 0)-np.nanmean(favg[0:4,:,:],axis = 0);

wcc = np.zeros((N,N));

for i in range(N):
    for j in range(N):
        ind = np.where((stimDist[j,:]<30) & (stimDist[i,:]>30))[0]
        if len(ind) > 0:
            wcc[j,i] = np.nanmean(amp[i,ind])

ind = np.where(data['iscell'][:,0]==1)[0]
plt.imshow(wcc[np.ix_(ind, ind)], vmin=-.5, vmax=.5, cmap='seismic')
plt.xlabel('Post-synaptic')
plt.ylabel('Pre-synaptic')
plt.title(folder)
plt.colorbar()
#%%
