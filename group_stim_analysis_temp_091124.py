# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:29:04 2024

@author: kayvon.daie
"""
siHeader = np.load(folder + r'/suite2p_photostim/plane0/siHeader.npy', allow_pickle=True).tolist()
#%%
favg = data['photostim']['favg_raw']
# for i in range(favg.shape[1]):
#      bl = np.nanmean(favg[0:4,i,:])
#      favg[:,i,:] = (favg[:,i,:]-bl)/bl
stimDist = data['photostim']['stimDist']
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
t = np.arange(0, dt_si * (favg.shape[0]), dt_si)
cl = 2
ind = np.where(stimDist[cl,:]<10)
plt.plot(t,np.nanmean(favg[:,cl,ind],axis=2))
ind = np.where((stimDist[cl,:]>5)&(stimDist[2,:]<10))
#plt.plot(t,np.nanmean(favg[:,cl,ind],axis=2)*2)
#ind = np.where((stimDist[cl,:]>10)&(stimDist[2,:]<15))
#plt.plot(t,np.nanmean(favg[:,cl,ind],axis=2)*9)
ind = np.where((stimDist[cl,:]>20)&(stimDist[2,:]<3000))
plt.plot(t,np.nanmean(favg[:,cl,ind],axis=2)*7)
plt.xlabel('Time (s)')
plt.tight_layout()
#%%
amp = np.nanmean(favg[9:11,:,:],axis = 0)-np.nanmean(favg[0:4,:,:],axis = 0);
#plt.plot(stimDist[cl,:],amp[cl,:],'.')
plt.plot(stimDist.flatten(),amp.flatten(),'k.',markersize=.5)
#%%
N = favg.shape[1]
W = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        ind = np.where((stimDist[i,:]<5)&(stimDist[j,:]>15))
        W[i,j] = np.nanmean(favg[9:11,j,ind])
        


