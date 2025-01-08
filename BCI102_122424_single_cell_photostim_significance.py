# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:37:52 2025

@author: kayvon.daie
"""

favg_raw = data['photostim']['favg_raw']
favg = np.zeros(favg_raw.shape)
umPerPix = 1200/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
stimDist = data['photostim']['stimDist']*umPerPix
#bl = np.percentile(Ftrace, 50, axis=1)
N = stimDist.shape[0]
 
 # Process photostimulation data
for i in range(N):
    favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[10:18, i]))/np.nanmean(favg_raw[10:18, i])

siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist() 
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
favg[19:23, :, :] = np.nan
#%%
t = np.arange(0, dt_si * (favg.shape[0]), dt_si)
t = t - t[23]
amp = np.nanmean(favg[24:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
gi = 15
plt.subplot(211)
plt.scatter(stimDist[:,gi],amp[:,gi])
plt.subplot(212)
ind = np.where((stimDist[:,gi]>30) & (stimDist[:,gi]<55) & (amp[:,gi]>.2))[0]
plt.plot(t[10:50],np.nanmean(favg[10:50,ind,gi],axis=1));
#plt.plot(favg[10:40,ind,gi]);

plt.title(str(len(ind)) + ' cells  ' + str(round(np.nanmean(stimDist[ind,gi]))) + ' um')
plt.tight_layout()

#%%
from scipy.stats import ttest_ind


#plt.scatter(stimDist[:,gi],-np.log(p_value))
bins = [30,100]
bins = np.linspace(0,200,10)
bins = np.arange(0,300,10)
bins = np.concatenate((np.arange(0, 100, 10), np.arange(100, 300, 25)))
G = 50
num_connect = np.zeros((len(bins)-1,G))
frac_e = np.zeros((len(bins)-1,G))
frac_i = np.zeros((len(bins)-1,G))
frac_connect = np.zeros((len(bins)-1,G))
for gi in range(G):
    Fstim = data['photostim']['Fstim'];
    seq = data['photostim']['seq']-1
    ind = np.where(seq == gi)[0]
    post = np.nanmean(Fstim[24:30, :, ind], axis=0) 
    pre  = np.nanmean(Fstim[10:19, :, ind], axis=0)
    t_stat, p_value = ttest_ind(post, pre, axis=1)
    for i in range(len(bins)-1):
        ind1 = np.where((stimDist[:,gi] > bins[i]) & (stimDist[:,gi]<bins[i+1]))[0]    
        
        num_connect[i] = len(np.where((p_value[ind1] < 0.05) & (amp[ind1,gi]>0))[0]) 
        #num_connect[i] = np.sum((p_value[ind1] < 0.05)) - (0.05*len(ind1))
        frac_connect[i] = np.nanmean(p_value[ind1] < 0.05)
        
        ind = np.where((stimDist[:,gi] > bins[i]) & (stimDist[:,gi]<bins[i+1]) & (p_value < 0.05) & (amp[:,gi]>0))[0]  
        frac_e[i,gi] = len(ind)/len(ind1)
        ind = np.where((stimDist[:,gi] > bins[i]) & (stimDist[:,gi]<bins[i+1]) & (p_value < 0.05) & (amp[:,gi]<0))[0]  
        frac_i[i,gi] = len(ind)/len(ind1)
        #plt.title(str(len(ind)) + '/' + str(len(ind1)*.001) + ' cells  ' + str(round(np.nanmean(stimDist[ind,gi]))) + ' um')
        #plt.plot(t[10:50],np.nanmean(favg[10:50,ind,gi],axis=1));
        #plt.show()
    
    
    #plt.plot(bins[0:-1],frac_e,'b.-')
    #plt.plot(bins[0:-1],frac_i,'r.-')
    #plt.plot([0,150],[.05,.05],'k:')

plt.bar(bins[0:2],np.nanmean(frac_e[0:2,:],axis=1),color=[.7,.7,.7],width=9)
plt.bar(bins[2:-1],np.nanmean(frac_e[2:,:],axis=1),color='k',width=9)
plt.bar(bins[0:-1],-np.nanmean(frac_i,axis=1),width=9,color='w',edgecolor='k')
plt.xlabel('Distance from photostim. target (um)')
plt.ylabel('Fraction significant')
#plt.plot([0,150],[.00,.00],'k:')