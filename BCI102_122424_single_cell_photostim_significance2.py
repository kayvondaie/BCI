# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:37:52 2025

@author: kayvon.daie
"""
siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist() 
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
favg_raw = data['photostim']['favg_raw']
favg = np.zeros(favg_raw.shape)
umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
iscell = np.load('//allen/aind/scratch/BCI/2p-raw/BCI102/122424/pophys/suite2p_spont/plane0/iscell.npy',allow_pickle=True).tolist()
stimDist = data['photostim']['stimDist']*umPerPix
#bl = np.percentile(Ftrace, 50, axis=1)
N = stimDist.shape[0]

 # Process photostimulation data
for i in range(N):
    favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[10:18, i]))/np.nanmean(favg_raw[10:18, i])

siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist() 
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
favg[19:23, :, :] = np.nan

favg = np.apply_along_axis(
lambda m: np.interp(
    np.arange(len(m)),
    np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
    m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
),
axis=0,
arr=favg
)

cells = np.where(np.asarray(iscell)[:,0]==1)[0]
cells = np.where(np.asarray(iscell)[:,1]>.1)[0]
stimDist = stimDist[cells,:]
favg = favg[:,cells,:]

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

Fstim = data['photostim']['Fstim'][:,cells,:];
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
pv = np.zeros((Fstim.shape[1],favg.shape[2]))

for gi in range(G):    
    seq = data['photostim']['seq']-1
    ind = np.where(seq == gi)[0][::]
    post = np.nanmean(Fstim[24:30, :, ind], axis=0) 
    pre  = np.nanmean(Fstim[10:19, :, ind], axis=0)
    t_stat, p_value = ttest_ind(post, pre, axis=1)
    pv[:,gi] = p_value
#%%
frac_e = np.zeros((len(bins)-1,))
frac_i = np.zeros((len(bins)-1,))
for i in range(len(bins)-1):
    ind = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]))[0]
    inde = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]) & (pv.flatten() < 0.05) & (amp.flatten()>0))[0]    
    indi = np.where((stimDist.flatten() > bins[i]) & (stimDist.flatten()<bins[i+1]) & (pv.flatten() < 0.05) & (amp.flatten()<0))[0]    
     
    num_connect[i] = len(np.where((pv.flatten()[ind1] < 0.05) & (amp.flatten()[ind1]>0))[0])          
    frac_i[i] = len(indi)/len(ind)
    frac_e[i] = len(inde)/len(ind)
    
plt.bar(bins[0:2],frac_e[0:2],color=[.7,.7,.7],width=9)
plt.bar(bins[2:-1],frac_e[2:],color='k',width=9)
plt.bar(bins[0:-1],-frac_i,width=9,color='w',edgecolor='k')
plt.xlabel('Distance from photostim. target (um)')
plt.ylabel('Fraction significant')
#%%


stimCells = np.argmin(stimDist,axis=0)
gi = 47;
ci = 0;
ind = np.where((pv[:,gi]<.01) & (stimDist[:,gi]>30) & (amp[:,gi]>0))[0]
a = np.argsort(-amp[ind,gi])
a = np.argsort(pv[ind,gi])
ind = ind[a]
plt.figure(figsize=(8,6))
#pf.show_rois(ops,stat,[stimCells[gi],ind[ci]])

plt.subplot(256)
plt.plot(t[10:60],favg[10:60,stimCells[gi],gi],'r')
for ci in range(4):
    plt.subplot(2,5,ci+2+5)
    plt.plot(t[10:60],favg[10:60,ind[ci],gi]-np.nanmean(favg[10:18,ind[ci],gi]),'b')
    plt.title(str(round(stimDist[ind[ci],gi])) + 'um')

ax = plt.subplot(2, 5, (1, 5))  # Merge columns 6 through 10
pf.show_rois_outline(ops,stat[cells],[stimCells[gi],ind[0],ind[1],ind[2],ind[3],ind[4]],[(1,0,0),(0,0,1),(0,0,1),(0,0,1),(0,0,1)],ax)

plt.tight_layout()
plt.show()

#%%
