# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:41:54 2024

@author: Kayvon Daie
"""
import data_dict_create_module as ddc
import numpy as np

ops = np.load(folder + r'suite2p_photostim2/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
Ftrace = np.load(folder +r'/suite2p_photostim2/plane0/F.npy', allow_pickle=True)
trip = np.std(Ftrace,axis=0)
trip = np.where(trip<20)[0]

extended_trip = np.concatenate((trip, trip + 1))
trip = np.unique(extended_trip)
trip[trip>Ftrace.shape[1]-1] = Ftrace.shape[1]-1

Ftrace[:,trip] = np.nan
stat = np.load(folder + r'/suite2p_photostim2/plane0/stat.npy', allow_pickle=True)
siHeader = np.load('//allen/aind/scratch/BCI/2p-raw/BCI79/042224/suite2p_photostim2/plane0/siHeader.npy', allow_pickle=True).tolist()
data = dict()
data['photostim'] = dict()
data['photostim']['Fstim'], data['photostim']['seq'], data['photostim']['favg'], data['photostim']['stimDist'], data['photostim']['stimPosition'], data['photostim']['centroidX'], data['photostim']['centroidY'], data['photostim']['slmDist'],data['photostim']['stimID'] = ddc.stimDist_single_cell(ops, Ftrace,siHeader,stat)
#%%
stimDist = data['photostim']['stimDist']
seq = data['photostim']['seq']
ci = 0;
dists = stimDist[ci,seq-1]
ind = np.where(dists < 20)
#ind = np.where(seq==ci+1)
stim_trace_for_group = np.zeros((1,Ftrace.shape[1]))[0]
stim_trace_for_group[ind[0]] = 1

plt.plot(-stim_trace_for_group*10,'r')
plt.plot(Ftrace[ci,:],'k')
plt.xlim((000,1270))
plt.show()

 

#%%
out = ddc.read_stim_file(folder,r'suite2p_photostim2/')
beam = -out['Beam']
stimOn = np.where(np.diff(beam)>.01)[0]

stimOn = stimOn[np.diff(stimOn, prepend=np.nan) >= 200]

#%%
from scipy import stats
import plotting_functions as pf

seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli'];
list_nums = seq.strip('[]').split();
seq = [int(num) for num in list_nums]
seq = np.asarray(seq)
seq = np.tile(seq,(1,10))[0]
seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
seq = seq[seqPos:Ftrace.shape[1]]



bl = np.std(Ftrace.T, axis=0)
bl = np.percentile(Ftrace,20,axis = 1)
bl = np.tile(bl, (Ftrace.T.shape[0], 1))
dff = (Ftrace.T - bl) / bl
dff = Ftrace.T
dff = dff.T
pre = 20
post = 80
Fstim = np.zeros((pre+post,Ftrace.shape[0],max(seq)))
P = []
num = np.zeros(max(seq),)
A = []
for ci in range(max(seq)):
    ind = np.where(seq==ci+1)[0]
    
    a = np.zeros((pre+post,Ftrace.shape[0],len(ind)))
    for ti in range(len(ind)):
        indd = np.arange(ind[ti]-pre,ind[ti]+post)
        indd[indd<0] = 0
        indd[indd>=Ftrace.shape[1]] = Ftrace.shape[1]-1
    
        b = dff[:,indd].T
        bl = np.tile(np.mean(b[0:19], axis=0), (b.shape[0], 1))
        b = (b - bl)/bl
        #b = b - bl
        a[:,:,ti] = b
    A.append(a)
    amp = np.nanmean(a[21:30,:,:],axis = 0) - np.nanmean(a[0:19,:,:],axis = 0)                
    h,p = stats.ttest_1samp(amp.T, 0)  
    k = np.where(stimDist[:,ci]>30)[0]
    num[ci] = np.nansum(p[k]<.05)
    P.append(p)

    Fstim[:,:,ci] = np.nanmean(a,axis = 2)
amp = np.nanmean(Fstim[21:23,:,:],axis = 0) - np.nanmean(Fstim[18:19,:,:],axis = 0)
pf.mean_bin_plot(stimDist.flatten(),amp.flatten(),255,1,1,'k')
#plt.xlim(0,100)
#plt.ylim(-.05,.05)
#%%
dt = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
t = np.arange(0,100+dt,dt)
t = t[0:Fstim.shape[0]]
t = t - t[pre]
plt.rcParams.update({'font.size': 8})

ci = 2
#plt.subplot(10,5,ci+1)
amp = np.nanmean(Fstim[21:23,:,:],axis = 0) - np.nanmean(Fstim[17:19,:,:],axis = 0)
ind = np.where(stimDist[:,ci]>30)[0]    
b = np.argsort(amp[:,ci])
b = np.argsort(stimDist[:,ci])
p = np.asarray(P)
#b = np.argsort(p[:,ci])
f = Fstim[:,:,ci]
#plt.plot(np.nanmean(f[:,b[-1]],axis=1))
num = -1;
plt.plot(t,f[:,b[num]],'k')
plt.title(str(stimDist[b[num],ci]))
#plt.plot(Fstim[:,b[21],ci])

#%%
amp = np.nanmean(Fstim[21:22,:,:],axis = 0) - np.nanmean(Fstim[15:20,:,:],axis = 0)
cl, stm = np.where((amp > 0.25) & (stimDist > 25))

i = 23
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(4,5))  # Adjust width and height as needed
plt.subplot(223)
plt.plot(Fstim[10:40,cl[i],stm[i]],'k-',linewidth=.5)
plt.subplot(224)
plt.imshow(A[stm[i]][10:40,cl[i],:].T,vmin=0,vmax=3)
plt.subplot(211)
plt.plot(Fstim[10:40,stm[i],stm[i]],'k-',linewidth=.5)
print(amp[cl[i],stm[i]])
print(stimDist[cl[i],stm[i]])
print(stm[i])

















