# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:41:54 2024

@author: Kayvon Daie
"""
import suite2p
import os;os.chdir('G:/My Drive/Python Scripts/BCI_analysis/')
import re
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import copy
import shutil
import data_dict_create_module as ddc

folder = r'//allen/aind/scratch/BCI/2p-raw/BCI79/042324/'
ops = np.load(folder + r'suite2p_photostim3/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
Ftrace = np.load(folder +r'/suite2p_photostim3/plane0/F.npy', allow_pickle=True)
trip = np.std(Ftrace,axis=0)
trip = np.where(trip<20)[0]

extended_trip = np.concatenate((trip, trip + 1))
trip = np.unique(extended_trip)
trip[trip>Ftrace.shape[1]-1] = Ftrace.shape[1]-1

Ftrace[:,trip] = np.nan
stat = np.load(folder + r'/suite2p_photostim3/plane0/stat.npy', allow_pickle=True)
siHeader = np.load('//allen/aind/scratch/BCI/2p-raw/BCI79/042324/suite2p_photostim3/plane0/siHeader.npy', allow_pickle=True).tolist()
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
out = ddc.read_stim_file(folder,r'suite2p_photostim3/')
beam = -out['Beam']
stimOn = np.where(np.diff(beam)>.05)[0]
one = np.array([stimOn[0]])
stimOn = np.concatenate((one,stimOn[np.diff(stimOn, prepend=np.nan) >= 2000]))

dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
dt_stim = 1/float(siHeader['metadata']['hPhotostim']['monitoringSampleRate'])
t_stim = np.arange(0,dt_stim*len(beam),dt_stim)

s2pfolder = folder + '/suite2p_photostim3/plane0/'
ops = np.load(s2pfolder + 'ops.npy', allow_pickle = True)
ops = ops.tolist()
strt_time = []
for i in range(len(ops['tiff_list'])):
    file = folder + ops['tiff_list'][i]
    header=extract_scanimage_metadata.extract_scanimage_metadata(file)
    strt_time.append(float(header['description_first_frame']['frameTimestamps_sec']))
strt_time = np.asarray(strt_time)
numFrames = ops['frames_per_file']
t_si = []
for i in range(len(numFrames)):
    a = np.arange(0,dt_si*numFrames[i],dt_si)
    a = a + strt_time[i] - strt_time[0]
    t_si.append(a.T)
t_si = np.asarray(t_si)

stimTime = t_stim[stimOn]
frame_indices = np.searchsorted(t_si, stimTime, side='left')
#%%
from scipy import stats
import plotting_functions as pf

seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli'];
list_nums = seq.strip('[]').split();
seq = [int(num) for num in list_nums]
seq = np.asarray(seq)
seq = np.tile(seq,(1,10))[0]
seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
seq = seq[seqPos:len(frame_indices)]



bl = np.nanstd(Ftrace.T, axis=0)
#bl = np.percentile(Ftrace,20,axis = 1)
bl = np.tile(bl, (Ftrace.T.shape[0], 1))
dff = (Ftrace.T - bl) / bl
#dff = Ftrace.T
dff = dff.T
pre = 20
post = 80
Fstim = np.zeros((pre+post,Ftrace.shape[0],max(seq)))
P = []
num = np.zeros(max(seq),)
A = []
for ci in range(max(seq)):
    ind = np.where(seq==ci+1)[0]
    ind = frame_indices[ind]
    a = np.zeros((pre+post,Ftrace.shape[0],len(ind)))
    for ti in range(len(ind)):
        indd = np.arange(ind[ti]-pre,ind[ti]+post)
        indd[indd<0] = 0
        indd[indd>=Ftrace.shape[1]] = Ftrace.shape[1]-1
    
        b = dff[:,indd].T
        bl = np.tile(np.nanmean(b[0:19], axis=0), (b.shape[0], 1))
        #b = (b - bl)/bl
        b = b - bl
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

ci = 6
#plt.subplot(10,5,ci+1)
amp = np.nanmean(Fstim[21:23,:,:],axis = 0) - np.nanmean(Fstim[17:19,:,:],axis = 0)
ind = np.where(stimDist[:,ci]>30)[0]    
b = np.argsort(-amp[:,ci])
#b = np.argsort(stimDist[:,ci])
p = np.asarray(P)
#b = np.argsort(p[:,ci])
f = Fstim[:,:,ci]
plt.subplot(211)
plt.plot(f[:,b[0]],'k')
num = 23;
plt.subplot(212)
plt.plot(t,f[:,b[num]],'k')
plt.title(str(stimDist[b[num],ci]))
#plt.plot(Fstim[:,b[21],ci])

#%%
amp = np.nanmean(Fstim[21:30,:,:],axis = 0) - np.nanmean(Fstim[15:20,:,:],axis = 0)
cl, stm = np.where((amp > 0.5) & (stimDist > 25))
stim_cells = np.argmin(stimDist, axis=0)
i = 12
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(4,5))  # Adjust width and height as needed
plt.subplot(212)
plt.plot(t,Fstim[:,cl[i],stm[i]],'k-',linewidth=.5)
plt.title('connected cell')
#plt.subplot(224)
#plt.imshow(A[stm[i]][10:40,cl[i],:].T,vmin=0,vmax=3)

plt.subplot(211)
plt.plot(t,Fstim[:,stim_cells[stm[i]],stm[i]],'k-',linewidth=.5)
plt.title('Stimulated cell')
plt.tight_layout()
print(amp[cl[i],stm[i]])
print(stimDist[cl[i],stm[i]])
print(stm[i])

#%%
plt.plot(stimDist.flatten(),amp.flatten(),'k.',markersize=.4)
plt.ylim(-1,3)

#%%
plt.rcParams.update({'font.size': 8})
ind = np.arange(10,50)
inhibitory_cell = 36
string = r'roi #'+ str(inhibitory_cell)
for i in range(Fstim.shape[2]):
    
    plt.subplot(4,5,i+1)
    plt.plot(t[ind],Fstim[ind,inhibitory_cell,i],'k',linewidth = .8)
    if np.mod(i,5) != 0:
        plt.tick_params(axis='y', labelleft=False)    # y-axis
    plt.ylim(-1.7,2.7)
    ax = plt.gca()  # Get the current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if i == 15:
        plt.xlabel('Time from photostim. (s)')
        plt.ylabel('DF/F')
    if i == 2:
        plt.title(string)
    plt.tight_layout()
plt.show()

plt.plot(stimDist[inhibitory_cell,:],amp[inhibitory_cell,:],'.')
plt.xlabel('Distance from stim. cell')
plt.ylabel('DF/F')
plt.show()

ind = np.arange(0,600)
plt.plot(t_si[ind],Ftrace[inhibitory_cell,ind],'k',linewidth=.5)
plt.xlabel('Time (s)')

#%%
ft = Ftrace[:,np.isnan(Ftrace[0,:])==0]
cc = np.corrcoef(ft)
other_inh_cells = np.where(cc[inhibitory_cell,:]>.2)[0]
ind = np.arange(0,600)

b = np.argsort(-cc[inhibitory_cell,:])
for i in range(20):
    plt.plot(Ftrace[b[i],ind]-(300*i),'k',linewidth=.5)
    
#%%
i = 15
plt.subplot(211)
plt.imshow(A[i][10:40,stim_cells[i],:].T,vmin=0,vmax=3)
plt.subplot(212)
plt.plot(Fstim[:,stim_cells[i],i])

#%%
inhibitory_cell_array = np.array([inhibitory_cell])  
cls_to_plot = np.concatenate((stim_cells, inhibitory_cell_array))
cls_to_plot = np.concatenate((stim_cells, other_inh_cells[0:5]))
num = Fstim.shape[2]
num2 = len(cls_to_plot)
for i in range(num2):
    for j in range(num):
        plt.subplot2grid((num2,num),(i,j))
        plt.plot(Fstim[15:30,cls_to_plot[i],j],'k',linewidth=.4)
        #plt.gca().set_visible(False)
        plt.axis('off')
        if i == j:
            plt.ylim(-1,6)
        else:
            plt.ylim(-1.5,1.5)
        if i > num:
            plt.ylim(-1,5)

#%%

cell_ind = np.arange(Ftrace.shape[0])
B = []
for i in range(Fstim.shape[2]):
    ind = np.where((stimDist[:,i]>100) & (stimDist[:,i]<150))[0]
    B.append(np.nanmean(Fstim[:,ind,i],axis = 1))
B1 = np.asarray(B) 
plt.plot(t,np.nanmean(B,axis = 0)) 
plt.title('All cells 100-150 um from photostim')  
plt.xlabel('Time from photostim. (s)')
plt.show()

B = []
for i in range(Fstim.shape[2]):
    ind = np.where((stimDist[:,i]>30) & (stimDist[:,i]<80))[0]
    B.append(np.nanmean(Fstim[:,ind,i],axis = 1))
B2 = np.asarray(B) 
plt.plot(t,np.nanmean(B,axis = 0)) 
plt.title('All cells 30-80 um from photostim')  
plt.xlabel('Time from photostim. (s)')
plt.show()

B = []
for i in range(Fstim.shape[2]):
    ind = np.where((stimDist[:,i]>0) & (stimDist[:,i]<10))[0]
    B.append(np.nanmean(Fstim[:,ind,i],axis = 1))
B3 = np.asarray(B) 
plt.plot(t,np.nanmean(B,axis = 0)) 
plt.title('All cells 0-20 um from photostim')  
plt.xlabel('Time from photostim. (s)')
plt.show()

B = []
for i in range(Fstim.shape[2]):
    ind = np.where((stimDist[:,i]>300) & (stimDist[:,i]<40000) & (cell_ind<300)&(iscell[:,0]==1))[0]
    B.append(np.nanmean(Fstim[:,ind,i],axis = 1))
B3 = np.asarray(B) 
plt.plot(t,np.nanmean(B,axis = 0)) 
plt.title('All cells >300 um from photostim')  
plt.xlabel('Time from photostim. (s)')
plt.show()

plt.plot(t,np.nanmean(50*B1,axis = 0)) 
plt.plot(t,np.nanmean(10*B2,axis = 0)) 
plt.plot(t,np.nanmean(B3,axis = 0)) 
    
#%%
i = 14
plt.plot(Fstim[:,stim_cells[i],i])
plt.plot(Fstim[:,76,i])



#%% find inhibitory cells again

ci = 8;
a = np.nanmean(Fstim[37:40,:,ci],axis=0)-np.nanmean(Fstim[33:35,:,ci],axis=0)

fs = np.nanmean(Fstim,axis=2)
cc = np.corrcoef(fs.T)
b = np.argsort(-cc[:,76])
#plt.plot(Fstim[:,b[20],3]);
b = np.argsort(-a)
plt.plot(stimDist[b[31:],:].flatten(),amp[b[31:],:].flatten(),'k.',markersize=.5)
plt.plot(stimDist[b[0:30],:].flatten(),amp[b[0:30],:].flatten(),'ro',markersize=2,markerfacecolor='w')


