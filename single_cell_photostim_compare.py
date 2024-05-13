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

folders = []
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI79/043024/')
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI79/050124/')
for i in range(len(folders)):
    folder = folders[i]
    ops = np.load(folder + r'suite2p_photostim/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
    Ftrace = np.load(folder +r'/suite2p_photostim/plane0/F.npy', allow_pickle=True)
    trip = np.std(Ftrace,axis=0)
    trip = np.where(trip<20)[0]
    
    extended_trip = np.concatenate((trip, trip + 1))
    trip = np.unique(extended_trip)
    trip[trip>Ftrace.shape[1]-1] = Ftrace.shape[1]-1
    
    Ftrace[:,trip] = np.nan
    stat = np.load(folder + r'/suite2p_photostim/plane0/stat.npy', allow_pickle=True)
    siHeader = np.load(folder + r'/suite2p_photostim/plane0/siHeader.npy', allow_pickle=True).tolist()
    data = dict()
    data['photostim'] = dict()
    data['photostim']['Fstim'], data['photostim']['seq'], data['photostim']['favg'],data['photostim']['stimDist'], data['photostim']['stimPosition'],data['photostim']['centroidX'], data['photostim']['centroidY'],data['photostim']['slmDist'],data['photostim']['stimID'], data['photostim']['Fstim_raw'],data['photostim']['favg_raw'] = ddc.stimDist_single_cell(ops, Ftrace,siHeader,stat)
    
    (data['photostim']['Fstim'], data['photostim']['seq'], data['photostim']['favg'],
     data['photostim']['stimDist'], data['photostim']['stimPosition'],
     data['photostim']['centroidX'], data['photostim']['centroidY'],
     data['photostim']['slmDist'], data['photostim']['stimID'],
     data['photostim']['Fstim_raw'], data['photostim']['favg_raw']) = ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat)    
    if i == 1:
        new = data;
    else:
        old = data;
#%%
SD = []
AMP = []
for i in range(2):
    if i == 0:
        data = old
    else:
        data = new
    stimDist = data['photostim']['stimDist']
    bl = np.nanstd(Ftrace,axis = 1)
    f = data['photostim']['favg_raw']
    ff = f*0
    N = f.shape[1]
    for i in range(N):
        a = f[:,i,:]
        a = (a-bl[i])/bl[i]
        ff[:,i,:] = a
    amp = np.nanmean(ff[8:14,:,:],axis =0) - np.nanmean(ff[0:4,:,:],axis =0)
    plt.plot(stimDist.flatten(),amp.flatten(),'k.',markersize = .5)   
    plt.show()
    bins = []
    bins.append((0,20));bins.append((30,80));bins.append((80,200));bins.append((200,10000));  
    for i in range(len(bins)):
        plt.subplot(2,2,i+1)
        A = []
        for j in range(ff.shape[2]):
            ind = np.where((stimDist[:,j]>bins[i][0])&(stimDist[:,j]<bins[i][1]))[0]
            A.append(np.nanmean(ff[:,ind,j],axis=1))
        A = np.asarray(A)        
        plt.plot(np.nanmean(A[:,0:30],axis=0))
    SD.append(stimDist)
    AMP.append(amp)
#%%
import plotting_functions as pf
sd_new = SD[1].flatten()
sd_old = SD[0].flatten()
amp_new = AMP[1].flatten()
amp_old = AMP[0].flatten()
bins = []
bins.append((0,20));bins.append((30,80));bins.append((80,200));bins.append((200,400));  

for i in range(len(bins)):
    plt.subplot(2,2,i+1)    
    ind = np.where((sd_new>bins[i][0])&(sd_new<bins[i][1]))[0]
    plt.plot(amp_old[ind],amp_new[ind],'k.',markersize=.5)
    #pf.mean_bin_plot(amp_old[ind],amp_new[ind],5,1,1,'k')
#%%
spont_old = np.load(folders[0] +r'/suite2p_photostim/plane0/F.npy', allow_pickle=True)
spont_new = np.load(folders[1] +r'/suite2p_photostim/plane0/F.npy', allow_pickle=True)
cc = np.corrcoef(spont_new)
cco = np.corrcoef(spont_old)
X = []
Y = []
Yo = []
Xo = []
for i in range(SD[0].shape[1]):
    stm_cl = np.argsort(SD[0][:,i])[0]
    ind = np.where(SD[0][:,i]>30)
    X.append(cc[ind,stm_cl]*AMP[1][stm_cl,i])
    Xo.append(cco[ind,stm_cl]*AMP[0][stm_cl,i])    
    Y.append(AMP[1][ind,i])
    Yo.append(AMP[0][ind,i])
Y = np.concatenate(Y,axis = 1)
X = np.concatenate(X,axis = 1)
Xo = np.concatenate(Xo,axis = 1)
Yo = np.concatenate(Yo,axis = 1)

num_bins = 5;
plt.subplot(131)
pf.mean_bin_plot(Xo,Yo,num_bins,1,1,'k')
plt.subplot(132)
pf.mean_bin_plot(X,Y,num_bins,1,1,'k')
plt.subplot(133)
pf.mean_bin_plot(X+Xo,Y-Yo,num_bins,1,1,'k')
