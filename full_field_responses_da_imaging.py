# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:33:08 2023

@author: Kayvon Daie
"""
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import matplotlib.pyplot as plt

folder = r'//allen/aind/scratch/BCI/2p-raw/BCINM_004/120623/'
data = ddc.load_data_dict(folder)
ops = np.load(data['dat_file']+'/ops.npy', allow_pickle=True).tolist()

tifs = ops['tiff_list']
DA = []
for i in range(len(tifs)):  
    print(i)
    vol = ScanImageTiffReader(folder + tifs[i]).data()
    avg = np.mean(np.mean(vol,axis=2),axis=1)
    DA.append(avg)
    
#%%
da = np.zeros((90,len(DA)))
for i in range(len(DA)):
    a = DA[i][1:180:2]
    da[:,i] = a - a[0]
    
plt.plot(np.nanmean(da,axis=1))
#%%
da_trace = np.array([])

# Iterate through DA and append each array to da_trace
for array in DA:
    da_trace = np.append(da_trace, array[1:-1:2])

dt = data['dt_si'];
t = np.arange(0, len(da_trace) * dt, dt)

#%%
pre = 60;
post = 100
da_rew = np.zeros((pre+post,len(DA)))

rew = data['trial_start']
for ri in range(len(rew)):
    rt = rew[ri]
    ind = np.where(t>rt)[0][0]
    ind = [i for i in range(ind-pre+1, ind+post+1)]  # Range ends at ind+60, so ind+61 is exclusive
    ind = [min(i, len(t)-1) for i in ind]
    a = da_trace[ind]
    da_rew[:,ri] = a - np.mean(a[0:pre])
    
tsta = np.arange(0, da_rew.shape[0] * dt, dt)    
tsta = tsta - tsta[pre]
plt.plot(tsta,np.mean(da_rew,axis=1))    