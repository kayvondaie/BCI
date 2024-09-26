# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:37:47 2024

@author: kayvon.daie
"""

import os;
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir)
os.chdir(relative_path)
import numpy as np
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import data_dict_create_module as ddc

folder1 = r'//allen/aind/scratch/BCI/2p-raw/BCI87/081924/'
folder2 = r'//allen/aind/scratch/BCI/2p-raw/BCI87/082024/'
#%%
spont1 = np.load(folder1 +r'/suite2p_spont/plane0/F.npy', allow_pickle=True)
bci1 = np.load(folder1 +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)

spont2 = np.load(folder2 +r'/suite2p_spont/plane0/F.npy', allow_pickle=True)
bci2 = np.load(folder2 +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
#%%
plt.plot(np.percentile(bci1,20,axis=1),np.percentile(bci2,20,axis=1),'.')
plt.plot((0,100),(0,100))
plt.xlabel('Day 1 median')
plt.ylabel('Day 2 median')
#%%
plt.subplot(121)
plt.plot(np.percentile(spont1[:,0:1000],20,axis=1),np.percentile(spont2[:,0:1000],20,axis=1),'.')
plt.plot((0,100),(0,100))
plt.xlabel('Day 1 median')
plt.ylabel('Day 2 median')
plt.subplot(122)
plt.plot(np.percentile(spont1[:,0:1000],20,axis=1),np.percentile(spont1[:,1001:],20,axis=1),'.')
plt.plot((0,100),(0,100))
plt.xlabel('Day 1 median')
plt.ylabel('Day 1 (half 2) median')
#%%

