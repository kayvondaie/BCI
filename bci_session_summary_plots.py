# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:25:02 2024

@author: kayvon.daie
"""
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

old_folder = '//allen/aind/scratch/BCI/2p-raw/BCINM_010/040424/'
folder = '//allen/aind/scratch/BCI/2p-raw/BCINM_010/040524/'
#folder = '//allen/aind/scratch/BCI/2p-raw/BCINM_013/050324/'
data = ddc.load_data_dict(folder)
old_data = ddc.load_data_dict(old_folder)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
base = siHeader['siBase']
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
dt_si = data['dt_si']
data['reward_time'], data['step_time'], data['trial_start']= ddc.create_zaber_info(folder,base,ops,dt_si)
data['dist'], data['conditioned_neuron_coordinates'], data['conditioned_neuron'] = ddc.find_conditioned_neurons(siHeader,stat)
#%%
X = data['centroidX']
Y = data['centroidY']
Xo = old_data['centroidX']
Yo = old_data['centroidY']
    
def find_closest_points(X, Y, Xo, Yo):
    # Convert lists to numpy arrays for efficient computation
    points = np.array(list(zip(X, Y)))
    other_points = np.array(list(zip(Xo, Yo)))

    # Initialize an array to store indices of closest points
    closest_indices = []
    nearest = []
    # Calculate the distance from each point to each other point in the other set
    for point in points:
        distances = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
        nearest.append(np.min(distances))
    nearest = np.asarray(nearest)
    return closest_indices, nearest

old_sort, nearest = find_closest_points(X, Y, Xo, Yo)
F = data['F']
Fo = old_data['F'][:,old_sort,:]
iscell = data['iscell']
cn = data['conditioned_neuron'][0]

cl_ind = np.where((iscell[:,0]==1) & (nearest < 5))[0]
cn = np.where(cl_ind == cn)[0][0]

F = F[:,cl_ind,:]
Fo = Fo[:,cl_ind,:]
dist = data['dist'][cl_ind]
f = np.nanmean(F,axis=2)
fo = np.nanmean(Fo,axis=2)
for ci in range(np.shape(f)[1]):
    f[:,ci] = f[:,ci] - np.nanmean(f[0:20,ci])
    fo[:,ci] = fo[:,ci] - np.nanmean(fo[0:20,ci])
#%% plot behavior stuff
rew_time = np.concatenate(data['reward_time']);
rew = np.isnan(rew_time)==0
#%%
import plotting_functions as pf

def boxoff():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


bins = 10;
rew_rate = np.convolve(rew, np.ones(bins)/bins, mode='full')
rew_rate = rew_rate[bins-1:-bins+1]
trial_start = data['trial_start'];
tind = np.arange(40,240)
tun = np.nanmean(f[tind,:],axis = 0)
tuno = np.nanmean(fo[tind,:],axis = 0)
delt = tun - tuno
df_closedloop = data['df_closedloop'][cl_ind,:]
cc = np.corrcoef(df_closedloop)

plt.subplot(321)
plt.plot(fo[:,cn],'k')
plt.plot(f[:,cn],'m')
plt.xlabel('Time from trial start (s)')
plt.ylabel('DF/F')
boxoff()

plt.subplot(322)
plt.plot(dist,delt,'ko',markerfacecolor = 'w',markersize=3)
plt.xlabel('Distance from CN (um)')
plt.ylabel('D Tuning')
boxoff()

plt.subplot(323)
plt.imshow(F[:,cn,:].T,vmin = -1,vmax = 6,aspect='auto')
plt.xlabel('Time from trial start (s)')
plt.ylabel('Trial #')
boxoff()

plt.subplot(324)
n = F.shape[1]
notcn = np.arange(1,n);
notcn = notcn[notcn!=cn]
pf.mean_bin_plot(cc[notcn,cn],delt[notcn],5,1,1,'k')
plt.xlabel('Correlation with CN')
plt.ylabel('D tuning')
boxoff()

plt.subplot(325);
plt.plot(rew_rate,'k')
plt.ylim((-.1,1.1))
plt.ylabel('Hit rate')
plt.xlabel('Trial #')
boxoff()

plt.subplot(326);
plt.plot(rew_time,'ko',markerfacecolor = 'w',markersize=3)
plt.xlabel('Trial #')
plt.ylabel('Time to reward (s)')
plt.tight_layout()
boxoff()

