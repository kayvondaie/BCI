# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:25:02 2024

@author: kayvon.daie
"""
#import os;os.chdir('H:/My Drive/Python Scripts/BCI_analysis/')
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

folder = '//allen/aind/scratch/BCI/2p-raw/BCI85/062424/'
data = ddc.main(folder)

#%%
dt_si = data['dt_si']
F = data['F']
iscell = data['iscell']
cn = data['conditioned_neuron'][0][0]
f = np.nanmean(F,axis=2)
for ci in range(np.shape(f)[1]):
    f[:,ci] = f[:,ci] - np.nanmean(f[0:20,ci])    
#%% plot behavior stuff
reward_time_arrays = data['reward_time']
reward_time_arrays_with_nans = [arr if len(arr) > 0 else np.array([np.nan]) for arr in reward_time_arrays]
rew_time = np.concatenate(reward_time_arrays_with_nans)
rew = np.isnan(rew_time)==0
#%%
import plotting_functions as pf
dt_si = data['dt_si']
def boxoff():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
df = data['df_closedloop']
t = np.arange(0,dt_si*df.shape[1] ,dt_si)
bins = 10;
rew_rate = np.convolve(rew, np.ones(bins)/bins, mode='full')
rew_rate = rew_rate[bins-1:-bins+1]
df_closedloop = data['df_closedloop']

plt.subplot(234)
plt.plot(t,df_closedloop[cn,:].T,'k',linewidth = .3)
boxoff()
plt.xlabel('Time (s)')

plt.subplot(236)
tsta = np.arange(0,dt_si*F.shape[0],dt_si)
plt.plot(tsta,f[:,cn],'m')
plt.xlabel('Time from trial start (s)')
plt.ylabel('DF/F')
boxoff()
plt.title(data['mouse']+' ' + data['session'])

# plt.subplot(422)
# plt.plot(dist,delt,'ko',markerfacecolor = 'w',markersize=3)
# plt.xlabel('Distance from CN (um)')
# plt.ylabel('D Tuning')
# boxoff()

plt.subplot(235)
plt.imshow(np.squeeze(F[:,cn,:]).T,vmin=2,vmax=8)
plt.xticks([40, 240], ['0', '10'])
plt.xlabel('Time from trial start (s)')
plt.ylabel('Trial #')
boxoff()


plt.subplot(231);
plt.plot(rew_rate,'k')
plt.ylim((-.1,1.1))
plt.ylabel('Hit rate')
plt.xlabel('Trial #')
plt.title(data['mouse'] + '  ' +  data['session'])
boxoff()

plt.subplot(232);
plt.plot(rew_time,'ko',markerfacecolor = 'w',markersize=3)
plt.xlabel('Trial #')
plt.ylabel('Time to reward (s)')
plt.tight_layout()
boxoff()

#%%
faft = np.nanmean(F[:,:,30:60],axis = 2)
fbef = np.nanmean(F[:,:,0:30],axis = 2)
for ci in range(np.shape(fbef)[1]):
    fbef[:,ci] = fbef[:,ci] - np.nanmean(fbef[0:20,ci])  
    faft[:,ci] = faft[:,ci] - np.nanmean(faft[0:20,ci])  
tind = np.arange(40,240)
tun_bef = np.nanmean(fbef[tind,:],axis = 0)
tun_aft = np.nanmean(faft[tind,:],axis = 0)
delt = tun_aft - tun_bef;

plt.subplot(1,3,1)
plt.plot(tsta,fbef[:,cn],'k')
plt.plot(tsta,faft[:,cn],'m')
plt.xlabel('Time from trial start (s)')
plt.ylabel('DF/F')
boxoff()

plt.subplot(1,3,2)
dist = data['dist']
plt.plot(dist,delt,'ko',markersize=4,markerfacecolor = 'w')
plt.xlabel('Distance from CN (um)')
plt.ylabel('D Tuning')
boxoff()

plt.subplot(1,3,3)
cc = np.corrcoef(df_closedloop)
plt.plot(cc[:,cn],delt,'ko',markersize=4,markerfacecolor = 'w')
plt.xlim(-.1,.3)
plt.xlabel('Correlation with CN')
plt.ylabel('D tuning')
boxoff()

plt.tight_layout()
plt.show()

#%%
plt.subplot(131)
plt.plot(t,np.nanmean(Ftrace_axons,axis = 0),'k',linewidth=.3)
plt.xlabel('Time (s)')
plt.ylabel('DFF avg. all axons')
boxoff()

plt.subplot(132)
plt.imshow(np.squeeze(np.nanmean(Faxons,axis =1)).T)
plt.xticks([40, 240], ['0', '10'])
plt.xlabel('Time from trial start (s)')
plt.ylabel('Trial #')
boxoff()

plt.subplot(133)
#%%
fig, axs = plt.subplots(2,1, figsize=(2, 4))  # Adjust the size as needed

# Compute the mean
a = np.nanmean(np.nanmean(Faxons, 0), 0)
bins = 10;
aa = np.convolve(a, np.ones(bins)/bins, mode='full')
aa = aa[bins-1:-bins+1]

# Plot the data on the first subplot
axs[0].plot(rew_rate, 'k')
axs[0].set_ylabel('Hit rate')

# Plot the data on the second subplot
#axs[1].plot(a, 'g')
axs[1].plot(aa,'g')
axs[1].set_ylabel('DFF LC axons')
axs[1].set_xlabel('Trial #')
boxoff()
plt.tight_layout()

#%%

# Normalize the images to the range [0, 1]
scl = 4
meanImg = ops['meanImg'] / np.max(ops['meanImg'])*scl
meanImg_chan2 = ops['meanImg_chan2'] / np.max(ops['meanImg_chan2'])*scl*2

# Create an empty RGB image
overlay = np.zeros((meanImg.shape[0], meanImg.shape[1], 3))

# Assign magenta (red + blue) to meanImg
overlay[:, :, 0] = meanImg  # Red channel
overlay[:, :, 2] = meanImg  # Blue channel

# Assign green to meanImg_chan2
overlay[:, :, 1] = meanImg_chan2  # Green channel

# Display the overlay image
plt.imshow(overlay)
plt.axis('off')  # Hide the axes
plt.show()
#%%
miss = np.where(np.isnan(rew_time)==1)[0]
hit = np.where(np.isnan(rew_time)==0)[0]

hit = hit[hit<40]
miss = miss[miss<40]

plt.plot(tsta,np.nanmean(np.nanmean(Faxons[:,:,miss],axis=2),axis=1),'k')
plt.plot(tsta,np.nanmean(np.nanmean(Faxons[:,:,hit],axis=2),axis=1),'r')
boxoff()