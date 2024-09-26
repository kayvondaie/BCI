# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:25:02 2024

@author: kayvon.daie
"""
import os;os.chdir('H:/My Drive/Python Scripts/BCI_analysis/')
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
import plotting_functions as pf
from collections import Counter

folders = []
parentdir = '//allen/aind/scratch/BCI/2p-raw/BCINM_017/'
folders.append(parentdir + '060324/')
folders.append(parentdir + '060424/')
folders.append(parentdir + '060624/')
folders.append(parentdir + '061024/')
folders.append(parentdir + '061124/')
plt.figure(figsize=(10, 3))  # Width: 10 inches, Height: 6 inches
for fi in range(len(folders)):
    
    folder = folders[fi]
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    file = folder + ops['filelist'][0][ops['filelist'][0].rfind('\\')+1:]
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
    folder_props = folder_props_fun.folder_props_fun(folder)
    
    # Extract base names from siFiles directly from the dictionary
    base_names = [base for file in folder_props['siFiles'] for base in folder_props['bases'] if file.startswith(base)]    
    base_counts = Counter(base_names)    
    siHeader['siBase'] = base_counts.most_common(1)[0][0]
        
    np.save(folder + r'/suite2p_BCI/plane0/siHeader.npy',siHeader)
    #folder = '//allen/aind/scratch/BCI/2p-raw/BCINM_013/050324/'
    try:
        data = ddc.load_data_dict(folder)
    except:
        data = ddc.main(folder)
    stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
    base = siHeader['siBase']
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    dt_si = data['dt_si']
    
    F = data['F']
    iscell = data['iscell']
    cn = data['conditioned_neuron'][0][0]
    f = np.nanmean(F,axis=2)
    for ci in range(np.shape(f)[1]):
        f[:,ci] = f[:,ci] - np.nanmean(f[0:20,ci])        
    reward_time_arrays = data['reward_time']
    reward_time_arrays_with_nans = [arr if len(arr) > 0 else np.array([np.nan]) for arr in reward_time_arrays]
    rew_time = np.concatenate(reward_time_arrays_with_nans)
    rew = np.isnan(rew_time)==0
    
    
    def boxoff():
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    df = data['df_closedloop']
    t = np.arange(0,dt_si*df.shape[1],dt_si)
    t = t[0:df.shape[1]]
    bins = 10;
    rew_rate = np.convolve(rew, np.ones(bins)/bins, mode='full')
    rew_rate = rew_rate[bins-1:-bins+1]
    df_closedloop = data['df_closedloop']
    
    plt.subplot(2,6,fi+1)
    plt.plot(rew_rate,'k')
    plt.ylim((-.1,1.1))
    plt.ylabel('Hit rate')
    plt.xlabel('Trial #')
    plt.tight_layout()
    boxoff()
    
    plt.subplot(2,6,fi+1+6)
    plt.plot(rew_time,'ko',markerfacecolor = 'w',markersize=3)
    plt.xlabel('Trial #')
    plt.ylabel('Time to reward (s)')
    plt.tight_layout()
    boxoff()
    


#%%
faft = np.nanmean(F[:,:,20:60],axis = 2)
fbef = np.nanmean(F[:,:,0:20],axis = 2)
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
Ftrace_axons = np.load(folder +r'/suite2p_ch1/plane0/F.npy', allow_pickle=True)
Faxons,_,_,_,_ = ddc.create_BCI_F(Ftrace_axons,ops,stat)
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
# Plot the data on the second subplot
#axs[1].plot(a, 'g')
axs[1].plot(a,'g')
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
img = ops['meanImg_chan2']
# Display the image
plt.imshow(img, cmap='gray',vmin=0,vmax=20)
from matplotlib.patches import Polygon
stat_axons = np.load(folder + r'/suite2p_ch1/plane0/stat.npy', allow_pickle=True)
# Loop through each ROI and draw a patch
cns = np.arange(0,100)
for roi in stat_axons[cns]:
    ypix = roi['ypix']  # Y-coordinates for the current ROI
    xpix = roi['xpix']  # X-coordinates for the current ROI

    # Create a set of (x, y) pairs for the current ROI
    polygon_points = list(zip(xpix, ypix))

    # Create a Polygon patch from these points
    polygon = Polygon(polygon_points, closed=True, fill=True, color='g', alpha=.5, edgecolor='white')

    # Add the patch to the current axes
    plt.gca().add_patch(polygon)
plt.axis('off')  # Hide the axes

# Show the plot with all ROIs overlaid
plt.show()


#%%
miss = np.where(np.isnan(rew_time)==1)[0]
hit = np.where(np.isnan(rew_time)==0)[0]

hit = hit[hit<50]
miss = miss[miss<50]

plt.plot(tsta,np.nanmean(np.nanmean(Faxons[:,:,miss],axis=2),axis=1),'k')
plt.plot(tsta,np.nanmean(np.nanmean(Faxons[:,:,hit],axis=2),axis=1),'r')
boxoff()
#%%

dt = data['dt_si']
t = np.arange(0, dt * (df.shape[1]), dt)
trial_strt = 0*t;
rew = 0*t
steps = data['step_time']
strt = data['trial_start']
rewT = data['reward_time']
vel = np.zeros((F.shape[0],F.shape[2]))
for i in range(len(steps)):
    if np.isnan(F[-1,0,i]):
        l = np.where(np.isnan(F[40:,0,i])==1)[0][0]+39;
    else:
        l = F.shape[0]
    v = np.zeros(l,)
    for si in range(len(steps[i])):
        ind = np.where(t>steps[i][si])[0][0]
        v[ind] = 1
    vel[0:l,i] = v
for i in range(len(strt)):
    ind = np.where(t>strt[i])[0][0]
    trial_strt[ind] = 1
    
for i in range(len(rewT)):
    ind = np.where(t>rewT[i])[0][0]
    rew[ind] = 1

pos = 0*t;
for i in range(len(pos)):
    pos[i] = pos[i-1]
    if vel[i] == 1:
        pos[i] = pos[i-1] + 1;
    if trial_strt[i]==1:
        pos[i] = 0
#%%
from scipy.signal import medfilt
k = np.where(np.nanmean(f[40:150,:],axis =0)>0)[0]
avg = np.nanmean(df[k,:],axis=0)
avg = avg+.93
avg = avg*100
ind = list(range(1400,1900))
lw = .6
plt.subplot(312)
plt.plot(t[ind],pos[ind],'k',linewidth = lw)
plt.ylabel('Port position')
plt.subplot(413)
plt.plot(t[ind],rew[ind],'k',linewidth = lw)
plt.xlabel('Time (s)')
plt.ylabel('Reward')
#plt.plot(trial_strt[ind],linewidth = lw)
plt.subplot(411)
plt.plot(t[ind],medfilt(avg[ind],11),'k',linewidth = lw)
plt.ylabel('LC Axons')
plt.show()