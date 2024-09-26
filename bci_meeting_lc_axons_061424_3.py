# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:25:02 2024

@author: kayvon.daie
"""
import os;
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir)
os.chdir(relative_path)
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
folders.append(parentdir + '060524/')
folders.append(parentdir + '060624/')
folders.append(parentdir + '061024/')
folders.append(parentdir + '061124/')
folders.append(parentdir + '061224/')
for fi in range(len(folders)):
    folder = folders[fi]
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    try:
        file = folder + ops['filelist'][0][ops['filelist'][0].rfind('\\')+1:]
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
    except:
        file = ops['filelist'][0][ops['filelist'][0].rfind('\\')+1:]
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
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
    
    plt.figure(figsize=(10, 5))  # Width: 10 inches, Height: 6 inches

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
    
    plt.subplot(235)
    plt.imshow(np.squeeze(F[:,cn,:]).T,vmin=0,vmax=5)
    plt.xticks([40, 240], ['0', '10'])
    plt.xlabel('Time from trial start (s)')
    plt.ylabel('Trial #')
    boxoff()
    
    
    plt.subplot(231);
    plt.plot(rew_rate,'k')
    plt.ylim((-.1,1.1))
    plt.ylabel('Hit rate')
    plt.xlabel('Trial #')
    boxoff()
    
    plt.subplot(232);
    plt.plot(rew_time,'ko',markerfacecolor = 'w',markersize=3)
    plt.xlabel('Trial #')
    plt.ylabel('Time to reward (s)')
    plt.tight_layout()
    boxoff()
    
        
    plt.subplot(2,3,3)
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
    
    plt.show()
    
    #%%
    plt.figure(figsize=(10,3.5))  # Width: 10 inches, Height: 6 inches

    faft = np.nanmean(F[:,:,20:50],axis = 2)
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
    #%%
    rew = np.zeros((F.shape[0],F.shape[2]))
    rewt = data['reward_time']
    for i in range(len(rew)):
        if i < len(rewt) and rewt[i].size > 0:  # Check if rewt[i] is not empty
            ind = np.where(t > rewt[i][0])[0]
            if ind.size > 0:  # Ensure ind is not empty
                rew[ind[0], i] = 1
    
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
    avg = np.nanmean(Faxons[:,0:,:],axis=1)
    vel_row = []
    avg_row=[];
    rew_row=[];
    axon_row = []
    for i in range(avg.shape[1]):
        avg_row.append(avg[:,i])
        vel_row.append(vel[:,i])
        rew_row.append(rew[:,i])
        axon_row.append(Faxons[:,:,i])
    
    
    axon_row = np.concatenate(axon_row)
    avg_row = np.concatenate(avg_row)
    vel_row = np.concatenate(vel_row)
    rew_row = np.concatenate(rew_row)
    ind = np.where(np.isnan(avg_row)==0)[0]
    avg_row = avg_row[ind]
    vel_row = vel_row[ind]
    rew_row = rew_row[ind]
    axon_row = axon_row[ind,:]

    tind = np.arange(11200, 12000)
    plt.plot(avg_row[0:3000],linewidth=.3);
    plt.plot(rew_row[0:3000],linewidth=.3)
    plt.show()
    

    plt.figure(figsize=(4, 4))  # Width: 10 inches, Height: 6 inches
    g=[]
    ind = np.where(vel_row==1)[0]

    #ind = ind[np.insert(np.diff(ind) >= 0, 0, True)]
    AVG = np.zeros((40,len(ind)))
    VEL = np.zeros((40,len(ind)))
    VTA = np.zeros((40,Faxons.shape[1],len(ind)))
    for i in range(len(ind)):    
        try:
            AVG[:,i] = avg_row[ind[i]-10:ind[i]+30].T
            VEL[:,i] = vel_row[ind[i]-10:ind[i]+30].T
            VTA[:,:,i] = axon_row[ind[i]-10:ind[i]+30,:]
        except:
            ok = 1;    
        
    ind = np.where(rew_row==1)[0]
    ind = ind[1:]
    AVG_rew = np.zeros((100,len(ind)))
    REW = np.zeros((100,len(ind)))
    RTA = np.zeros((100,Faxons.shape[1],len(ind)))
    for i in range(len(ind)):    
        AVG_rew[:,i] = avg_row[ind[i]-20:ind[i]+80].T
        REW[:,i] = rew_row[ind[i]-20:ind[i]+80].T
        RTA[:,:,i] = axon_row[ind[i]-20:ind[i]+80,:]
    rta = np.nanmean(RTA,axis = 2)       
 
    plt.subplot(221)
    tva = np.arange(0,dt_si*AVG.shape[0],dt_si)
    tva = tva - tva[10]
    plt.plot(tva,np.nanmean(AVG,axis=1),'k')    
    boxoff()
    plt.subplot(223)
    plt.plot(tva,np.nanmean(VEL,axis=1),'k')    
    plt.xlabel('Time from step (s)')
    boxoff()
    
    plt.subplot(222)
    tr = np.arange(0,dt_si*REW.shape[0],dt_si)
    tr = tr[0:AVG_rew.shape[0]]
    tr = tr - tr[20]
    plt.plot(tr,np.nanmean(AVG_rew,axis=1),'k')    
    boxoff()
    plt.subplot(224)
    plt.plot(tr,np.nanmean(REW,axis=1),'k')    
    plt.xlabel('Time from reward (s)')
    boxoff()
    plt.tight_layout()
#%%

import matplotlib.pyplot as plt
import numpy as np

def overlay_rois(folder, roi_indices, ax):
    stat_axons = np.load(folder + r'/suite2p_ch1/plane0/stat.npy', allow_pickle=True)
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    img = ops['meanImg_chan2']

    # Create an empty RGBA image with the same shape as the original image
    rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)

    # Set the grayscale image to the RGB channels of the RGBA image
    normalized_img = img / img.max()  # Normalizing the image to [0, 1] range
    rgba_img[..., 0] = normalized_img
    rgba_img[..., 1] = normalized_img
    rgba_img[..., 2] = normalized_img
    rgba_img[..., 3] = 1.0  # Fully opaque initially

    # Create an overlay mask for the ROIs
    overlay = np.zeros_like(rgba_img)

    # Initialize bounding box coordinates
    min_y, min_x = float('inf'), float('inf')
    max_y, max_x = float('-inf'), float('-inf')

    # Loop through each ROI and fill the overlay
    for idx in roi_indices:
        roi = stat_axons[idx]
        ypix = roi['ypix']  # Y-coordinates for the current ROI
        xpix = roi['xpix']  # X-coordinates for the current ROI
        overlay[ypix, xpix, 1] = 1  # Set the green channel to 1 for ROI pixels
        overlay[ypix, xpix, 3] = 0.5  # Set the alpha channel to 0.5 for ROI pixels

        # Update bounding box coordinates
        min_y, max_y = min(min_y, ypix.min()), max(max_y, ypix.max())
        min_x, max_x = min(min_x, xpix.min()), max(max_x, xpix.max())

    # Display the grayscale image
    ax.imshow(img, cmap='gray', vmin=0, vmax=20)

    # Overlay the RGBA image
    ax.imshow(overlay, alpha=1)

    # Zoom into the ROI
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 10, max_y + 10)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates

    # Hide the axes
    ax.axis('off')

    plt.show()



indices_sorted_by_size = sorted(range(len(stat_axons)), key=lambda i: stat_axons[i]['ypix'].size, reverse=True)
index_of_largest_roi = indices_sorted_by_size[0]
plt.figure(figsize=(14, 4))  # Width: 10 inches, Height: 6 inches
tind = np.arange(7000,10000)
plt.subplot(211)
plt.plot(t[tind],avg_row[tind],'k',linewidth=.3);
plt.plot(t[tind],rew_row[tind],'r',linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('DFF avg. all axons')
boxoff()
plt.subplot(212)
plt.plot(t[tind],avg_row[tind],'k',linewidth=.3);
plt.plot(t[tind],vel_row[tind]/2,'r',linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('DFF avg. all axons')
boxoff()
plt.show()


for i in range(5):
    fig = plt.figure(figsize=(10, 2))
    axon_ind = i;    
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, :1])  # First row, first column (1/4th width)
    ax2 = fig.add_subplot(gs[0, 1:])  # First row, columns 2-4 (3/4ths width)    
    tind = np.arange(7000,8000)    
    ax2.plot(t[tind],axon_row[tind,indices_sorted_by_size[axon_ind]],'k',linewidth=.3);
    ax2.plot(t[tind],rew_row[tind],'b',linewidth=1)
    ax2.plot(t[tind],vel_row[tind]/2,'r',linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('DFF')
    boxoff()
    
    overlay_rois(folder,[indices_sorted_by_size[axon_ind]],ax1)
    plt.tight_layout()
    
    import matplotlib.pyplot as plt
    import numpy as np
#%%
fig = plt.figure(figsize=(5,5))
num = 40;
offset = 0
tind = np.arange(7000,12000)  
for i in range(num):
    a = Ftrace[i,tind]
    a = a - min(a)
    plt.plot(t[tind],a+offset,'k',linewidth = .3)
    offset = max(Ftrace[i,tind]+offset)

boxoff()      
    
    