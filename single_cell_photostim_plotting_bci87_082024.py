# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:15:01 2024

@author: kayvon.daie
"""
Ftrace = np.load(folder +r'/suite2p_photostim_single/plane0/F.npy', allow_pickle=True)
bl = np.nanmedian(Ftrace,axis=1)
favg_raw = data['photostim']['favg_raw']
favg = 0*favg_raw
for i in range(favg.shape[1]):
    for gi in range(favg.shape[2]):
        favg[:,i,gi] = (favg_raw[:,i,gi] - np.nanmean(favg_raw[0:4,i,gi]))/bl[i]

#%%
stimDist = data['photostim']['stimDist']
gi = 1;
bins = (0,10,25,60,160,300)
avg = np.zeros((favg.shape[0],len(bins)-1,favg.shape[2]))
for i in range(len(bins)-1):
    for gi in range(favg.shape[2]):
        ind = np.where((stimDist[:,gi]>bins[i]) & (stimDist[:,gi]<bins[i+1]))[0]
        avg[:,i,gi] = np.nanmean(favg[:,ind,gi],axis = 1);

plt.plot(np.nanmean(avg[0:30,4,:],axis=1))

#%%
amp = np.nanmean(favg[8:10,:,:],axis=0)
for i in range(40):
    plt.plot(stimDist[i,:],amp[i,:],'.')
    plt.title(str(i))
    plt.show()
#%%
for i in range(40):
    plt.plot(np.nanmean(favg[0:35,i,:],axis=1))
    plt.title(str(i))
    plt.show()

#%%
vol=ScanImageTiffReader('//allen/aind/scratch/BCI/2p-raw/BCI87/082024/two_color_00002.tif').data();
red = np.nanmean(vol[1::2,:,:],axis=0);
green = np.nanmean(vol[0::2,:,:],axis=0);
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
green = ops['meanImg']
#%%
red_zoom = np.zeros((window*2,window*2,40))
green_zoom = np.zeros((window*2,window*2,40))
k = np.nanmean(favg[0:35,:,:],axis=2)
target_amp=np.zeros((40,))
target_response = np.zeros((favg.shape[0],40))
for ci in range(0,40):
    
    x = int(round(data['centroidX'][ci]))
    y = int(round(data['centroidY'][ci]))
    
    red_zoom[:,:,ci] = red[y-window-5:y+window-5, x-window:x+window]
    green_zoom[:,:,ci] = green[y-window-5:y+window-5, x-window:x+window]
    
    window = 20
    
    # Create a 1x4 grid of subplots with a fixed figure size
    #fig, axs = plt.subplots(1, 5, figsize=(12, 4))
    
    # First subplot: green channel image
    # axs[0].imshow(green[y-window:y+window, x-window:x+window], cmap='gray', vmin=0, vmax=150)
    # axs[0].set_aspect('equal', adjustable='box')
    # axs[0].axis('off')  # Hide axes for image
    # axs[0].set_title(str(ci))
    
    
    # # Second subplot: red channel image
    # axs[1].imshow(red[y-window:y+window, x-window:x+window], cmap='gray')
    # axs[1].set_aspect('equal', adjustable='box')
    # axs[1].axis('off')  # Hide axes for image
    
    # # Third subplot: favg_raw line plot
    # #axs[2].plot(favg_raw[:, ci, ci], 'k', linewidth=1)
    # axs[2].plot(k[:, ci], 'k', linewidth=1)
    # #axs[2].set_xlim(0, len(favg_raw[:, ci, ci]))
    # axs[2].set_aspect('auto')  # Set aspect to auto for line plot
    
    # # Fourth subplot: f line plot
    # axs[3].plot(f[:, ci], 'k', linewidth=1)
    # axs[3].set_xlim(0, len(f[:, ci]))
    # axs[3].set_aspect('auto')  # Set aspect to auto for line plot
        
    # axs[4].scatter(stimDist[ci,:],amp[ci,:],color='k')
    
    
    # plt.tight_layout()  # Adjust layout to ensure plots do not overlap
    # plt.show()
    
    target_response[:,ci] = favg[:,ci,ci]
    target_amp[ci] = np.nanmean(favg[8:10,ci,ci])
#%%
aa = np.nanmean(k[8:10,:],axis=0) - np.nanmean(k[0:3,:],axis=0);
b = np.argsort(aa[0:40])

plt.subplot(331)
plt.plot(np.nanmean(k[:,b[-5:]],axis=1),'k');
plt.subplot(332)
plt.imshow(np.nanmean(red_zoom[:,:,b[-5:]],axis=2),cmap='gray')
plt.subplot(333)
plt.imshow(np.nanmean(green_zoom[:,:,b[-5:]],axis=2),cmap='gray')

plt.subplot(334)
plt.plot(np.nanmean(k[:,b[0:5]],axis=1),'k');
plt.subplot(335)
plt.imshow(np.nanmean(red_zoom[:,:,b[0:5]],axis=2),cmap='gray')
plt.subplot(336)
plt.imshow(np.nanmean(green_zoom[:,:,b[0:5]],axis=2),cmap='gray')

b = np.argsort(target_amp)
plt.subplot(337)
plt.plot(np.nanmean(target_response[:,b[-10:]],axis=1),'k');
plt.subplot(338)
plt.imshow(np.nanmean(red_zoom[:,:,b[-10:]],axis=2),cmap='gray')
plt.subplot(339)
plt.imshow(np.nanmean(green_zoom[:,:,b[-10:]],axis=2),cmap='gray')