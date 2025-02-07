# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:13:35 2023

@author: scanimage
"""
from scipy.signal import medfilt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)

mpl.rcParams['figure.dpi'] = 300
iscell = data['iscell']
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ind = np.where(iscell[:,0]==1)[0]
df = Ftrace[ind,:]
cn = data['conditioned_neuron'][0][0]
f = data['F'];
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:19,i])
    f[:,i] = f[:,i] - bl    
tune = np.mean(f[120:250,:],axis = 0)

lens = 10;
evts = []
for ci in range(df.shape[0]):
    a = Ftrace[ci,:];
    b = np.convolve(a,np.ones(lens),'same')/lens;
    noise = np.std(b-a)*3;
    bl = np.percentile(a,20);
    evts.append(np.mean(a > (bl + noise)))
plt.plot(tune,evts,'o',markerfacecolor = 'w')
plt.show()

try:
    del str
except:
    a=[]


cns = np.where(((tune) < .2) & (np.asarray(evts) > .08) )[0]
cns = cns[cns != cn]

fig, axs = plt.subplots(12, 5, figsize=(2.5, 5))  # Adjust figsize as needed
# [rest of your plotting code]
axs = axs.ravel()

# Plotting the first set of data
for i in range(20):
    axs[i].plot(f[:, cns[i]], 'k', linewidth=.2)
    axs[i].axis('off')
    axs[i].set_title(str(cns[i]), fontsize=6)

# Plotting the second set of data
for i in range(20, 40):
    axs[i].plot(df[cns[i-20], :], 'k', linewidth=.05)
    axs[i].axis('off')
    axs[i].set_title(str(cns[i-20]), fontsize=6)

# Plotting the images
img = ops['meanImg']
win = 10
for i in range(40, 60):
    x = np.round(data['centroidX'][cns[i-40]])
    y = np.round(data['centroidY'][cns[i-40]])
    x = int(x)
    y = int(y)
    a = img[y - win:y + win, x - win:x + win]
    axs[i].imshow(a, vmin=0, vmax=200, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(str(cns[i-40]+1), fontsize=6)
print(cns[0:19]+1)
plt.tight_layout()
plt.show()

#%%
win = 5
brightness = np.zeros(favg.shape[1])
brightness2 = np.zeros(favg.shape[1])
ss = stat[ind]
for i in range(f.shape[1]):
    x = np.round(data['centroidX'][i])
    y = np.round(data['centroidY'][i])
    x = int(x)
    y = int(y)
    xpix = ss[i]['xpix']
    ypix = ss[i]['ypix']
    a = img[y - win:y + win, x - win:x + win]
    brightness[i] = np.nanmean(a)
    a = img[ypix, xpix]
    brightness2[i] = np.nanmean(a)

plt.plot(brightness2)
thr = 300;
plt.plot((0,favg.shape[1]),(thr,thr),'k:')
too_bright = np.where(brightness2>thr)[0]

stim_cells = np.setdiff1d(stim_cells,too_bright)
num_cells = 100
stim_cells = stim_cells[0:num_cells]
print('0:' + str(max(stim_cells)+1))
too_bright = too_bright[too_bright<max(stim_cells)]
print(too_bright+1)
