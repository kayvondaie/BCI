# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:13:35 2023

@author: scanimage
"""
from scipy.signal import medfilt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
df = Ftrace
cn = data['conditioned_neuron'][0][0]
lens = 30;
evts = []
B = Ftrace*0
for ci in range(Ftrace.shape[0]):
    a = Ftrace[ci,:];
    b = np.convolve(a,np.ones(lens),'same')/lens;
    noise = np.std(b-a)*3;
    bl = np.percentile(a,20);
    evts.append(np.mean(a > (bl + noise)))
    B[ci,:] = b

siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist() 
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
if dt_si < 0.05:
    post = int(round(10/0.05 * 0.05/dt_si))
    pre = int(round(2/0.05 * 0.05/dt_si))
else:
    post = 200
    pre = 40
    
F2,_,_,_,_ = ddc.create_BCI_F(B,ops,stat,pre,post);
#%%

f = F2;
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:40:pre,i])
    f[:,i] = f[:,i] - bl    
tune = np.mean(f[40:pre+40,:],axis = 0)
plt.plot(tune,evts,'o',markerfacecolor = 'w')
plt.show()
#%%
try:
    del str
except:
    a=[]

iscell = data['iscell']
cns = np.where((np.abs(tune) < .1) & (np.asarray(evts) > .04))[0]
cns = np.where(((tune) > .15) & (np.asarray(evts) > .03))[0]


fig, axs = plt.subplots(12, 5, figsize=(5, 10))  # Adjust figsize as needed
# [rest of your plotting code]
axs = axs.ravel()

# Plotting the first set of data
for i in range(20):
    axs[i].plot(f[:, cns[i]], 'k', linewidth=.2)
    axs[i].axis('off')
    axs[i].set_title(str(cns[i]), fontsize=6)

# Plotting the second set of data
for i in range(20, 40):
    axs[i].plot(B[cns[i-20], :], 'k', linewidth=.05)
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
    axs[i].imshow(a, vmin=0, vmax=60, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(str(cns[i-40]), fontsize=6)
print(cns[0:19]+1)
plt.tight_layout()
plt.show()
