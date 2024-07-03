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
cn = data['conditioned_neuron'][0][0]
f = data['F'];
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:19,i])
    f[:,i] = f[:,i] - bl    
tune = np.mean(f,axis = 0)
#%%
evts = np.zeros((N,))
for ci in range(N):
    df = data['df_closedloop'][ci,:]
    df = medfilt(df, 21)
    df = np.diff(df)    
    evts[ci] = len(np.where(df>.2)[0])
plt.plot(tune,evts,'o',markerfacecolor = 'w')
plt.show()
#%%
len = 10;
evts = []
for ci in range(Ftrace.shape[0]):
    a = Ftrace[ci,:];
    b = np.convolve(a,np.ones(len),'same')/len;
    noise = np.std(b-a)*3;
    bl = np.percentile(a,20);
    evts.append(np.mean(a > (bl + noise)))
#%%
iscell = data['iscell']
cns = np.where((np.abs(tune)>.05) & (np.asarray(evts)>.1))[0]

#cns = np.where(((tune)<.05) & (iscell[:,0]==1) & (evts>1000))[0]
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.plot(f[:,cns[i]],'k',linewidth=.2)
    plt.axis('off')
    plt.title(str(cns[i]),fontsize=6)
plt.show()

df = data['df_closedloop']    
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.plot(df[cns[i],:],'k',linewidth=.05)
    plt.axis('off')
    plt.title(str(cns[i]),fontsize=6)
plt.show()

ops = np.load(data['dat_file']+'/ops.npy', allow_pickle=True).tolist()
img = ops['meanImg']
win = 10
for i in range(20):
    x = np.round(data['centroidX'][cns[i]])
    y = np.round(data['centroidY'][cns[i]])
    x = int(x)
    y = int(y)
    plt.subplot(4,5,i+1)
    a = img[y-win:y+win,x-win:x+win]
    plt.axis('off')
    plt.imshow(a,vmin = 0,vmax=300,cmap='gray')
    plt.title(str(cns[i]),fontsize=6)

plt.show()

print(cns[0:19]+1)