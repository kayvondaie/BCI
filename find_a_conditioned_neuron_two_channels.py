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

#cn = data['conditioned_neuron'][0][0]
cn = 0
f = data['F'][0::1,:];
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:19,i])
    f[:,i] = f[:,i] - bl    
tune = np.mean(f[0:100,:],axis = 0)
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
iscell = data['iscell']
cns = np.where((tune>.0) & (iscell[:,0]==1))[0]
#cns = np.where( (iscell[:,0]==1))[0]
numpix = np.zeros((1,f.shape[1]))[0]
for i in range(len(numpix)):
    numpix[i] = len(stat[i]['xpix'])
#cns = np.where((numpix>25) & (tune>.1))[0]
#cns = np.where(tune>.09)[0]
#cns = np.where(((tune)<.05) & (iscell[:,0]==1) & (evts>1000))[0]
b = np.argsort(tune[0:100])
cns = b[0:20]
cns = np.concatenate([b[0:10], b[-11:]])
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
win = 15
for i in range(20):
    x = np.round(data['centroidX'][cns[i]])
    y = np.round(data['centroidY'][cns[i]])
    x = int(x)
    y = int(y)
    plt.subplot(4,5,i+1)
    a = img[y-win:y+win,x-win:x+win]
    plt.axis('off')
    plt.imshow(a,vmin = 0,vmax=400,cmap='gray')
    plt.title(str(cns[i]),fontsize=6)

plt.show()

print(cns[0:19]+1)

img = ops['meanImg']
# Display the image
plt.imshow(img, cmap='gray',vmin=0,vmax=50)

# Loop through each ROI and draw a patch
for roi in stat[cns[0:20]]:
    ypix = roi['ypix']  # Y-coordinates for the current ROI
    xpix = roi['xpix']  # X-coordinates for the current ROI

    # Create a set of (x, y) pairs for the current ROI
    polygon_points = list(zip(xpix, ypix))

    # Create a Polygon patch from these points
    polygon = Polygon(polygon_points, closed=True, fill=True, color='red', alpha=1, edgecolor='white')

    # Add the patch to the current axes
    plt.gca().add_patch(polygon)

# Show the plot with all ROIs overlaid
plt.show()
