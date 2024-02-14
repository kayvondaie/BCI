# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:43:10 2023

@author: scanimage
"""


import scipy.io as spio
import numpy as np
import extract_scanimage_metadata
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
F = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)

from scipy.io import savemat
savemat(folder +r'/suite2p_BCI/plane0/' + 'stat.mat',{'stat': stat})
savemat(folder +r'/suite2p_BCI/plane0/' + 'meanImg.mat',{'meanImg': ops['meanImg']})
savemat(folder +r'/suite2p_BCI/plane0/' + 'iscell.mat',{'iscell': iscell})

#%%
F_trial_strt = [];
strt = 0;
dff = 0*F
for i in range(np.shape(F)[0]):
    #bl = np.percentile(F[i,:],50)
    bl = np.std(F[i,:])
    dff[i,:] = (F[i,:] - bl)/bl
for i in range(len(ops['frames_per_file'])):
    ind = list(range(strt,strt+ops['frames_per_file'][i]))    
    f = dff[:,ind]
    F_trial_strt.append(f)
    strt = ind[-1]+1
    

f_first_ten = np.full((240,np.shape(F)[0],len(ops['frames_per_file'])),np.nan)
pre = np.full((np.shape(F)[0],40),np.nan)
for i in range(len(ops['frames_per_file'])):
    f = F_trial_strt[i]
    if i > 0:
        pre = F_trial_strt[i-1][:,-40:]
    pad = np.full((np.shape(F)[0],200),np.nan)
    f = np.concatenate((pre,f),axis = 1)
    f = np.concatenate((f,pad),axis = 1)
    f = f[:,0:240]
    f_first_ten[:,:,i] = np.transpose(f)

            
favg = np.nanmean(f_first_ten,axis = 2)
for i in range(F.shape[0]):
    favg[:,i] = favg[:,i] - np.nanmean(favg[1:20,i])
    
    
#%%
folder = ops['data_path'][0]
file = folder + ops['tiff_list'][0]
data = extract_scanimage_metadata.extract_scanimage_metadata(file)
cnName = data['metadata']['hIntegrationRoiManager']['outputChannelsRoiNames']
g = [i for i in range(len(cnName)) 
     if cnName.startswith("'",i)]
cnName = cnName[g[0]+1:g[1]]

rois = data['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']
a = []
for i in range(len(rois)):
    name = rois[i]['name']
    a.append(cnName == name)
    
indices = [i for i, x in enumerate(a) if x]
cnPos = rois[indices[0]]['scanfields']['centerXY'];

deg = data['metadata']['hRoiManager']['imagingFovDeg']
g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
for i in gg:
    g.append(i)
g = np.sort(g)
num = [];
for i in range(len(g)-1):
    num.append(float(deg[g[i]+1:g[i+1]]))
dim = int(data['metadata']['hRoiManager']['linesPerFrame']),int(data['metadata']['hRoiManager']['pixelsPerLine'])
degRange = np.max(num) - np.min(num)
pixPerDeg = dim[0]/degRange

cnPosPix = np.array(np.array(cnPos)-num[0])*pixPerDeg

centroidX = []
centroidY = []
dist = []
for i in range(len(stat)):
    centroidX.append(np.mean(stat[i]['xpix']))
    centroidY.append(np.mean(stat[i]['ypix']))
    dx = centroidX[i] - cnPosPix[0]
    dy = centroidY[i] - cnPosPix[1]
    d = np.sqrt(dx**2+dy**2)
    dist.append(d)
dist = np.asarray(dist)
a = np.min(dist)    
cn = np.where(dist==a)[0][0]
print(cn)


#%%
evt = np.zeros((1,F.shape[0]))[0]
tuning = np.mean(favg,axis = 0)
for i in range(F.shape[0]):
    a = dff[i,:]
    b = np.std(a)*1.5
    a = np.diff(a)
    evt[i] = len(np.where(a>b)[0])
cns = np.where((np.abs(tuning)<.05) & (evt>10))[0]
num = 10;
cnsMat = cns[0:num-1] + 1

for i in range(num-1):
    plt.subplot(5,2,i+1)
    plt.plot(favg[:,cns[i]],'k')
    plt.ylim([-.2,.8])
    plt.title(str(cnsMat[i]))
plt.show()

for i in range(num-1):
    plt.subplot(5,2,i+1)
    plt.plot(dff[cns[i],:],'k',linewidth=.2)    
    plt.title(str(cnsMat[i]))
plt.show()

img = ops['meanImg']
win = 30;
for i in range(num-1):
    plt.subplot(5,2,i+1)    
    idx = (int(centroidX[cns[i]])-win,int(centroidX[cns[i]])+win)
    idy = (int(centroidY[cns[i]])-win,int(centroidY[cns[i]])+win)
    plt.imshow(img[idy[0]:idy[1],idx[0]:idx[1]], cmap = 'gray')
    plt.title(str(cnsMat[i]))
plt.show()
print(cnsMat)





