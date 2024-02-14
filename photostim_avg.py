# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:12:53 2023

@author: scanimage
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
F = np.load(folder + ops['save_folder'] + r'/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + ops['save_folder'] + r'/plane0/ops.npy', allow_pickle=True).tolist()
stat = np.load('D:/KD/BCI_data/BCI_2022/BCI48/032823/suite2p_neuron_2/plane0/stat.npy', allow_pickle=True)

numTrl = len(ops['frames_per_file']);
timepts = 45;
numCls = F.shape[0]
Fstim = np.full((timepts,numCls,numTrl),np.nan)
strt = 0;
dff = 0*F
pre = 5;
post = 20
for ti in range(numTrl):
    pre_pad = np.arange(strt-5,strt)
    ind = list(range(strt,strt+ops['frames_per_file'][ti]))
    strt = ind[-1]+1
    post_pad = np.arange(ind[-1]+1,ind[-1]+20)
    ind = np.concatenate((pre_pad,np.asarray(ind)),axis=0)
    ind = np.concatenate((ind,post_pad),axis = 0)
    ind[ind > F.shape[1]-1] = F.shape[1]-1;
    ind[ind < 0] = 0
    a = F[:,ind].T
    bl = np.tile(np.mean(a[0:pre,:],axis = 0),(a.shape[0],1))
    a = (a-bl) / bl
    Fstim[0:a.shape[0],:,ti] = a
#%%
import extract_scanimage_metadata
folder = ops['data_path'][0]
file = folder + ops['tiff_list'][0]
data = extract_scanimage_metadata.extract_scanimage_metadata(file)
photostim_groups = data['metadata']['json']['RoiGroups']['photostimRoiGroups']
seq = data['metadata']['hPhotostim']['sequenceSelectedStimuli'];
list_nums = seq.strip('[]').split();
seq = [int(num) for num in list_nums]
seqPos = int(data['metadata']['hPhotostim']['sequencePosition'])-1;
seq = seq[seqPos:Fstim.shape[2]]
seq = np.asarray(seq)
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

centroidX = []
centroidY = []
for i in range(len(stat)):
    centroidX.append(np.mean(stat[i]['xpix']))
    centroidY.append(np.mean(stat[i]['ypix']))

favg = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
stimDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
for gi in range(len(photostim_groups)):
    coordinates = photostim_groups[gi]['rois'][1]['scanfields']['slmPattern']
    xy = np.asarray(coordinates)[:,:2] + photostim_groups[gi]['rois'][1]['scanfields']['centerXY']
    stimPos = np.zeros(np.shape(xy))
    for i in range(np.shape(xy)[0]):
        stimPos[i,:] = np.array(xy[i,:]-num[0])*pixPerDeg
    sd = np.zeros([np.shape(xy)[0],f_avg.shape[1]])        
    for i in range(np.shape(xy)[0]):
        for j in range(f_avg.shape[1]):
            sd[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
    stimDist[:,gi] = np.min(sd,axis=0)
    ind = np.where(seq == gi+1)[0]
    favg[:,:,gi] = np.nanmean(Fstim[:,:,ind],axis = 2)

