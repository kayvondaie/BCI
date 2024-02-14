# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:18:50 2023

@author: scanimage
"""
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
# Load the .mat file
file = folder + r'neuron14_22_BCIparams_48.mat'
bci_params = scipy.io.loadmat(file)
bci_params = bci_params['a']



stim_time = spio.loadmat(folder + '/BCI_params_py.mat')
stim_time=stim_time['stim_time'][0]
stim_time = stim_time*3600*24;
stim_time = stim_time - stim_time[0];
stim_time = stim_time+stim_time[1]

photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups']
power = photostim_groups['rois'][1]['scanfields']['powers']
coordinates = photostim_groups['rois'][1]['scanfields']['slmPattern']
xy = np.asarray(coordinates)[:,:2] + photostim_groups['rois'][1]['scanfields']['centerXY']
stimPos = np.zeros(np.shape(xy))
for i in range(np.shape(xy)[0]):
    print(i)
    stimPos[i,:] = np.array(xy[i,:]-num[0])*pixPerDeg

stimDist = np.zeros([np.shape(xy)[0],len(stat)])
for i in range(np.shape(xy)[0]):
    for j in range(len(stat)):
        stimDist[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
stimDist = np.min(stimDist,axis=0)
a = np.sort(stimDist)
b = np.argsort(stimDist)

dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
t_si = np.arange(0,dt_si * np.shape(F)[1],dt_si)
stm_cls = np.where(stimDist<10)[0]
#ind = b[0:9]
ax = plt.subplot(1,2,1)
plt.plot(t_si[0000:2000],np.nanmean(F[stm_cls,0000:2000],axis = 0),'k',linewidth = .5)
y = np.ones((10,))/2;
xl = plt.xlim()
ind = np.where((stim_time < xl[1]) & (stim_time > xl[0]))[0]
stim_time = stim_time[ind]
for i in range(len(stim_time)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [stim_time[i], stim_time[i]+3, stim_time[i]+3, stim_time[i]]
    y = [0, 0, 4, 4]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)



ax = plt.subplot(1,2,2)
cns = data['conditioned_neuron'][0]
plt.plot(t_si[0:2000],np.nanmean(F[cns,0:2000],axis = 0),'k',linewidth = .5)
for i in range(len(stim_time)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [stim_time[i], stim_time[i]+3, stim_time[i]+3, stim_time[i]]
    y = [1, 1, 3, 3]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)

plt.show()

#%%
stim_time = spio.loadmat(folder + '/BCI_params_py.mat')
stim_time=stim_time['stim_time'][0]
stim_time = stim_time*3600*24;
stim_time = stim_time - stim_time[0];
stim_time = stim_time+stim_time[1]
stim_time = stim_time[~np.isnan(stim_time)]
pre = 60;
post = 120;
sta = np.zeros((np.shape(F)[0],pre+post,len(stim_time)))
for i in range(len(stim_time)):
    ind = np.where(t_si>stim_time[i]-i*.005)[0][0]    
    ind = np.array(range(ind-pre,ind+post));
    ind[ind<0] = 0;
    ind[ind>np.shape(F)[1]-1] = np.shape(F)[1]-1;
    sta[:,:,i] = F[:,ind]

for i in range(len(stim_time)):
    for j in range(sta.shape[0]):
        a = sta[j,:,i]
        bl = np.nanmean(a[0:30])
        sta[j,:,i] = a - bl
#%%
late = (190,250)
early = (0,160)
plt.subplot(2,2,1)
plt.plot(np.nanmean(np.nanmean(sta[stm_cls,:,:],axis = 2),axis = 0))

plt.subplot(2,2,3)
a = (np.nanmean(sta[stm_cls,:,:],axis = 0))
a = a - np.tile(np.mean(a[0:30, :], axis=0), (a.shape[0], 1))
plt.imshow(a.T,vmin = 0,vmax = 1)

plt.subplot(2,2,2)
#cns = np.where((stimDist>440)&(stimDist<4000)&(iscell[:,0]==1))[0]
plt.plot(np.nanmean(np.nanmean(sta[cns,:,early[0]:early[1]],axis = 2),axis = 0))
plt.plot(np.nanmean(np.nanmean(sta[cns,:,late[0]:late[1]],axis = 2),axis = 0))

plt.subplot(2,2,4)
a = (np.nanmean(sta[cns,:,:],axis = 0))
a = a - np.tile(np.mean(a[0:10, :], axis=0), (a.shape[0], 1))
plt.imshow(a.T,vmin = 0,vmax = .6)

#%%
fig = plt.figure(figsize=(8, 8))
ind = np.where(iscell[:,0]==1)[0]
b = np.argsort(stimDist[ind])
avg = np.nanmean(sta[ind[b],60:100,:],axis = 2)
plt.imshow(avg,vmin=0,vmax=.3)
plt.show()
plt.plot(stimDist[ind[b]],np.mean(avg,axis = 1),'ko',markerfacecolor = 'w', markersize = 3)
plt.plot((30,30),(0,1))
plt.plot((45,45),(0,1))