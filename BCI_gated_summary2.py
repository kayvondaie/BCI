# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:09:21 2023

@author: scanimage
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:18:50 2023

@author: scanimage
"""
import scipy.io as spio
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import matplotlib.patches as patches

import numpy as np
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folde
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
# Load the .mat file
if 'stim_time' in data:
    stim_time = data['stim_time']
else:
    a = spio.loadmat(folder + '/BCI_params_py.mat')
    stim_time = a['stim_time'][0]
    #stim_group = a['stim_group'][0]
    #stim_time = stim_time*3600*24;
    #stim_time = stim_time[1:-1]
    #stim_time = stim_time - stim_time[0];
    stim_time = stim_time+.8
    stim_time = stim_time[~np.isnan(stim_time)]
    #stim_time = stim_time[stim_group==2]

        
iscell = data['iscell']
deg = siHeader['metadata']['hRoiManager']['imagingFovDeg']
g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
for i in gg:
    g.append(i)
g = np.sort(g)
num = [];
for i in range(len(g)-1):
    num.append(float(deg[g[i]+1:g[i+1]]))
dim = int(siHeader['metadata']['hRoiManager']['linesPerFrame']),int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
degRange = np.max(num) - np.min(num)
pixPerDeg = dim[0]/degRange

centroidX = data['centroidX']
centroidY = data['centroidY']

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
F = data['df_closedloop']
t_si = np.arange(0,dt_si * np.shape(F)[1],dt_si)
stim_time= stim_time[stim_time<t_si[-1]]
stm_cls = np.where(stimDist<4)[0]
#ind = b[0:9]
ax = plt.subplot(2,1,1)
xlim = (000,2000)
plt.plot(t_si[xlim[0]:xlim[1]],np.nanmean(F[stm_cls,xlim[0]:xlim[1]],axis = 0),'k',linewidth = .5)
y = np.ones((10,))/2;
xl = plt.xlim()
ind = np.where((stim_time < xl[1]) & (stim_time > xl[0]))[0]
st = stim_time[ind]
for i in range(len(st)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [st[i], st[i]+3, st[i]+3, st[i]]
    y = [0, 0, 4, 4]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)



ax = plt.subplot(2,1,2)
cns = data['conditioned_neuron'][0]
cns = cns[iscell[cns,0]==1]
cns = cns[0]
#plt.plot(t_si[0:4000],np.nanmean(F[cns,0:4000],axis = 0),'k',linewidth = .5)
plt.plot(t_si[xlim[0]:xlim[1]],F[cns,xlim[0]:xlim[1]],'k',linewidth = .5)
for i in range(len(st)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [st[i], st[i]+3, st[i]+3, st[i]]
    y = [0,0, 5, 5]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)

plt.show()

#%%

pre = 10;
post = 120;
sta = np.zeros((np.shape(F)[0],pre+post,len(stim_time)))
for i in range(len(stim_time)):    
    ind = np.where(t_si>stim_time[i]-i*.004)[0][0]    
    ind = np.array(range(ind-pre,ind+post));
    ind[ind<0] = 0;
    ind[ind>np.shape(F)[1]-1] = np.shape(F)[1]-1;
    sta[:,:,i] = F[:,ind]

for i in range(len(stim_time)):
    for j in range(sta.shape[0]):
        a = sta[j,:,i]
        bl = np.nanmean(a[pre-10:pre])
        sta[j,:,i] = a - bl
#%%
plt.rcParams['font.size'] = 8
fig = plt.figure(figsize=(6,8))
axes = fig.subplots(4,3).flat
late = (130,150)
early = (0,120)
tsta = np.arange(0,dt_si * np.shape(sta)[1],dt_si)
tsta = tsta - tsta[pre]
titles = list()
titles.append('Early')
titles.append('Late')
axes[0].plot(tsta,np.nanmean(np.nanmean(sta[stm_cls,:,early[0]:early[1]],axis = 2),axis = 0))
axes[0].plot(tsta,np.nanmean(np.nanmean(sta[stm_cls,:,late[0]:late[1]],axis = 2),axis = 0))
axes[0].set_title('Stim. neurons')
axes[0].set_xlabel('Distance from stim. (s)')
axes[0].legend(titles)
axes[0].set_ylabel('DF/F')

axes[1].plot(tsta,np.nanmean(sta[cns,:,early[0]:early[1]],axis = 1))
axes[1].plot(tsta,np.nanmean(sta[cns,:,late[0]:late[1]],axis = 1))
axes[1].set_title('CN')
axes[1].set_xlabel('Distance from stim. (s)')
axes[1].set_ylabel('DF/F')
axes[1].legend(titles)

ind = np.where(iscell[:,0]==1)[0]
b = np.argsort(stimDist[ind])
avg = np.nanmean(sta[:,10:70,:],axis = 2)
axes[2].plot(stimDist[ind],np.nanmean(avg[ind],axis = 1),'ko',markerfacecolor = 'w', markersize = 3)
axes[2].plot(stimDist[cns],np.nanmean(avg[cns,:]),'ko',markerfacecolor = 'm', markersize = 8)
axes[2].set_ylabel('Response amp.')
axes[2].set_xlabel('Distance from stim.')

#plt.plot((30,30),(0,1))
#plt.plot((45,45),(0,1))

#axes[3].plot(np.nanmean(np.nanmean(sta[cns,:,late[0]:late[1]],axis = 2),axis = 0))
#axes[3].plot(np.nanmean(np.nanmean(sta[stm_cls,:,late[0]:late[1]],axis = 2),axis = 0)/4-.05)
#a = (np.nanmean(sta[cns,:,:],axis = 0))
#a = a - np.tile(np.mean(a[0:10, :], axis=0), (a.shape[0], 1))
#plt.imshow(a.T,vmin = 0,vmax = 1)
a = (np.nanmean(sta[stm_cls,:,:],axis = 0))
a = a - np.tile(np.mean(a[0:30, :], axis=0), (a.shape[0], 1))
axes[3].imshow(a.T,vmin = -1,vmax = 1,extent=[0, .8, 0, 1],cmap = 'bwr')
axes[3].set_title('Stim. neurons')
axes[3].set_ylabel('Trial #')

a = sta[cns,:,:]
a = a - np.tile(np.mean(a[0:30, :], axis=0), (a.shape[0], 1))
axes[4].imshow(a.T,vmin = -1,vmax = 1, extent=[0, .8, 0, 1],cmap = 'bwr')
axes[4].set_aspect('equal')  # Set the aspect ratio to make the image square
axes[4].set_title('CN')
axes[4].set_ylabel('Trial #')

bins = 40
y = np.mean(sta[cns,11:30,:],axis = 0)
#y = np.mean(sta[stm_cls,11:50,:],axis = 0)
if y.ndim>1:
    y = np.mean(y,axis = 0)
a = np.convolve(y, np.ones((bins,1)).flatten())
a = a[:len(y)]
axes[5].plot(a/bins,'ko-',markerfacecolor='w',markersize=3)
axes[5].set_ylabel('CN response amp.')
axes[5].set_xlabel('Trial #')
axes[5].set_xlim(bins-1,len(y))


for ii in range(2):
    ax = plt.subplot(4,2,5+ii)
    if ii == 1:
        xlim = (30000,32000)
    else:
        xlim = (600,2600)
    plt.plot(t_si[xlim[0]:xlim[1]],np.nanmean(F[stm_cls,xlim[0]:xlim[1]],axis = 0),'k',linewidth = .5)
    y = np.ones((10,))/2;
    xl = plt.xlim()
    ind = np.where((stim_time < xl[1]) & (stim_time > xl[0]))[0]
    st = stim_time[ind]
    yl = (0,4)
    for i in range(len(st)):
        #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
        x = [st[i], st[i]+3, st[i]+3, st[i]]
        y = [yl[0],yl[0],yl[1],yl[1]]
        p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
        ax.add_patch(p)
    plt.ylim(yl)
    plt.title(titles[ii])
    plt.xlabel('Time (s)')
    plt.ylabel('Stim neurons (DF/F)')
    ax = plt.subplot(4,2,7+ii)
    cns = data['conditioned_neuron'][0]
    cns = cns[iscell[cns,0]==1]
    cns = cns[0]
    #plt.plot(t_si[0:4000],np.nanmean(F[cns,0:4000],axis = 0),'k',linewidth = .5)
    plt.plot(t_si[xlim[0]:xlim[1]],F[cns,xlim[0]:xlim[1]],'k',linewidth = .5)
    yl = (-1,8)
    for i in range(len(st)):
        #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
        x = [st[i], st[i]+3, st[i]+3, st[i]]
        y = [yl[0],yl[0],yl[1],yl[1]]
        p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
        ax.add_patch(p)
    plt.ylim(yl)
    plt.title(titles[ii])
    plt.xlabel('Time (s)')
    plt.ylabel('CN (DF/F)')
fig.subplots_adjust(left=0.05, right=0.96, bottom=0.05, top=0.96, wspace=0.6, hspace=0.6)
plt.savefig(folder+data['mouse']+"_"+data['session']+r'BCI_gated_summary.pdf', format='pdf')
plt.show()
