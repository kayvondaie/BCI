# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:01:49 2023

@author: kayvon.daie
"""
import os;os.chdir(r'H:/My Drive/Python Scripts/BCI_analysis/')
import data_dict_create_module as ddc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import matplotlib.patches as patches


folder = r'E:/BCI49/071123/'
data = ddc.load_data_dict(folder)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
ops = np.load(folder +r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
zaber = np.load(folder + folder[-7:-1]+r'-bpod_zaber.npy',allow_pickle=True).tolist()
Ftrace = data['df_closedloop']
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folde
# Load the .mat file
#%%
F_trial_strt = [];
Fraw_trial_strt = [];
strt = 0;
dff = 0*Ftrace
for i in range(np.shape(Ftrace)[0]):
    bl = np.std(Ftrace[i,:])
    dff[i,:] = (Ftrace[i,:] - bl)/bl
for i in range(len(ops['frames_per_file'])):
    ind = list(range(strt,strt+ops['frames_per_file'][i]))    
    f = dff[:,ind]
    F_trial_strt.append(f)
    f = Ftrace[:,ind]
    Fraw_trial_strt.append(f)
    strt = ind[-1]+1
#%%
xlim = (250,1250)
F = data['df_closedloop']
t_si = np.arange(0,dt_si * np.shape(F)[1],dt_si)
xl = (t_si[xlim[0]],t_si[xlim[1]])
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
cns = data['conditioned_neuron'][0]
cns = cns[iscell[cns,0]==1]
cns = cns[0]
VEL = []
FF = []
REW = []
for trl in range(0,30):
    f = F_trial_strt[trl][cns,:]
    FF.append(f)
    dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
    t_si = np.arange(0,dt_si * np.shape(f)[0],dt_si)
    steps = zaber['zaber_move_forward'][trl];
    #steps = data['step_time']
    vel = np.zeros(len(f),)
    rew = np.zeros(len(f),)
    for i in range(len(steps)):
        if steps[i] < t_si[-1]:
            offset= zaber['scanimage_first_frame_offset'][i]
            ind = np.where(t_si>steps[i]-offset)[0][0]
            vel[ind] = 1
    rewT = zaber['reward_L'][trl]
    ind = np.where(t_si>rewT-offset)[0][0]
    rew[ind] = 1
    REW.append(rew)
    VEL.append((vel))
VEL = np.hstack(VEL)
REW = np.hstack(REW)
FF = np.hstack(FF)
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
F = data['df_closedloop']
t_si = np.arange(0,dt_si * np.shape(F)[1],dt_si)
t_si = t_si[0:len(VEL)]    
F = F[0:len(VEL)]

ax = plt.subplot(3,1,2)
plt.plot(t_si[xlim[0]:xlim[1]],FF[xlim[0]:xlim[1]]+1.5,'k',linewidth = .5)
y = np.ones((10,))/2;
xl = plt.xlim()
ind = np.where((stim_time < xl[1]) & (stim_time > xl[0]))[0]
st = stim_time[ind]
for i in range(len(st)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [st[i], st[i]+3.5, st[i]+3.6, st[i]]
    y = [0, 0, 4, 4]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)
plt.xticks(plt.xticks()[0], [""]*len(plt.xticks()[0]))
plt.xlim(xl)    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = plt.subplot(3,1,3)
plt.plot(t_si[xlim[0]:xlim[1]],-REW[xlim[0]:xlim[1]]/5,'b',linewidth = 1)
plt.plot(t_si[xlim[0]:xlim[1]],VEL[xlim[0]:xlim[1]],'k',linewidth = .5)
for i in range(len(st)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [st[i], st[i]+3.5, st[i]+3.5, st[i]]
    y = [0,0, 1.3, 1.3]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)
plt.xlim(xl)    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
        
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

stim_time= stim_time[stim_time<t_si[-1]]
stm_cls = np.where(stimDist<4)[0]
#ind = b[0:9]
ax = plt.subplot(3,1,1)
plt.plot(t_si[xlim[0]:xlim[1]],np.nanmean(F[stm_cls,xlim[0]:xlim[1]],axis = 0),'k',linewidth = .5)
for i in range(len(st)):
    #plt.plot(stim_time[i] * np.array([1, 1]),np.array([.5,4]),'m')
    x = [st[i], st[i]+3.5, st[i]+3.5, st[i]]
    y = [0, 0, 4, 4]
    p = patches.Polygon(xy=list(zip(x,y)), facecolor='r', alpha=0.2)
    ax.add_patch(p)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(plt.xticks()[0], [""]*len(plt.xticks()[0]))
plt.xlim(xl)    

#%%
fig,ax=plt.subplots()
im = ax.imshow(ops['meanImg'],cmap = 'gray',vmin = 0,vmax = 300)
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
cns = data['conditioned_neuron'][0]
cns = cns[iscell[cns,0]==1][0]
ax.plot(data['centroidX'][cns],data['centroidY'][cns],'ro',markerfacecolor = 'none')

stm_cls = np.where(stimDist<10)[0]
cns = stm_cls[0:12]
cns = np.delete(cns, [9, 10])
for i in range(len(cns)):
    ax.plot(data['centroidX'][cns[i]],data['centroidY'][cns[i]],'mo',markerfacecolor = 'none')

plt.show()

plt.show()
