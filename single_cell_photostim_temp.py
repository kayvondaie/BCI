# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:05:04 2023

@author: kayvon.daie
"""
import numpy as np
import matplotlib.pyplot as plt
import extract_scanimage_metadata
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
AMP = np.zeros((30,30,2))
for I in range(2):
    if I == 0:
        epochstr = 'photostim'
    else:
        epochstr = 'photostim2'
    siHeader = np.load(folder + r'/suite2p_'+ epochstr +'/plane0/siHeader.npy', allow_pickle=True).tolist()    
    Ftrace = np.load(folder +r'/suite2p_'+ epochstr +'/plane0/F.npy', allow_pickle=True)
    ops = np.load(folder +r'/suite2p_'+ epochstr +'/plane0/ops.npy', allow_pickle=True).tolist()
    out = ddc.read_stim_file(folder, r'suite2p_'+ epochstr +'/')
    dt_st = 1 / float(siHeader['metadata']['hPhotostim']['monitoringSampleRate'])
    t_st = np.arange(0,dt_st*len(out['Beam']),dt_st)
    y = np.diff(-out['Y'])
    y = y/np.percentile(y,99)
    ind = np.where(y > .5)[0]
    diff_ind = np.diff(ind)
    ind = ind[np.insert(diff_ind >= 2000, 0, True)]
    data[epochstr]['stim_time'] = t_st[ind]
    
    num = 90000;
    plt.plot(y[0:num])
    k = np.zeros((len(y),1));
    stim_time = data[epochstr]['stim_time']
    k[ind] = 10;
    plt.plot(k[0:num],'ro')
    plt.show()
    
    seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli'];
    list_nums = seq.strip('[]').split();
    seq = [int(num) for num in list_nums]
    seq = np.asarray(seq).T
    seq = np.hstack((seq, seq, seq, seq))
    seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
    seq = seq[seqPos:len(ind)]
    stimDist = data[epochstr]['stimDist']
    cl = np.where(stimDist[:,grp]==np.min(stimDist[:,grp]))[0]
    dt_si = data['dt_si']
    #t_si = np.arange(0, dt_si*(Ftrace.shape[1]), dt_si)
    
    strt = np.zeros((len(ops['tiff_list']),1))
    frm = np.zeros((len(ops['tiff_list']),1))
    frms = np.zeros((len(ops['tiff_list']),1))
    for i in range(len(ops['tiff_list'])):
        file = folder + ops['tiff_list'][i]                
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
        strt[i] = siHeader['description_first_frame']['frameTimestamps_sec']
        frm[i] = siHeader['description_first_frame']['frameNumbers']
        frms[i] = siHeader['shape'][0]
    t = []
    for i in range(len(ops['tiff_list'])):
        t_i = np.arange(0, dt_si*(frms[i]), dt_si) + strt[i]
        t.append(t_i)
    t_si = np.concatenate(t)
    F = 0*Ftrace
    for i in range(Ftrace.shape[0]):
        a = Ftrace[i,:]
        bl = np.percentile(a,20)
        F[i,:] = (a-bl)/bl
    pre = 10;
    post = 20;
    favg = np.zeros((pre+post,Ftrace.shape[0],max(seq)))
    for grp in range(max(seq)):
        st = stim_time[np.where(seq==grp+1)[0]]
        a = np.zeros((pre+post,Ftrace.shape[0],len(st)))
        for tr in range(len(st)):
            ind = np.where(t_si > st[tr])[0]
            if len(ind) > 0:
                ind = ind[0]
                ind = np.arange(ind - pre, ind+ post)
                ind[ind < 0] = 0
                ind[ind>Ftrace.shape[1]-1] = Ftrace.shape[1]-1
                a[:,:,tr] = F[:,ind].T
        favg[:,:,grp] = np.nanmean(a,axis=2)
    stmCls = np.zeros(favg.shape[2])
    for i in range(len(stmCls)):
        stmCls[i] = np.where(stimDist[:,i]==np.min(stimDist[:,i]))[0][0]
        
    amp = favg[:,stmCls.astype(int),:]
    amp = np.nanmean(amp[pre+1:pre+6,:,:],axis=0)-np.nanmean(amp[pre-5:pre-2,:,:],axis=0)
    plt.subplot(1,2,1)
    plt.imshow(amp,vmin=-.5,vmax=.5,cmap='RdBu_r')
    plt.colorbar(aspect=1)
    plt.subplot(1,2,2)
    plt.plot(dist.flatten(),amp.flatten(),'ko',markersize=5,markerfacecolor='none')
    plt.show()
    AMP[:,:,I] = amp
#%%
plt.scatter(AMP[:,:,0].flatten(),AMP[:,:,1].flatten())
plt.plot((-.25,2),[-.25,2],'k')


#%%
y = (AMP[:,:,1]-AMP[:,:,0]).flatten()
x = dist.flatten()
pf.mean_bin_plot(x,y,11,1,1,'k')
#%%
ex = np.ones((30, 30))
np.fill_diagonal(ex, np.nan)
ex_ind = np.where(np.isnan(ex.flatten()))
dx = AMP[:,:,1].flatten()[ex_ind]-AMP[:,:,0].flatten()[ex_ind]

mask = np.eye(30)==0
inp_pre = np.mean(AMP[:,:,0]*mask,axis = 1)
inp_post = np.mean(AMP[:,:,1]*mask,axis = 1)
plt.subplot(1,2,1)
plt.plot(dx,inp_post-inp_pre,'ko')
plt.title('Input strength vs excitability')
out_pre = np.mean(AMP[:,:,0]*mask,axis = 0)
out_post = np.mean(AMP[:,:,1]*mask,axis = 0)
plt.subplot(1,2,2)
plt.plot(dx,out_post-out_pre,'ko')
#pf.mean_bin_plot(dx,out_post-out_pre,3,1,1,'k')
plt.title('Output strength vs excitability')
#%%
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)        
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()

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

bci_stimDist = np.zeros([np.shape(xy)[0],len(stat)])
for i in range(np.shape(xy)[0]):
    for j in range(len(stat)):
        bci_stimDist[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
bci_stimDist = np.min(bci_stimDist,axis=0)
bci_stm_cls = np.where(bci_stimDist < 5)[0]
ind = np.argsort(bci_stimDist[stmCls.astype(int)])
#plt.plot(stimDist[stmCls],dx)
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Black -> White -> Orange
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = "custom_div_cmap"
cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=100)
plt.imshow(AMP[ind[:,None],ind,1]-AMP[ind[:,None],ind,0],vmin=-1,vmax=1,cmap = cm)
#%%
plt.plot(bci_stimDist[stmCls.astype(int)],inp_post,'mo',markerfacecolor = 'w')
plt.plot(bci_stimDist[stmCls.astype(int)],inp_pre,'ko',markerfacecolor = 'none')
plt.show()
#%%
adj_bci_stimDist = bci_stimDist[stmCls.astype(int)]

stm_ind = np.where(adj_bci_stimDist < 5)[0]
non_ind = np.where(adj_bci_stimDist >= 5)[0]
mask_stm = np.eye(len(stm_ind))==0
mask_non = np.eye(len(non_ind))==0
w_pre  = np.zeros((2,2))
w_pre[0,0] = np.mean(AMP[stm_ind[:,None],stm_ind,0]*mask_stm)
w_pre[0,1] = np.mean(AMP[stm_ind[:,None],non_ind,0])
w_pre[1,0] = np.mean(AMP[non_ind[:,None],stm_ind,0])
w_pre[1,1] = np.mean(AMP[non_ind[:,None],non_ind,0]*mask_non)

w_post  = np.zeros((2,2))
w_post[0,0] = np.mean(AMP[stm_ind[:,None],stm_ind,1]*mask_stm)
w_post[0,1] = np.mean(AMP[stm_ind[:,None],non_ind,1])
w_post[1,0] = np.mean(AMP[non_ind[:,None],stm_ind,1])
w_post[1,1] = np.mean(AMP[non_ind[:,None],non_ind,1]*mask_non)

plt.subplot(121)
plt.imshow(w_pre,vmin=-.1,vmax=.1,cmap = cm)
plt.subplot(122)
plt.imshow(w_post,vmin=-.1,vmax=.1,cmap = cm)
#%%
plt.plot(bci_stimDist[stmCls.astype(int)],dx,'ko')
plt.xlabel('Distance from BCI stim neurons')
plt.ylabel('$\Delta$ response to direct photostim')