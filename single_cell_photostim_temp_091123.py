# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:50:02 2023

@author: kayvon.daie
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

favg = data['photostim']['favg']
favg2 = data['photostim2']['favg']
stimDist = data['photostim']['stimDist']

stmCls = np.argmin(stimDist, axis=0)

amp = np.median(favg[8:10,stmCls,:],axis = 0) - np.median(favg[3:5,stmCls,:],axis = 0)
amp2 = np.median(favg2[8:10,stmCls,:],axis = 0) - np.median(favg2[3:5,stmCls,:],axis = 0)

plt.subplot(1,2,1)
plt.imshow(amp,vmin = 0,vmax = .2)
plt.subplot(1,2,2)
plt.imshow(amp2,vmin = 0,vmax = .2)

#%%
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)        
#Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
#ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()

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
coordinates = photostim_groups[0]['rois'][1]['scanfields']['slmPattern']
xy = np.asarray(coordinates)[:,:2] + photostim_groups[0]['rois'][1]['scanfields']['centerXY']
stimPos = np.zeros(np.shape(xy))
for i in range(np.shape(xy)[0]):
    print(i)
    stimPos[i,:] = np.array(xy[i,:]-num[0])*pixPerDeg

bci_stimDist = np.zeros([np.shape(xy)[0],len(stat)])
for i in range(np.shape(xy)[0]):
    for j in range(len(stat)):
        bci_stimDist[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
bci_stimDist = np.min(bci_stimDist,axis=0)

plt.scatter(np.diag(amp),np.diag(amp2))
xlims = plt.xlim()
plt.plot(plt.xlim(),plt.xlim(), '--', color='gray')

non = amp - np.diag(np.diag(amp))
non2 = amp2 - np.diag(np.diag(amp2))
plt.show()
plt.subplot(121)

plt.plot(bci_stimDist[stmCls],np.diag(amp2)-np.diag(amp),'o')
plt.subplot(122)
non = np.sum(non,axis = 1)
non2 = np.sum(non2,axis =1)
plt.plot(bci_stimDist[stmCls],non2-non,'o')
#%%
plt.subplot(121)
pf.mean_bin_plot(bci_stimDist[stmCls],np.abs(non),2,1,1,'k')
pf.mean_bin_plot(bci_stimDist[stmCls],np.abs(non2),2,1,1,'m')
plt.subplot(122)
pf.mean_bin_plot(bci_stimDist[stmCls],np.abs(np.diag(amp)),2,1,1,'k')
pf.mean_bin_plot(bci_stimDist[stmCls],np.abs(np.diag(amp2)),2,1,1,'m')
plt.show()
#%%
dt_si = data['dt_si']
sd = data['photostim']['stimDist']
psp = np.zeros((favg.shape[0],10))
psp2 = np.zeros((favg.shape[0],10))
direct = np.zeros((favg.shape[0],10))
direct2 = np.zeros((favg.shape[0],10))
t = np.arange(0, dt_si*(12), dt_si)
t = t - t[3]
for grp in range(10):
    cl = np.where(sd[:,grp] == min(sd[:,grp]))[0];print(sd[cl,grp])
    ind = np.arange(0,10)
    ind = [x for x in ind if x != grp]
    d = sd[stmCls[ind],grp]  
    ind = [value for i, value in enumerate(ind) if d[i] > 30]
    print(len(ind))
    psp[:,grp]= (np.nanmean(favg[:,stmCls[ind],grp],axis = 1))
    psp2[:,grp]= (np.nanmean(favg2[:,stmCls[ind],grp],axis = 1))
    direct[:,grp] = favg[:,stmCls[grp],grp]
    direct2[:,grp] = favg2[:,stmCls[grp],grp]
psp2 = np.mean(psp2,axis=1)
psp2 = psp2 - np.mean(psp2[3:5])
psp = np.mean(psp,axis=1)
psp = psp - np.mean(psp[3:5])
direct = np.mean(direct,axis=1)
direct2 = np.mean(direct2,axis=1)
direct = direct - np.mean(direct[3:5])
direct2 = direct2 - np.mean(direct2[3:5])
plt.subplot(122)
plt.plot(t,psp[3:15],'k')
plt.plot(t,psp2[3:15],'m')
plt.subplot(121)
plt.plot(t,direct[3:15],'k')
plt.plot(t,direct2[3:15],'m')
#%%
sd = data['photostim']['stimDist']
psp = np.zeros((favg.shape[0],10))
psp2 = np.zeros((favg.shape[0],10))
direct = np.zeros((favg.shape[0],10))
direct2 = np.zeros((favg.shape[0],10))
t = np.arange(0, dt_si*(12), dt_si)
t = t - t[3]
for grp in range(10,20):
    cl = np.where(sd[:,grp] == min(sd[:,grp]))[0];print(sd[cl,grp])
    ind = np.arange(10,20)
    ind = [x for x in ind if x != grp]
    d = sd[stmCls[ind],grp]   
    ind = [value for i, value in enumerate(ind) if d[i] > 50]
    psp[:,grp-10]= (np.nanmean(favg[:,stmCls[ind],grp],axis = 1))
    psp2[:,grp-10]= (np.nanmean(favg2[:,stmCls[ind],grp],axis = 1))
    direct[:,grp-10] = favg[:,stmCls[grp],grp]
    direct2[:,grp-10] = favg2[:,stmCls[grp],grp]
psp2 = np.mean(psp2,axis=1)
psp2 = psp2 - np.mean(psp2[3:5])
psp = np.mean(psp,axis=1)
psp = psp - np.mean(psp[3:5])
direct = np.mean(direct,axis=1)
direct2 = np.mean(direct2,axis=1)
direct = direct - np.mean(direct[3:5])
direct2 = direct2 - np.mean(direct2[3:5])
plt.subplot(122)
plt.plot(t,psp[3:15],'k')
plt.plot(t,psp2[3:15],'m')
plt.subplot(121)
plt.plot(t,direct[3:15],'k')
plt.plot(t,direct2[3:15],'m')
#%%
sd = data['photostim']['stimDist']
psp = np.zeros((favg.shape[0],10))
psp2 = np.zeros((favg.shape[0],10))
direct = np.zeros((favg.shape[0],10))
direct2 = np.zeros((favg.shape[0],10))
t = np.arange(0, dt_si*(12), dt_si)
t = t - t[3]
for grp in range(10,20):
    cl = np.where(sd[:,grp] == min(sd[:,grp]))[0];print(sd[cl,grp])
    ind = np.arange(0,10)
    ind = [x for x in ind if x != grp]
    d = sd[stmCls[ind],grp]    
    ind = [value for i, value in enumerate(ind) if d[i] > 50]
    psp[:,grp-10]= (np.nanmean(favg[:,stmCls[ind],grp],axis = 1))
    psp2[:,grp-10]= (np.nanmean(favg2[:,stmCls[ind],grp],axis = 1))
    direct[:,grp-10] = favg[:,stmCls[grp],grp]
    direct2[:,grp-10] = favg2[:,stmCls[grp],grp]
psp2 = np.mean(psp2,axis=1)
psp2 = psp2 - np.mean(psp2[3:5])
psp = np.mean(psp,axis=1)
psp = psp - np.mean(psp[3:5])
direct = np.mean(direct,axis=1)
direct2 = np.mean(direct2,axis=1)
direct = direct - np.mean(direct[3:5])
direct2 = direct2 - np.mean(direct2[3:5])
plt.subplot(122)
plt.plot(t,psp[3:15],'k')
plt.plot(t,psp2[3:15],'m')
plt.subplot(121)
plt.plot(t,direct[3:15],'k')
plt.plot(t,direct2[3:15],'m')
#%%
non = amp - np.diag(np.diag(amp))
non2 = amp2 - np.diag(np.diag(amp2))
plt.subplot(1,3,1)
plt.imshow(non[0:10,0:10],vmin = 0,vmax = 1,cmap = 'Reds')
plt.subplot(1,3,2)
plt.imshow(non2[0:10,0:10],vmin = 0,vmax = 1,cmap='Reds')
plt.subplot(1,3,3)
d = non2[0:10,0:10]-non[0:10,0:10]
b = np.argsort(np.sum(d, axis=0))
a = np.diag(amp2)-np.diag(amp)
a = a[0:10]
b = np.argsort(a)
d = d[b, :][:, b]
plt.imshow(d,vmin = -.5,vmax = .5,cmap='RdBu_r')
#%%
import plotting_functions as pf
plt.subplot(121)
x = stimDist[:,0:10].reshape(-1,1)
y = np.nanmean(favg2[5:11:,:,0:10],axis=0).reshape(-1,1)-np.nanmean(favg[5:11,:,0:10],axis = 0).reshape(-1,1);
pf.mean_bin_plot(x[0:1500],y[0:1500],5,1,1,'k')
plt.ylim([-.03,.03])
plt.subplot(122)
x = stimDist[:,10:20].reshape(-1,1)
y = np.nanmean(favg2[5:11:,:,10:20],axis=0).reshape(-1,1)-np.nanmean(favg[5:11,:,10:20],axis = 0).reshape(-1,1);
pf.mean_bin_plot(x[0:1500],y[0:1500],5,1,1,'k')
plt.ylim([-.03,.03])
#%%
dt_si = data['dt_si']
F = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)

t_si = np.arange(0, dt_si*(F.shape[1]), dt_si)
num =2000
for i in range(20):
    a = F[stmCls[i],:]
    bl = np.percentile(a,20)
    a = (a-bl)/bl
    if i < 10:
        plt.plot(t_si[0:num],a[0:num]-i*3,linewidth = .5)
    else:
        plt.plot(t_si[0:num],a[0:num]-i*3,color = 'gray',linewidth = .5)
plt.show()
#%%
for j in range(20):
    for i in range(20):
        plt.subplot(20,20,i+1 + 20*j)
        plt.plot(favg[0:16,stmCls[j],i],'k')
        if i == j:
            plt.ylim(-1,5)
        else:
            plt.ylim(-1,1)
        plt.axis('off')
#%%
for j in range(10):
    for i in range(10):
        plt.subplot(10,10,i+1 + 10*j)
        plt.plot(favg2[0:16,stmCls[j],i],'k')
        if i == j:
            plt.ylim(-1,5)
        else:
            plt.ylim(-1,1)
        plt.axis('off')

