# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:17:30 2023

@author: scanimage
"""
stm_cls = list()
for si in range(2):
    photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups'][si]
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
    stm_cls.append(np.where(stimDist<10)[0])
    #ind = b[0:9]

#%%

    
pre = 20;
post = 120;
sta = np.zeros((np.shape(F)[0],pre+post,200,2))
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
F = data['df_closedloop']
t_si = np.arange(0,dt_si * np.shape(F)[1],dt_si)
for gi in range(2):
    a = spio.loadmat(folder + '/BCI_params_py.mat')
    stim_time = a['stim_time'][0]
    stim_group = a['stim_group'][0]
    #stim_time = stim_time*3600*24;
    #stim_time = stim_time[1:-1]
    #stim_time = stim_time - stim_time[0];
    stim_time = stim_time+.8
    stim_time = stim_time[~np.isnan(stim_time)]
    stim_time = stim_time[stim_group==gi+1]
    stim_time= stim_time[stim_time<t_si[-1]]

    for i in range(len(stim_time)):    
        ind = np.where(t_si>stim_time[i]-i*.004)[0][0]    
        ind = np.array(range(ind-pre,ind+post));
        ind[ind<0] = 0;
        ind[ind>np.shape(F)[1]-1] = np.shape(F)[1]-1;
        sta[:,:,i,gi] = F[:,ind]
    
    for i in range(len(stim_time)):
        for j in range(sta.shape[0]):
            a = sta[j,:,i,gi]
            bl = np.nanmean(a[0:pre-10])
            sta[j,:,i,gi] = a - bl


#%%
early = (0,10)
late = (10,40)
for ei in range(2):
    if ei == 0:
        a = early
    else:
        a = late        
    plt.subplot(2,2,1)
    plt.plot(np.nanmean(np.nanmean(sta[stm_cls[0],:,a[0]:a[1],0],axis=2),axis=0))
    plt.subplot(2,2,2)
    plt.plot(np.nanmean(np.nanmean(sta[stm_cls[0],:,a[0]:a[1],1],axis=2),axis=0))
    plt.subplot(2,2,3)
    plt.plot(np.nanmean(np.nanmean(sta[stm_cls[1],:,a[0]:a[1],0],axis=2),axis=0))
    plt.subplot(2,2,4)
    plt.plot(np.nanmean(np.nanmean(sta[stm_cls[1],:,a[0]:a[1],1],axis=2),axis=0))
plt.show()