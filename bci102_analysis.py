# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:19:44 2024

@author: kayvon.daie
"""

old_folder = r'//allen/aind/scratch/BCI/2p-raw/BCI102/120424/pophys/'
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI102/120524/pophys/'
ps_old = ddc.load_data_dict(old_folder, subset='photostim')
ps = ddc.load_data_dict(folder, subset='photostim')
#%%
data = dict()
old = dict()
data['photostim'] = ps;
old['photostim'] = ps_old
stimDist, amp, favg = stim.stim_amp(data)
stimDisto, ampo, favgo = stim.stim_amp(old)
#%%
siHeader = np.load(folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist()
siHeadero = np.load(old_folder + r'/suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).tolist()
rois = siHeader['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']
roiso = siHeadero['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']


N = len(rois)
xyo = np.zeros((N,2));
roinum = np.zeros(N,)
for i in range(N):
    xy[i,:] = rois[i]['scanfields']['centerXY']
    name = rois[i]['name']
    try:
        ind = np.where(np.array(list(name))=='_')[0][-1]
        roinum[i] = float(name[ind+1:])
    except:
        ind = np.where(np.array(list(name))==' ')[0][-1]
        roinum[i] = float(name[ind+1:])

N = len(roiso)
xyo = np.zeros((N,2));
roinumo = np.zeros(N,)
for i in range(N):
    xyo[i,:] = roiso[i]['scanfields']['centerXY']
    name = roiso[i]['name']
    ind = np.where(np.array(list(name))=='_')[0][-1]
    roinumo[i] = float(name[ind+1:])
#%%
ng = amp.shape[0]
x = np.zeros(ng)
for ci in range(ng):
    