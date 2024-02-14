# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:14:38 2023

@author: kayvon.daie
"""

stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)        
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()


#%%'
F = Ftrace*0
for i in range(F.shape[0]):
    a = Ftrace[i,:]
    bl = np.percentile(a,20)
    F[i,:] = (a-bl)/bl
ind = np.where(iscell[:,0]==1)[0]
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
ten_minutes = np.round((600/dt_si))


plt.subplot(1,2,1)
plt.imshow(ops['meanImg'],cmap = 'gray',vmin = 0,vmax = 300)
plt.gca().axis('off')
plt.title(folder)

plt.subplot(1,2,2)
plt.imshow(F[ind,0:15000], cmap='viridis', aspect='auto',vmin=0,vmax=2)
plt.xticks([0,ten_minutes],['0','10'])
plt.colorbar(shrink=.3)
plt.xlabel('Time (min.)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%
folder = r'G:/My Drive/Learning rules/BCI_data/dataset ai229 & ai228/'
file = r'BCI_58-072323.npy'
data = np.load(folder+file,allow_pickle=True).tolist()
#%%
Ftrace=data['BCI_1']['Ftrace']
F = Ftrace*0
for i in range(F.shape[0]):
    a = Ftrace[i,:]
    bl = np.percentile(a,20)
    F[i,:] = (a-bl)/bl
iscell = data['iscell']
ind = np.where(iscell==1)[0]
dt_si = data['dt_si']
ten_minutes = np.round((600/dt_si))

plt.subplot(1,2,1)
plt.imshow(data['mean_image'],cmap = 'gray',vmin = 0,vmax = 300)
plt.gca().axis('off')
plt.title(folder)

plt.subplot(1,2,2)
plt.imshow(F[ind,0:15000], cmap='viridis', aspect='auto',vmin=0,vmax=1)
plt.xticks([0,ten_minutes],['0','10'])
plt.colorbar(shrink=.3)
plt.xlabel('Time (min.)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()