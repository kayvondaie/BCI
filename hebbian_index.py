import suite2p
import os;os.chdir('G:/My Drive/Python Scripts/BCI_analysis/')
import re
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import copy
import shutil
import data_dict_create_module as ddc

folders = []
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI88/091824/')
folders.append(r'//allen/aind/scratch/BCI/2p-raw/BCI88/091924/')
old = ddc.load_data_dict(folders[0])
new = ddc.load_data_dict(folders[1])
Ftrace = np.load(folders[0] +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
#%%
SD = []
AMP = []
for i in range(2):
    if i == 0:
        data = old
    else:
        data = new
    stimDist = data['photostim']['stimDist']
    bl = np.nanstd(Ftrace,axis = 1)
    f = data['photostim']['favg_raw']
    ff = f*0
    N = f.shape[1]
    for i in range(N):
        a = f[:,i,:]
        a = (a-bl[i])/bl[i]
        ff[:,i,:] = a
    amp = np.nanmean(ff[12:30,:,:],axis =0) - np.nanmean(ff[0:4,:,:],axis =0)
    plt.plot(stimDist.flatten(),amp.flatten(),'k.',markersize = .5)   
    plt.show()
    bins = []
    bins.append((0,20));bins.append((30,80));bins.append((80,200));bins.append((200,10000));  
    for i in range(len(bins)):
        plt.subplot(2,2,i+1)
        A = []
        for j in range(ff.shape[2]):
            ind = np.where((stimDist[:,j]>bins[i][0])&(stimDist[:,j]<bins[i][1]))[0]
            A.append(np.nanmean(ff[:,ind,j],axis=1))
        A = np.asarray(A)        
        plt.plot(np.nanmean(A[:,0:30],axis=0))
    SD.append(stimDist)
    AMP.append(amp)
#%%
import plotting_functions as pf
sd_new = SD[1].flatten()
sd_old = SD[0].flatten()
amp_new = AMP[1].flatten()
amp_old = AMP[0].flatten()
bins = []
bins.append((0,20));bins.append((30,80));bins.append((80,200));bins.append((200,400));  

for i in range(len(bins)):
    plt.subplot(2,2,i+1)    
    ind = np.where((sd_new>bins[i][0])&(sd_new<bins[i][1]))[0]
    plt.plot(amp_old[ind],amp_new[ind],'k.',markersize=.5)
    #pf.mean_bin_plot(amp_old[ind],amp_new[ind],5,1,1,'k')
#%%
df_new = new['df_closedloop']
df_old = old['df_closedloop']
cc = np.corrcoef(df_new)
cco = np.corrcoef(df_old)

F = new['F']
Fo = old['F']
k = np.nanmean(F[60:120,:,:],axis= 0)
ko = np.nanmean(Fo[60:120,:,:],axis=0)
cc = np.corrcoef(k)
cco = np.corrcoef(ko)

X = []
Y = []
Yo = []
Xo = []
for i in range(SD[0].shape[1]):
    stm_cl = np.argsort(SD[0][:,i])[0]
    stm_cl = np.where(SD[0][:,i]<10)[0]
    ind = np.where(SD[0][:,i]>15)[0]
    
    cc_sub = cco[np.ix_(ind, stm_cl)]
    amp_sub = AMP[1][stm_cl,i]
    x = np.dot(cc_sub,amp_sub)
    X.append(x.T)
    
    
    cc_sub = cco[np.ix_(ind, stm_cl)]
    amp_sub = AMP[0][stm_cl,i]
    xo = np.dot(cc_sub,amp_sub)
    Xo.append(xo.T)
    
    ind = np.where(SD[0][:,i]>15)
    Y.append(AMP[1][ind,i])
    Yo.append(AMP[0][ind,i])
Y = np.concatenate(Y,axis = 1)
X = np.concatenate(X,axis = 0)
Xo = np.concatenate(Xo,axis = 0)
Yo = np.concatenate(Yo,axis = 1)

num_bins = 9;
plt.subplot(131)
pf.mean_bin_plot(Xo,Yo,num_bins,1,1,'k')
plt.subplot(132)
pf.mean_bin_plot(X,Y,num_bins,1,1,'k')
plt.subplot(133)
pf.mean_bin_plot(X+Xo,Y-Yo,num_bins,1,1,'k')
plt.tight_layout()