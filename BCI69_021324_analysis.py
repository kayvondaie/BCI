 -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os;os.chdir('G:/My Drive/Python Scripts/BCI_analysis/')
import data_dict_create_module as ddc
import numpy as np
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

folder = r'Z:/2p-raw/BCI69/021324/'
old_folder = r'Z:/2p-raw/BCI69/020924/'
data = ddc.load_data_dict(folder)
data_old = ddc.load_data_dict(old_folder)
cns = np.where(data['iscell'][:,0]==1)[0]
df = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
df = df[cns,:].T
dfo = np.load(old_folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
dfo = dfo[cns,:].T
mat_contents = loadmat(folder + r'vector.mat')
vector = mat_contents['vector'][0]
#%%

#df = data['df_closedloop'][cns,:].T
F = data_old['Fraw'][:,cns,0:];
f = np.nanmean(F,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:39,i])
    f[:,i] = f[:,i] - bl    
up = vector*0
down = vector*0
up[0::2] = vector[0::2]
down[1::2] = vector[1::2]
plt.subplot(311)
plt.plot(np.dot(f,up))
plt.plot(-np.dot(f,down))
plt.subplot(312)
plt.plot(np.dot(f,vector))
plt.subplot(313)
plt.plot(np.dot(df[0:5000,:],up),'r',linewidth = .2)
plt.plot(-np.dot(df[0:5000,:],down),'b',linewidth = .2)
#%%

trl = 122
plt.plot(np.dot(F[:,:,trl],up),'r')
plt.plot(-np.dot(F[:,:,trl],down),'b')
