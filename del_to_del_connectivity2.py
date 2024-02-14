# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:15:42 2023

@author: kayvon.daie
"""

folder = r'G:/My Drive/Learning rules/BCI_data/dataset ai229 & ai228/'
file = r'BCI_58-072323.npy'
old_file = r'BCI_58-072223.npy'
data = np.load(folder+file,allow_pickle=True).tolist()
old = np.load(folder+old_file,allow_pickle=True).tolist()
#%%
import plotting_functions as pf
favg = data['photostim']['favg']
favgo = old['photostim']['favg']
stimDist = data['photostim']['stimDist']
f = np.nanmean(data['BCI_1']['F'],axis = 2);
fo = np.nanmean(old['BCI_1']['F'],axis = 2);
for i in range(f.shape[1]):    
    f[:,i] = f[:,i] - np.nanmean(f[0:19,i])
    fo[:,i] = fo[:,i] - np.nanmean(fo[0:19,i])

delt = np.nanmean(f[39:-1,:],axis = 0) - np.nanmean(fo[39:-1,:],axis = 0)

Y = np.nanmean(favg[9:15,:,:],axis = 0)
Yo = np.nanmean(favgo[9:15,:,:],axis = 0)

X = np.zeros((1,Y.shape[1]))[0]
Xo = np.zeros((1,Y.shape[1]))[0]
C = np.zeros((1,Y.shape[1]))[0]
Co = np.zeros((1,Y.shape[1]))[0]
for i in range(Y.shape[1]):
    near = np.where(stimDist[:,i]<30)
    far = np.where((stimDist[:,i]>30)&(stimDist[:,i]<100))
    X[i] = np.dot(Y[near,i].flatten(),delt[near])
    Xo[i] = np.dot(Yo[near,i].flatten(),delt[near])
    
    C[i] = np.dot(Y[far,i].flatten(),delt[far])
    Co[i] = np.dot(Yo[far,i].flatten(),delt[far])

pf.mean_bin_plot(X+Xo, C-Co, 5, 1, 1, 'k')