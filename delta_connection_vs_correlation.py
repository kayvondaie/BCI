# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:27:17 2023

@author: kayvon.daie
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:15:42 2023

@author: kayvon.daie
"""
import numpy as np
folder = r'G:/My Drive/Learning rules/BCI_data/dataset ai229 & ai228/'
file = r'BCI_58-072323.npy'
old_file = r'BCI_58-072223.npy'
data = np.load(folder+file,allow_pickle=True).tolist()
old = np.load(folder+old_file,allow_pickle=True).tolist()
#%%
import plotting_functions as pf
favg = data['photostim']['favg']
favgo = old['photostim']['favg']
X = data['photostim']['stimDist']
Xo = old['photostim']['stimDist']
f = np.nanmean(data['BCI_1']['F'],axis = 2);
fo = np.nanmean(old['BCI_1']['F'],axis = 2);
FO = old['BCI_1']['F'];
ko = np.mean(FO[40:80,:,:],axis = 0)
cco = np.corrcoef(ko)
for i in range(f.shape[1]):    
    f[:,i] = f[:,i] - np.nanmean(f[0:19,i])
    fo[:,i] = fo[:,i] - np.nanmean(fo[0:19,i])

delt = np.nanmean(f[39:-1,:],axis = 0) - np.nanmean(fo[39:-1,:],axis = 0)

Y = np.nanmean(favg[9:15,:,:],axis = 0)
Yo = np.nanmean(favgo[9:15,:,:],axis = 0)

stm = np.dot(cco,Y*(X<30))
stmo = np.dot(cco,Yo*(Xo<30))

eff = Y*(X>30)
effo = Yo*(Xo>30)

pf.mean_bin_plot(stm+stmo,eff-effo,6,1,1,'k')
