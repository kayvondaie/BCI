# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:22:37 2024

@author: kayvon.daie
"""
import numpy as np
import data_dict_create_module as ddc
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI94/090524/'
bci1 = np.load(folder +r'/suite2p_BCI2/plane0/F.npy', allow_pickle=True)
bci2 = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
data = ddc.load_data_dict(folder)
cn = data['conditioned_neuron'][0][0]
plt.subplot(121)
plt.plot(bci1[cn,:],linewidth=.3)
plt.subplot(122)
plt.plot(bci2[cn,:],linewidth=.3)
#%%
fcn = np.concatenate((bci1[cn,:],bci2[cn,:]))
#plt.plot(fcn,linewidth=.3)
ker = np.ones((10,))
k = np.convolve(fcn,ker)
plt.plot(k,linewidth=.3)
plt.xlim(10,8000)
#%%
bci1 = np.load(folder +r'/suite2p_BCI2/plane0/F.npy', allow_pickle=True)
ops1 = np.load(folder + r'/suite2p_BCI2/plane0/ops.npy', allow_pickle=True).tolist()
stat1 = np.load(folder + r'/suite2p_BCI2/plane0/stat.npy', allow_pickle=True)
data1 = dict()
data1['F'], data1['Fraw'],data1['df_closedloop'],data1['centroidX'],data1['centroidY'] = ddc.create_BCI_F(bci1,ops1,stat1);
#%%
F1 = data1['F']
F2 = data['F']
fcn1 = np.nanmean(F1[:,cn,:],axis=1)
fcn2 = np.nanmean(F2[:,cn,0:20],axis=1)
plt.plot(fcn1,'k',label='low thr epoch')
plt.plot(fcn2,'m',label='high thr epoch')
plt.legend()
plt.show()
#%%
volt1 = np.nanmean(F1[0:,cn,:],axis=0);
volt2 = np.nanmean(F2[0:,cn,:],axis=0);
volt = np.concatenate((volt1.T, volt2.T))
plt.plot(volt,'o')
plt.plot([len(volt1)] * 2, plt.ylim(), 'k-')
bins = 10
ker = np.ones((bins,))
k = np.convolve(volt,ker/bins)
plt.plot(k[bins:],'k',linewidth=.3)
#%%
import numpy as np

def fun(x, BCI_threshold):
    return (x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)[0]) * 3.3

