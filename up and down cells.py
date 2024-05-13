# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:41:50 2024

@author: Kayvon Daie
"""

import os;os.chdir('G:/My Drive/Python Scripts/BCI_analysis/')
import data_dict_create_module as ddc
import numpy as np
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt

folder = r'Z:/2p-raw/BCI69/021624/'
data = ddc.load_data_dict(folder)
f = data['F'];
f = np.nanmean(f,axis = 2)
N = f.shape[1]
for i in range(N):
    bl = np.nanmean(f[0:19,i])
    f[:,i] = f[:,i] - bl   


b = np.argsort(np.nanmean(f[40:80,:],axis=0))[::-1]
num = 10
plt.plot(np.nanmean(f[:,b[1:10:2]],axis=1))
plt.plot(np.nanmean(f[:,b[0:10:2]],axis=1))
plt.show()
df = data['df_closedloop']
iscell = data['iscell']
df = df[iscell[:,0]==1,:].T
f = f[:,iscell[:,0]==1]
#u,s,V = np.linalg.svd(df)
#%%
from scipy.io import savemat

tun = np.nanmean(f[40:80,:],axis=0)
up=0*tun
down=0*tun
up[0:-1:2] = tun[0:-1:2]
down[1:-1:2] = -tun[1:-1:2]
vec = up + down
plt.subplot(411)
plt.plot(np.dot(f,up),'r')
plt.plot(np.dot(f,down),'b')
yl = plt.ylim()
plt.subplot(412)
plt.plot(np.dot(f,vec),'k')
plt.ylim(yl)
plt.subplot(413)
plt.plot(np.dot(df[0:1000,:],up)-50,'r',linewidth = .3)
plt.plot(np.dot(df[0:1000,:],down)+50,'b',linewidth = .3)
yl = plt.ylim()
plt.subplot(414)
plt.plot(np.dot(df[0:1000,:],vec),'k',linewidth = .3)
plt.ylim(yl)
vector = vec
bci69_folder = 'H:/My Drive/Learning rules/BCI_data/BCI_69/'
savemat(bci69_folder +data['mouse']+r'_'+data['session']+ '_vector.mat', {'vector': vector})

#%%

v = vec/np.linalg.norm(vec)
proj = np.dot(df,v)
proj2 = np.dot(df,V.T)
S = np.var(proj2,axis=0)
plt.semilogy(S)
st = np.var(proj)
plt.plot((0,239),(st,st))
plt.show()




