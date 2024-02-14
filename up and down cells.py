# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:41:50 2024

@author: Kayvon Daie
"""
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
u,s,V = np.linalg.svd(df)
#%%
from scipy.io import savemat
import numpy as np
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
savemat(folder + 'vector.mat', {'vector': vector})
#%%

v = vec/np.linalg.norm(vec)
proj = np.dot(df,v)
proj2 = np.dot(df,V.T)
S = np.var(proj2,axis=0)
plt.semilogy(S)
st = np.var(proj)
plt.plot((0,239),(st,st))
plt.show()




