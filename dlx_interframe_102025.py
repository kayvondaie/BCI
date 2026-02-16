# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:33:48 2025

@author: kayvon.daie
"""

favg = data['photostim']['favg']
gi = 10;
gii = np.where(targs == targs[gi])[0]
stimDist = data['photostim']['stimDist']
targs = np.argmin(stimDist,0)
plt.subplot(121);
plt.plot(favg[:,targs[gi],gii[0]])
plt.plot(favg[:,targs[gi],gii[1]])
#%%
from scipy.signal import medfilt
f = favg[0:70,:,gi]
pop = np.where(np.var(f,0) < 100)[0]
f = f[:,pop]
forig = f.copy()
f[np.isnan(f)] = 0
win = 11;
for i in range(f.shape[1]):
    f[:,i] = medfilt(f[:,i],win)
f[:,targs[gi]] = 0
u,s,v = np.linalg.svd(f)
v = v.T;
vp = v[:,1] * (v[:,1]>0)
vn = -v[:,1] * (v[:,1]<0)
plt.plot(f @ vp,'b.-')
plt.plot(f @ vn,'r.-')



#%%
b = np.argsort(v[:,1]);
i = 4
plt.plot(f[0:,b[i]])
plt.plot(forig[0:,b[i]])



#%%
from scipy.signal import medfilt
favg = data['photostim']['favg']
stimDist = data['photostim']['stimDist']
targs = np.argmin(stimDist,0)
F = []
for gi in range(favg.shape[2]):
    
    f = favg[0:70,:,gi]
    f[:,targs[gi]] = 0
    pop = np.where(np.var(f,0) < 100)[0]
    f = f[:,pop]
    forig = f.copy()
    f[np.isnan(f)] = 0
    for i in range(f.shape[1]):
        f[:,i] = medfilt(f[:,i],5)
    u,s,v = np.linalg.svd(f)
    v = v.T;
    F.append(f @ v[:,1])
    F[gi] = -F[gi] * np.sign(F[gi][15])
#%%
ff = np.stack(F)
plt.plot(np.nanmean(ff[0:40,:],0))
plt.plot(np.nanmean(ff[40:,:],0))
plt.legend(('Long','Short'))
#%%
A = []
for gi in range(stimDist.shape[1]):
    ind = np.where((stimDist[:,gi]>20)&(stimDist[:,gi]<100))[0];
    a = np.nanmedian(favg[:,ind,gi],1);
    A.append(a)
A = np.stack(A)
plt.plot(np.nanmedian(A[0:40,0:50],0));plt.plot(np.nanmedian(A[41:,0:50],0))
#%%
A = []
for gi in range(stimDist.shape[1]):
    A.append(favg[:,targs[gi],gi])
A = np.stack(A)
plt.plot(np.nanmean(A[0:40,0:50],0));
plt.plot(np.nanmean(A[41:,0:50],0))    