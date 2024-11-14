# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:01:20 2024

@author: kayvon.daie
"""


favg = data['photostim']['favg_raw']
amp = np.nanmean(favg[11:15,:,:],axis = 0)-np.nanmean(favg[0:4,:,:],axis = 0);
#amp[np.abs(amp)>2] = 0
slmDist[cl,gi]
stimDist = data['photostim']['stimDist']
x = []
y = []
for gi in range(100):
    cl = np.where(stimDist[:,gi]<10)[0]
    x.append(slmDist[cl,gi])
    y.append(amp[cl,gi])
x = np.concatenate(x);
y = np.concatenate(y);
import plotting_functions as pf
plt.subplot(211)
plt.plot(stimDist.flatten(),amp.flatten(),'k.')
plt.subplot(212)
pf.mean_bin_plot(x, y,13,1,1,'k')
plt.xlabel('Dist from SLM center')
plt.ylabel('Response (DF/F) of target cell')

