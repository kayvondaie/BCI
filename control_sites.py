# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:17:15 2025

@author: kayvon.daie
"""

stimCells = np.argmin(stimDist,axis=0)
stimAmp = np.diag(amp[stimCells,:])
amp = AMP[1]

bins = np.concatenate((np.arange(0, 100, 6), np.arange(100, 300, 25)))
for ei in range(2):
    if ei == 0:
        ind = np.where(stimAmp>1)[0]
    elif ei == 1:
        ind = np.where(stimAmp<0)[0]
    A = np.zeros((len(bins),))
    dist = np.zeros((len(bins),))
    for i in range(len(bins)-1):
        indd = np.where((stimDist[:,ind].flatten() > bins[i]) & (stimDist[:,ind].flatten().flatten()<bins[i+1]))[0]
        A[i] = np.nanmean(amp[:,ind].flatten()[indd])
        dist[i] = np.nanmean(stimDist[:,ind].flatten()[indd])
    plt.plot(dist[:-1],A[:-1],'.-')
plt.plot((30,30),plt.ylim(),'k:')
plt.plot(plt.xlim(),(0,0),'k:')