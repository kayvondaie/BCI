# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:02:06 2024

@author: kayvon.daie
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
dt = 0.05
tau = 0.05
t=np.arange(0,2+dt,dt)
tonic = np.ones((len(t),))
tonic[0:5] = 0

ra = np.zeros((len(t),))
ra_dot = np.zeros((len(t),))
rs = np.zeros((len(t),))
pert_alm = np.zeros((len(t),))
inp = np.zeros((len(t),));inp[4:5] = 10
tt = 1


for i in range(1,len(tonic)):
    rs[i] = rs[i-1] + dt/tau*(tonic[i] + ra_dot[i-1]*1)
    ra[i] = ra[i-1] + dt/tau*(-ra[i-1] + inp[i]*0 + pert_alm[i] + rs[i])
    if tt == 1:
        if (i>10) & (i<15):
            ra[i] = 0
    ra_dot[i] = ra[i] - ra[i-1]
    if ra_dot[i]>0:
        ra_dot[i] = 0

plt.subplot(3,1,1)
plt.plot(rs)
plt.subplot(3,1,2)
plt.plot(ra)
plt.subplot(3,1,3)
plt.plot(ra_dot)
    