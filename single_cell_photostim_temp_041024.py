# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:41:54 2024

@author: Kayvon Daie
"""

ci = 12;
gi = np.where(stimDist[ci,:] == np.min(stimDist[ci,:]))[0];
ind = np.where(seq==gi+1)

stim_trace_for_group = np.zeros((1,Ftrace.shape[1]))[0]
stim_trace_for_group[ind[0]] = 1

plt.plot(Ftrace[ci,:])
plt.plot(stim_trace_for_group*100)
plt.xlim((0,1000))
plt.show()
