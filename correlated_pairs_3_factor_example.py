# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:56:27 2025

@author: kayvon.daie
"""

gi = 4;
cl = np.argsort(stimDist[:,gi])[0]
plt.subplot(121)
plt.plot(favg[10:55,cl,gi])

cl = np.where((stimDist[:,gi]>30) & (AMP[1][:,gi]>.15))[0]
plt.subplot(122)
plt.plot(favg2[10:55,cl[0],gi])
plt.plot(favg1[10:55,cl[0],gi])
#%%
for gi in range(4,5):
    try:
        cl = np.argsort(stimDist[:,gi])[0]
        cl_targ = cl.copy()
        plt.subplot(121)
        plt.plot(favg[10:55,cl,gi])
        
        cl = np.where((stimDist[:,gi]>30) & ((AMP[1][:,gi] - AMP[0][:,gi])>.15))[0]
        plt.subplot(122)
        plt.plot(favg2[10:55,cl[0],gi])
        plt.plot(favg1[10:55,cl[0],gi])
        print((AMP[1][:,gi] - AMP[0][:,gi])[cl[0]])
        plt.show()
    except:
        continue
#%%
df = data['df_closedloop']
cc = np.corrcoef(df[cl_targ,:],df[cl[0],:])