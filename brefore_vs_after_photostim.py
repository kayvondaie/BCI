# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:23:37 2025

@author: kayvon.daie
"""

stimDist = data['photostim2']['stimDist']
for i in range(4):
    if i == 0:
        ind = np.where((stimDist.flatten()>0)&(stimDist.flatten()<10))
        title = '0 < x < 10 um'
    elif i == 1:
        ind = np.where((stimDist.flatten()>10)&(stimDist.flatten()<30))
        title = '10 < x < 30'
    elif i == 2:
        ind = np.where((stimDist.flatten()>30)&(stimDist.flatten()<50))
        title = '30 < x < 50'
    elif i == 3:
        ind = np.where((stimDist.flatten()>50)&(stimDist.flatten()<100))
        title = '50 < x < 100'
    plt.subplot(2,2,i+1)
    plt.plot(AMP[0].flatten()[ind],AMP[1].flatten()[ind],'k.')
    xl = plt.xlim()
    plt.plot(xl,xl,'r:')
    plt.xlabel('Pre learning')
    plt.ylabel('Post learning')
    plt.title(title)
plt.tight_layout()
            