# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:17:24 2023

@author: kayvon.daie
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:28:51 2023

@author: scanimage
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
fig,ax=plt.subplots()
im = ax.imshow(ops['meanImg'],cmap = 'gray',vmin = 0,vmax = 55)
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
cns = np.where(iscell[:,0]>-1)[0]
#cns = cns[0:100]
for i in range(len(cns)):
    ax.plot(data['centroidX'][cns[i]],data['centroidY'][cns[i]],'ro',markerfacecolor = 'none')

stimPos = data['photostim']['stimPosition']
col = r'mb'
for i in range(20):
    x = data['photostim']['stimPosition'][0][0,i]
    y = data['photostim']['stimPosition'][0][1,i]
    ind = 0;
    if i > 9:
        ind = 1
    str = col[ind] + '.'
    ax.plot(x,y,str,markerfacecolor = 'none')
plt.show()