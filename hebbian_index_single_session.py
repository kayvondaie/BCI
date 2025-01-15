# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:11:41 2025

@author: kayvon.daie
"""

#%%
plt.figure(figsize=(8,4))  # Set figure size to 10x10 inches
df = data['df_closedloop']
cc = np.corrcoef(df)
F = data['F']
ko = np.nanmean(F[120:200,:,0:10],axis=0)
k = np.nanmean(F[120:200,:,0:],axis=0)
cc = np.corrcoef(k)
cco = np.corrcoef(ko)
ei = 1;
X = []
Y = []
Yo = []
for gi in range(stimDist.shape[1]):
    cl = np.where((stimDist[:,gi]<30) & (AMP[ei][:,gi]> .3))[0]
    #plt.plot(favg[0:80,cl,gi])
    
    x = np.nanmean(cc[cl,:],axis=0)    
    nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<1000))
    y = AMP[1][nontarg,gi]
    yo = AMP[0][nontarg,gi]
    #plt.scatter(x[nontarg],amp[nontarg,gi])
    X.append(x[nontarg])
    Y.append(y)
    Yo.append(yo)

X = np.concatenate(X)
Y = np.concatenate(Y,axis=1)
Yo = np.concatenate(Yo,axis=1)
plt.subplot(221)
pf.mean_bin_plot(X,Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('Before learning')

plt.subplot(223)
pf.mean_bin_plot(X,Y,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('After learning')

plt.subplot(122)
pf.mean_bin_plot(X,Y-Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()

