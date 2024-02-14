# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:43:02 2023

@author: scanimage
"""

import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
data = np.load('F:/BCI48/032923/data_BCI48_032923.npy',allow_pickle=True).tolist()
df = data['df_closedloop'];
#%%
u,s,v = np.linalg.svd(df.T)
fa = FactorAnalysis(n_components=20)  # Specify the number of factors (components) you want to extract
fa.fit(df.T)  # Replace X with your data
loadings = fa.components_
transformed_data = fa.transform(df.T)  # Replace X with your data

vr_fa = np.var((loadings@df).T,axis=0)
vr_svd = np.var(v@df,axis=1)
#%%
plt.semilogy(vr_svd[0:22],'o-')
plt.semilogy(np.sqrt(vr_fa[0:22]),'o-')
plt.show()
#%%
F = np.nanmean(data['F'],axis=2)
N = F.shape[1]
for i in range(N):
    F[:,i] = F[:,i] - np.nanmean(F[0:19,i])

#%%
proj_svd = (F@v.T);
