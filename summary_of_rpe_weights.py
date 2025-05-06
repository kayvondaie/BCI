# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:23:20 2025

@author: kayvon.daie
"""

hi = HI.copy()
weight = np.asarray(FIT)
weight = weight / max(weight)
for i in range(len(HI)):
    hi[i] = hi[i] * 1
ind = np.where(weight > -133331)[0]    
hi = np.asarray(hi)
hi[np.isnan(hi)]=0
k = np.nanmean(np.nanmean(hi,axis=2),axis=1)
good = np.where((np.abs(np.asarray(DW))<.2) & (np.abs(k)<.00015))[0]
hi = hi[good,:,:]
hi_avg = np.nanmedian(hi,axis = 0)
hi_z = hi_avg / np.nanstd(hi,axis = 0)
sns.heatmap(hi_z*1000, annot=True, xticklabels=['rt', 'rt_rpe', 'hit', 'hit_rpe', 'miss', 'miss_rpe'], yticklabels=['step', 'reward', 'go cue'], cmap='coolwarm', center=0)
#%%
from scipy.stats import ttest_1samp
import numpy as np
betas = hi.copy()
nsessions, nrows, ncols = betas.shape
weight = np.asarray(FIT)
# store p-values
pvals = np.zeros((nrows, ncols))
ind = np.where(weight > .0)[0]
for i in range(nrows):
    for j in range(ncols):
        this_beta = betas[ind, i, j]   # slice across sessions
        tstat, pval = ttest_1samp(this_beta, 0, nan_policy='omit')
        pvals[i, j] = pval

# pvals[i,j] gives the p-value for (i,j) testing mean(beta) ≠ 0 across sessions
sns.heatmap(-np.log(pvals), annot=True, xticklabels=['rt', 'rt_rpe', 'hit', 'hit_rpe', 'miss', 'miss_rpe'], yticklabels=['step', 'reward', 'go cue'], cmap='coolwarm', center=0)
#%%


# Compute correlation matrix between beta types
corr_matrix = np.corrcoef(betas_flat.T)

import matplotlib.pyplot as plt

plt.imshow(corr_matrix, vmin=-1, vmax=1)
plt.colorbar(label='correlation')
plt.title('Correlation between beta weights')
plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
betas = hi.copy()
betas_flat = betas.reshape(betas.shape[0], -1)   # shape (36 sessions, 18 betas)
# --- Your data ---
u, s, vt = np.linalg.svd(betas_flat, full_matrices=False)
v = vt.T  # shape (18 features, 18 modes)
v[:,0] = v[:,0] * np.sign(np.nanmean(u[:,0]))
# First mode
mode1 = v[:,0]  # 18-length vector

# Reshape into (3,6)
mode1_reshaped = mode1.reshape(3,6)

# --- Create figure ---
fig = plt.figure(figsize=(6,8))

# Top plot (small cumulative variance plot)
ax1 = fig.add_axes([0.15, 0.75, 0.7, 0.2])  # [left, bottom, width, height]
ax1.plot(np.cumsum(s**2)/np.sum(s**2), 'k.-')
ax1.set_ylabel('Cumulative Variance')

# Bottom plot (square heatmap)
ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.65])  # wider and taller
sns.heatmap(mode1_reshaped, annot=True,
            xticklabels=['rt', 'rt_rpe', 'hit', 'hit_rpe', 'miss', 'miss_rpe'],
            yticklabels=['step', 'reward', 'go cue'],
            cmap='coolwarm', center=0,
            ax=ax2)
ax2.set_title('Mode 1 loading onto behavioral features')

plt.show()
#%%
# After cleaning betas_flat
session_norms = np.linalg.norm(betas_flat, axis=1)

plt.figure()
plt.plot(session_norms, 'ko-')
plt.xlabel('Session index')
plt.ylabel('Norm of beta vector')
plt.title('Session beta magnitudes')
plt.show()
