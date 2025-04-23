# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:50:06 2025

@author: kayvon.daie
"""


mouse = 'BCI103'
session = '012225'
folder = folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
#%%
data = np.load(folder + 'data_main.npy',allow_pickle=True)
data['photostim'] = np.load(folder + 'data_photostim.npy',allow_pickle=True)
data['photostim2'] = np.load(folder + 'data_photostim2.npy',allow_pickle=True)
#%%
AMP = []
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
for epoch_i in range(2):
    if epoch_i == 0:
        stimDist = data['photostim']['stimDist']*umPerPix 

        favg_raw = data['photostim']['favg_raw']
    else:
        stimDist = data['photostim2']['stimDist']*umPerPix 
        favg_raw = data['photostim2']['favg_raw']
    favg = favg_raw*0
    for i in range(favg.shape[1]):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])
    favg[8:13, :, :] = np.nan
    
    favg = np.apply_along_axis(
    lambda m: np.interp(
        np.arange(len(m)),
        np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
        m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
    ),
    axis=0,
    arr=favg
    )

    amp = np.nanmean(favg[13:20, :, :], axis=0) - np.nanmean(favg[0:6, :, :], axis=0)
    AMP.append(amp)
    plt.plot(np.nanmean(np.nanmean(favg[0:40,:,:],axis=2),axis=1))
#%%
from scipy.stats import pearsonr
import numpy as np

def get_indices_around_steps(tsta, steps, pre=0, post=0):
    indices = np.searchsorted(tsta, steps)
    all_indices = []

    for idx in indices:
        # Avoid going out of bounds
        start = max(idx - pre, 0)
        end = min(idx + post + 1, len(tsta))  # +1 because slicing is exclusive
        all_indices.extend(range(start, end))
    
    return np.unique(all_indices)

pairwise_mode = 'noise_corr'#dot_prod, noise_corr


dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
tsta = np.arange(0,12,data['dt_si'])
tsta=tsta-tsta[int(2/dt_si)]

# Initialize arrays
kstep = np.zeros((F.shape[1], trl))
krewards = np.zeros((F.shape[1], trl))

# Loop through trials and calculate kstep for steps and rewards separately
for ti in range(trl):
    # Steps regressor
    steps = data['step_time'][ti]
    if len(steps) > 0:
        indices_steps = get_indices_around_steps(tsta, steps, pre=10, post=0)
        indices_steps = indices_steps[indices_steps < F.shape[0]]
        kstep[:, ti] = np.nanmean(F[indices_steps, :, ti], axis=0)

    # Rewards regressor
    rewards = data['reward_time'][ti]
    if len(rewards) > 0:
        indices_rewards = get_indices_around_steps(tsta, rewards, pre=10, post=10)
        indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
        krewards[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
ts = np.where((tsta>0) & (tsta<2))[0]  
k = np.nanmean(F[ts[0]:ts[-1],:,:],axis=0);

kstep[np.isnan(kstep)] = 0
krewards[np.isnan(krewards)] = 0
k[np.isnan(k)] = 0

trial_bins = np.arange(0,F.shape[2],10)
trial_bins = np.linspace(0,F.shape[2],10).astype(int)
cc = np.corrcoef(kstep)
CCstep = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
CCrew = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
CCts = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))

rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
st = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
rt = rt - st;
hit = np.isnan(rt)==0;
hit_bin = np.zeros((len(trial_bins)-1,))
rt_bin = np.zeros((len(trial_bins),))
for i in range(len(trial_bins)-1):
    ind = np.arange(trial_bins[i],trial_bins[i+1])
    hit_bin[i] = np.nanmean(hit[ind]);
    rt_bin[i] = np.nanmean(rt[ind]);
    if pairwise_mode == 'noise_corr':    
        CCrew[:,:,i] = np.corrcoef(krewards[:,ind])
        CCstep[:,:,i] = np.corrcoef(kstep[:,ind])
        CCts[:,:,i] = np.corrcoef(k[:,ind]);
    elif pairwise_mode == 'dot_prod':
        CCrew[:,:,i] = np.dot(krewards[:,ind],krewards[:,ind].T)
        CCstep[:,:,i] = np.dot(kstep[:,ind],kstep[:,ind].T)
        CCts[:,:,i] = np.dot(k[:,ind],k[:,ind].T);
# Preallocate combined CC with interleaved shape
CC = np.zeros((cc.shape[0], cc.shape[1], CCstep.shape[2]*3))

# Interleave step and reward correlations
CC[:, :, 0::3] = CCstep
CC[:, :, 1::3] = CCrew
CC[:, :, 2::3] = CCts



import plotting_functions as pf


XX = []
for i in range(CC.shape[2]):
    X = []
    X2 = []
    Y = []
    Yo = []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:,gi]<10) & (AMP[0][:,gi]> .1) * ((AMP[1][:,gi]> .1)))[0]
        #plt.plot(favg[0:80,cl,gi])
        if len(cl)>0:
            x = np.nanmean(CC[cl,:,i],axis=0)
            
            # A = AMP[0][cl,gi] + AMP[1][cl,gi]
            # B = CC[cl,:,i]
            # x = np.dot(A.T,B)  
                
            nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<1000))
            y = AMP[1][nontarg,gi]
            yo = AMP[0][nontarg,gi]
            Y.append(y)
            Yo.append(yo)
            X.append(x[nontarg])
    
    
    
    X = np.concatenate(X)
    Y = np.concatenate(Y,axis=1)
    Yo = np.concatenate(Yo,axis=1)
    XX.append(X)

X = np.asarray(XX)
X[np.isnan(X)==1] = 0
Y[np.isnan(Y)==1] = 0
Yo[np.isnan(Yo)==1] = 0
X_T = X.T  # Shape: (82045, 13)
Y_T = Y.T.ravel() - Yo.T.ravel() # Shape: (82045,) — ravel to make it 1D

# Compute pseudoinverse solution
beta = np.linalg.pinv(X_T) @ Y_T  # (13, 1)

Y_pred = np.dot(beta.T,X_T.T)
plt.figure(figsize=(6,3))
plt.subplot(121)
pf.mean_bin_plot(Y_pred,Y_T,5,1,1,'k')
plt.xlabel(r'Predicted  ' + r'$\Delta W$')
plt.ylabel('$\Delta W$')

plt.subplot(122)
plt.plot(trial_bins[0:],beta[0::3][:],'ko-')
plt.plot(trial_bins[0:],beta[1::3][:],'bo-')
plt.plot(trial_bins[0:],beta[2::3][:],'mo-')
plt.xlabel('Trials')
plt.ylabel('Regression coeff.')
plt.plot(plt.xlim(),(0,0),'k:')
plt.legend(['Steps', 'Rew.', 'Go'],fontsize=6)
plt.tight_layout()
corr_coef, p_value = pearsonr(Y_pred, Y_T)
plt.show()

# a = beta[0::3][:-2];b = beta[1::3][:-2];c = beta[2::3][:-2];plt.plot(trial_bins[1:-1],a+b+c,'ko-')
# plt.xlabel('Trials')
# plt.ylabel('Avg. Regression coeffs.')

from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions as pf

kf = KFold(n_splits=5, shuffle=True)

corr_train, p_train = [], []
corr_test, p_test = [], []

# Arrays to store combined test set predictions and actual values
Y_test_all = np.array([])
Y_test_pred_all = np.array([])

for train_idx, test_idx in kf.split(X_T):
    # Split data
    X_train, X_test = X_T[train_idx], X_T[test_idx]
    Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
    
    # Fit regression on training set
    beta_cv = np.linalg.pinv(X_train) @ Y_train
    
    # Predict on train/test
    Y_train_pred = X_train @ beta_cv
    Y_test_pred = X_test @ beta_cv
    
    # Pearson correlations
    r_train, pval_train = pearsonr(Y_train_pred, Y_train)
    r_test, pval_test = pearsonr(Y_test_pred, Y_test)
    
    # Save correlations and p-values
    corr_train.append(r_train)
    p_train.append(pval_train)
    corr_test.append(r_test)
    p_test.append(pval_test)
    
    # Collect predictions and actual Y from test set
    Y_test_all = np.concatenate([Y_test_all, Y_test])
    Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])

# Report average correlation & significance
print("Cross-validation results (mean ± SD):")
print(f"Train correlation: {np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
print(f"Train p-value: {np.mean(p_train):.3e}")

print(f"Test correlation: {np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")
print(f"Test p-value: {np.mean(p_test):.3e}")

# Plotting test set predictions vs actual using mean_bin_plot
plt.figure(figsize=(7,3.5))
plt.subplot(121)
pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
plt.xlabel('Predicted Y (test)')
plt.ylabel('Actual Y (test)')
plt.title('Cross-validated predictions vs actual')

plt.subplot(122)
a = beta[0::3][:-2];b = beta[1::3][:-2];c = beta[2::3][:-2];
a = beta[0::3][:];b = beta[1::3][:];c = beta[2::3][:];
coefs = [a,b,c]
plt.plot(trial_bins[:-1],(a)[:-1]*1000,'ko-');
plt.plot(trial_bins[:-1],rt_bin[:-1]);
plt.xlabel('Trial #');
plt.ylabel('Coefficient (AU) / Rew Time (s)')

plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# Compute Singular Value Decomposition (SVD)
U, s, Vh = np.linalg.svd(X_T, full_matrices=False)

# Normalize singular values (variance explained)
variance_explained = s**2 / np.sum(s**2)

# Cumulative variance explained
cum_variance_explained = np.cumsum(variance_explained)

# Plot singular values and cumulative variance
plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.plot(range(1, len(s)+1), s, 'ko-', markersize=5)
plt.xlabel('Dimension')
plt.ylabel('Singular values')
plt.title('Singular values of X')

plt.subplot(122)
plt.plot(range(1, len(s)+1), cum_variance_explained*100, 'bo-', markersize=5)
plt.xlabel('Dimension')
plt.ylabel('Cumulative % variance explained')
plt.title('Cumulative variance explained')

plt.tight_layout()
plt.show()


#%%
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Parameters
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

r2_scores_train = []
r2_scores_test = []

for train_index, test_index in kf.split(X_T):
    X_train, X_test = X_T[train_index], X_T[test_index]
    Y_train, Y_test = Y_T[train_index], Y_T[test_index]

    # Fit model on training set
    beta_cv = np.linalg.pinv(X_train) @ Y_train

    # Predict and evaluate on training and test sets
    Y_train_pred = X_train @ beta_cv
    Y_test_pred = X_test @ beta_cv

    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)

    r2_scores_train.append(r2_train)
    r2_scores_test.append(r2_test)

# Display results
print("Cross-validation results:")
print(f"Mean R² (train): {np.mean(r2_scores_train):.3f} ± {np.std(r2_scores_train):.3f}")
print(f"Mean R² (test):  {np.mean(r2_scores_test):.3f} ± {np.std(r2_scores_test):.3f}")



#%%
Yo = np.concatenate(Yo,axis=1)
plt.subplot(231)
pf.mean_bin_plot(X,Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('Before learning')

plt.subplot(234)
pf.mean_bin_plot(X,Y,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$W_{i,j}$')
plt.title('After learning')

plt.subplot(132)
pf.mean_bin_plot(X,Y-Yo,5,1,1,'k')
plt.xlabel('Pre-post correlation')
plt.ylabel('$\Delta W_{i,j}$')
plt.tight_layout()

#%%

plt.scatter(X_T[:,3], Y_T, alpha=0.1)
plt.xlabel('X feature 0')
plt.ylabel('Y response')
plt.title('Check the relationship directly')
plt.show()
