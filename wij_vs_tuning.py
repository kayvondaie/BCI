# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:39:48 2025

@author: kayvon.daie
"""


import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import ttest_1samp
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso
import re
Wij,CVAL,PVAL = [],[],[]

mice = ["BCI102","BCI105","BCI106","BCI109","BCI103","BCI104","BCI93","BCI107"]

for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    si = 1
    
    fit_type      = 'ridge'     #ridge, pinv, lasso
    
    for sii in range(0,len(session_inds)):     
    #for sii in range(si,si+1):
        try:
            num_bins      =  2000         # number of bins to calculate correlations
            print(sii)
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
            except:
                continue        
            BCI_thresholds = data['BCI_thresholds']        
            
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        
            dt_si = data['dt_si']
            F = data['F']
            if num_bins > F.shape[2]:
                num_bins = F.shape[2]
            trl = F.shape[2]
            tsta = np.arange(0,12,data['dt_si'])
            tsta=tsta-tsta[int(2/dt_si)]
            if epoch == 'steps':
                epoch = 'step'
            
           
            # --- Replace step_time and reward_time with parsed versions if needed ---
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            
            kstep = np.zeros((F.shape[1], trl))
            Q = np.zeros((F.shape[1], trl))
            krew, kpre_rew, kts = [np.zeros((F.shape[1], trl)) for _ in range(3)]
            one_sec = np.round(1/dt_si).astype(int)
            kpre = np.nanmean(F[0:one_sec,:,:],0)
    
            # --- Replace step_time and reward_time with parsed versions if needed ---
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
    
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=3*one_sec)
                    indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                    krew[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
                    
                    indices_rewards = get_indices_around_steps(tsta, rewards, pre=one_sec, post=0)
                    indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                    kpre_rew[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
                    
                    if rewards <= 10:
                        indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=1)
                        indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                        kts[:, ti] = np.nanmean(F[one_sec*2:indices_rewards[1], :, ti], axis=0)
                    else:
                        kts[:, ti] = np.nanmean(F[one_sec*2:, :, ti], axis=0)
                        
    
            tunings = [kpre, kts, kpre_rew, krew]
            tunings = [np.nanmean(k, axis=1) for k in tunings]
            tunings = [np.ones(len(tunings[0]))] + tunings
    
            N = F.shape[1]
            CC = np.zeros((N,N,len(tunings)**2))
            I = 0
            for i in range(len(tunings)):
                for j in range(len(tunings)):
                    outer_prod = np.outer(tunings[i], tunings[j])
                    outer_prod[np.isnan(outer_prod)] = 0
                    CC[:,:,I] = outer_prod
                    I = I + 1;
            CC[:,:,0] = 0                
            XX = []
            for i in range(CC.shape[2]):
                X = []
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
                
                
                if len(X) == 0:
                    print('something wrong with ' + folder)
                    continue 
                X = np.concatenate(X)
                Y = np.concatenate(Y,axis=1)
                Yo = np.concatenate(Yo,axis=1)
                XX.append(X)
        
            X = np.asarray(XX)
            if len(X) == 0:
                print('something wrong with ' + folder)
                continue 
            X[np.isnan(X)==1] = 0
            
            Y[np.isnan(Y)==1] = 0
            Yo[np.isnan(Yo)==1] = 0
            X_T = X.T  # Shape: (82045, 13)
            Y_T = Y.T.ravel() - Yo.T.ravel() # Shape: (82045,) — ravel to make it 1D
            Y_T = Y.T.ravel() 
        
            # Compute pseudoinverse solution
            if fit_type == 'pinv':
                beta = np.linalg.pinv(X_T) @ Y_T  # (13, 1)
            elif fit_type == 'ridge':
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha, fit_intercept=False)
                ridge.fit(X_T, Y_T)
                beta = ridge.coef_
           
                
            from sklearn.linear_model import LinearRegression
            from scipy.stats import zscore
            import numpy as np
     
            Y_pred = np.dot(beta.T,X_T.T)
           
        
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
            best_alphas = []
            for train_idx, test_idx in kf.split(X_T):
                # Split data
                X_train, X_test = X_T[train_idx], X_T[test_idx]
                Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
                
                # Fit regression on training set            
                if fit_type == 'pinv':
                    beta_cv = np.linalg.pinv(X_train) @ Y_train
                elif fit_type == 'ridge':                
    
                    alphas_to_try = np.logspace(-4, 2, 30)  # You can adjust the range if needed
                    
                    # Inner CV to select alpha within the training set
                    ridge_cv = RidgeCV(alphas=alphas_to_try, fit_intercept=False)
                    ridge_cv.fit(X_train, Y_train)
                    best_alpha = ridge_cv.alpha_
                    
                    # Refit with the best alpha on the same training fold
                    ridge = Ridge(alpha=best_alpha, fit_intercept=False)
                    ridge.fit(X_train, Y_train)
                    beta_cv = ridge.coef_
                    
                elif fit_type == 'lasso':
                    # Inner CV to select alpha for Lasso
                    lasso_cv = LassoCV(alphas=alphas_to_try, fit_intercept=False, max_iter=10000)
                    lasso_cv.fit(X_train, Y_train)
                    best_alpha = lasso_cv.alpha_
                    best_alphas.append(best_alpha)
            
                    # Fit Lasso with selected alpha
                    lasso = Lasso(alpha=best_alpha, fit_intercept=False, max_iter=10000)
                    lasso.fit(X_train, Y_train)
                    beta_cv = lasso.coef_
    
               
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
            print(f"Test p-value: {np.exp(np.mean(np.log(p_test))):.3e}")
            # Plotting test set predictions vs actual using mean_bin_plot
            PVAL.append(np.exp(np.mean(np.log(p_test))))
            RVAL.append(np.mean(corr_test))
            plt.figure(figsize=(8,4))        
            plt.subplot(121)
            pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
            plt.xlabel(r'$HI r_i r_j$')
            plt.ylabel('$\Delta W$')
            plt.title('Cross-validated predictions vs actual')
            
            plt.subplot(122)
            labels = ["all", "pre", "ts", "pre_rew", "rew"]  # "ones" first
            n_types = len(labels)
    
            # Reshape beta back to matrix form (pre x post)
            beta_matrix = beta.reshape((n_types, n_types))
    
            # Plot
            im = plt.imshow(beta_matrix.T, cmap='bwr', vmin=-np.max(np.abs(beta_matrix)), vmax=np.max(np.abs(beta_matrix)))
            plt.colorbar(im, label='Regression Coefficient (β)')
    
            plt.xticks(range(n_types), labels, rotation=45)
            plt.yticks(range(n_types), labels)
            plt.ylabel("Post-synaptic Tuning Type")
            plt.xlabel("Pre-synaptic Tuning Type")        
            plt.gca().xaxis.set_ticks_position('top')   # Move x-axis ticks to top
            plt.gca().xaxis.set_label_position('top')   # Move x-axis label to top
            plt.tight_layout()
            plt.show()
            
            Wij.append(beta_matrix)
            CVAL.append(np.nanmean(corr_test))
            PVAL.append(np.exp(np.mean(np.log(p_test))))
        except:
            continue

#%%
w = np.nanmean(np.stack(Wij),0)
im = plt.imshow(w.T, cmap='bwr', vmin=-np.max(np.abs(w)), vmax=np.max(np.abs(w)))
plt.colorbar(im, label='Regression Coefficient (β)')

plt.xticks(range(n_types), labels, rotation=45)
plt.yticks(range(n_types), labels)
plt.ylabel("Post-synaptic Tuning Type")
plt.xlabel("Pre-synaptic Tuning Type")        
plt.gca().xaxis.set_ticks_position('top')   # Move x-axis ticks to top
plt.gca().xaxis.set_label_position('top')   # Move x-axis label to top
plt.tight_layout()
plt.show()    
#%%
ind = np.where(np.array(PVAL)<.05)[0]
selected_Wij = [Wij[i] for i in ind]
w = np.nanmean(np.stack(selected_Wij), axis=0)
im = plt.imshow(w.T, cmap='bwr', vmin=-np.max(np.abs(w)), vmax=np.max(np.abs(w)))
plt.colorbar(im, label='Regression Coefficient (β)')

plt.xticks(range(n_types), labels, rotation=45)
plt.yticks(range(n_types), labels)
plt.ylabel("Post-synaptic Tuning Type")
plt.xlabel("Pre-synaptic Tuning Type")        
plt.gca().xaxis.set_ticks_position('top')   # Move x-axis ticks to top
plt.gca().xaxis.set_label_position('top')   # Move x-axis label to top
plt.tight_layout()
plt.show()          
     
