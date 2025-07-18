# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:48:35 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_017"],'010112',has_pophys=False)
#%%
import sys
sys.path.append('./LC_axon_analysis')  # or use full path if needed
from scipy.signal import medfilt
from scipy.signal import correlate
from axon_helper_module import *

processing_mode = 'one'
si = 12
if processing_mode == 'all':
    inds = np.arange(0,len(sessions))
else:
    inds = np.arange(si,si+1)
AXON_REW, AXON_TS = [], []
SESSION = []
import bci_time_series as bts
for i in inds:
    print(i)
    data = get_axon_data_dict(sessions,i)
    
    # --- New cell: Correlation matrix, SVD, and distribution of correlations ---
    try:
        
        rt = data['reward_time'];dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        dfaxon = data['ch1']['df_closedloop']
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
        
        # Drop timepoints with NaNs across all cells to avoid SVD/corr issues
        valid_t = ~np.isnan(dfaxon).any(axis=0)
        X = dfaxon[:, valid_t]
        
        # Center each neuron's activity
        X = X - np.nanmean(X, axis=1, keepdims=True)
        
        # Correlation matrix
        C = np.corrcoef(X)
        
        # Singular Value Decomposition
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var_explained = (S**2) / np.sum(S**2)
        cumulative_var = np.cumsum(var_explained) * 100  # in percent
        
        
        # Pairwise correlation values (excluding diagonal)
        triu_inds = np.triu_indices_from(C, k=1)
        pairwise_corrs = C[triu_inds]
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Correlation matrix heatmap
        im = axes[0].imshow(C, vmin=-1, vmax=1, cmap='bwr', aspect='auto')
        axes[0].set_title('Pairwise correlation matrix')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Cumulative variance explained (SVD)
        axes[1].plot(np.arange(1, len(S)+1), cumulative_var, 'ko-')
        axes[1].set_title('Cumulative variance explained')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('% Variance explained')
        axes[1].set_xticks(np.arange(1, len(S)+1))
        axes[1].set_ylim(0, 105)
        
        
        # Pairwise correlation histogram
        axes[2].hist(pairwise_corrs, bins=30, color='gray', edgecolor='black')
        axes[2].set_title('Pairwise correlation distribution')
        axes[2].set_xlabel('Correlation coefficient')
        axes[2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    except:
        continue
