# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:13:36 2025

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
import re
DX,DY = [],[]
mice = ["BCI102","BCI103","BCI105","BCI106","BCI109","BCI104","BCI107","BCI88","BCI93"]
mice = ["BCI102"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    #session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
    si = 11
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'ridge'     #ridge, pinv
    alpha         =  .1        #only used for ridge
    epoch         =  'step'  # reward, step, trial_start
    
    for sii in range(0,len(session_inds)):        
    #for sii in range(si,si+1):
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
        try:
            cn = data['conditioned_neuron'][0][0]
        except:
            cn = 0;
        dfcn = data['df_closedloop'][cn,:]
        
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
        
        # Initialize arrays
        kstep = np.zeros((F.shape[1], trl))
        Q = np.zeros((F.shape[1], trl))
        krewards = np.zeros((F.shape[1], trl))    
        step_raw = data['step_time']
    
        # --- Replace step_time and reward_time with parsed versions if needed ---
        data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
        data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
    
        # --- Compute step/reward regressors ---
        for ti in range(trl):
            # Steps regressor
            steps = data['step_time'][ti]
            if len(steps) > 0:
                indices_steps = get_indices_around_steps(tsta, steps, pre=10, post=0)
                indices_steps = indices_steps[indices_steps < F.shape[0]]
                kstep[:, ti] = np.nanmean(F[indices_steps, :, ti], axis=0)
                
                indices_first_step = get_indices_around_steps(tsta, [steps[0]], pre=10, post=0)
                indices_first_step = indices_first_step[indices_first_step < F.shape[0]]
                Q[:,ti] = np.nanmean(F[indices_first_step,:,ti],axis=0)
    
            # Rewards regressor
            rewards = data['reward_time'][ti]
            if len(rewards) > 0:
                indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                krewards[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
    
        # Go cue regressor
        ts = np.where((tsta > 0) & (tsta < 2))[0]
        k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
    
    
        kstep[np.isnan(kstep)] = 0
        krewards[np.isnan(krewards)] = 0
        k[np.isnan(k)] = 0
        #num_bins = F.shape[2]

        trial_bins = np.arange(0,F.shape[2],10)
        trial_bins = np.linspace(0,F.shape[2],num_bins).astype(int)
        CCts = np.zeros((F.shape[1],F.shape[1],len(trial_bins)))
        
      
    
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        hit = np.isnan(rt)==0;
        rt[np.isnan(rt)] = 30;
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
            folder, data, rt, dt_si)
        rpe = compute_rpe(rt, baseline=10, window=20, fill_value=50)

        hit_bin = np.zeros((len(trial_bins),))
        rt_bin = np.zeros((len(trial_bins),))
        avg_dot_bin = np.zeros((len(trial_bins),))
        thr_bin = np.zeros((len(trial_bins),))
        rpe_bin = np.zeros((len(trial_bins),))
        
      
        
        
        for i in range(len(trial_bins) - 1):
            #ind = i
            ind = np.arange(trial_bins[i],trial_bins[i+1])
            hit_bin[i] = np.nanmean(hit[ind])
            rt_bin[i] = np.nanmean(rt[ind])
            thr_bin[i] = np.nanmax(BCI_thresholds[1, ind])
            rpe_bin[i] = np.nanmean(rpe[ind])
            
            if epoch == 'step' or epoch == 'reward':
                if epoch == 'step':
                    step_list = [s for s in data['step_time'][ind] if len(s) > 0]
                if epoch == 'reward':
                    step_list = [s for s in data['reward_time'][ind] if len(s) > 0]
                if len(step_list) > 0:
                    steps = np.concatenate(step_list)
                    indices_steps = get_indices_around_steps(tsta, steps, pre=20, post=0)
                    indices_steps = indices_steps[indices_steps < F.shape[0]]
                    tpts = indices_steps
                else:
                    tpts = np.array([], dtype=int)
            elif epoch == 'trial_start':                    
                tpts = np.where(np.isnan(F[:,0,ind]) == 0)[0]
                #tpts = np.where((tsta > 0) & (tsta < 2))[0]
            k_concat = F[:,:,np.ravel(ind)][np.ravel(tpts),:,:]
            k_concat = k_concat.transpose(2, 0, 1).reshape(-1, k_concat.shape[1])
            k_concat = k_concat[np.where(np.isnan(k_concat[:,0])==0)[0],:];
            if pairwise_mode == 'noise_corr':
                CCts[:, :, i]   = np.corrcoef(k_concat.T)
             
        
            elif pairwise_mode == 'dot_prod':
                CCts[:, :, i]   = k_concat.T @ k_concat
                
            elif pairwise_mode == 'dot_prod_no_mean':
                avg_f = np.nanmean(np.nanmean(F,axis=2),axis=0)
                for ci in range(F.shape[1]):
                    k_concat[:,ci] = k_concat[:,ci] - avg_f[ci]
                CCts[:, :, i]   = k_concat.T @ k_concat
             
    
        
        CC = CCts
    
        import plotting_functions as pf
        XX = []
        for i in range(CC.shape[2]):
            X = []
            Y = []
            Yo = []
            d_direct = []
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
                    
                    a = np.ones(yo.shape)*np.nanmean(AMP[1][cl,gi] - AMP[0][cl,gi],0)
                    d_direct.append(a)
                    
                    X.append(x[nontarg])
            
            
            if len(X) == 0:
                print('something wrong with ' + folder)
                continue 
            X = np.concatenate(X)
            d_direct = np.concatenate(d_direct,axis=1)
            Y = np.concatenate(Y,axis=1)
            Yo = np.concatenate(Yo,axis=1)
            XX.append(X)
    
        X = np.asarray(XX)
        pf.mean_bin_plot(d_direct,Y-Yo,21)
        plt.show()
        DX.append(d_direct)
        DY.append(Y-Yo)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

pf.mean_bin_plot(np.concatenate(DX,1),np.concatenate(DY,1))
plt.xlabel('$\Delta direct$');
plt.ylabel('$\Delta W_{cc}$')
# Flatten the data
x = np.concatenate(DX, axis=1).flatten()
y = np.concatenate(DY, axis=1).flatten()

# Remove any NaNs
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]

# Compute regression
slope, intercept, r_value, p_value, stderr = linregress(x, y)

# Plot regression line
x_vals = np.linspace(np.min(x), np.max(x), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, 'k--', label=f'slope = {slope:.3f}, p = {p_value:.3g}')
plt.legend()

# Add axis labels again (in case plot was refreshed)
plt.xlabel('$\Delta direct$')
plt.ylabel('$\Delta W_{cc}$')
plt.title('Change in Target vs. Change in Non-target Response')
plt.tight_layout()
plt.show()
