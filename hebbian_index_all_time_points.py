# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:39:48 2025

@author: kayvon.daie
"""

import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29
folder = str(list_of_dirs[si])

#%%
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import ttest_1samp
import re

HI, HIb, HIc = [], [], []
RT, RPE, HIT = [], [], []
HIT_RATE, D_HIT_RATE, DOT, TRL, THR = [], [], [], [], []
CC_RPE, CC_RT, CC_MIS, CORR_RPE, CORR_RT = [], [], [], [], []
RT_WINDOW, HIT_WINDOW, THR_WINDOW = [], [], []
PTRL, PVAL, RVAL = [], [], []
Ddirect, Dindirect, CCdirect = [], [], []
NUM_STEPS, TST, CN, MOUSE, SESSION = [], [], [], [], []

mice = ["BCI102"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy'] == True))[0]
    si = 7

    pairwise_mode = 'dot_prod_no_mean'
    n_avg = 5
    

    for sii in range(si, si + 1):
        num_bins = 2000
        print(sii)
        mouse = list_of_dirs['Mouse'][session_inds[sii]]
        session = list_of_dirs['Session'][session_inds[sii]]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        photostim_keys = ['stimDist', 'favg_raw']
        bci_keys = ['df_closedloop', 'F', 'mouse', 'session', 'conditioned_neuron',
                    'dt_si', 'step_time', 'reward_time', 'BCI_thresholds']
        try:
            data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
        except:
            continue

        BCI_thresholds = data['BCI_thresholds']
        cn = data['conditioned_neuron'][0][0]
        dfcn = data['df_closedloop'][cn, :]

        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)

        dt_si = data['dt_si']
        F = data['F']
        if num_bins > F.shape[2]:
            num_bins = F.shape[2]
        trl = F.shape[2]
        tsta = np.arange(0, 12, data['dt_si'])
        tsta = tsta - tsta[int(2 / dt_si)]
        if epoch == 'steps':
            epoch = 'step'

        # Initialize arrays
        kstep = np.zeros((F.shape[1], trl))
        Q = np.zeros((F.shape[1], trl))
        krewards = np.zeros((F.shape[1], trl))
        step_raw = data['step_time']

        data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
        data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
            folder, data, rt, dt_si)
        for ti in range(trl):
            steps = data['step_time'][ti]
            if len(steps) > 0:
                indices_steps = get_indices_around_steps(tsta, steps, pre=10, post=0)
                indices_steps = indices_steps[indices_steps < F.shape[0]]
                kstep[:, ti] = np.nanmean(F[indices_steps, :, ti], axis=0)

                indices_first_step = get_indices_around_steps(tsta, [steps[0]], pre=10, post=0)
                indices_first_step = indices_first_step[indices_first_step < F.shape[0]]
                Q[:, ti] = np.nanmean(F[indices_first_step, :, ti], axis=0)

            rewards = data['reward_time'][ti]
            if len(rewards) > 0:
                indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                krewards[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)

        ts = np.where((tsta > 0) & (tsta < 2))[0]
        k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)

        kstep[np.isnan(kstep)] = 0
        krewards[np.isnan(krewards)] = 0
        k[np.isnan(k)] = 0

        trial_bins = np.linspace(0, F.shape[2], num_bins).astype(int)
        CCts = np.zeros((F.shape[1], F.shape[1], len(trial_bins)))

        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        hit = ~np.isnan(rt)
        rt[np.isnan(rt)] = 30
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
        rpe = compute_rpe(rt, baseline=10, window=20, fill_value=50)

        hit_bin = np.zeros((len(trial_bins),))
        rt_bin = np.zeros((len(trial_bins),))
        avg_dot_bin = np.zeros((len(trial_bins),))
        thr_bin = np.zeros((len(trial_bins),))
        rpe_bin = np.zeros((len(trial_bins),))
        df = data['df_closedloop']
        df[np.isnan(df)] = 0
       
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr
        
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr
        from tqdm import tqdm  # <-- progress bar
        
        b, c, intercept, ptrl = [], [], [], []
        XX = []
        
        n_timepoints = df.shape[1]
        n_bins = n_timepoints // n_avg
        
        for bin_idx in tqdm(range(n_bins), desc="Computing regression", unit="bin"):
            start = bin_idx * n_avg
            end = start + n_avg
            if end > n_timepoints:
                break
        
            r = np.nanmean(df[:, start:end], axis=1)
            if np.any(np.isnan(r)):
                continue
        
            CC = np.outer(r, r)
        
            X, Y, Yo = [], [], []
        
            for gi in range(stimDist.shape[1]):
                cl = np.where((stimDist[:, gi] < 10) & (AMP[0][:, gi] > 0.1) & (AMP[1][:, gi] > 0.1))[0]
                if len(cl) == 0:
                    continue
        
                x = np.nanmean(CC[cl, :], axis=0)
                nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))[0]
                if len(nontarg) == 0:
                    continue
        
                y = AMP[1][nontarg, gi]
                yo = AMP[0][nontarg, gi]
        
                X.append(x[nontarg])
                Y.append(y)
                Yo.append(yo)
        
            if len(X) == 0:
                continue
        
            X = np.concatenate(X)
            Y = np.concatenate(Y)
            Yo = np.concatenate(Yo)
            XX.append(X)
        
            ΔAMP = Y - Yo
            mask = ~np.isnan(X) & ~np.isnan(ΔAMP)
            X_clean = X[mask].reshape(-1, 1)
            Y_clean = ΔAMP[mask]
        
            if len(X_clean) < 2:
                b.append(np.nan)
                c.append(np.nan)
                ptrl.append(np.nan)
                intercept.append(np.nan)
                continue
        
            model = LinearRegression().fit(X_clean, Y_clean)
            b.append(model.coef_[0])
            intercept.append(model.intercept_)
            c.append(np.corrcoef(Y_clean, X_clean.ravel())[0, 1])
            _, pval = pearsonr(Y_clean, X_clean.ravel())
            ptrl.append(pval)
        import numpy as np

        def bin_array(arr, n_avg):
            arr = arr[:(len(arr) // n_avg) * n_avg]  # truncate to full bins
            return arr.reshape(-1, n_avg).mean(axis=1)
        
        step_vector_binned = bin_array(step_vector[:], n_avg)
        reward_vector_binned = bin_array(reward_vector[:], n_avg)

        # Use binned step_vector
        pf.mean_bin_plot(b, step_vector_binned, 5)
#%%
num = 50
rpe = step_vector_binned[0:] - .05*np.convolve(step_vector_binned,np.ones(num,))[num-1:]        
#rpe = reward_vector_binned[0:] - np.convolve(reward_vector_binned,np.ones(num,))[num-1:]        
pf.mean_bin_plot(b, rpe, 5)
