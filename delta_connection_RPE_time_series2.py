# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:37:20 2025

@author: kayvon.daie
"""


import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import seaborn as sns
list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
HI = []
RT = []
HIT = []
HIa= []
HIb = []
HIc = []
DOT = []
TRL = []
THR = []
RPE = []
RPE_FIT = []
session_inds = np.where((list_of_dirs['Mouse'] == 'BCI102') & (list_of_dirs['Has data_main.npy']==True))[0]
#session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
si = 3

pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
fit_type      = 'ridge'     #ridge, pinv
alpha         =  100        #only used for ridge
num_bins      =  10        # number of bins to calculate correlations
tau_elig      =  2
shuffle       =  0
#for sii in range(len(session_inds)):
for sii in range(si,si+1):
    print(sii)
    mouse = list_of_dirs['Mouse'][session_inds[sii]]
    session = list_of_dirs['Session'][session_inds[sii]]
    folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
    photostim_keys = ['stimDist', 'favg_raw']
    bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
    data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
    BCI_thresholds = data['BCI_thresholds']
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
        dt_si = data['dt_si']
        after = np.floor(0.2/dt_si)
        before = np.floor(0.2/dt_si)
        artifact = np.nanmean(np.nanmean(favg_raw,axis=2),axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > .5)[0]
        artifact = artifact[artifact<40]
        pre = (int(artifact[0]-before),int(artifact[0]-2))
        post = (int(artifact[-1]+2),int(artifact[-1]+after))
        favg[artifact, :, :] = np.nan
        
        favg[0:30,:] = np.apply_along_axis(
        lambda m: np.interp(
            np.arange(len(m)),
            np.where(~np.isnan(m))[0] if np.where(~np.isnan(m))[0].size > 0 else [0],  # Ensure sample points are non-empty
            m[~np.isnan(m)] if np.where(~np.isnan(m))[0].size > 0 else [0]             # Use a fallback value if empty
        ),
        axis=0,
        arr=favg[0:30,:]
        )
    
        amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
        AMP.append(amp)
        #plt.plot(np.nanmean(np.nanmean(favg[0:40,:,:],axis=2),axis=1))
    
    from scipy.stats import pearsonr
    import numpy as np

    
    dt_si = data['dt_si']
    F = data['F']
    trl = F.shape[2]
    tsta = np.arange(0,12,data['dt_si'])
    tsta=tsta-tsta[int(2/dt_si)]

    # Initialize arrays
    kstep = np.zeros((F.shape[1], trl))
    krewards = np.zeros((F.shape[1], trl))

    import numpy as np
    import re

    step_raw = data['step_time']

    import numpy as np
    import re

    def parse_hdf5_array_string(array_raw, trl):
        if isinstance(array_raw, str):
            # Match both non-empty and empty arrays
            pattern = r'array\(\[([^\]]*)\](?:, dtype=float64)?\)'
            matches = re.findall(pattern, array_raw.replace('\n', ''))

            parsed = []
            for match in matches:
                try:
                    if match.strip() == '':
                        parsed.append(np.array([]))
                    else:
                        arr = np.fromstring(match, sep=',')
                        parsed.append(arr)
                except Exception as e:
                    print("Skipping array due to error:", e)

            # Pad to match number of trials
            pad_len = trl - len(parsed)
            if pad_len > 0:
                parsed += [np.array([])] * pad_len

            return np.array(parsed, dtype=object)

        else:
            # Already a list/array
            if len(array_raw) < trl:
                pad_len = trl - len(array_raw)
                return np.array(list(array_raw) + [np.array([])] * pad_len, dtype=object)
            return array_raw


    # --- Replace step_time and reward_time with parsed versions if needed ---
    data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
    data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
    
    
    
           
   

    from bci_time_series import *
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    import plotting_functions as pf
    import itertools
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # === Load behavioral data ===
    step_vector, hit_vector, trial_start_vector = bci_time_series_fun(folder, data, rt, dt_si)
    df = data['df_closedloop']

    # === Define parameter grid ===
    step_gains = [0.5, 1.0, 1.5]
    hit_gains = [0.5, 1.0, 1.5]
    step_taus = [int(round(x / dt_si)) for x in [30, 60, 130]]
    hit_taus  = [int(round(x / dt_si)) for x in [30, 60, 130]]
    param_grid = list(itertools.product(step_gains, hit_gains, step_taus, hit_taus))

    # === Optimize parameters ===
    print("Searching parameter space...")
    best_score = -np.inf
    for step_gain, hit_gain, step_tau, hit_tau in tqdm(param_grid):
        # RPE
        step_rate = compute_ema(step_vector, round(10 / dt_si), np.nanmean(step_vector))
        step_rate_slow = compute_ema(step_vector, step_tau, np.nanmean(step_vector))
        hit_rate = compute_ema(hit_vector, round(10 / dt_si), 0)
        hit_rate_slow = compute_ema(hit_vector, hit_tau, 0)

        rpe_step = (step_rate - np.nanmean(step_rate)) / np.nanstd(step_rate) \
                 - step_gain * (step_rate_slow - np.nanmean(step_rate_slow)) / np.nanstd(step_rate_slow)
        rpe_hit = (hit_rate - np.nanmean(hit_rate)) / np.nanstd(hit_rate) \
                - hit_gain * (hit_rate_slow - np.nanmean(hit_rate_slow)) / np.nanstd(hit_rate_slow)

        df_steps = df * rpe_step
        df_rew   = df * rpe_hit
        CCstep = df @ df_steps.T
        CCrew  = df @ df_rew.T

        Xstep, Xrew, Y, Yo = [], [], [], []
        for gi in range(stimDist.shape[1]):
            cl = np.where((stimDist[:, gi] < 30) & (AMP[0][:, gi] > .10) & (AMP[1][:, gi] > .10))[0]
            if len(cl) == 0: continue
            xstep = np.nanmean(CCstep[cl, :], axis=0)
            xrew  = np.nanmean(CCrew[cl, :], axis=0)
            nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))
            y, yo = AMP[1][nontarg, gi], AMP[0][nontarg, gi]
            Y.append(y); Yo.append(yo)
            Xstep.append(xstep[nontarg])
            Xrew.append(xrew[nontarg])

        if not Xstep or not Y:
            continue

        Xstep = np.nan_to_num(np.concatenate(Xstep))
        Xrew  = np.nan_to_num(np.concatenate(Xrew))
        Y     = np.nan_to_num(np.concatenate(Y, axis=1))
        Yo    = np.nan_to_num(np.concatenate(Yo, axis=1))

        X_T = np.vstack([Xstep, Xrew]).T
        Y_T = Y.T.ravel() - Yo.T.ravel()
        X_T = (X_T - np.mean(X_T, axis=0)) / np.std(X_T, axis=0)
        Y_T = (Y_T - np.mean(Y_T)) / np.std(Y_T)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        r_vals = []
        for train_idx, test_idx in kf.split(X_T):
            ridge = Ridge(alpha=100, fit_intercept=False)
            ridge.fit(X_T[train_idx], Y_T[train_idx])
            y_pred = ridge.predict(X_T[test_idx])
            r, _ = pearsonr(Y_T[test_idx], y_pred)
            r_vals.append(r)
        mean_r = np.mean(r_vals)

        if mean_r > best_score:
            best_score = mean_r
            best_params = (step_gain, hit_gain, step_tau, hit_tau)

    # === Use best parameters ===
    step_gain, hit_gain, step_tau, hit_tau = best_params
    print(f"\nUsing best params: step_gain={step_gain}, hit_gain={hit_gain}, step_tau={step_tau*dt_si:.1f}s, hit_tau={hit_tau*dt_si:.1f}s")

    # === Recompute with best params ===
    step_rate = compute_ema(step_vector, round(10 / dt_si), np.nanmean(step_vector))
    step_rate_slow = compute_ema(step_vector, step_tau, np.nanmean(step_vector))
    hit_rate = compute_ema(hit_vector, round(10 / dt_si), 0)
    hit_rate_slow = compute_ema(hit_vector, hit_tau, 0)

    step_rate_z = (step_rate - np.nanmean(step_rate)) / np.nanstd(step_rate)
    step_rate_slow_z = (step_rate_slow - np.nanmean(step_rate_slow)) / np.nanstd(step_rate_slow)
    rpe_step = step_rate_z - step_gain * step_rate_slow_z

    hit_rate_z = (hit_rate - np.nanmean(hit_rate)) / np.nanstd(hit_rate)
    hit_rate_slow_z = (hit_rate_slow - np.nanmean(hit_rate_slow)) / np.nanstd(hit_rate_slow)
    rpe_hit = hit_rate_z - hit_gain * hit_rate_slow_z

    df_steps = df * rpe_step
    df_rew   = df * rpe_hit
    CCstep = df @ df_steps.T
    CCrew  = df @ df_rew.T

    Xstep, Xrew, Y, Yo = [], [], [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < 30) & (AMP[0][:, gi] > .10) & (AMP[1][:, gi] > .10))[0]
        if len(cl) == 0: continue
        xstep = np.nanmean(CCstep[cl, :], axis=0)
        xrew  = np.nanmean(CCrew[cl, :], axis=0)
        nontarg = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000))
        y, yo = AMP[1][nontarg, gi], AMP[0][nontarg, gi]
        Y.append(y); Yo.append(yo)
        Xstep.append(xstep[nontarg])
        Xrew.append(xrew[nontarg])

    Xstep = np.nan_to_num(np.concatenate(Xstep))
    Xrew  = np.nan_to_num(np.concatenate(Xrew))
    Y     = np.nan_to_num(np.concatenate(Y, axis=1))
    Yo    = np.nan_to_num(np.concatenate(Yo, axis=1))

    X_T = np.vstack([Xstep, Xrew]).T
    Y_T = Y.T.ravel() - Yo.T.ravel()
    X_T = (X_T - np.mean(X_T, axis=0)) / np.std(X_T, axis=0)
    Y_T = (Y_T - np.mean(Y_T)) / np.std(Y_T)

    # === Final model fit & plot ===
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_test_all, y_pred_all, beta_list = [], [], []
    r_list, p_list = [], []

    for train_idx, test_idx in kf.split(X_T):
        ridge = Ridge(alpha=100, fit_intercept=False)
        ridge.fit(X_T[train_idx], Y_T[train_idx])
        Y_pred = ridge.predict(X_T[test_idx])

        y_test_all.append(Y_T[test_idx])
        y_pred_all.append(Y_pred)
        beta_list.append(ridge.coef_)

        r, p = pearsonr(Y_T[test_idx], Y_pred)
        r_list.append(r)
        p_list.append(p)

    y_test_all = np.concatenate(y_test_all)
    y_pred_all = np.concatenate(y_pred_all)
    beta_avg = np.mean(beta_list, axis=0)

    print(f"\nFinal Test R:      {pearsonr(y_test_all, y_pred_all)[0]:.3f}")
    print(f"Final Betas:       Step={beta_avg[0]:.3f}, Hit={beta_avg[1]:.3f}")
    # === Final plotting ===
    import matplotlib.pyplot as plt

    # Compute overall correlation on combined test data
    r_full, p_full = pearsonr(y_test_all, y_pred_all)

    # Normalize inputs for interpretability
    X_T_z = (X_T - np.mean(X_T, axis=0)) / np.std(X_T, axis=0)
    ridge_z = Ridge(alpha=100, fit_intercept=False)
    ridge_z.fit(X_T_z, Y_T)
    beta_z = ridge_z.coef_

    # === Plotting ===
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), constrained_layout=True)

    # 1. Step RPE over time
    axs[0].plot(rpe_step, color='k', linewidth=1)
    axs[0].set_title("Step RPE (z-scored)")
    axs[0].set_ylabel("RPE")
    axs[0].set_xlabel("Frame")

    # 2. Beta weights
    axs[1].bar(['Step RPE', 'Hit RPE'], beta_avg, color='gray', edgecolor='k')
    axs[1].set_title("Average Regression Coefficients")
    axs[1].set_ylabel("ΔW per unit regressor")
    for i, b in enumerate(beta_z):
        axs[1].text(i, b, f'{b:.2e}', ha='center', va='bottom' if b > 0 else 'top', fontsize=8)

    # 3. Mean bin plot with valid p-value
    plt.sca(axs[2])
    pf.mean_bin_plot(y_pred_all, y_test_all, 5, 1, 1, 'k')
    axs[2].set_title(f"Predicted vs. Actual ΔW\nr = {r_full:.3f}, p = {p_full:.1e}")
    axs[2].set_ylabel("ΔW (true)")
    axs[2].set_xlabel("ΔW (predicted)")

    plt.suptitle("RPE-Based Model of Synaptic Change", fontsize=14)
    plt.show()

