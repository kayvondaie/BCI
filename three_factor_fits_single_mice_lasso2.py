# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:37:20 2025

@author: kayvon.daie
"""


import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import bci_time_series as bts
from BCI_data_helpers import *
list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]
mice = ["BCI105","BCI106","BCI109","BCI102"]
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
FIT = []
GRP = []
RPE_FIT = []
DW = []
XALL,YALL = [],[]
B,XPRE3 = [],[]

for mi in range(len(mice)):
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean, dot_prod_z
    fit_type      = 'lasso'     #ridge, pinv
    alpha         =  100        #only used for ridge
    num_bins      =  5        # number of bins to calculate correlations
    tau_elig      =  10
    shuffle       =  0
    plotting      =  0
    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) & (list_of_dirs['Has data_main.npy']==True))[0]
    for sii in range(len(session_inds)):
    #for sii in range(si,si+1):
        try:
            print(sii)
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"Skipping session {mouse} {session} — file not found.")
                continue  # <--- Skip to next session
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
                after = np.floor(0.4/dt_si)
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
        
            def get_indices_around_steps(tsta, steps, pre=0, post=0):
                indices = np.searchsorted(tsta, steps)
                all_indices = []
        
                for idx in indices:
                    # Avoid going out of bounds
                    start = max(idx - pre, 0)
                    end = min(idx + post + 1, len(tsta))  # +1 because slicing is exclusive
                    all_indices.extend(range(start, end))
                
                return np.unique(all_indices)
        
        
        
            dt_si = data['dt_si']
            F = data['F']
            #num_bins = F.shape[2]
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
        
            # --- Compute step/reward regressors ---
            for ti in range(trl):
                # Steps regressor
                # steps = data['step_time'][ti]
                # if len(steps) > 0:
                #     indices_steps = get_indices_around_steps(tsta, steps, pre=10, post=0)
                #     indices_steps = indices_steps[indices_steps < F.shape[0]]
                #     kstep[:, ti] = np.nanmean(F[indices_steps, :, ti], axis=0)
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices_rewards = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                    kstep[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
        
                # Rewards regressor
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
                    krewards[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)
                    
                    
        
            # Go cue regressor
            ts = np.where((tsta > 0) & (tsta < 2))[0]
            k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
            
            ts = np.where((tsta > -10) & (tsta < 0))[0]
            kpre = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
        
        
            kstep[np.isnan(kstep)] = 0
            krewards[np.isnan(krewards)] = 0
            k[np.isnan(k)] = 0
        
            trial_bins = np.arange(0,F.shape[2],10)
            trial_bins = np.linspace(0,F.shape[2],num_bins).astype(int)
            cc = np.corrcoef(kstep)
            CCstep = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
            CCrew = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
            CCts = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
            CCpre = np.zeros((cc.shape[0], cc.shape[1], len(trial_bins)))

            
            def centered_dot(A):
                A_centered = A - np.nanmean(A, axis=1, keepdims=True)
                return A_centered @ A_centered.T
            
            # helper: zscore each column (neuron)
            def zscore_mat(X):
                mu = np.nanmean(X, axis=0, keepdims=True)
                sd = np.nanstd(X, axis=0, keepdims=True)
                sd[sd == 0] = 1  # avoid div by zero
                return (X) / sd

            
        
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            hit = np.isnan(rt)==0;
            rt[np.isnan(rt)] = 30
            hit_bin = np.zeros((len(trial_bins),))
            miss_bin = np.zeros((len(trial_bins),))
            rt_bin = np.zeros((len(trial_bins),))
            avg_dot_bin = np.zeros((len(trial_bins),))
            thr_bin = np.zeros((len(trial_bins),))
            rpe_bin = np.zeros((len(trial_bins),))
            hit_rpe_bin = np.zeros((len(trial_bins),))
            miss_rpe_bin = np.zeros((len(trial_bins),))
            
            def compute_rpe(rt, baseline=3.0, tau=10, fill_value=np.nan):
                rpe = np.full_like(rt, np.nan, dtype=np.float64)
                rt_clean = np.where(np.isnan(rt), fill_value, rt)
        
                for i in range(len(rt)):
                    if i == 0:
                        avg = baseline
                    else:
                        start = max(0, i - tau)
                        avg = np.nanmean(rt_clean[start:i]) if i > start else baseline
                    rpe[i] = rt_clean[i] - avg
                return rpe
            rt_rpe = -compute_rpe(rt, baseline=2.0, tau=tau_elig, fill_value=10)
            hit_rpe = compute_rpe(hit, baseline=1, tau=tau_elig, fill_value=0)
            miss_rpe = compute_rpe(hit==0, baseline=0, tau=tau_elig, fill_value=1)
            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
                folder, data, rt, dt_si)
            #trials = np.arange(0,len(hit))
            #miss_rpe = np.convolve((hit==0),np.exp(-trials/tau_elig))[0:len(trials)]
            
            
            for i in range(len(trial_bins)-1):
                ind = np.arange(trial_bins[i],trial_bins[i+1])
                hit_bin[i] = np.nanmean(hit[ind]);
                miss_bin[i] = np.nanmean(hit[ind]==0);
                rt_bin[i] = np.nanmean(rt[ind]);
                avg_dot_bin[i] = np.nanmean(centered_dot(k[:, ind]))
                thr_bin[i] = np.nanmean(BCI_thresholds[1,ind])
                rpe_bin[i] = np.nanmean(rt_rpe[ind])
                miss_rpe_bin[i] = np.nanmean(miss_rpe[ind])
                hit_rpe_bin[i] = np.nanmean(hit_rpe[ind])
                if pairwise_mode == 'noise_corr':    
                    CCrew[:,:,i] = np.corrcoef(krewards[:,ind])
                    CCstep[:,:,i] = np.corrcoef(kstep[:,ind])
                    CCts[:,:,i] = np.corrcoef(k[:,ind]);
                elif pairwise_mode == 'dot_prod':
                    CCrew[:,:,i] = np.dot(krewards[:,ind],krewards[:,ind].T)
                    CCstep[:,:,i] = np.dot(kstep[:,ind],kstep[:,ind].T)
                    CCts[:,:,i] = np.dot(k[:,ind],k[:,ind].T);
                    CCpre[:,:,i] = np.dot(kpre[:,ind],kpre[:,ind].T);
                elif pairwise_mode == 'dot_prod_z':
                    # zscore each neuron to remove mean and variance bias
                    kr = zscore_mat(krewards[:, ind])
                    ks = zscore_mat(kstep[:, ind])
                    kt = zscore_mat(k[:, ind])
                    kp = zscore_mat(kpre[:, ind])
                
                    CCrew[:, :, i] = np.dot(kr, kr.T)
                    CCstep[:, :, i] = np.dot(ks, ks.T)
                    CCts[:, :, i] = np.dot(kt, kt.T)
                    CCpre[:, :, i] = np.dot(kp, kp.T)

                elif pairwise_mode == 'dot_prod_no_mean':
                    CCrew[:, :, i] = centered_dot(krewards[:, ind])
                    CCstep[:, :, i] = centered_dot(kstep[:, ind])
                    CCts[:, :, i] = centered_dot(k[:, ind])
                    CCpre[:,:,i] = centered_dot(kpre[:,ind]);
            # Preallocate combined CC with interleaved shape
            CC = np.zeros((cc.shape[0], cc.shape[1], CCstep.shape[2]*3))
        
            # Interleave step and reward correlations
            CC = np.zeros((cc.shape[0], cc.shape[1], CCstep.shape[2]*4))
            CC[:, :, 0::4] = CCstep
            CC[:, :, 1::4] = CCrew
            CC[:, :, 2::4] = CCts
            CC[:, :, 3::4] = CCpre

        
            #CC = CCrew;
        
        
            import plotting_functions as pf
        
        
            XX = []
            XXstep = []
            XXrew = []
            XXts = []
            XXpre = []
            for i in range(CCstep.shape[2]):
                X = []
                Xstep = []
                Xrew = []
                Xts = []
                Xpre = []
                X2 = []
                Y = []
                Yo = []
                for gi in range(stimDist.shape[1]):
                    cl = np.where((stimDist[:,gi]<10) & (AMP[0][:,gi]> .1) * ((AMP[1][:,gi]> .1)))[0]
                    #plt.plot(favg[0:80,cl,gi])
                    if len(cl)>0:
                        x = np.nanmean(CC[cl,:,i],axis=0)
                        xstep = np.nanmean(CCstep[cl,:,i],axis=0)
                        xrew = np.nanmean(CCrew[cl,:,i],axis=0)
                        xts = np.nanmean(CCts[cl,:,i],axis=0)
                        xpre = np.nanmean(CCpre[cl,:,i],axis=0)
                        # A = AMP[0][cl,gi] + AMP[1][cl,gi]
                        # B = CC[cl,:,i]
                        # x = np.dot(A.T,B)  
                            
                        nontarg = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<1000))
                        y = AMP[1][nontarg,gi]
                        yo = AMP[0][nontarg,gi]
                        Y.append(y)
                        Yo.append(yo)
                        X.append(x[nontarg])
                        
                        Xstep.append(xstep[nontarg])
                        Xrew.append(xrew[nontarg])
                        Xts.append(xts[nontarg])
                        Xpre.append(xpre[nontarg])
                
                
                
                X = np.concatenate(X)
                Xstep = np.concatenate(Xstep)
                Xrew = np.concatenate(Xrew)
                Xts = np.concatenate(Xts)
                Xpre = np.concatenate(Xpre)
        
                Y = np.concatenate(Y,axis=1)
                Yo = np.concatenate(Yo,axis=1)
                XX.append(X)
                XXstep.append(Xstep)
                XXrew.append(Xrew)
                XXts.append(Xts)
                XXpre.append(Xpre)
                
        
            X = np.asarray(XX)
            
            Xstep = np.asarray(XXstep)
            Xrew = np.asarray(XXrew)
            Xts = np.asarray(XXts)
            Xpre = np.asarray(XXpre)
            
            X[np.isnan(X)==1] = 0
            Xstep[np.isnan(Xstep)==1] = 0
            Xrew[np.isnan(Xrew)==1] = 0
            Xts[np.isnan(Xts)==1] = 0
            Xpre[np.isnan(Xpre)==1] = 0
            Y[np.isnan(Y)==1] = 0
            Yo[np.isnan(Yo)==1] = 0
            X_T = X.T  # Shape: (82045, 13)
            Xstep_T = Xstep.T  # Shape: (82045, 13)
            Xrew_T = Xrew.T  # Shape: (82045, 13)
            Xts_T = Xts.T  # Shape: (82045, 13)
            Xpre_T = Xpre.T  # Shape: (82045, 13)
            Y_T = Y.T.ravel() - Yo.T.ravel() # Shape: (82045,) — ravel to make it 1D
            ones_bin = np.ones(rt_bin.shape)
            # Stack behavioral features into one array: shape = (features, trial_bins)
            behavior_features = np.vstack([
                hit_bin,
                hit_rpe_bin,
                10-rt_bin,
                rpe_bin,       
                ones_bin,
            ]).T
            
            B.append(hit_bin)
            stds = np.nanstd(behavior_features, axis=0, keepdims=True)
            stds[stds == 0] = 1
            behavior_features /= stds
        
        
            Xstep_mod  = Xstep_T @ behavior_features
            Xrew_mod   = Xrew_T  @ behavior_features
            Xts_mod    = Xts_T   @ behavior_features
            Xpre_mod   = Xpre_T   @ behavior_features
        
            
            #X_T =X_T[:,:-1] @ behavior_features[:-1,:]
            
            X_T = np.hstack([Xpre_mod, Xts_mod, Xstep_mod, Xrew_mod])
            X_T[np.isnan(X_T)==1] = 0
            
            # Z-score X_T before regression (important for Lasso)
            X_T_mean = np.mean(X_T, axis=0, keepdims=True)
            X_T_std = np.std(X_T, axis=0, keepdims=True)
            X_T_std[X_T_std == 0] = 1  # prevent divide by zero
            
            X_T = (X_T - X_T_mean) / X_T_std

        
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
                
                # Fit based on fit_type
                if fit_type == 'pinv':
                    beta_cv = np.linalg.pinv(X_train) @ Y_train
                    beta = beta_cv  # <-- to make rest of code compatible
                elif fit_type == 'ridge':
                    ridge = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True)
                    ridge.fit(X_train, Y_train)
                    beta_cv = ridge.coef_
                    beta = beta_cv  # <-- to make rest of code compatible
                elif fit_type == 'lasso':
                    alphas = np.logspace(-10, -4, 10)
                    lasso = LassoCV(alphas=alphas, fit_intercept=False, cv=5, max_iter=1000)
                    lasso.fit(X_train, Y_train)
                    beta_cv = lasso.coef_
                    beta = beta_cv  # <-- to make rest of code compatible
                    print(f"Best alpha (lasso): {lasso.alpha_}")

                
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
        
            n_features = behavior_features.shape[1]
            beta_reshaped = beta.reshape(4, n_features)

        
            HI.append(beta_reshaped)
            RT.append(np.nanmean(rt))
            HIT.append(hit)
            FIT.append(np.mean(corr_test))
            GRP.append(favg.shape[2])
            ind = np.where(stimDist> 30);
            dw = np.nanmean(AMP[1][ind] - AMP[0][ind])
            DW.append(dw)
            XALL.append(X_T)
            YALL.append(Y_T)
            XPRE3.append(Xpre_T)
            
            if plotting == 1:
                # Report average correlation & significance
                print("Cross-validation results (mean ± SD):")
                print(f"Train correlation: {np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
                print(f"Train p-value: {np.mean(p_train):.3e}")
        
                print(f"Test correlation: {np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")
                print(f"Test p-value: {np.mean(p_test):.3e}")
                print(f"Test p-value: {np.exp(np.mean(np.log(p_test))):.3e}")
                
                
                # Plotting test set predictions vs actual using mean_bin_plot
                plt.figure(figsize=(6,8))
                plt.subplot(221)
                pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
                plt.xlabel(r'$Behav._t r_{j,t} r_{j,t}$')
                plt.ylabel('$\Delta W$')
                plt.title(data['mouse'] + ' ' + data['session'] + r'  hit = ' + str(round(np.nanmean(hit)*100)) + '%')
            
                # plt.subplot(222)
                # cn = data['conditioned_neuron'][0][0]
                # plt.plot(np.nanmean(F[:,cn,:],axis=1))
                # CNTUN.append(np.nanmean(F[ts,cn,0:10])-np.nanmean(F[0:ts[0],cn,0:10]))
                # plt.title(str(favg.shape[2]) + ' groups ' + f"{np.nanmean(rt):.1f}" + 'sec')
            
            
                
                plt.subplot(212)
                sns.heatmap(beta_reshaped*1000, annot=True, xticklabels=[
                    'rt', 'rt_rpe', 'hit', 'hit_rpe',
                ], yticklabels=['pre','go cue', 'late', 'reward'], cmap='coolwarm', center=0)
                plt.title(r'$\beta$ weights: CC source × behavior feature')
                plt.xlabel('Behavioral feature')
                plt.ylabel('Trial epoch')
                plt.tight_layout()
                plt.show()
            if shuffle == 1:
                shuffle_test(X_T,Y_T)
        except Exception as e:
            print(f"Failed on {mouse} {session}: {e}")
            continue
hi = HI.copy()

# === Grand Fit After Session Loop with 2 Subplots + Full CV Analysis ===

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_functions as pf  # Your custom module

# --- Stack all sessions ---
if mouse == "BCI102":
    X_concat = np.vstack(XALL[3:])
    Y_concat = np.concatenate(YALL[3:])
else:
    aaa = [np.nanstd(x) for x in YALL]
    ind = np.where(np.array(aaa) < .5)[0]  
    XALL = [XALL[i] for i in ind]
    YALL = [YALL[i] for i in ind]
    X_concat = np.vstack(XALL)
    Y_concat = np.concatenate(YALL)


# si = 9
# X_concat = XALL[si]
# Y_concat = YALL[si]

# --- Remove NaNs and infs ---
mask = np.isfinite(Y_concat) & np.all(np.isfinite(X_concat), axis=1)
X_concat = X_concat[mask]
Y_concat = Y_concat[mask]

# --- Z-score across rows ---
X_concat -= np.nanmean(X_concat, axis=0, keepdims=True)
X_concat /= (np.nanstd(X_concat, axis=0, keepdims=True) + 1e-10)
Y_concat -= np.nanmean(Y_concat)
Y_concat /= (np.nanstd(Y_concat) + 1e-10)

alphas = np.logspace(-10, -4, 10)
lasso = LassoCV(alphas=alphas, fit_intercept=False, cv=5, max_iter=10000)
lasso.fit(X_concat, Y_concat)
beta_cv = lasso.coef_
beta = beta_cv  # <-- to make rest of code compatible

# --- Predict using full fit ---
Y_pred = X_concat @ beta
beta_reshaped = beta.reshape(4, n_features)  # 3 trial epochs × 6 behavioral regressors

# === PLOTTING: mean_bin_plot + β matrix ===
fig, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1.2]})

plt.sca(axs[0])
axs[0].set_title("Mouse " + mouse)

sns.heatmap(beta_reshaped * 1000, annot=True, fmt=".1f", ax=axs[1],
            xticklabels=['Hit', '$RPE_{hit}$', 'Speed', '$RPE_{speed}$'],
            yticklabels=['pre','early', 'late', 'reward'],
            cmap='coolwarm', center=0)
axs[1].set_title(r'$\beta$ weights: CC source $\times$ behavior feature')
axs[1].set_xlabel('Behavioral feature')
axs[1].set_ylabel('Trial epoch')

# === CROSS-VALIDATION ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
corr_train, corr_test = [], []
p_train, p_test = [], []
Y_test_all, Y_test_pred_all = [], []

for train_idx, test_idx in kf.split(X_concat):
    X_train, X_test = X_concat[train_idx], X_concat[test_idx]
    Y_train, Y_test = Y_concat[train_idx], Y_concat[test_idx]
    
    lasso_cv = LassoCV(alphas=alphas, fit_intercept=False)
    lasso_cv.fit(X_train, Y_train)
    
    Y_pred_train = X_train @ lasso_cv.coef_
    Y_pred_test = X_test @ lasso_cv.coef_
    
    r_tr, p_tr = pearsonr(Y_pred_train, Y_train)
    r_te, p_te = pearsonr(Y_pred_test, Y_test)
    
    corr_train.append(r_tr)
    corr_test.append(r_te)
    p_train.append(p_tr)
    p_test.append(p_te)
    
    Y_test_all.append(Y_test)
    Y_test_pred_all.append(Y_pred_test)

# --- Concatenate all test folds ---
Y_test_all = np.concatenate(Y_test_all)
Y_test_pred_all = np.concatenate(Y_test_pred_all)

# --- Final test correlation (all pooled test data) ---
r_final, p_final = pearsonr(Y_test_pred_all, Y_test_all)

# === Report ===
print("\nCross-validation results (mean ± SD):")
print(f"Train correlation: {np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
print(f"Train p-value: {np.mean(p_train):.3e}")

print(f"Test correlation: {np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")
print(f"Test p-value: {np.mean(p_test):.3e}")
print(f"Test p-value (geometric mean): {np.exp(np.mean(np.log(p_test))):.3e}")

print(f"Final test correlation (pooled): {r_final:.3f}")
print(f"Final test p-value (pooled): {p_final:.3e}")

# === Optional: Final prediction vs actual plot ===
pf.mean_bin_plot(Y_test_pred_all, Y_test_all)
plt.xlabel(r'Predicted $\Delta W_{ij}$')
plt.ylabel(r'True $\Delta W_{ij}$')
plt.title("Mouse " + mouse)
plt.tight_layout()
plt.show()
  
  
#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:38:07 2025

@author: kayvon.daie
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_functions as pf  # Your custom module


# --- Stack all sessions ---
if mouse == "BCI102":
    X_concat = np.vstack(XALL[3:])
    Y_concat = np.concatenate(YALL[3:])
    valid_idx = list(range(3, len(XALL)))
else:
    aaa = [np.nanstd(x) for x in YALL]
    ind = np.where(np.array(aaa) < 10)[0]
    X_concat = np.vstack([XALL[i] for i in ind])
    Y_concat = np.concatenate([YALL[i] for i in ind])
    valid_idx = ind

# --- Remove NaNs and infs ---
mask = np.isfinite(Y_concat) & np.all(np.isfinite(X_concat), axis=1)
X_concat = X_concat[mask]
Y_concat = Y_concat[mask]

# --- Z-score across rows (global scaling) ---
X_mean, X_std = np.nanmean(X_concat, axis=0, keepdims=True), np.nanstd(X_concat, axis=0, keepdims=True) + 1e-10
Y_mean, Y_std = np.nanmean(Y_concat), np.nanstd(Y_concat) + 1e-10

X_concat = (X_concat - X_mean) / X_std
Y_concat = (Y_concat - Y_mean) / Y_std

# --- Also scale per-session data consistently ---
XALL_session = []
YALL_session = []
for i in valid_idx:
    Xi = (XALL[i] - X_mean) / X_std
    Yi = (YALL[i] - Y_mean) / Y_std
    # drop NaNs
    mask_i = np.isfinite(Yi) & np.all(np.isfinite(Xi), axis=1)
    XALL_session.append(Xi[mask_i])
    YALL_session.append(Yi[mask_i])


from sklearn.linear_model import Ridge

# === Group Ridge Regression ===
def group_ridge(XALL, YALL, lam=10.0, alpha=100.0, max_iter=10):
    n_features = XALL[0].shape[1]
    S = len(XALL)

    # --- pooled (unweighted) beta ---
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(np.vstack(XALL), np.concatenate(YALL))
    beta_pooled = ridge.coef_.copy()   # <-- plain pooled fit

    # init group beta from pooled
    beta_global = beta_pooled.copy()
    betas = [beta_global.copy() for _ in range(S)]

    for it in range(max_iter):
        for s in range(S):
            Xs, Ys = XALL[s], YALL[s]
            # augment with penalty toward global beta
            X_aug = np.vstack([Xs, np.sqrt(lam) * np.eye(n_features)])
            y_aug = np.concatenate([Ys, np.sqrt(lam) * beta_global])
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(X_aug, y_aug)
            betas[s] = ridge.coef_
        beta_global = np.mean(betas, axis=0)

    return betas, beta_global, beta_pooled



# --- Run group ridge ---
betas, beta_global, beta_pooled = group_ridge(XALL_session, YALL_session, lam=100.0, alpha=100.0, max_iter=20)
print("Group ridge finished. Global beta shape:", beta_global.shape)


# --- Predict using global beta ---
Y_pred = X_concat @ beta_global
beta_reshaped = beta_pooled.reshape(4, 4)

# === PLOTTING: mean_bin_plot + β matrix ===
fig, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1.2]})
plt.sca(axs[0])
axs[0].set_title("Mouse " + mouse)

sns.heatmap(beta_reshaped * 1000, annot=True, fmt=".1f", ax=axs[1],
            xticklabels=['Hit', '$RPE_{hit}$', 'Speed', '$RPE_{RT}$'],
            yticklabels=['pre','early', 'late', 'reward'],
            cmap='coolwarm', center=0)
axs[1].set_title(r'$\beta$ weights: CC source $\times$ behavior feature')
axs[1].set_xlabel('Behavioral feature')
axs[1].set_ylabel('Trial epoch')


# === CROSS-VALIDATION (with pooled global beta) ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
corr_train, corr_test = [], []
p_train, p_test = [], []
Y_test_all, Y_test_pred_all = [], []

for train_idx, test_idx in kf.split(X_concat):
    X_train, X_test = X_concat[train_idx], X_concat[test_idx]
    Y_train, Y_test = Y_concat[train_idx], Y_concat[test_idx]

    # just use pooled ridgeCV inside CV for reference
    ridge_cv = RidgeCV(alphas=np.logspace(-2, 6, 20), fit_intercept=False)
    ridge_cv.fit(X_train, Y_train)

    Y_pred_train = X_train @ ridge_cv.coef_
    Y_pred_test = X_test @ ridge_cv.coef_

    r_tr, p_tr = pearsonr(Y_pred_train, Y_train)
    r_te, p_te = pearsonr(Y_pred_test, Y_test)

    corr_train.append(r_tr)
    corr_test.append(r_te)
    p_train.append(p_tr)
    p_test.append(p_te)

    Y_test_all.append(Y_test)
    Y_test_pred_all.append(Y_pred_test)

Y_test_all = np.concatenate(Y_test_all)
Y_test_pred_all = np.concatenate(Y_test_pred_all)
r_final, p_final = pearsonr(Y_test_pred_all, Y_test_all)

print("\nCross-validation results (mean ± SD):")
print(f"Train correlation: {np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
print(f"Test correlation: {np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")
print(f"Final pooled test correlation: {r_final:.3f}")

# === Optional: Final prediction vs actual plot ===
pf.mean_bin_plot(Y_test_pred_all, Y_test_all)
plt.xlabel(r'Predicted $\Delta W_{ij}$')
plt.ylabel(r'True $\Delta W_{ij}$')
plt.title("Mouse " + mouse)
plt.tight_layout()
plt.show()



#%%

# === Compare global vs per-session fits ===
n_sessions = len(XALL)
fig, axs = plt.subplots(3, n_sessions, figsize=(3*n_sessions, 8))

if n_sessions == 1:
    axs = np.array([[axs[0]], [axs[1]]])  # ensure 2D array for consistency

psession = np.zeros(len(XALL),)
ppooled = np.zeros(len(XALL),)
pglobal = np.zeros(len(XALL),)
for i, (X_i, Y_i) in enumerate(zip(XALL, YALL)):
    if i not in ind:
        continue

    # --- Apply the same scaling as used in the grand fit ---
    X_i_scaled = (X_i - np.nanmean(X_concat, axis=0, keepdims=True)) / (
        np.nanstd(X_concat, axis=0, keepdims=True) + 1e-10
    )
    Y_i_scaled = (Y_i - np.nanmean(Y_concat)) / (np.nanstd(Y_concat) + 1e-10)

    # --- Mask bad rows ---
    mask_i = np.isfinite(Y_i_scaled) & np.all(np.isfinite(X_i_scaled), axis=1)
    X_i_scaled = X_i_scaled[mask_i]
    Y_i_scaled = Y_i_scaled[mask_i]

    # --- Predictions with global beta ---
    Y_pred_global = X_i_scaled @ beta_global
    
    Y_pred_pooled = X_i_scaled @ beta_pooled

    # --- Predictions with session-specific beta ---
    Y_pred_session = X_i_scaled @ betas[i]

    r,a = pearsonr(Y_pred_pooled,Y_i_scaled)
    ppooled[i] = a
    r,a = pearsonr(Y_pred_session,Y_i_scaled)
    psession[i] = a
    r,a = pearsonr(Y_pred_global,Y_i_scaled)
    pglobal[i] = a
    # --- Plot (row 0 = global, row 1 = session-specific) ---
    plt.sca(axs[0, i])
    pf.mean_bin_plot(Y_pred_global, Y_i_scaled, col=5, pltt=1, A=1, color='k')
    plt.xlabel(r'Predicted $\Delta W_{ij}$')
    plt.ylabel(r'True $\Delta W_{ij}$')
    plt.title(f"Session {i} — Global β")

    plt.sca(axs[2, i])
    pf.mean_bin_plot(Y_pred_session, Y_i_scaled, col=5, pltt=1, A=1, color='k')
    plt.xlabel(r'Predicted $\Delta W_{ij}$')
    plt.ylabel(r'True $\Delta W_{ij}$')
    plt.title(f"Session {i} — Session β")
    
    plt.sca(axs[1, i])
    pf.mean_bin_plot(Y_pred_pooled, Y_i_scaled, col=5, pltt=1, A=1, color='k')
    plt.xlabel(r'Predicted $\Delta W_{ij}$')
    plt.ylabel(r'True $\Delta W_{ij}$')
    plt.title(f"Session {i} — Pooled β")

plt.tight_layout()
plt.show()
#plt.scatter(-np.log(psession),-np.log(ppooled))
#%%
cc = np.zeros((len(HI),len(HI)))
dd = np.zeros((len(HI),len(HI)))
for i in range(len(HI)):
    for j in range(len(HI)):
        cc[i,j],p = pearsonr(HI[i].flatten(),HI[j].flatten())         
        dd[i,j] = np.abs(i-j)
ind = np.where(dd.flatten()>0)        
#plt.scatter(dd.flatten()[ind],cc.flatten()[ind])
pf.mean_bin_plot(dd.flatten()[ind],cc.flatten()[ind],3)