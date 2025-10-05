# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os
plt.figure(figsize = (12,4))
for crei in range(3):
        
    if crei == 0:
        mice = ["BCINM_034", "BCINM_031"];creline = ['5-HT']
    elif crei == 1:
        mice = ["BCINM_027","BCINM_017"];creline = ['NE']
    elif crei == 2:
        mice = ["BCINM_024","BCINM_021"];creline = ['Ach']
    
    sessions = session_counting.counter2(mice,'010112',has_pophys=False)

    from scipy.signal import medfilt, correlate
    from scipy.stats import pearsonr
    p = r"C:\Users\kayvon.daie\Documents\GitHub\BCI\LC_axon_analysis"
    assert os.path.isdir(p), f"Not a directory: {p}"
    if p not in sys.path:
        sys.path.insert(0, p)  # use insert so this path is searched first
    from axon_helper_module import *  # or: from axon_helper_module import whatever_you_need
    from BCI_data_helpers import *
    import bci_time_series as bts
    
    processing_mode = 'all'
    si = 10
    inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)
    XCORR, LAGS, SESSION = [], [], []
    RTA_low,RTA_high,CC_RPE,PP_RPE = [],[],[],[]
    RPE,RT,RPE_all,RT_all= [],[],[],[]
    RTA,STA = [],[]
    num = 1000
    plot = 1
    
    def detrend_across_trials(data, order=1, lam=1e-6):
        """
        Remove slow drift across trials.
        Fits a polynomial trend across trials to the trial-wise mean fluorescence.
        Subtracts that trend from each trial (same value for all timepoints in a trial).
        
        Parameters:
            data: array (timepoints, trials)
            order: polynomial order (1 = linear)
            lam: ridge penalty (small value to stabilize)
        
        Returns:
            Detrended data (same shape)
        """
        T, N = data.shape
        trial_means = np.nanmean(data, axis=0)  # (trials,)
        x = np.arange(N)
        
        # Design matrix
        X = np.column_stack([x**k for k in range(1, order+1)] + [np.ones_like(x)])
        I = np.eye(X.shape[1])
    
        mask = np.isfinite(trial_means)
        if mask.sum() < (order + 1):
            trend = np.full(N, np.nanmean(trial_means))
        else:
            beta = np.linalg.solve(X[mask].T @ X[mask] + lam * I, X[mask].T @ trial_means[mask])
            trend = X @ beta  # (trials,)
    
        # Subtract trend from all timepoints in each trial
        return data - trend[None, :]
    
    
    for i in inds:
        try:
            print(i)
            mouse = sessions['Mouse'][i]
            session = sessions['Session'][i]
        
            # Load data
            try:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
                main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
                data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
            except:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
                main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
                data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
        
            # Timing and behavioral signals
            dt_si = data['dt_si']
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            rt[np.isnan(rt)] = 20
            dfaxon = data['ch1']['df_closedloop']
        
            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
        
            # Compute RPE
            rpe = compute_rpe(rt == 20, baseline=1, window=20, fill_value=50)
        
            # Reward-aligned responses
            rta = reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4))
            sta = reward_aligned_responses(dfaxon, trial_start_vector, dt_si, window=(-2, 4))
        
            # Get rewarded frame indices
            rewarded_frames = np.where(reward_vector > 0)[0]
            pre_pts = int(2 / dt_si)
            post_pts = int(4 / dt_si)
            valid_reward_frames = rewarded_frames[
                (rewarded_frames - pre_pts >= 0) &
                (rewarded_frames + post_pts < dfaxon.shape[1])
            ]
            valid_reward_frames = valid_reward_frames[:rta.shape[2]]  # match rta
        
            # --- Map reward frames to trial indices using trial_start_vector ---
            trial_starts = np.where(trial_start_vector > 0)[0]  # one per trial
            assert len(trial_starts) == len(rpe), "Trial count mismatch"
            
            # Match number of reward-aligned frames to rta
            valid_reward_frames = valid_reward_frames[:rta.shape[2]]
            
            # Find the trial each reward frame belongs to
            reward_trial_inds = np.searchsorted(trial_starts, valid_reward_frames, side='right') - 1
            
            # Keep only valid trial indices
            reward_trial_inds = reward_trial_inds[(reward_trial_inds >= 0) & (reward_trial_inds < len(rpe))]
            
            # Get corresponding RPE values
            reward_trial_inds = np.arange(0,len(rpe))
            rpe_rewarded = rpe[reward_trial_inds]
        
        
            # Sanity check
        
            # Average across neurons → (time, trials)
            hit = np.where(rt!=20)[0];
            a = data['ch1']['F'][:,:,:]
            mean_rta = np.nanmean(a, axis=1)  # shape (timepoints, trials)
            mean_rta = detrend_across_trials(mean_rta, order=1)  # remove linear drift
            mean_rta[:, 0:-1] = mean_rta[:, 1:]  # optional: shift forward by one trial
    
        
            # Bin trials into low / medium / high RPE groups
            rpe_rewarded = 10 - rt
            percentiles = np.percentile(rpe_rewarded, [33, 66])
            low_inds  = np.where(rpe_rewarded <= percentiles[0])[0]
            med_inds  = np.where((rpe_rewarded > percentiles[0]) & (rpe_rewarded <= percentiles[1]))[0]
            high_inds = np.where(rpe_rewarded > percentiles[1])[0]
        
            # Time vector
            n_timepoints = mean_rta.shape[0]
            time = np.linspace(-2, 10, n_timepoints)
            
            # for ii in range(mean_rta.shape[1]):
            #     bl = np.nanmean(mean_rta[35:45,ii])
            #     mean_rta[:,ii] = mean_rta[:,ii] - bl; 
            
            # Plot PSTHs
       
            #RTA_low.append(np.nanmean(mean_rta[:,low_inds],1))
            #RTA_high.append(np.nanmean(mean_rta[:,high_inds],1))
            RTA_low.append(mean_rta[:,low_inds])        
            RTA_high.append(mean_rta[:,high_inds])
            # plt.show()
            pp,cc = [],[]
            for ti in range(mean_rta.shape[0]):
                a = mean_rta[ti, :]
                a[np.isnan(a)] = 0
                r, p = pearsonr(a, rpe_rewarded)
                cc.append(r)
                pp.append(p)
            CC_RPE.append(cc)
            PP_RPE.append(pp)
            RPE.append(rpe_rewarded)
            RPE_all.append(rpe)
            RT_all.append(rt)
            RTA.append(mean_rta)
            STA.append(np.nanmean(sta,axis=1))
            RT.append(rt[reward_trial_inds])
        except:
            continue

    plt.subplot(1,3,crei+1)    
    for i in range(2):
        if i == 1:
            color = 'r'
            a = np.concatenate(RTA_high,1)
        else:
            color = 'b'        
            a = np.concatenate(RTA_low,1)
        #a = a - np.tile(np.mean(a[:20, :], axis=0), (a.shape[0], 1))
        trace = np.nanmean(a, 1)
        sem   = np.nanstd(a, 1) / np.sqrt(a.shape[1])
        plt.plot(time, trace, color=color)
        plt.fill_between(time, trace - sem, trace + sem, color=color, alpha=0.3)
    plt.title(creline)
plt.tight_layout()

