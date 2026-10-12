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
import re
HI, HIb, HIc = [], [], []
RT, RPE, HIT = [], [], []
HIT_RATE,D_HIT_RATE,DOT, TRL, THR = [], [], [], [], []
CC_RPE, CC_RT, CC_MIS, CORR_RPE, CORR_RT = [], [], [], [], []
RT_WINDOW, HIT_WINDOW, THR_WINDOW = [], [], []
PTRL, PVAL, RVAL = [], [], []
Ddirect, Dindirect, CCdirect = [], [], []
NUM_STEPS,TST, CN, MOUSE, SESSION = [], [], [], [],[]

mice = ["BCI102","BCI105","BCI106","BCI109","BCI103","BCI104","BCI93","BCI107"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    #session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
    si = 11
    
    pairwise_mode = 'dot_prod_no_mean'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'ridge'     #ridge, pinv
    alpha         =  .1        #only used for ridge
    epoch         =  'reward'  # reward, step, trial_start
    
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
        cn = data['conditioned_neuron'][0][0]
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
        rpe = compute_rpe(rt, baseline=10, tau=20, fill_value=50)

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
    
        # Compute pseudoinverse solution
        if fit_type == 'pinv':
            beta = np.linalg.pinv(X_T) @ Y_T  # (13, 1)
        elif fit_type == 'ridge':
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha, fit_intercept=False)
            ridge.fit(X_T, Y_T)
            beta = ridge.coef_
        elif fit_type == 'slr':
            from sklearn.linear_model import LinearRegression
            b,c,intercept,ptrl = [],[],[],[]
            inds = np.arange(0,X.shape[0])[:]
            for i in range(len(inds)):
                x = X[inds[i], :].reshape(-1, 1)
                y = (Y - Yo).flatten()
                model = LinearRegression().fit(x, y)
                b.append(model.coef_[0])
                c.append(np.corrcoef((Y-Yo).flatten(),X[inds[i],:])[0,1])
                _, d = pearsonr((Y-Yo).flatten(),X[inds[i],:])
                ptrl.append(d)
                intercept.append(model.intercept_)
            
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
    
        for train_idx, test_idx in kf.split(X_T):
            # Split data
            X_train, X_test = X_T[train_idx], X_T[test_idx]
            Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
            
            # Fit regression on training set            
            if fit_type == 'pinv':
                beta_cv = np.linalg.pinv(X_train) @ Y_train
            elif fit_type == 'ridge':
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha, fit_intercept=False)
                ridge.fit(X_train, Y_train)
                beta_cv = ridge.coef_
            elif fit_type == 'slr':
                from sklearn.linear_model import LinearRegression
                b,c,intercept,ptrl = [],[],[],[]
                inds = np.arange(0,X.shape[0])[:]
                for i in range(len(inds)):
                    x = X_train
                    y = Y_train
                    model = LinearRegression().fit(x, y)
                    b.append(model.coef_[0])
                    c.append(np.corrcoef((Y-Yo).flatten(),X[inds[i],:])[0,1])
                    _, d = pearsonr((Y-Yo).flatten(),X[inds[i],:])
                    ptrl.append(d)
                    intercept.append(model.intercept_)
                beta_cv = np.array(b)
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
        plt.figure(figsize=(8,6))
        plt.subplot(231)
        pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
        plt.xlabel(r'$HI r_i r_j$')
        plt.ylabel('$\Delta W$')
        plt.title('Cross-validated predictions vs actual')
     
        
        from sklearn.linear_model import LinearRegression
        b,c,intercept,ptrl = [],[],[],[]
        inds = np.arange(0,X.shape[0])[:]
        for i in range(len(inds)):
            x = X[inds[i], :].reshape(-1, 1)
            y = (Y - Yo).flatten()
            model = LinearRegression().fit(x, y)
            b.append(model.coef_[0])
            c.append(np.corrcoef((Y-Yo).flatten(),X[inds[i],:])[0,1])
            _, d = pearsonr((Y-Yo).flatten(),X[inds[i],:])
            ptrl.append(d)
            intercept.append(model.intercept_)
        
       
        PTRL.append(ptrl)
        HIb.append(np.asarray(b))
        HIc.append(c)
        HIT.append(hit_bin)
        DOT.append(avg_dot_bin.T)
        TRL.append(trial_bins.T)
        THR.append(thr_bin.T)
        RPE.append(rpe.T)    
        RT.append(rt)
        
        plt.subplot(232)
        plt.plot(trial_bins, np.asarray(b) / max(np.abs(np.asarray(b)))*10, 'k', label='HI')
        plt.plot(trial_bins, rpe_bin, 'b', label='RPE')
        plt.plot(trial_bins, rt_bin, 'r', label='rew time')
        plt.legend(fontsize=4, loc='upper left')  # adjust fontsize and position
        plt.title(mouse + ' ' + session)
        plt.xlabel('Trial #')
    
        
       
        
       
        HIT_RATE.append(np.nanmean(hit[0:40]))
        D_HIT_RATE.append(np.nanmean(hit[20:70]) - np.nanmean(hit[0:20]))
        try:
            cn = data['conditioned_neuron'][0][0]
        except:
            cn = []    
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii-1]]
            session = list_of_dirs['Session'][session_inds[sii-1]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'    
            old = ddct.load_hdf5(folder,bci_keys = ['F'],photostim_keys = [])
            Fo = old['F']
            plt.subplot(4,3,7);
            plt.plot(np.nanmean(F[:,cn,:],axis=1),'m')
            if Fo.shape[1] == F.shape[1]:
                plt.plot(np.nanmean(Fo[:,cn,:],axis=1),'k')
        except:
            continue
        CN.append(cn)    
        plt.tight_layout()
        
        plt.subplot(4,3,8)
        plt.plot(hit)
        
        plt.subplot(4,3,11)
        smooth = 1
        fcn_post = np.nanmean(F[:,cn,:],axis=0)
        fcn_post = np.convolve(fcn_post,np.ones(smooth,)/smooth)
        plt.plot(fcn_post[smooth:-smooth])
      
        # plt.subplot(236)
        # plt.scatter(rpe_bin,rt_bin)
        # AVG_RPE.append(np.nanmean(rpe[0:40]))
        window = 10
        corr_rpe = np.zeros(len(b)-window)
        corr_rt = np.zeros(len(b)-window)
        rt_window = np.zeros(len(b)-window)
        hit_window = np.zeros(len(b)-window)
        thr_window = np.zeros(len(b)-window)
        for i in range(len(b)-window):
            corr_rpe[i] = np.corrcoef(b[i:i+window],rpe_bin[i:i+window])[0,1]
            corr_rt[i] = np.corrcoef(b[i:i+window],rt_bin[i:i+window])[0,1]
            rt_window[i] = np.nanmean(rt_bin[i:i+window])
            hit_window[i] = np.nanmean(hit_bin[i:i+window])
            thr_window[i] = np.nanmax(thr_bin[i:i+window])
            
        plt.subplot(236)
        #plt.plot(trial_bins[:-window], corr_rpe,'k')
        plt.plot(trial_bins[:-window], corr_rpe,'k');
        plt.plot(trial_bins[:-window],(thr_window-np.nanmean(thr_window))/np.nanmean(thr_window))
        plt.xlabel('Trial #')
        plt.ylabel('HI corr with RPE')
        
        # plt.subplot(235)
        # pf.mean_bin_plot(rt_window,corr_rpe,5,1,1,'k')
        
        CORR_RPE.append(corr_rpe)
        CORR_RT.append(corr_rt)
        RT_WINDOW.append(rt_window) 
        HIT_WINDOW.append(hit_window)
        THR_WINDOW.append(thr_window)
        # Use flattening to compare individual (cell, stim target) entries
        Ddirect.append(np.nanmean((AMP[1] - AMP[0]).flatten()[np.where(stimDist.flatten() < 10)[0]]))
        Dindirect.append(np.nanmean((AMP[1] - AMP[0]).flatten()[np.where(stimDist.flatten() > 30)[0]]))
        MOUSE.append(mouse)
        SESSION.append(session)

        
        ind = np.where((stimDist.flatten()>0)&(stimDist.flatten()<10))
        CCdirect.append(np.corrcoef(AMP[0].flatten()[ind],AMP[1].flatten()[ind])[0,1])
        
        dfcn_clean = np.nan_to_num(dfcn)
        step_vector_clean = np.nan_to_num(step_vector)
        early_trial = np.where(trial_start_vector == 1)[0][10]
        early_trial = len(dfcn_clean)
        xcorr = correlate(dfcn_clean[0:early_trial], step_vector_clean[0:early_trial], mode='full')

        lags = np.arange(-len(dfcn[0:early_trial])+1, len(dfcn[0:early_trial])) * dt_si
        mid = len(lags) // 2
        xwindow = int(np.round(2/dt_si))
        plt.subplot(4,3,10)
        plt.plot(lags[mid - xwindow:mid + xwindow],
                 xcorr[mid - xwindow:mid + xwindow], color='teal')
        
        trial_mismatch_score, smooth_step_trial, target_similarity_trial, _ = compute_trial_mismatch_score(
            df_closedloop=data['df_closedloop'],
            target_activity_pattern=Q,
            trial_start_vector=trial_start_vector,
            step_vector=step_vector,
            tau=400,
            target_window=1,
            scale_factor=5
        )
#        pf.mean_bin_plot(trial_mismatch_score,b,5,1,1,'k')
#        plt.show()
        
        
        plt.subplot(233)
        cc_rpe = (np.corrcoef(rpe_bin[:-1],np.asarray(b)[:-1])[1,0])
        cc_rt = (np.corrcoef(rt_bin[:-1],np.asarray(b)[:-1])[1,0])
        cc_mis = (np.corrcoef(target_similarity_trial[:-1],np.asarray(b)[:-1])[1,0])
        plt.bar(['RPE', 'RT','Mismatch'], [cc_rpe, cc_rt, cc_mis],color = 'k')
        plt.ylabel('Correlation with b')
        plt.title('Correlation of b with RPE and RT')
        plt.tight_layout()
        plt.show()
        
        TST.append(target_similarity_trial)
        CC_RPE.append(cc_rpe)
        CC_RT.append(cc_rt)
        CC_MIS.append(cc_mis)
        NUM_STEPS.append([len(x) for x in data['step_time']])
#%%
hib = [hib[:]/np.nanstd(hib[0:]) for hib in HIb]
_,_,p = pf.mean_bin_plot(np.concatenate(TST),np.concatenate(hib),5,1,1,'k');
print(p)
plt.xlabel('Expected reward')
plt.ylabel('Hebbian index at reward')
x = np.array(CC_MIS);x=x[~np.isnan(x)]
h, p = ttest_1samp(x, popmean=0)
print(p)
#%%
cc,x,y = [],[],[]
for i in range(len(RT)):
    rt = RT[i]
    rpe = RPE[i]
    hib = HIb[i]
    hib = hib/np.nanstd(hib)    
    tst = TST[i]        
    ind = np.where((rt > 0) & (hib!=0))[0]
    ind = np.where((rt > 0) & (hib!=0) & (np.isnan(hib)==0))[0]
    rpe = compute_rpe(rt[ind], baseline=10, tau=1, fill_value=0)
    cc.append(np.corrcoef(rpe[:-1],tst[ind][1:])[0,1])
    x.append(rpe[:-1])
    y.append(tst[ind][1:])
    #plt.scatter(hib[ind],rpe[ind])

pf.mean_bin_plot(np.concatenate(y),np.concatenate(x),4)
c,p = pearsonr(np.concatenate(x),np.concatenate(y))
np.nanmean(np.array(cc)<0)
#%%
#%% What predicts the sign/magnitude of session-level correlations?

session_features = []

for idx in range(len(HIb)):
    session_features.append({
        'cc_rpe': CC_RPE[idx],
        'cc_rt': CC_RT[idx],
        'cc_mis': CC_MIS[idx],
        'abs_cc_rpe': abs(CC_RPE[idx]) if np.isfinite(CC_RPE[idx]) else np.nan,
        'abs_cc_rt': abs(CC_RT[idx]) if np.isfinite(CC_RT[idx]) else np.nan,
        'abs_cc_mis': abs(CC_MIS[idx]) if np.isfinite(CC_MIS[idx]) else np.nan,
        'mean_thr': np.nanmean(THR[idx]),
        'mean_hit': np.nanmean(HIT[idx]),
        'mean_tst': np.nanmean(TST[idx]),
        'd_hit': D_HIT_RATE[idx] if idx < len(D_HIT_RATE) else np.nan,
        'mouse': MOUSE[idx],
    })

df_sess = pd.DataFrame(session_features)
df_sess = df_sess.dropna()

print("="*70)
print("WHAT PREDICTS WHICH RULE A SESSION USES?")
print("="*70)

# Test what predicts RPE usage (positive vs negative)
print("\n1. What predicts sign of RPE correlation?")
for var in ['mean_thr', 'mean_hit', 'mean_tst', 'd_hit']:
    r, p = spearmanr(df_sess[var], df_sess['cc_rpe'])
    print(f"   {var:12s}: r={r:+.3f}, p={p:.4f}")

# Test what predicts MAGNITUDE of relationship
print("\n2. What predicts STRENGTH of relationships?")
print("   (Does anything predict when rules are strong?)")
for var in ['mean_thr', 'mean_hit', 'mean_tst']:
    r_rpe, p_rpe = spearmanr(df_sess[var], df_sess['abs_cc_rpe'])
    r_rt, p_rt = spearmanr(df_sess[var], df_sess['abs_cc_rt'])
    r_mis, p_mis = spearmanr(df_sess[var], df_sess['abs_cc_mis'])
    
    print(f"\n   {var}:")
    print(f"     vs |CC_RPE|: r={r_rpe:+.3f}, p={p_rpe:.4f}")
    print(f"     vs |CC_RT|:  r={r_rt:+.3f}, p={p_rt:.4f}")
    print(f"     vs |CC_MIS|: r={r_mis:+.3f}, p={p_mis:.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sign of RPE effect vs performance
axes[0, 0].scatter(df_sess['mean_hit'], df_sess['cc_rpe'], s=100, alpha=0.7)
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_xlabel('Mean Hit Rate')
axes[0, 0].set_ylabel('CC_RPE (HI~RPE correlation)')
axes[0, 0].set_title('Does performance predict RPE rule sign?')

# Magnitude vs threshold
axes[0, 1].scatter(df_sess['mean_thr'], df_sess['abs_cc_rpe'], s=100, alpha=0.7, label='RPE')
axes[0, 1].scatter(df_sess['mean_thr'], df_sess['abs_cc_mis'], s=100, alpha=0.7, label='TST')
axes[0, 1].set_xlabel('Mean Threshold')
axes[0, 1].set_ylabel('|Correlation|')
axes[0, 1].set_title('Does threshold predict rule strength?')
axes[0, 1].legend()

# Compare magnitudes across features
axes[1, 0].bar(['RPE', 'RT', 'TST'], 
              [df_sess['abs_cc_rpe'].mean(), df_sess['abs_cc_rt'].mean(), df_sess['abs_cc_mis'].mean()],
              alpha=0.7)
axes[1, 0].set_ylabel('Mean |Correlation|')
axes[1, 0].set_title('Which feature has strongest relationships?')

# Distribution of correlation signs
axes[1, 1].hist(df_sess['cc_rpe'], bins=15, alpha=0.5, label='RPE')
axes[1, 1].hist(df_sess['cc_rt'], bins=15, alpha=0.5, label='RT')
axes[1, 1].hist(df_sess['cc_mis'], bins=15, alpha=0.5, label='TST')
axes[1, 1].axvline(0, color='k', linestyle='--')
axes[1, 1].set_xlabel('Correlation with HI')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Distribution of session-level correlations')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
#%% Visualize the learning rate effect

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Main finding: d_hit vs CC_RPE
axes[0].scatter(df_sess['d_hit'], df_sess['cc_rpe'], s=100, alpha=0.7)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_xlabel('Learning Rate (Δ Hit Rate)')
axes[0].set_ylabel('RPE Correlation (CC_RPE)')
axes[0].set_title(f'Learning Rate Predicts RPE Rule\nr={-0.407:.3f}, p=0.017')

# Add quadrant labels
axes[0].text(0.05, 0.1, 'Improving\n+ Error-driven', transform=axes[0].transAxes, 
            ha='left', va='top', fontsize=10, alpha=0.5)
axes[0].text(0.05, 0.9, 'Improving\n+ Success-driven', transform=axes[0].transAxes,
            ha='left', va='bottom', fontsize=10, alpha=0.5)

# Does this also affect RT rule?
r_rt, p_rt = spearmanr(df_sess['d_hit'], df_sess['cc_rt'])
axes[1].scatter(df_sess['d_hit'], df_sess['cc_rt'], s=100, alpha=0.7)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Learning Rate (Δ Hit Rate)')
axes[1].set_ylabel('RT Correlation (CC_RT)')
axes[1].set_title(f'RT Rule\nr={r_rt:.3f}, p={p_rt:.3f}')

# Does this affect TST rule?
r_mis, p_mis = spearmanr(df_sess['d_hit'], df_sess['cc_mis'])
axes[2].scatter(df_sess['d_hit'], df_sess['cc_mis'], s=100, alpha=0.7)
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[2].set_xlabel('Learning Rate (Δ Hit Rate)')
axes[2].set_ylabel('TST Correlation (CC_MIS)')
axes[2].set_title(f'TST Rule\nr={r_mis:.3f}, p={p_mis:.3f}')

plt.tight_layout()
plt.show()

# Split sessions by learning phase
df_sess['learning_phase'] = df_sess['d_hit'] > 0

print("\n" + "="*70)
print("PLASTICITY RULES BY LEARNING PHASE")
print("="*70)

for phase in [True, False]:
    label = "IMPROVING" if phase else "PLATEAU/DECLINING"
    subset = df_sess[df_sess['learning_phase'] == phase]
    
    print(f"\n{label} sessions (n={len(subset)}):")
    print(f"  Mean CC_RPE: {subset['cc_rpe'].mean():+.3f}")
    print(f"  Mean CC_RT:  {subset['cc_rt'].mean():+.3f}")
    print(f"  Mean CC_MIS: {subset['cc_mis'].mean():+.3f}")


#%% Final publication figure - CONSISTENT COLORS

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

color_improving = '#0173B2'  # Blue
color_plateau = '#DE8F05'     # Orange

# Panel A: Main effect - FIXED to match B/C
ax1 = fig.add_subplot(gs[0, :])
improving_mask = df_sess['d_hit'] > 0
plateau_mask = df_sess['d_hit'] <= 0

ax1.scatter(df_sess[improving_mask]['d_hit'], df_sess[improving_mask]['cc_rpe'], 
           c=color_improving, s=100, alpha=0.7, edgecolors='black', label='Improving (n=10)')
ax1.scatter(df_sess[plateau_mask]['d_hit'], df_sess[plateau_mask]['cc_rpe'], 
           c=color_plateau, s=100, alpha=0.7, edgecolors='black', label='Plateau (n=24)')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=2)
ax1.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=2)
ax1.set_xlabel('Learning Rate (Δ Hit Rate)', fontsize=12)
ax1.set_ylabel('RPE→HI Correlation', fontsize=12)
ax1.set_title('A. Learning Phase Predicts Plasticity Strategy\n(r=-0.41, p=0.017)', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Panel B: Mean correlations by phase
ax2 = fig.add_subplot(gs[1, 0])
improving = df_sess[df_sess['d_hit'] > 0]
plateau = df_sess[df_sess['d_hit'] <= 0]

x = np.arange(3)
width = 0.35
ax2.bar(x - width/2, [improving['cc_rpe'].mean(), improving['cc_rt'].mean(), improving['cc_mis'].mean()],
       width, label='Improving (n=10)', color=color_improving, alpha=0.7, edgecolor='black')
ax2.bar(x + width/2, [plateau['cc_rpe'].mean(), plateau['cc_rt'].mean(), plateau['cc_mis'].mean()],
       width, label='Plateau (n=24)', color=color_plateau, alpha=0.7, edgecolor='black')
ax2.axhline(0, color='k', linestyle='-', linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(['RPE', 'RT', 'TST'])
ax2.set_ylabel('Mean Correlation with HI', fontsize=12)
ax2.set_title('B. Plasticity Rules by Learning Phase', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Panel C: TST effect is plateau-driven
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(improving['cc_mis'], np.ones(len(improving))*1, s=100, alpha=0.7, 
           color=color_improving, edgecolors='black', label=f'Improving: {improving["cc_mis"].mean():.3f}')
ax3.scatter(plateau['cc_mis'], np.ones(len(plateau))*0, s=100, alpha=0.7,
           color=color_plateau, edgecolors='black', label=f'Plateau: {plateau["cc_mis"].mean():.3f}')
ax3.axvline(0, color='k', linestyle='--', linewidth=2)
ax3.set_xlabel('TST→HI Correlation', fontsize=12)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Plateau', 'Improving'])
ax3.set_title('C. TST Effect is Consolidation-Specific', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# Panel D & E: Example sessions
ax4 = fig.add_subplot(gs[2, 0])
improving_idx = improving.index[0]
if improving_idx < len(HIb):
    ax4.plot(HIb[improving_idx], color=color_improving, linewidth=2)
    ax4.set_xlabel('Trial', fontsize=10)
    ax4.set_ylabel('Hebbian Index', fontsize=10)
    ax4.set_title(f'D. Example Improving Session\n(Δhit={improving.iloc[0]["d_hit"]:.2f})', 
                 fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
plateau_idx = plateau.index[0]
if plateau_idx < len(HIb):
    ax5.plot(HIb[plateau_idx], color=color_plateau, linewidth=2)
    ax5.set_xlabel('Trial', fontsize=10)
    ax5.set_ylabel('Hebbian Index', fontsize=10)
    ax5.set_title(f'E. Example Plateau Session\n(Δhit={plateau.iloc[0]["d_hit"]:.2f})', 
                 fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3)

plt.suptitle('Adaptive Plasticity Rules Switch Between Learning and Consolidation', 
            fontsize=16, fontweight='bold', y=0.995)
plt.show()