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
si = 5

pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
fit_type      = 'ridge'     #ridge, pinv
alpha         =  100        #only used for ridge
num_bins      =  30        # number of bins to calculate correlations
tau_elig      =  10
shuffle       =  0

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
        steps = data['step_time'][ti]
        if len(steps) > 0:
            indices_steps = get_indices_around_steps(tsta, steps, pre=10, post=0)
            indices_steps = indices_steps[indices_steps < F.shape[0]]
            kstep[:, ti] = np.nanmean(F[indices_steps, :, ti], axis=0)

        # Rewards regressor
        rewards = data['reward_time'][ti]
        if len(rewards) > 0:
            indices_rewards = get_indices_around_steps(tsta, rewards, pre=1, post=10)
            indices_rewards = indices_rewards[indices_rewards < F.shape[0]]
            krewards[:, ti] = np.nanmean(F[indices_rewards, :, ti], axis=0)

    # Go cue regressor
    ts = np.where((tsta > 0) & (tsta < 10))[0]
    k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)


    kstep[np.isnan(kstep)] = 0
    krewards[np.isnan(krewards)] = 0
    k[np.isnan(k)] = 0

    trial_bins = np.arange(0,F.shape[2],10)
    trial_bins = np.linspace(0,F.shape[2],num_bins).astype(int)
    cc = np.corrcoef(kstep)
    CCstep = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
    CCrew = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
    CCts = np.zeros((cc.shape[0],cc.shape[1],len(trial_bins)))
    
    def centered_dot(A):
        A_centered = A - A.mean(axis=1, keepdims=True)
        return A_centered @ A_centered.T

    

    rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
    hit = np.isnan(rt)==0;
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
    hit_rpe = compute_rpe(hit, baseline=0, tau=tau_elig, fill_value=0)
    miss_rpe = compute_rpe(hit==0, baseline=0, tau=tau_elig, fill_value=1)
    
    
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
        elif pairwise_mode == 'dot_prod_no_mean':
            CCrew[:, :, i] = centered_dot(krewards[:, ind])
            CCstep[:, :, i] = centered_dot(kstep[:, ind])
            CCts[:, :, i] = centered_dot(k[:, ind])
    # Preallocate combined CC with interleaved shape
    CC = np.zeros((cc.shape[0], cc.shape[1], CCstep.shape[2]*3))

    # Interleave step and reward correlations
    CC[:, :, 0::3] = CCstep
    CC[:, :, 1::3] = CCrew
    CC[:, :, 2::3] = CCts

    #CC = CCrew;


    import plotting_functions as pf


    XX = []
    XXstep = []
    XXrew = []
    XXts = []
    for i in range(CCstep.shape[2]):
        X = []
        Xstep = []
        Xrew = []
        Xts = []
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
        
        
        
        X = np.concatenate(X)
        Xstep = np.concatenate(Xstep)
        Xrew = np.concatenate(Xrew)
        Xts = np.concatenate(Xts)

        Y = np.concatenate(Y,axis=1)
        Yo = np.concatenate(Yo,axis=1)
        XX.append(X)
        XXstep.append(Xstep)
        XXrew.append(Xrew)
        XXts.append(Xts)
        

    X = np.asarray(XX)
    
    Xstep = np.asarray(XXstep)
    Xrew = np.asarray(XXrew)
    Xts = np.asarray(XXts)
    
    X[np.isnan(X)==1] = 0
    Xstep[np.isnan(Xstep)==1] = 0
    Xrew[np.isnan(Xrew)==1] = 0
    Xts[np.isnan(Xts)==1] = 0
    Y[np.isnan(Y)==1] = 0
    Yo[np.isnan(Yo)==1] = 0
    X_T = X.T  # Shape: (82045, 13)
    Xstep_T = Xstep.T  # Shape: (82045, 13)
    Xrew_T = Xrew.T  # Shape: (82045, 13)
    Xts_T = Xts.T  # Shape: (82045, 13)
    Y_T = Y.T.ravel() - Yo.T.ravel() # Shape: (82045,) — ravel to make it 1D

    # Stack behavioral features into one array: shape = (features, trial_bins)
    behavior_features = np.vstack([
        rt_bin,
        rpe_bin,        # Time-to-reward RPE
        hit_bin,        # Hit rate
        hit_rpe_bin,
        miss_bin,       # Miss rate        
        miss_rpe_bin    # Miss RPE
    ]).T
    
    
    stds = np.nanstd(behavior_features, axis=0, keepdims=True)
    stds[stds == 0] = 1
    behavior_features /= stds


    Xstep_mod = Xstep_T @ behavior_features
    Xrew_mod  = Xrew_T  @ behavior_features
    Xts_mod   = Xts_T   @ behavior_features

    
    #X_T =X_T[:,:-1] @ behavior_features[:-1,:]
    
    X_T = np.hstack([Xstep_mod, Xrew_mod, Xts_mod])
    X_T[np.isnan(X_T)==1] = 0
    
    # Compute pseudoinverse solution
    if fit_type == 'pinv':
        beta = np.linalg.pinv(X_T) @ Y_T  # (13, 1)
    elif fit_type == 'ridge':
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha, fit_intercept=False)
        ridge.fit(X_T, Y_T)
        beta = ridge.coef_
        
    from sklearn.linear_model import RidgeCV

    # Try a range of alphas (log scale is typical)
    alphas = np.logspace(-2, 6, 20)
    
    ridge = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True)
    ridge.fit(X_T, Y_T)
    beta = ridge.coef_
    
    print(f"Best alpha: {ridge.alpha_}")



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
    print(f"Test p-value: {np.exp(np.mean(np.log(p_test))):.3e}")
    # Plotting test set predictions vs actual using mean_bin_plot
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
    plt.xlabel(r'$Behav._t r_{j,t} r_{j,t}$')
    plt.ylabel('$\Delta W$')
    plt.title(data['mouse'] + ' ' + data['session'])

    n_features = behavior_features.shape[1]
    beta_reshaped = beta.reshape(3, n_features)  # 3 = step, reward, go
    
    plt.subplot(212)
    sns.heatmap(beta_reshaped*1000, annot=True, xticklabels=[
        'rt', 'rt_rpe', 'hit', 'hit_rpe', 'miss', 'miss_rpe'
    ], yticklabels=['step', 'reward', 'go cue'], cmap='coolwarm', center=0)
    plt.title(r'$\beta$ weights: CC source × behavior feature')
    plt.xlabel('Behavioral feature')
    plt.ylabel('Trial epoch')
    plt.tight_layout()
    plt.show()
     
    if shuffle == 1:
        shuffle_test(X_T,Y_T)


#%%

def shuffle_test(X_T,Y_T):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV
    from scipy.stats import pearsonr, norm
    import numpy as np
    import matplotlib.pyplot as plt
    import plotting_functions as pf
    
    def fit_and_evaluate(X_T, Y_T, alphas=np.logspace(-2, 6, 20), n_splits=5, shuffle=False):
        """
        Fits ridge regression and evaluates cross-validated performance.
        If shuffle=True, shuffle Y_T relative to X_T before fitting.
        Reports train/test r and p-values.
        """
        if shuffle:
            Y_T = np.random.permutation(Y_T)
        
        ridge = RidgeCV(alphas=alphas, fit_intercept=False)
        ridge.fit(X_T, Y_T)
        beta = ridge.coef_
    
        kf = KFold(n_splits=n_splits, shuffle=True)
        corr_train, p_train = [], []
        corr_test, p_test = [], []
    
        Y_test_all = np.array([])
        Y_test_pred_all = np.array([])
    
        for train_idx, test_idx in kf.split(X_T):
            X_train, X_test = X_T[train_idx], X_T[test_idx]
            Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
    
            beta_cv = np.linalg.pinv(X_train) @ Y_train
    
            Y_train_pred = X_train @ beta_cv
            Y_test_pred = X_test @ beta_cv
    
            r_train, pval_train = pearsonr(Y_train_pred, Y_train)
            r_test, pval_test = pearsonr(Y_test_pred, Y_test)
    
            corr_train.append(r_train)
            p_train.append(pval_train)
            corr_test.append(r_test)
            p_test.append(pval_test)
    
            Y_test_all = np.concatenate([Y_test_all, Y_test])
            Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])
    
        return {
            'beta': beta,
            'train_corr': np.mean(corr_train),
            'train_std': np.std(corr_train),
            'test_corr': np.mean(corr_test),
            'test_std': np.std(corr_test),
            'train_pval': np.mean(p_train),
            'test_pval': np.mean(p_test),
            'Y_test': Y_test_all,
            'Y_test_pred': Y_test_pred_all
        }
    
    # --- Real fit ---
    real_results = fit_and_evaluate(X_T, Y_T, shuffle=False)
    
    # --- Shuffled fits ---
    n_shuffles = 100
    shuffled_test_r = []
    
    for _ in range(n_shuffles):
        shuff_result = fit_and_evaluate(X_T, Y_T, shuffle=True)
        shuffled_test_r.append(shuff_result['test_corr'])
    
    shuffled_test_r = np.array(shuffled_test_r)
    
    # --- Print results ---
    print(f"Real train r: {real_results['train_corr']:.4f}")
    print(f"Real test r: {real_results['test_corr']:.4f}")
    
    # Permutation-based p-value
    perm_pval = np.mean(shuffled_test_r >= real_results['test_corr'])
    print(f"Permutation p-value (real > shuffle): {perm_pval:.4f}")
    
    # --- Plot real vs shuffled predictions ---
    plt.figure(figsize=(5,5))
    pf.mean_bin_plot(real_results['Y_test_pred'], real_results['Y_test'], 5, 1, 1, 'k')
    plt.xlabel('Predicted ΔW (real)')
    plt.ylabel('Actual ΔW')
    plt.title('Real fit')
    plt.show()
    
    # --- Plot shuffle distribution ---
    plt.figure(figsize=(6,4))
    
    # Histogram of shuffled test r
    plt.hist(shuffled_test_r, bins=20, density=True, color='lightgray', edgecolor='k', label='Shuffled')
    
    # Gaussian fit overlay
    mu, std = np.mean(shuffled_test_r), np.std(shuffled_test_r)
    x = np.linspace(mu - 4*std, mu + 4*std, 200)
    plt.plot(x, norm.pdf(x, mu, std), 'k--', label='Gaussian fit')
    
    # Vertical line for real test r
    plt.axvline(real_results['test_corr'], color='red', linestyle='--', label='Real')
    
    plt.xlabel('Cross-validated test $r$')
    plt.ylabel('Probability density')
    plt.title('Real vs shuffled test correlation')
    plt.legend()
    plt.tight_layout()
    plt.show()
