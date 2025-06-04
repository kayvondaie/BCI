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
        if num_bins > F.shape[2]:
            num_bins = F.shape[2]
        trl = F.shape[2]
        tsta = np.arange(0,12,data['dt_si'])
        tsta=tsta-tsta[int(2/dt_si)]
        if epoch == 'steps':
            epoch = 'step'
        
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
        ts = np.where((tsta > 0) & (tsta < 2))[0]
        k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
    
    
        kstep[np.isnan(kstep)] = 0
        krewards[np.isnan(krewards)] = 0
        k[np.isnan(k)] = 0
        #num_bins = F.shape[2]

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
        rt[np.isnan(rt)] = 30;
        hit_bin = np.zeros((len(trial_bins),))
        rt_bin = np.zeros((len(trial_bins),))
        avg_dot_bin = np.zeros((len(trial_bins),))
        thr_bin = np.zeros((len(trial_bins),))
        rpe_bin = np.zeros((len(trial_bins),))
        

        def compute_rpe(rt, baseline=3.0, window=10, fill_value=np.nan):
            rpe = np.full_like(rt, np.nan, dtype=np.float64)
            rt_clean = np.where(np.isnan(rt), fill_value, rt)
    
            for i in range(len(rt)):
                if i == 0:
                    avg = baseline
                else:
                    start = max(0, i - window)
                    avg = np.nanmean(rt_clean[start:i]) if i > start else baseline
                rpe[i] = avg - rt_clean[i]
            return rpe
        rpe = compute_rpe(rt, baseline=10, window=20, fill_value=50)
        
        
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
                CCstep[:, :, i] = np.corrcoef(k_concat.T)
                CCrew[:, :, i]  = np.corrcoef(k_concat.T)
        
            elif pairwise_mode == 'dot_prod':
                CCts[:, :, i]   = k_concat.T @ k_concat
                CCstep[:, :, i] = k_concat.T @ k_concat
                CCrew[:, :, i]  = k_concat.T @ k_concat
                
            elif pairwise_mode == 'dot_prod_no_mean':
                avg_f = np.nanmean(np.nanmean(F,axis=2),axis=0)
                for ci in range(F.shape[1]):
                    k_concat[:,ci] = k_concat[:,ci] - avg_f[ci]
                CCts[:, :, i]   = k_concat.T @ k_concat
                CCstep[:, :, i] = k_concat.T @ k_concat
                CCrew[:, :, i]  = k_concat.T @ k_concat

    
        # Preallocate combined CC with interleaved shape
        CC = np.zeros((cc.shape[0], cc.shape[1], CCstep.shape[2]*3))
    
        # Interleave step and reward correlations
        CC[:, :, 0::3] = CCstep
        CC[:, :, 1::3] = CCrew
        CC[:, :, 2::3] = CCts
        
        CC = CCts
    
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
    
        
        plt.subplot(233)
        cc_rpe = (np.corrcoef(rpe_bin[:-1],np.asarray(b)[:-1])[1,0])
        cc_rt = (np.corrcoef(rt_bin[:-1],np.asarray(b)[:-1])[1,0])
        plt.bar(['RPE', 'RT'], [cc_rpe, cc_rt],color = 'k')
        plt.ylabel('Correlation with b')
        plt.title('Correlation of b with RPE and RT')
        
        CC_RPE.append(cc_rpe)
        CC_RT.append(cc_rt)
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
            plt.subplot(234);
            plt.plot(np.nanmean(F[:,cn,:],axis=1),'m')
            if Fo.shape[1] == F.shape[1]:
                plt.plot(np.nanmean(Fo[:,cn,:],axis=1),'k')
        except:
            continue
        CN.append(cn)    
        plt.tight_layout()
        
        plt.subplot(235)
        plt.plot(hit)
      
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



