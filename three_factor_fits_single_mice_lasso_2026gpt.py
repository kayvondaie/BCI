# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:37:20 2025

@author: kayvon.daie
"""

#%%
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
import re
from collections import OrderedDict
import traceback


list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]
#mice = ["BCI102","BCI109","BCI105","BCI106"]
HI, RT, HIT, HIa, HIb, HIc, DOT, TRL, THR, RPE, FIT, GRP, RPE_FIT, DW = ([] for _ in range(14))
XALL, YALL, B, XPRE3, PVAL, BETA = ([] for _ in range(6))

# Top-level design configuration (used in session loop and baseline block)
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_SPECS = [
    {"name": "hit", "source": "hit_bin"},
    {"name": "hit_rpe", "source": "hit_rpe_bin"},
    {"name": "10-rt", "source": lambda d: 10 - np.asarray(d["rt_bin"], dtype=float)},
    {"name": "rpe", "source": "rpe_bin"},
    {"name": "thr", "source": "thr_bin"},   # <-- FIX: remove level="trial"
    {"name": "2-factor", "source": lambda d: np.ones_like(np.asarray(d["hit_bin"], dtype=float))},
]

for mi in range(len(mice)):
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean, dot_prod_z
    fit_type      = 'pinv'     #ridge, pinv
    alpha         =  100        #only used for ridge
    lasso_alphas = np.logspace(-6, 0, 60)  # only if you don't know alpha_max scale

    num_bins      =  10        # number of bins to calculate correlations
    tau_elig      =  10         # number of trials for eligibility trace in RPE calculation
    shuffle       =  0
    plotting      =  1
    fitting       =  1
    correct_direct = True
    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) & (list_of_dirs['Has data_main.npy']==True))[0]
    si = 2;
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
            BCI_thresholds = np.asarray(data["BCI_thresholds"], dtype=float)
            thr = BCI_thresholds[1, :]            
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i-1]            
            
            # optional: back-fill if it starts with NaN
            if np.isnan(thr[0]):
                first = thr[np.isfinite(thr)]
                if first.size > 0:
                    thr[0] = first[0]
            
            BCI_thresholds[1, :] = thr



            AMP = []
            siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
            umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
            
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
           
            dt_si = data['dt_si']
            F = data['F']
            #num_bins = F.shape[2]
            trl = F.shape[2]
            tsta = np.arange(0,12,data['dt_si'])
            tsta=tsta-tsta[int(2/dt_si)]
        
            # Initialize arrays
            kstep = np.zeros((F.shape[1], trl))
            krewards = np.zeros((F.shape[1], trl))
        
        
            step_raw = data['step_time']
      
           
        
        
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
        
            # Fix zscore_mat (in case you use dot_prod_z)
            def zscore_mat(X):
                mu = np.nanmean(X, axis=0, keepdims=True)
                sd = np.nanstd(X, axis=0, keepdims=True)
                sd[sd == 0] = 1
                return (X ) / sd
            
            # Make robust bin edges (num_bins bins => num_bins+1 edges)
            trl = F.shape[2]
            trial_bins = np.unique(np.linspace(0, trl, num_bins + 1).astype(int))
            nb = len(trial_bins) - 1
            
            # Preallocate as NaN (safer than zeros for skipped bins)
            cc = np.corrcoef(kstep)  # only used for shapes
            CCstep = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCrew  = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCts   = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCpre  = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            
            # Trial-level outcome vectors
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0  # your choice
          
            
            rt_rpe   = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe  =  compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)
            miss_rpe =  compute_rpe((~hit).astype(float), baseline=0.0, tau=tau_elig, fill_value=1.0)
            
            (hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin, hit_rpe_bin, miss_rpe_bin,
             CCrew, CCstep, CCts, CCpre, CC) = compute_binned_behaviors_and_pairwise(
                nb=nb,
                trial_bins=trial_bins,
                hit=hit,
                rt_filled=rt_filled,
                rt_rpe=rt_rpe,
                miss_rpe=miss_rpe,
                hit_rpe=hit_rpe,
                BCI_thresholds=BCI_thresholds,
                pairwise_mode=pairwise_mode,
                krewards=krewards,
                kstep=kstep,
                k=k,
                kpre=kpre,
                CCrew=CCrew,
                CCstep=CCstep,
                CCts=CCts,
                CCpre=CCpre,
                cc=cc,  # optional
                centered_dot=centered_dot,  # only needed for dot_prod_no_mean
                zscore_mat=zscore_mat,      # only needed for dot_prod_z
                return_interleaved_CC=True
            )

            
            Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_stim, used_bins = build_pairwise_XY_per_bin(
                CCstep, CCrew, CCts, CCpre,
                stimDist, AMP,
                nb=nb,
                hit_bin=hit_bin,
                dist_target_lt=10,
                dist_nontarg_min=30,
                dist_nontarg_max=1000,
                amp0_thr=0.1,
                amp1_thr=0.1
            )
            
            X_stim = np.asarray(X_stim, dtype=float).ravel()
            X_stim = np.nan_to_num(X_stim, nan=0.0, posinf=0.0, neginf=0.0)
            
            # (optional but recommended) z-score nuisance so its scale is reasonable
            xs_mu = np.mean(X_stim)
            xs_sd = np.std(X_stim)
            if xs_sd == 0:
                xs_sd = 1.0
            X_stim_z = (X_stim - xs_mu) / xs_sd
            
            
     
            # sanity check: pairs match target length
            if Xstep_T.shape[0] != Y_T.shape[0]:
                raise ValueError(f"X/Y mismatch: X has {Xstep_T.shape[0]} pairs, Y has {Y_T.shape[0]} pairs.")
            
            # ----------------------------
            # Configurable design spec (edit only this block)
            # ----------------------------
            EPOCH_TO_XT = OrderedDict([
                ("pre", Xpre_T),
                ("go_cue", Xts_T),
                ("late", Xstep_T),
                ("reward", Xrew_T),
            ])
            ZSCORE_BEHAVIOR = [spec["name"] for spec in BEHAVIOR_SPECS if spec["name"] != "2-factor"]

            binned_dict = {
                "hit_bin": hit_bin,
                "hit_rpe_bin": hit_rpe_bin,
                "rt_bin": rt_bin,
                "rpe_bin": rpe_bin,
                "thr_bin": thr_bin,
                "miss_bin": miss_bin,
                "avg_dot_bin": avg_dot_bin,
                "miss_rpe_bin": miss_rpe_bin,
            }
            trial_dict = {
                "BCI_thresholds": np.asarray(data["BCI_thresholds"]).squeeze(),
            }
            zscore_cols_idx = [
                i for i, spec in enumerate(BEHAVIOR_SPECS)
                if spec["name"] in ZSCORE_BEHAVIOR
            ]
            Bz, behavior_names = build_behavior_matrix(
                binned_dict=binned_dict,
                trial_dict=trial_dict,
                trial_bins=trial_bins,
                used_bins=used_bins,
                behavior_specs=BEHAVIOR_SPECS,
                zscore_cols_idx=zscore_cols_idx,
            )
            X_T = build_design_matrix(EPOCH_TO_XT, Bz, EPOCH_ORDER)
            if correct_direct == True:
                X_T = np.hstack([X_T, X_stim_z[:, None]])  # keep X_stim in last column

            # final check
            assert X_T.shape[0] == Y_T.shape[0]

            
            # ----------------------------
            # Cross-validated fit
            # ----------------------------
            from sklearn.model_selection import KFold, GroupKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LassoCV, RidgeCV
            
            print("X_T finite:", np.all(np.isfinite(X_T)), "Y_T finite:", np.all(np.isfinite(Y_T)))
            print("X_T shape:", X_T.shape, "Y_T shape:", Y_T.shape)
            print("X_T min/max:", np.nanmin(X_T), np.nanmax(X_T))
            print("Any zero-variance cols:", np.sum(np.nanstd(X_T, axis=0) == 0), "/", X_T.shape[1])
            bad_rows = np.where(~np.isfinite(Y_T) | ~np.all(np.isfinite(X_T), axis=1))[0]
            print("bad rows:", bad_rows[:10], "count:", bad_rows.size)

            
            # single-session: plain KFold
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iter = outer_cv.split(X_T, Y_T)
            
            corr_train, p_train = [], []
            corr_test, p_test = [], []
            Y_test_all = np.array([])
            Y_test_pred_all = np.array([])
            beta_reshaped = None
            if fitting == 1:
                for train_idx, test_idx in split_iter:
                    X_train, X_test = X_T[train_idx], X_T[test_idx]
                    Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
                    
                    n_epochs = len(EPOCH_ORDER)
                    n_features = len(behavior_names)
                    n_base = n_epochs * n_features
                    
                    # --- regress nuisance out FIRST (raw Y), train-only fit ---
                    if correct_direct == True:
                        X_stim_train = X_train[:, -1]
                        X_stim_test  = X_test[:,  -1]
                    
                        # drop nuisance column for main regression
                        X_train = X_train[:, :n_base]
                        X_test  = X_test[:,  :n_base]
                    
                        xs_mu = X_stim_train.mean()
                        xs_sd = X_stim_train.std()
                        if xs_sd == 0:
                            xs_sd = 1.0
                    
                        xtr = (X_stim_train - xs_mu) / xs_sd
                        xte = (X_stim_test  - xs_mu) / xs_sd
                    
                        Xn_tr = np.column_stack([np.ones_like(xtr), xtr])
                        coef_nuis, _, _, _ = np.linalg.lstsq(Xn_tr, Y_train, rcond=None)
                        b0, a0 = coef_nuis
                    
                        Y_train = Y_train - (b0 + a0 * xtr)
                        Y_test  = Y_test  - (b0 + a0 * xte)
                    
                    # --- THEN z-score Y using TRAIN stats only ---
                    mu = Y_train.mean()
                    sigma = Y_train.std()
                    if sigma == 0:
                        sigma = 1.0
                    Y_train = (Y_train - mu) / sigma
                    Y_test  = (Y_test  - mu) / sigma



                    if fit_type == 'pinv':
                        beta_cv = np.linalg.pinv(X_train) @ Y_train
                        Y_train_pred = X_train @ beta_cv
                        Y_test_pred  = X_test  @ beta_cv
                        beta = beta_cv

                
                    elif fit_type == 'ridge':
                        alphas = np.logspace(-10, -4, 10)
                        ridge = RidgeCV(alphas=alphas, fit_intercept=True)
                        ridge_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', ridge)])
                        ridge_pipe.fit(X_train, Y_train)
                    
                        scaler = ridge_pipe.named_steps['scaler']
                        ridge_m = ridge_pipe.named_steps['ridge']
                    
                        Y_train_pred = (
                            ridge_m.intercept_
                            + scaler.transform(X_train) @ ridge_m.coef_
                        )
                        Y_test_pred  = (
                            ridge_m.intercept_
                            + scaler.transform(X_test)  @ ridge_m.coef_
                        )

                    
                        beta = ridge_m.coef_

                
                    elif fit_type == 'lasso':
                        # alpha_max anchored grid (computed on scaled X_train)
                        Xs = StandardScaler().fit_transform(X_train)
                        yc = Y_train - np.mean(Y_train)
                        alpha_max = np.max(np.abs(Xs.T @ yc)) / Xs.shape[0]
                
                        #lasso_alphas_fold = np.logspace(np.log10(alpha_max) - 2.5, np.log10(alpha_max), 50)
                        lasso_alphas_fold = np.logspace(-10,-3, 10)
                
                        lasso_model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('lassocv', LassoCV(
                                alphas=lasso_alphas_fold,
                                fit_intercept=True,
                                cv=5,
                                max_iter=1000,
                                tol=1e-4,
                                n_jobs=-1
                            ))
                        ])                
                        lasso_model.fit(X_train, Y_train)

                        scaler = lasso_model.named_steps['scaler']
                        lasso_m = lasso_model.named_steps['lassocv']
                        
                        Y_train_pred = lasso_m.intercept_ + scaler.transform(X_train) @ lasso_m.coef_
                        Y_test_pred  = lasso_m.intercept_ + scaler.transform(X_test)  @ lasso_m.coef_

                        beta = lasso_m.coef_

                        print(f"alpha_max={alpha_max:.3e} alpha*={lasso_model.named_steps['lassocv'].alpha_:.3e} "
                              f"(ratio={lasso_model.named_steps['lassocv'].alpha_/alpha_max:.3e}) | nnz={(beta!=0).sum()}/{beta.size}")
                
                    # correlations
                    r_train, pval_train = pearsonr(Y_train_pred, Y_train)
                    r_test,  pval_test  = pearsonr(Y_test_pred,  Y_test)
                
                    corr_train.append(r_train); p_train.append(pval_train)
                    corr_test.append(r_test);   p_test.append(pval_test)
                
                    Y_test_all = np.concatenate([Y_test_all, Y_test])
                    Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])
                
                n_epochs = len(EPOCH_ORDER)
                n_features = len(behavior_names)
                # if beta.size != n_epochs * n_features:
                    # raise ValueError(f"beta size mismatch: got {beta.size}, expected {n_epochs * n_features}")
                beta_reshaped = beta[:n_base].reshape(n_epochs, n_features)
                
                if beta_reshaped is not None:
                    BETA.append(beta_reshaped)
         
        
            if beta_reshaped is not None:
                HI.append(beta_reshaped)
            RT.append(np.nanmean(rt))
            HIT.append(hit)
            FIT.append(np.mean(corr_test) if len(corr_test) > 0 else np.nan)            
            GRP.append(stimDist.shape[1])
            ind = np.where(stimDist> 30);
            dw = np.nanmean(AMP[1][ind] - AMP[0][ind])
            DW.append(dw)
            XALL.append(X_T)
            YALL.append(Y_T)
            XPRE3.append(Xpre_T)
            PVAL.append(np.exp(np.mean(np.log(p_test))) if len(p_test) > 0 else np.nan)
            
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
                plt.subplot(211)
                pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
                plt.xlabel(r'$Behav._t r_{j,t} r_{j,t}$')
                plt.ylabel('$\Delta W$')
                plt.title(data['mouse'] + ' ' + data['session'] + r'  hit = ' + str(round(np.nanmean(hit)*100)) + '%')
            
                # plt.subplot(222)
                # cn = data['conditioned_neuron'][0][0]
                # plt.plot(np.nanmean(F[:,cn,:],axis=1))
                # CNTUN.append(np.nanmean(F[ts,cn,0:10])-np.nanmean(F[0:ts[0],cn,0:10]))
                # plt.title(str(favg.shape[2]) + ' groups ' + f"{np.nanmean(rt):.1f}" + 'sec')

            
                
                if beta_reshaped is not None:
                    plt.subplot(212)
                    sns.heatmap(beta_reshaped*1, annot=True, xticklabels=[
                        *behavior_names
                    ], yticklabels=EPOCH_ORDER, cmap='coolwarm', center=0)
                    plt.title(r'$\beta$ weights: CC source × behavior feature')
                    plt.xlabel('Behavioral feature')
                    plt.ylabel('Trial epoch')
                plt.tight_layout()
                plt.show()
            if shuffle == 1:
                shuffle_test(X_T,Y_T)
        except Exception as e:
            print(f"Failed on {mouse} {session}: {e}")
            traceback.print_exc()
            continue
hi = HI.copy()
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr

import plotting_functions as pf  # has mean_bin_plot

# ----------------------------
# Stack + clean
# ----------------------------

# --- Stack all sessions ---
# if mouse == "BCI102":
#     X_concat = np.vstack(XALL[3:])
#     Y_concat = np.concatenate(YALL[3:])
# else:
aaa = [np.nanstd(x) for x in YALL]
ind = np.where(np.array(aaa) < .5)[0]  
XALL = [XALL[i] for i in ind]
YALL = [YALL[i] for i in ind]
X_concat = np.vstack(XALL)
Y_concat = np.concatenate(YALL)


mask = np.isfinite(Y_concat) & np.all(np.isfinite(X_concat), axis=1)
X_concat = X_concat[mask]
Y_concat = Y_concat[mask]

# ----------------------------
# Split
# ----------------------------
ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(ss.split(X_concat))

X_train, X_test = X_concat[train_idx], X_concat[test_idx]
Y_train, Y_test = Y_concat[train_idx], Y_concat[test_idx]

# ----------------------------
# Define base vs nuisance (assumes X_stim is last column when correct_direct=True)
# ----------------------------
behavior_names = [spec["name"] for spec in BEHAVIOR_SPECS]
n_epochs   = len(EPOCH_ORDER)
n_features = len(behavior_names)
n_base     = n_epochs * n_features

if correct_direct == True:
    # nuisance vectors
    X_stim_train = X_train[:, -1]
    X_stim_test  = X_test[:,  -1]

    # base design only (drop nuisance column)
    X_train = X_train[:, :n_base]
    X_test  = X_test[:,  :n_base]

    # regress nuisance out of Y using TRAIN ONLY (with intercept), on raw Y
    xs_mu = X_stim_train.mean()
    xs_sd = X_stim_train.std()
    if xs_sd == 0:
        xs_sd = 1.0

    xtr = (X_stim_train - xs_mu) / xs_sd
    xte = (X_stim_test  - xs_mu) / xs_sd

    Xn_tr = np.column_stack([np.ones_like(xtr), xtr])
    coef_nuis, _, _, _ = np.linalg.lstsq(Xn_tr, Y_train, rcond=None)
    b0, a0 = coef_nuis[0], coef_nuis[1]

    Y_train = Y_train - (b0 + a0 * xtr)
    Y_test  = Y_test  - (b0 + a0 * xte)

# ----------------------------
# Z-score Y using TRAIN stats only (after nuisance regression if enabled)
# ----------------------------
mu, sigma = Y_train.mean(), Y_train.std()
if sigma == 0:
    sigma = 1.0
Y_train_z = (Y_train - mu) / sigma
Y_test_z  = (Y_test  - mu) / sigma

# ----------------------------
# FAST model (fixed alpha) on BASE columns only
# ----------------------------
alpha_fast = 10**(-3.3)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=alpha_fast, fit_intercept=True, max_iter=500, tol=1e-3))
])
model.fit(X_train, Y_train_z)

scaler = model.named_steps["scaler"]
lasso_m = model.named_steps["lasso"]

Y_pred_train = lasso_m.intercept_ + scaler.transform(X_train) @ lasso_m.coef_
Y_pred_test  = lasso_m.intercept_ + scaler.transform(X_test)  @ lasso_m.coef_

r_tr, p_tr = pearsonr(Y_pred_train, Y_train_z) if np.std(Y_pred_train) > 0 else (np.nan, np.nan)
r_te, p_te = pearsonr(Y_pred_test,  Y_test_z)  if np.std(Y_pred_test)  > 0 else (np.nan, np.nan)

beta = lasso_m.coef_
print("X shape (base):", X_train.shape[1], "features | nnz:", (beta != 0).sum(), "/", beta.size, " alpha:", alpha_fast)
print(f"Train r={r_tr:.3f} (p={p_tr:.2e}) | Test r={r_te:.3f} (p={p_te:.2e})")

beta_reshaped = beta[:n_base].reshape(n_epochs, n_features)


# ----------------------------
# Plot (mean_bin_plot + beta heatmap)
# ----------------------------
plt.figure(figsize=(6, 8))

plt.subplot(211)
pf.mean_bin_plot(Y_pred_test, Y_test_z, 5, 1, 1, 'k')
plt.xlabel(r'Predicted (z-scored $\Delta W$)')
plt.ylabel(r'True (z-scored $\Delta W$)')
plt.title(f"{mouse} fast baseline | alpha={alpha_fast} | r={r_te:.3f}")


plt.subplot(212)
sns.heatmap(beta_reshaped*1, annot=True, xticklabels=[
    *behavior_names
], yticklabels=EPOCH_ORDER, cmap='coolwarm', center=0)
plt.title(r'$\beta$ weights: CC source × behavior feature')
plt.xlabel('Behavioral feature')
plt.ylabel('Trial epoch')
plt.tight_layout()
plt.tight_layout()
plt.show()
