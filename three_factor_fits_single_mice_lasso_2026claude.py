#%% ============================================================================
# CELL 1: Imports and Setup (run once)
# ============================================================================
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import re
from collections import OrderedDict
import traceback
import plotting_functions as pf
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration (edit as needed, run before session loop)
# ============================================================================

# TOGGLE: Choose analysis method
USE_TRIAL_LEVEL = True  # Set to False to use original binned approach

# Mice to analyze
mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]

# Analysis parameters
pairwise_mode = 'dot_prod'  # dot_prod, noise_corr, dot_prod_no_mean, dot_prod_z
fit_type = 'pinv'           # pinv, ridge, lasso
alpha = 1e-3                 # only used for ridge/lasso
num_bins = 10               # only used for binned approach
tau_elig = 10
shuffle = 0
plotting = 1
fitting = 1
correct_direct = True

# Epoch and behavior definitions
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_SPECS = [
    {"name": "hit", "source": "hit_bin"},
    {"name": "hit_rpe", "source": "hit_rpe_bin"},
    {"name": "10-rt", "source": lambda d: 10 - np.asarray(d["rt_bin"], dtype=float)},
    {"name": "rpe", "source": "rpe_bin"},
    {"name": "thr", "source": "thr_bin"},
    {"name": "2-factor", "source": lambda d: np.ones_like(np.asarray(d["hit_bin"], dtype=float))},
]

# Initialize storage
HI, RT, HIT, HIa, HIb, HIc, DOT, TRL, THR, RPE, FIT, GRP, RPE_FIT, DW = ([] for _ in range(14))
XALL, YALL, B, XPRE3, PVAL, BETA = ([] for _ in range(6))

print(f"Configuration set: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'}")
print(f"Analyzing mice: {mice}")

mice = ["BCI102"]
si = 10
for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) & 
                           (list_of_dirs['Has data_main.npy']==True))[0]
    
    #for sii in range(si,si+1):
    for sii in range(len(session_inds)):
        try:
            print(f"\n{'='*70}")
            print(f"Mouse {mi+1}/{len(mice)}: {mouse}, Session {sii+1}/{len(session_inds)}")
            print(f"Method: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'}")
            print(f"{'='*70}")
            
            # ================================================================
            # Load data
            # ================================================================
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron',
                       'dt_si','step_time','reward_time','BCI_thresholds']
            
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"Skipping — file not found.")
                continue
            
            # ================================================================
            # Preprocess
            # ================================================================
            
            # Fix thresholds
            BCI_thresholds = np.asarray(data["BCI_thresholds"], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i-1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr
            
            # Photostim
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            
            # Neural data
            dt_si = data['dt_si']
            F = data['F']
            #num_bins = F.shape[2]
            trl = F.shape[2]
            tsta = np.arange(0,12,data['dt_si'])
            tsta=tsta-tsta[int(2/dt_si)]
            
            # Parse times
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            
            # Compute epoch activity
            kstep = np.zeros((F.shape[1], trl))
            krewards = np.zeros((F.shape[1], trl))
            
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    # Late epoch
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < F.shape[0]]
                    kstep[:, ti] = np.nanmean(F[indices, :, ti], axis=0)
                    
                    # Reward epoch
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < F.shape[0]]
                    krewards[:, ti] = np.nanmean(F[indices, :, ti], axis=0)
            
            # Go cue epoch
            ts = np.where((tsta > 0) & (tsta < 2))[0]
            k = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
            
            # Pre-trial epoch
            ts = np.where((tsta > -10) & (tsta < 0))[0]
            kpre = np.nanmean(F[ts[0]:ts[-1], :, :], axis=0)
            
            kstep[np.isnan(kstep)] = 0
            krewards[np.isnan(krewards)] = 0
            k[np.isnan(k)] = 0
            kpre[np.isnan(kpre)] = 0
            
            # Behavioral variables
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)
            miss_rpe = compute_rpe((~hit).astype(float), baseline=0.0, tau=tau_elig, fill_value=1.0)
            
            # ================================================================
            # Feature construction: TRIAL-LEVEL vs BINNED
            # ================================================================
            
            
            if USE_TRIAL_LEVEL:
                print("Computing TRIAL-LEVEL features...")
                
                # Use original slow-but-reliable version
                CC_dict = compute_trial_level_cc_with_epochs_FAST(  # ← Remove "_FAST"
                    F=F,
                    tsta=tsta,
                    dt_si=dt_si,
                    reward_time=data['reward_time'],
                    step_time=data['step_time'],
                    pairwise_mode=pairwise_mode,
                    centered_dot=centered_dot,
                    zscore_mat=zscore_mat,
                )
                
                t_cc = time.time() - t0
                print(f"  CC computation: {t_cc:.2f}s")
                
                # Trial-level behavior - WITH Z-SCORING (your fixed version)
                behavior_dict_raw = {
                    'hit': hit.astype(float),
                    'hit_rpe': hit_rpe,
                    '10-rt': 10 - rt_filled,
                    'rpe': rt_rpe,
                    'thr': BCI_thresholds[1, :],
                    '2-factor': np.ones(trl),
                }
                
                behavior_dict = {}
                for name, values in behavior_dict_raw.items():
                    if name == '2-factor':
                        behavior_dict[name] = values
                    else:
                        mu, sigma = np.nanmean(values), np.nanstd(values)
                        if sigma == 0 or not np.isfinite(sigma):
                            sigma = 1.0
                        behavior_dict[name] = (values - mu) / sigma
                
                # Build features
                t0 = time.time()
                X_T, Y_T, feature_names = build_trial_level_features(
                    CC_dict=CC_dict,
                    behavior_dict=behavior_dict,
                    stimDist=stimDist,
                    AMP=AMP,
                    dist_target_lt=10,
                    dist_nontarg_min=30,
                    dist_nontarg_max=1000,
                    amp0_thr=0.1,
                    amp1_thr=0.1,
                )
                t_feat = time.time() - t0
                
                # Z-score features by epoch to normalize scales
                print("\n  Z-scoring features by epoch...")
                for epoch in EPOCH_ORDER:
                    epoch_idx = [i for i, name in enumerate(feature_names) if name.startswith(epoch)]
                    if len(epoch_idx) > 0:
                        epoch_data = X_T[:, epoch_idx]
                        mu = epoch_data.mean(axis=0)
                        sigma = epoch_data.std(axis=0)
                        sigma[sigma == 0] = 1.0
                        X_T[:, epoch_idx] = (epoch_data - mu) / sigma
                        print(f"    {epoch}: rescaled {len(epoch_idx)} features")
                
                print(f"  Feature building: {t_feat:.2f}s")
                print(f"  Total: {t_cc + t_feat:.2f}s")
                print(f"  Features: {X_T.shape}, Pairs: {Y_T.shape[0]}")
            else:
                print("Computing BINNED features...")
                
                # Bin setup
                trial_bins = np.unique(np.linspace(0, trl, num_bins + 1).astype(int))
                nb = len(trial_bins) - 1
                
                cc = np.corrcoef(kstep)
                CCstep = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
                CCrew = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
                CCts = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
                CCpre = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
                
                # Compute binned CC and behavior
                (hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin, 
                 hit_rpe_bin, miss_rpe_bin, CCrew, CCstep, CCts, CCpre, CC) = \
                    compute_binned_behaviors_and_pairwise(
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
                        cc=cc,
                        centered_dot=centered_dot,
                        zscore_mat=zscore_mat,
                        return_interleaved_CC=True
                    )
                
                # Build pairwise features
                Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_stim, used_bins = \
                    build_pairwise_XY_per_bin(
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
                
                # Build design matrix
                EPOCH_TO_XT = OrderedDict([
                    ("pre", Xpre_T),
                    ("go_cue", Xts_T),
                    ("late", Xstep_T),
                    ("reward", Xrew_T),
                ])
                
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
                
                ZSCORE_BEHAVIOR = [s["name"] for s in BEHAVIOR_SPECS if s["name"] != "2-factor"]
                zscore_cols_idx = [i for i, s in enumerate(BEHAVIOR_SPECS) if s["name"] in ZSCORE_BEHAVIOR]
                
                Bz, feature_names = build_behavior_matrix(
                    binned_dict=binned_dict,
                    trial_dict=trial_dict,
                    trial_bins=trial_bins,
                    used_bins=used_bins,
                    behavior_specs=BEHAVIOR_SPECS,
                    zscore_cols_idx=zscore_cols_idx,
                )
                
                X_T = build_design_matrix(EPOCH_TO_XT, Bz, EPOCH_ORDER)
                
                # Add nuisance regressor
                X_stim = np.asarray(X_stim, dtype=float).ravel()
                X_stim = np.nan_to_num(X_stim, nan=0.0)
                xs_mu, xs_sd = np.mean(X_stim), np.std(X_stim)
                if xs_sd == 0:
                    xs_sd = 1.0
                X_stim_z = (X_stim - xs_mu) / xs_sd
                
                if correct_direct:
                    X_T = np.hstack([X_T, X_stim_z[:, None]])
                
                print(f"  Features: {X_T.shape}, Pairs: {Y_T.shape[0]}")
            
            # ================================================================
            # Fitting (same for both methods)
            # ================================================================
            
            # In CELL 3, replace the entire fitting section with this:

            if fitting != 1:
                continue
            
            assert X_T.shape[0] == Y_T.shape[0], f"X/Y mismatch: {X_T.shape} vs {Y_T.shape}"
            
            # Check for nuisance regressor (trial-level shouldn't have it)
            n_epochs = len(EPOCH_ORDER)
            n_features_behav = len([s["name"] for s in BEHAVIOR_SPECS])
            n_base = n_epochs * n_features_behav
            
            if X_T.shape[1] > n_base:
                print(f"  WARNING: X has {X_T.shape[1]} features, expected {n_base}")
                print(f"  Removing extra columns (nuisance regressors)")
                X_T = X_T[:, :n_base]
            
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            corr_train, p_train = [], []
            corr_test, p_test = [], []
            Y_test_all = np.array([])
            Y_test_pred_all = np.array([])
            beta_final = None
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_T, Y_T)):
                X_train, X_test = X_T[train_idx], X_T[test_idx]
                Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
                
                # Sanity checks
                assert X_train.shape[0] == Y_train.shape[0], f"Train mismatch: X={X_train.shape}, Y={Y_train.shape}"
                assert X_test.shape[0] == Y_test.shape[0], f"Test mismatch: X={X_test.shape}, Y={Y_test.shape}"
                
                # Z-score Y
                mu, sigma = Y_train.mean(), Y_train.std()
                if sigma == 0 or not np.isfinite(sigma):
                    sigma = 1.0
                Y_train_z = (Y_train - mu) / sigma
                Y_test_z = (Y_test - mu) / sigma
                
                # Fit
                if fit_type == 'pinv':
                    beta = np.linalg.pinv(X_train) @ Y_train_z
                    Y_train_pred = X_train @ beta
                    Y_test_pred = X_test @ beta
                    
                elif fit_type == 'ridge':
                    ridge = RidgeCV(alphas=np.logspace(-10, -4, 10), fit_intercept=True)
                    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', ridge)])
                    pipe.fit(X_train, Y_train_z)
                    Y_train_pred = pipe.predict(X_train)
                    Y_test_pred = pipe.predict(X_test)
                    beta = pipe.named_steps['ridge'].coef_
                
                elif fit_type == 'lasso':
                    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, tol=1e-4)
                    pipe = Pipeline([('scaler', StandardScaler()), ('lasso', lasso)])
                    pipe.fit(X_train, Y_train_z)
                    Y_train_pred = pipe.predict(X_train)
                    Y_test_pred = pipe.predict(X_test)
                    beta = pipe.named_steps['lasso'].coef_
    
                else:
                    raise ValueError(f"Unknown fit_type: {fit_type}")
                                        
                # More sanity checks
                assert Y_train_pred.shape == Y_train_z.shape, f"Train pred mismatch: {Y_train_pred.shape} vs {Y_train_z.shape}"
                assert Y_test_pred.shape == Y_test_z.shape, f"Test pred mismatch: {Y_test_pred.shape} vs {Y_test_z.shape}"
                
                # Correlations
                if np.std(Y_train_pred) > 0:
                    r_train, pval_train = pearsonr(Y_train_pred, Y_train_z)
                else:
                    r_train, pval_train = 0.0, 1.0
                
                if np.std(Y_test_pred) > 0:
                    r_test, pval_test = pearsonr(Y_test_pred, Y_test_z)
                else:
                    r_test, pval_test = 0.0, 1.0
                
                corr_train.append(r_train)
                p_train.append(pval_train)
                corr_test.append(r_test)
                p_test.append(pval_test)
                
                Y_test_all = np.concatenate([Y_test_all, Y_test_z])
                Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])
                
                if fold_idx == 0:  # Save beta from first fold
                    beta_final = beta
            
            # Reshape beta for plotting
            if beta_final is not None:
                beta_reshaped = beta_final[:n_base].reshape(n_epochs, n_features_behav)
            else:
                beta_reshaped = None
            
            # Store results
            if beta_reshaped is not None:
                HI.append(beta_reshaped)
                BETA.append(beta_reshaped)
            
            RT.append(np.nanmean(rt))
            HIT.append(hit)
            FIT.append(np.mean(corr_test) if len(corr_test) > 0 else np.nan)
            GRP.append(stimDist.shape[1])
            ind = np.where(stimDist > 30)
            DW.append(np.nanmean(AMP[1][ind] - AMP[0][ind]) if len(ind[0]) > 0 else np.nan)
            XALL.append(X_T)
            YALL.append(Y_T)
            PVAL.append(np.exp(np.mean(np.log(p_test))) if len(p_test) > 0 else np.nan)
            
            # Plot
            if plotting == 1:
                print(f"\nResults:")
                print(f"  Train: r={np.mean(corr_train):.3f} ± {np.std(corr_train):.3f}")
                print(f"  Test:  r={np.mean(corr_test):.3f} ± {np.std(corr_test):.3f}")
                
                plt.figure(figsize=(6, 8))
                plt.subplot(211)
                pf.mean_bin_plot(Y_test_pred_all, Y_test_all, 5, 1, 1, 'k')
                plt.xlabel(r'$Predicted \Delta W$')
                plt.ylabel(r'$\Delta W$')
                plt.title(f"{mouse} {session} | hit={np.nanmean(hit)*100:.0f}% | r={np.mean(corr_test):.3f}")
                
                if beta_reshaped is not None:
                    plt.subplot(212)
                    behavior_names_plot = [s["name"] for s in BEHAVIOR_SPECS]
                    sns.heatmap(beta_reshaped, annot=True, fmt='.2f',
                               xticklabels=behavior_names_plot, yticklabels=EPOCH_ORDER,
                               cmap='coolwarm', center=0, vmin=-0.1, vmax=0.1)
                    plt.title(r'$\beta$ weights')
                
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\n{'='*70}")
print(f"COMPLETE: Processed {len(BETA)} sessions")
print(f"Method: {'TRIAL-LEVEL' if USE_TRIAL_LEVEL else 'BINNED'}")
print(f"{'='*70}")

#%% ============================================================================
# CELL 4: Quick comparison of methods (run after both)
# ============================================================================

# After running Cell 3 with both USE_TRIAL_LEVEL settings, compare:

print("Comparison:")
print(f"  Sessions: {len(FIT)}")
print(f"  Mean test r: {np.mean([f for f in FIT if np.isfinite(f)]):.4f}")
print(f"  Median test r: {np.median([f for f in FIT if np.isfinite(f)]):.4f}")
print(f"  Significant (p<0.05): {sum([p < 0.05 for p in PVAL if np.isfinite(p)])}/{len(PVAL)}")