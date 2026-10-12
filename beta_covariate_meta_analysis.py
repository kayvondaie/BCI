#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import re
from collections import OrderedDict
import traceback
import plotting_functions as pf
from BCI_data_helpers import *
import os
import datetime

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

pairwise_mode = 'dot_prod'
fit_type = 'pinv'
num_bins = 10
tau_elig = 10
shuffle = 0
plotting = 0
fitting = 1

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_NAMES = ['hit', 'hit_rpe', '10-rt', 'rpe', '2-factor']

# Storage for meta-analysis
all_sessions = []  # list of dicts, one per session

print(f"Analyzing mice: {mice}")

#%% ============================================================================
# CELL 3: Main loop — collect betas + session-level covariates
# ============================================================================

for mi in range(len(mice)):
    mouse = mice[mi]
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            # ==============================================================
            # Load data
            # ==============================================================
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time',
                'reward_time', 'BCI_thresholds',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping — file not found.")
                continue

            # ==============================================================
            # Preprocess (same as _2026.py)
            # ==============================================================
            BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
            thr = BCI_thresholds[1, :]
            for i in range(1, thr.size):
                if np.isnan(thr[i]):
                    thr[i] = thr[i - 1]
            if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
                thr[0] = thr[np.isfinite(thr)][0]
            BCI_thresholds[1, :] = thr

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Epoch activity
            kstep = np.zeros((n_neurons, trl))
            krewards = np.zeros((n_neurons, trl))

            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < F.shape[0]]
                    kstep[:, ti] = np.nanmean(F[indices, :, ti], axis=0)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < F.shape[0]]
                    krewards[:, ti] = np.nanmean(F[indices, :, ti], axis=0)

            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            k = np.nanmean(F[ts_go[0]:ts_go[-1], :, :], axis=0)

            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            kpre = np.nanmean(F[ts_pre[0]:ts_pre[-1], :, :], axis=0)

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

            # ==============================================================
            # Binned pairwise features (proven approach from _2026.py)
            # ==============================================================
            trial_bins = np.unique(np.linspace(0, trl, num_bins + 1).astype(int))
            nb = len(trial_bins) - 1

            cc = np.corrcoef(kstep)
            CCstep = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCrew = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCts = np.full((cc.shape[0], cc.shape[1], nb), np.nan)
            CCpre = np.full((cc.shape[0], cc.shape[1], nb), np.nan)

            (hit_bin, miss_bin, rt_bin, avg_dot_bin, thr_bin, rpe_bin,
             hit_rpe_bin, miss_rpe_bin, CCrew, CCstep, CCts, CCpre, CC) = \
                compute_binned_behaviors_and_pairwise(
                    nb=nb, trial_bins=trial_bins, hit=hit, rt_filled=rt_filled,
                    rt_rpe=rt_rpe, miss_rpe=miss_rpe, hit_rpe=hit_rpe,
                    BCI_thresholds=BCI_thresholds, pairwise_mode=pairwise_mode,
                    krewards=krewards, kstep=kstep, k=k, kpre=kpre,
                    CCrew=CCrew, CCstep=CCstep, CCts=CCts, CCpre=CCpre,
                    cc=cc, centered_dot=centered_dot, zscore_mat=zscore_mat,
                    return_interleaved_CC=True,
                )

            Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_stim, used_bins = \
                build_pairwise_XY_per_bin(
                    CCstep, CCrew, CCts, CCpre, stimDist, AMP,
                    nb=nb, hit_bin=hit_bin,
                    dist_target_lt=10, dist_nontarg_min=30,
                    dist_nontarg_max=1000, amp0_thr=0.1, amp1_thr=0.1,
                )

            # Build behavior matrix and design matrix
            kept_bin_mask = np.isfinite(hit_bin)
            hit_bin_k = hit_bin[kept_bin_mask]
            hit_rpe_bin_k = hit_rpe_bin[kept_bin_mask]
            rt_bin_k = rt_bin[kept_bin_mask]
            rpe_bin_k = rpe_bin[kept_bin_mask]
            ones_bin_k = np.ones_like(hit_bin_k)

            behavior_features = np.vstack([
                hit_bin_k, hit_rpe_bin_k, 10 - rt_bin_k, rpe_bin_k, ones_bin_k,
            ]).T

            Bz = behavior_features.copy()
            muB = np.nanmean(Bz[:, :4], axis=0, keepdims=True)
            sdB = np.nanstd(Bz[:, :4], axis=0, keepdims=True)
            sdB[sdB == 0] = 1.0
            Bz[:, :4] = (Bz[:, :4] - muB) / sdB

            Xstep_mod = Xstep_T @ Bz
            Xrew_mod = Xrew_T @ Bz
            Xts_mod = Xts_T @ Bz
            Xpre_mod = Xpre_T @ Bz
            X_T = np.hstack([Xpre_mod, Xts_mod, Xstep_mod, Xrew_mod])

            assert X_T.shape[0] == Y_T.shape[0]

            # ==============================================================
            # Fit (5-fold CV, pinv)
            # ==============================================================
            if not fitting:
                continue

            n_features = behavior_features.shape[1]
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            corr_train, corr_test, p_test = [], [], []
            Y_test_all, Y_test_pred_all = np.array([]), np.array([])

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_T, Y_T)):
                X_train, X_test = X_T[train_idx], X_T[test_idx]
                Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]

                mu, sigma = Y_train.mean(), Y_train.std()
                if sigma == 0:
                    sigma = 1.0
                Y_train_z = (Y_train - mu) / sigma
                Y_test_z = (Y_test - mu) / sigma

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

                r_train, _ = pearsonr(Y_train_pred, Y_train_z) if np.std(Y_train_pred) > 0 else (0.0, 1.0)
                r_test, pval_test = pearsonr(Y_test_pred, Y_test_z) if np.std(Y_test_pred) > 0 else (0.0, 1.0)

                corr_train.append(r_train)
                corr_test.append(r_test)
                p_test.append(pval_test)
                Y_test_all = np.concatenate([Y_test_all, Y_test_z])
                Y_test_pred_all = np.concatenate([Y_test_pred_all, Y_test_pred])

            beta_reshaped = beta[:n_features * 4].reshape(4, n_features)
            mean_test_r = np.mean(corr_test)
            geo_mean_p = np.exp(np.mean(np.log(p_test))) if len(p_test) > 0 else np.nan

            # ==============================================================
            # Session-level covariates
            # ==============================================================

            # --- Performance ---
            hit_rate = np.nanmean(hit)
            mean_rt = np.nanmean(rt)
            std_rt = np.nanstd(rt_filled)
            # Learning slope: linear fit of hit rate across trial thirds
            thirds = np.array_split(hit.astype(float), 3)
            hit_by_third = [np.nanmean(t) for t in thirds]
            learning_slope = hit_by_third[2] - hit_by_third[0] if len(hit_by_third) == 3 else np.nan

            # --- BCI neuron identity ---
            cn_indices = data['conditioned_neuron']
            if hasattr(cn_indices, '__len__') and len(cn_indices) > 0:
                if hasattr(cn_indices[0], '__len__'):
                    cn_idx = cn_indices[0][0] if len(cn_indices[0]) > 0 else 0
                else:
                    cn_idx = cn_indices[0]
            else:
                cn_idx = 0
            cn_idx = int(cn_idx)

            # CN baseline activity (pre-trial epoch)
            cn_baseline = np.nanmean(kpre[cn_idx, :]) if cn_idx < n_neurons else np.nan
            # CN go-cue activity
            cn_go_activity = np.nanmean(k[cn_idx, :]) if cn_idx < n_neurons else np.nan
            # CN correlation with population (mean pairwise corr)
            if cn_idx < n_neurons:
                df_cl = data['df_closedloop']
                if df_cl.ndim == 2 and df_cl.shape[0] == n_neurons:
                    cn_trace = df_cl[cn_idx, :]
                    pop_mean = np.nanmean(df_cl, axis=0)
                    finite = np.isfinite(cn_trace) & np.isfinite(pop_mean)
                    if np.sum(finite) > 10:
                        cn_pop_corr, _ = pearsonr(cn_trace[finite], pop_mean[finite])
                    else:
                        cn_pop_corr = np.nan
                else:
                    cn_pop_corr = np.nan
            else:
                cn_pop_corr = np.nan

            # --- Population activity ---
            df_cl = data['df_closedloop']
            if df_cl.ndim == 2:
                # Mean and std of population activity
                pop_mean_activity = np.nanmean(df_cl)
                pop_std_activity = np.nanstd(np.nanmean(df_cl, axis=0))

                # Mean pairwise correlation (subsample for speed)
                n_samp = min(n_neurons, 100)
                rng = np.random.default_rng(42)
                idx_sub = rng.choice(n_neurons, n_samp, replace=False) if n_neurons > n_samp else np.arange(n_neurons)
                sub = df_cl[idx_sub, :]
                sub = sub[:, np.all(np.isfinite(sub), axis=0)]
                if sub.shape[1] > 10:
                    cc_pop = np.corrcoef(sub)
                    triu = np.triu_indices(cc_pop.shape[0], k=1)
                    mean_pairwise_corr = np.nanmean(cc_pop[triu])
                else:
                    mean_pairwise_corr = np.nan

                # Dimensionality (effective rank from SVD)
                sub_clean = sub - np.nanmean(sub, axis=1, keepdims=True)
                try:
                    s = np.linalg.svd(sub_clean, compute_uv=False)
                    s = s[s > 0]
                    p_s = s / s.sum()
                    eff_dim = np.exp(-np.sum(p_s * np.log(p_s)))
                except:
                    eff_dim = np.nan

                # Epoch-specific mean population activity
                pop_pre_mean = np.nanmean(kpre)
                pop_go_mean = np.nanmean(k)
                pop_late_mean = np.nanmean(kstep)
                pop_rew_mean = np.nanmean(krewards)

                # Population variance per epoch (across-neuron variance, averaged over trials)
                pop_pre_var = np.nanmean(np.nanvar(kpre, axis=0))
                pop_go_var = np.nanmean(np.nanvar(k, axis=0))
                pop_late_var = np.nanmean(np.nanvar(kstep, axis=0))
                pop_rew_var = np.nanmean(np.nanvar(krewards, axis=0))
            else:
                pop_mean_activity = np.nan
                pop_std_activity = np.nan
                mean_pairwise_corr = np.nan
                eff_dim = np.nan
                pop_pre_mean = pop_go_mean = pop_late_mean = pop_rew_mean = np.nan
                pop_pre_var = pop_go_var = pop_late_var = pop_rew_var = np.nan

            # --- Photostim / connectivity ---
            n_stim_groups = stimDist.shape[1]
            ind_far = np.where(stimDist > 30)
            mean_dw = np.nanmean(AMP[1][ind_far] - AMP[0][ind_far]) if len(ind_far[0]) > 0 else np.nan
            mean_amp_pre = np.nanmean(AMP[0]) if len(AMP) > 0 else np.nan
            mean_amp_post = np.nanmean(AMP[1]) if len(AMP) > 1 else np.nan

            # --- Session metadata ---
            try:
                session_date = datetime.datetime.strptime(session, "%m%d%y").date()
            except:
                session_date = None
            # Session number within mouse
            session_num = sii

            # Mean BCI threshold
            mean_thr = np.nanmean(BCI_thresholds[1, :])

            # ==============================================================
            # Store everything
            # ==============================================================
            session_result = {
                # Identity
                'mouse': mouse,
                'session': session,
                'session_date': session_date,
                'session_num': session_num,
                # Fit results
                'beta': beta_reshaped,
                'beta_flat': beta_reshaped.ravel(),
                'mean_test_r': mean_test_r,
                'mean_train_r': np.mean(corr_train),
                'geo_mean_p': geo_mean_p,
                'n_pairs': Y_T.shape[0],
                # Performance
                'hit_rate': hit_rate,
                'mean_rt': mean_rt,
                'std_rt': std_rt,
                'learning_slope': learning_slope,
                # BCI neuron
                'cn_idx': cn_idx,
                'cn_baseline': cn_baseline,
                'cn_go_activity': cn_go_activity,
                'cn_pop_corr': cn_pop_corr,
                # Population
                'n_neurons': n_neurons,
                'n_trials': trl,
                'pop_mean_activity': pop_mean_activity,
                'pop_std_activity': pop_std_activity,
                'mean_pairwise_corr': mean_pairwise_corr,
                'eff_dim': eff_dim,
                'pop_pre_mean': pop_pre_mean,
                'pop_go_mean': pop_go_mean,
                'pop_late_mean': pop_late_mean,
                'pop_rew_mean': pop_rew_mean,
                'pop_pre_var': pop_pre_var,
                'pop_go_var': pop_go_var,
                'pop_late_var': pop_late_var,
                'pop_rew_var': pop_rew_var,
                # Photostim
                'n_stim_groups': n_stim_groups,
                'mean_dw': mean_dw,
                'mean_amp_pre': mean_amp_pre,
                'mean_amp_post': mean_amp_post,
                'mean_thr': mean_thr,
            }

            all_sessions.append(session_result)
            print(f"  r={mean_test_r:.3f}, p={geo_mean_p:.2e}, hit={hit_rate*100:.0f}%, "
                  f"n_neurons={n_neurons}, n_pairs={Y_T.shape[0]}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\n{'='*70}")
print(f"COMPLETE: {len(all_sessions)} sessions collected")
print(f"{'='*70}")

#%% ============================================================================
# CELL 4: Save results
# ============================================================================
save_path = os.path.join(RESULTS_DIR, 'session_results.npy')
np.save(save_path, all_sessions, allow_pickle=True)
print(f"Saved {len(all_sessions)} sessions to {save_path}")

#%% ============================================================================
# CELL 5: Load results and run meta-analysis
# ============================================================================

# Load (or use from memory if just ran)
save_path = os.path.join(RESULTS_DIR, 'session_results.npy')
all_sessions = np.load(save_path, allow_pickle=True).tolist()
print(f"Loaded {len(all_sessions)} sessions")

# Filter to sessions with good fits
min_r = 0.0  # include all for now; raise to e.g. 0.05 to filter weak fits
good = [s for s in all_sessions if np.isfinite(s['mean_test_r']) and s['mean_test_r'] > min_r]
print(f"Using {len(good)} sessions (test r > {min_r})")

# Stack betas: (n_sessions, 4, 5) -> (n_sessions, 20)
beta_stack = np.array([s['beta_flat'] for s in good])
n_sess = beta_stack.shape[0]

# Build covariate matrix
cov_names = [
    'hit_rate', 'mean_rt', 'std_rt', 'learning_slope',
    'cn_baseline', 'cn_go_activity', 'cn_pop_corr',
    'n_neurons', 'n_trials',
    'pop_mean_activity', 'pop_std_activity', 'mean_pairwise_corr', 'eff_dim',
    'pop_pre_mean', 'pop_go_mean', 'pop_late_mean', 'pop_rew_mean',
    'pop_pre_var', 'pop_go_var', 'pop_late_var', 'pop_rew_var',
    'n_stim_groups', 'mean_dw', 'mean_amp_pre', 'mean_amp_post', 'mean_thr',
]

cov_matrix = np.array([[s[name] for name in cov_names] for s in good])
mouse_ids = np.array([s['mouse'] for s in good])
test_r = np.array([s['mean_test_r'] for s in good])

# Feature labels for betas
beta_labels = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]

#%% ============================================================================
# CELL 6: PCA on betas — is there low-dimensional structure?
# ============================================================================
from sklearn.decomposition import PCA

# Z-score betas across sessions before PCA
beta_z = (beta_stack - np.nanmean(beta_stack, axis=0)) / (np.nanstd(beta_stack, axis=0) + 1e-10)

pca = PCA()
scores = pca.fit_transform(beta_z)
var_explained = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scree plot
axes[0].bar(range(1, len(var_explained) + 1), np.cumsum(var_explained), color='steelblue')
axes[0].set_xlabel('PC')
axes[0].set_ylabel('Cumulative variance explained')
axes[0].set_title('PCA of beta matrices')
axes[0].axhline(0.8, color='r', ls='--', alpha=0.5)

# PC1 vs PC2, colored by mouse
unique_mice = np.unique(mouse_ids)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_mice)))
for mi_idx, m in enumerate(unique_mice):
    mask = mouse_ids == m
    axes[1].scatter(scores[mask, 0], scores[mask, 1], c=[colors[mi_idx]],
                    label=m, alpha=0.7, s=40)
axes[1].set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
axes[1].set_title('Sessions in beta-PC space')
axes[1].legend(fontsize=8)

# PC1 loadings — which beta components vary most?
pc1_loadings = pca.components_[0]
sort_idx = np.argsort(np.abs(pc1_loadings))[::-1]
axes[2].barh(range(len(pc1_loadings)), pc1_loadings[sort_idx], color='steelblue')
axes[2].set_yticks(range(len(pc1_loadings)))
axes[2].set_yticklabels([beta_labels[i] for i in sort_idx], fontsize=7)
axes[2].set_xlabel('PC1 loading')
axes[2].set_title('Which betas vary most across sessions?')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'beta_pca.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Correlate covariates with beta PCs and individual betas
# ============================================================================

# Correlate each covariate with each beta and with PC1/PC2
n_cov = cov_matrix.shape[1]
n_beta = beta_stack.shape[1]

# Covariate vs beta correlation matrix
corr_cov_beta = np.full((n_cov, n_beta), np.nan)
pval_cov_beta = np.full((n_cov, n_beta), np.nan)

for ci in range(n_cov):
    for bi in range(n_beta):
        finite = np.isfinite(cov_matrix[:, ci]) & np.isfinite(beta_stack[:, bi])
        if np.sum(finite) > 5:
            r, p = pearsonr(cov_matrix[finite, ci], beta_stack[finite, bi])
            corr_cov_beta[ci, bi] = r
            pval_cov_beta[ci, bi] = p

# Covariate vs beta PCs
corr_cov_pc = np.full((n_cov, min(5, scores.shape[1])), np.nan)
pval_cov_pc = np.full((n_cov, min(5, scores.shape[1])), np.nan)

for ci in range(n_cov):
    for pi in range(min(5, scores.shape[1])):
        finite = np.isfinite(cov_matrix[:, ci])
        if np.sum(finite) > 5:
            r, p = pearsonr(cov_matrix[finite, ci], scores[finite, pi])
            corr_cov_pc[ci, pi] = r
            pval_cov_pc[ci, pi] = p

# Plot: covariate × beta correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Full matrix
im = axes[0].imshow(corr_cov_beta, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[0].set_xticks(range(n_beta))
axes[0].set_xticklabels(beta_labels, rotation=90, fontsize=7)
axes[0].set_yticks(range(n_cov))
axes[0].set_yticklabels(cov_names, fontsize=8)
axes[0].set_title(f'Covariate-Beta correlations (n={n_sess} sessions)')
plt.colorbar(im, ax=axes[0], fraction=0.02)

# Mark significant ones
for ci in range(n_cov):
    for bi in range(n_beta):
        if pval_cov_beta[ci, bi] < 0.05:
            axes[0].text(bi, ci, '*', ha='center', va='center', fontsize=8, fontweight='bold')

# Covariate vs PCs
im2 = axes[1].imshow(corr_cov_pc, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[1].set_xticks(range(corr_cov_pc.shape[1]))
axes[1].set_xticklabels([f'PC{i+1}' for i in range(corr_cov_pc.shape[1])], fontsize=9)
axes[1].set_yticks(range(n_cov))
axes[1].set_yticklabels(cov_names, fontsize=8)
axes[1].set_title('Covariate-PC correlations')
plt.colorbar(im2, ax=axes[1], fraction=0.02)

for ci in range(n_cov):
    for pi in range(corr_cov_pc.shape[1]):
        if pval_cov_pc[ci, pi] < 0.05:
            axes[1].text(pi, ci, '*', ha='center', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'beta_covariate_summary.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Top covariate hits — scatter plots for strongest relationships
# ============================================================================

# Find the strongest covariate-PC correlations
n_top = 6
abs_corr_pc1 = np.abs(corr_cov_pc[:, 0])
abs_corr_pc1[~np.isfinite(abs_corr_pc1)] = 0
top_cov_idx = np.argsort(abs_corr_pc1)[::-1][:n_top]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.ravel()

for i, ci in enumerate(top_cov_idx):
    ax = axes[i]
    x = cov_matrix[:, ci]
    y = scores[:, 0]
    finite = np.isfinite(x) & np.isfinite(y)

    for mi_idx, m in enumerate(unique_mice):
        mask = (mouse_ids == m) & finite
        ax.scatter(x[mask], y[mask], c=[colors[mi_idx]], label=m, alpha=0.7, s=30)

    r_val = corr_cov_pc[ci, 0]
    p_val = pval_cov_pc[ci, 0]
    ax.set_xlabel(cov_names[ci], fontsize=9)
    ax.set_ylabel('Beta PC1', fontsize=9)
    ax.set_title(f'r={r_val:.2f}, p={p_val:.3f}', fontsize=10)

    # Fit line
    if np.sum(finite) > 5:
        z = np.polyfit(x[finite], y[finite], 1)
        xline = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), 50)
        ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5)

axes[0].legend(fontsize=6, loc='best')
plt.suptitle('Top covariates predicting beta PC1', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'top_covariate_scatters.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 9: Mean beta heatmap and consistency across mice
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# Top row: per-mouse mean beta
for mi_idx, m in enumerate(unique_mice):
    if mi_idx >= 4:
        break
    mask = mouse_ids == m
    if np.sum(mask) < 2:
        continue
    mean_beta = np.nanmean(beta_stack[mask], axis=0).reshape(4, len(BEHAVIOR_NAMES))
    ax = axes[0, mi_idx]
    sns.heatmap(mean_beta, annot=True, fmt='.2f', ax=ax,
                xticklabels=BEHAVIOR_NAMES, yticklabels=EPOCH_ORDER,
                cmap='coolwarm', center=0, vmin=-0.15, vmax=0.15, cbar=False)
    ax.set_title(f'{m} (n={np.sum(mask)})')

# Handle remaining mice
for mi_idx, m in enumerate(unique_mice):
    if mi_idx < 4:
        continue
    if mi_idx - 4 >= 4:
        break
    mask = mouse_ids == m
    if np.sum(mask) < 2:
        continue
    mean_beta = np.nanmean(beta_stack[mask], axis=0).reshape(4, len(BEHAVIOR_NAMES))
    ax = axes[1, mi_idx - 4]
    sns.heatmap(mean_beta, annot=True, fmt='.2f', ax=ax,
                xticklabels=BEHAVIOR_NAMES, yticklabels=EPOCH_ORDER,
                cmap='coolwarm', center=0, vmin=-0.15, vmax=0.15, cbar=False)
    ax.set_title(f'{m} (n={np.sum(mask)})')

# Grand mean
ax = axes[1, -1]
grand_mean = np.nanmean(beta_stack, axis=0).reshape(4, len(BEHAVIOR_NAMES))
sns.heatmap(grand_mean, annot=True, fmt='.2f', ax=ax,
            xticklabels=BEHAVIOR_NAMES, yticklabels=EPOCH_ORDER,
            cmap='coolwarm', center=0, vmin=-0.15, vmax=0.15)
ax.set_title(f'Grand mean (n={n_sess})')

plt.suptitle('Mean beta by mouse', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'beta_by_mouse.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 10: Print summary of significant covariate-beta relationships
# ============================================================================

print("\n" + "="*70)
print("SIGNIFICANT COVARIATE-BETA RELATIONSHIPS (p < 0.05)")
print("="*70)

sig_pairs = []
for ci in range(n_cov):
    for bi in range(n_beta):
        if pval_cov_beta[ci, bi] < 0.05:
            sig_pairs.append((cov_names[ci], beta_labels[bi],
                            corr_cov_beta[ci, bi], pval_cov_beta[ci, bi]))

sig_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"\n{'Covariate':<25} {'Beta':<20} {'r':>8} {'p':>10}")
print("-" * 65)
for cov, beta_name, r, p in sig_pairs:
    print(f"{cov:<25} {beta_name:<20} {r:>8.3f} {p:>10.4f}")

print(f"\nTotal significant: {len(sig_pairs)} / {n_cov * n_beta}")

print("\n" + "="*70)
print("SIGNIFICANT COVARIATE-PC RELATIONSHIPS (p < 0.05)")
print("="*70)
for pi in range(min(3, corr_cov_pc.shape[1])):
    print(f"\nPC{pi+1} ({var_explained[pi]*100:.1f}% variance):")
    for ci in range(n_cov):
        if pval_cov_pc[ci, pi] < 0.05:
            print(f"  {cov_names[ci]:<25} r={corr_cov_pc[ci, pi]:.3f}  p={pval_cov_pc[ci, pi]:.4f}")
