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

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

# Override zscore_mat to match the working _2026.py version:
# divide by std WITHOUT subtracting mean
def zscore_mat_std_only(X):
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1
    return X / sd

print("Setup complete!")

#%% ============================================================================
# CELL 2: Run collection with BOTH pairwise modes
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

fit_type = 'pinv'
num_bins = 10
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_NAMES = ['hit', 'hit_rpe', '10-rt', 'rpe', '2-factor']

MODES = ['dot_prod', 'dot_prod_z']
results_by_mode = {mode: [] for mode in MODES}

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

            # Load data
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

            # Preprocess (identical to meta_analysis)
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

            # Signal vs noise decomposition for each epoch
            # Signal correlation = outer product of trial-mean activity
            # Noise = residual after removing mean
            epoch_arrays = {'pre': kpre, 'go_cue': k, 'late': kstep, 'reward': krewards}
            signal_frac = {}
            for ep_name, ep_data in epoch_arrays.items():
                mu_vec = np.nanmean(ep_data, axis=1)  # mean across trials
                signal_cov = np.outer(mu_vec, mu_vec)
                total_cov = ep_data @ ep_data.T / max(trl, 1)
                # fraction of total pairwise variance explained by signal
                total_norm = np.sum(total_cov ** 2)
                if total_norm > 0:
                    signal_frac[ep_name] = np.sum(signal_cov ** 2) / total_norm
                else:
                    signal_frac[ep_name] = np.nan

            # Population covariates (same as before)
            hit_rate = np.nanmean(hit)
            mean_rt = np.nanmean(rt)
            df_cl = data['df_closedloop']
            pop_mean_activity = np.nanmean(df_cl) if df_cl.ndim == 2 else np.nan

            n_samp = min(n_neurons, 100)
            rng = np.random.default_rng(42)
            idx_sub = rng.choice(n_neurons, n_samp, replace=False) if n_neurons > n_samp else np.arange(n_neurons)
            if df_cl.ndim == 2:
                sub = df_cl[idx_sub, :]
                sub = sub[:, np.all(np.isfinite(sub), axis=0)]
                if sub.shape[1] > 10:
                    cc_pop = np.corrcoef(sub)
                    triu = np.triu_indices(cc_pop.shape[0], k=1)
                    mean_pairwise_corr = np.nanmean(cc_pop[triu])
                else:
                    mean_pairwise_corr = np.nan
            else:
                mean_pairwise_corr = np.nan

            # ============================================================
            # Fit under both modes
            # ============================================================
            for pairwise_mode in MODES:
                try:
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
                            cc=cc, centered_dot=centered_dot,
                            zscore_mat=zscore_mat_std_only,
                            return_interleaved_CC=True,
                        )

                    Xstep_T, Xrew_T, Xts_T, Xpre_T, Y_T, X_stim, used_bins = \
                        build_pairwise_XY_per_bin(
                            CCstep, CCrew, CCts, CCpre, stimDist, AMP,
                            nb=nb, hit_bin=hit_bin,
                            dist_target_lt=10, dist_nontarg_min=30,
                            dist_nontarg_max=1000, amp0_thr=0.1, amp1_thr=0.1,
                        )

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

                    X_T = np.hstack([Xpre_T @ Bz, Xts_T @ Bz, Xstep_T @ Bz, Xrew_T @ Bz])
                    assert X_T.shape[0] == Y_T.shape[0]

                    # Fit
                    n_features = behavior_features.shape[1]
                    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    corr_test, p_test = [], []

                    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_T, Y_T)):
                        X_train, X_test = X_T[train_idx], X_T[test_idx]
                        Y_train, Y_test = Y_T[train_idx], Y_T[test_idx]
                        mu, sigma = Y_train.mean(), Y_train.std()
                        if sigma == 0:
                            sigma = 1.0
                        Y_train_z = (Y_train - mu) / sigma
                        Y_test_z = (Y_test - mu) / sigma

                        beta = np.linalg.pinv(X_train) @ Y_train_z
                        Y_test_pred = X_test @ beta

                        if np.std(Y_test_pred) > 0:
                            r_test, pval_test = pearsonr(Y_test_pred, Y_test_z)
                        else:
                            r_test, pval_test = 0.0, 1.0
                        corr_test.append(r_test)
                        p_test.append(pval_test)

                    beta_reshaped = beta[:n_features * 4].reshape(4, n_features)

                    results_by_mode[pairwise_mode].append({
                        'mouse': mouse,
                        'session': session,
                        'beta': beta_reshaped,
                        'beta_flat': beta_reshaped.ravel(),
                        'mean_test_r': np.mean(corr_test),
                        'geo_mean_p': np.exp(np.mean(np.log(p_test))),
                        'n_pairs': Y_T.shape[0],
                        'hit_rate': hit_rate,
                        'mean_rt': mean_rt,
                        'mean_pairwise_corr': mean_pairwise_corr,
                        'pop_mean_activity': pop_mean_activity,
                        'n_neurons': n_neurons,
                        'n_trials': trl,
                        'signal_frac': signal_frac,
                    })

                    print(f"  {pairwise_mode}: r={np.mean(corr_test):.3f}")

                except Exception as e:
                    print(f"  {pairwise_mode} FAILED: {e}")
                    continue

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\n{'='*70}")
for mode in MODES:
    print(f"{mode}: {len(results_by_mode[mode])} sessions")
print(f"{'='*70}")

#%% ============================================================================
# CELL 3: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'followup_results.npy'), results_by_mode, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 4: Load and compare modes
# ============================================================================
results_by_mode = np.load(os.path.join(RESULTS_DIR, 'followup_results.npy'), allow_pickle=True).item()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel 1: Test r comparison ---
ax = axes[0]
for mode in MODES:
    rs = [s['mean_test_r'] for s in results_by_mode[mode]]
    ax.hist(rs, bins=15, alpha=0.5, label=f"{mode} (med={np.median(rs):.3f})")
ax.set_xlabel('Test r')
ax.set_ylabel('Count')
ax.set_title('Single-session fit quality')
ax.legend(fontsize=8)
ax.axvline(0, color='k', ls='--', alpha=0.3)

# --- Panel 2: Beta stability (cross-session std of each beta) ---
ax = axes[1]
for mi, mode in enumerate(MODES):
    betas = np.array([s['beta_flat'] for s in results_by_mode[mode]])
    beta_std = np.std(betas, axis=0)
    beta_labels = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]
    x = np.arange(len(beta_std)) + mi * 0.35
    ax.bar(x, beta_std, width=0.35, alpha=0.7, label=mode)
ax.set_xticks(np.arange(len(beta_labels)) + 0.175)
ax.set_xticklabels(beta_labels, rotation=90, fontsize=6)
ax.set_ylabel('Std of beta across sessions')
ax.set_title('Beta stability by mode')
ax.legend(fontsize=8)

# --- Panel 3: Does mean_pairwise_corr still predict beta PC1 under dot_prod_z? ---
ax = axes[2]
from sklearn.decomposition import PCA
colors_mode = ['steelblue', 'coral']
for mi, mode in enumerate(MODES):
    data_list = results_by_mode[mode]
    betas = np.array([s['beta_flat'] for s in data_list])
    mpc = np.array([s['mean_pairwise_corr'] for s in data_list])

    betas_z = (betas - betas.mean(axis=0)) / (betas.std(axis=0) + 1e-10)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(betas_z)

    finite = np.isfinite(mpc)
    if np.sum(finite) > 5:
        r, p = pearsonr(mpc[finite], scores[finite, 0])
        ax.scatter(mpc[finite], scores[finite, 0], alpha=0.5, c=colors_mode[mi],
                   label=f"{mode}: r={r:.2f}, p={p:.3f}", s=30)

ax.set_xlabel('Mean pairwise correlation')
ax.set_ylabel('Beta PC1')
ax.set_title('Does pairwise corr still predict betas\nafter variance normalization?')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mode_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 5: Signal vs noise fraction — does it explain beta variation?
# ============================================================================

# Use dot_prod results (the mode where fits are strong)
dp_results = results_by_mode['dot_prod']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

beta_stack = np.array([s['beta_flat'] for s in dp_results])
beta_z = (beta_stack - beta_stack.mean(axis=0)) / (beta_stack.std(axis=0) + 1e-10)
pca = PCA(n_components=5)
scores = pca.fit_transform(beta_z)

# Top row: signal fraction per epoch vs beta PCs
for ei, epoch in enumerate(EPOCH_ORDER):
    ax = axes[0, ei]
    sig_frac = np.array([s['signal_frac'].get(epoch, np.nan) for s in dp_results])
    finite = np.isfinite(sig_frac)

    if np.sum(finite) > 5:
        r, p = pearsonr(sig_frac[finite], scores[finite, 0])
        ax.scatter(sig_frac[finite], scores[finite, 0], alpha=0.6, s=25, c='steelblue')
        z = np.polyfit(sig_frac[finite], scores[finite, 0], 1)
        xline = np.linspace(np.nanmin(sig_frac[finite]), np.nanmax(sig_frac[finite]), 50)
        ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5)
        ax.set_title(f'{epoch}\nr={r:.2f}, p={p:.3f}')
    else:
        ax.set_title(f'{epoch}\ninsufficient data')
    ax.set_xlabel('Signal fraction')
    ax.set_ylabel('Beta PC1')

# Bottom row: signal fraction vs specific epoch betas
for ei, epoch in enumerate(EPOCH_ORDER):
    ax = axes[1, ei]
    sig_frac = np.array([s['signal_frac'].get(epoch, np.nan) for s in dp_results])
    # Get the mean absolute beta for this epoch
    epoch_betas = beta_stack[:, ei * len(BEHAVIOR_NAMES):(ei + 1) * len(BEHAVIOR_NAMES)]
    epoch_beta_norm = np.sqrt(np.sum(epoch_betas ** 2, axis=1))

    finite = np.isfinite(sig_frac)
    if np.sum(finite) > 5:
        r, p = pearsonr(sig_frac[finite], epoch_beta_norm[finite])
        ax.scatter(sig_frac[finite], epoch_beta_norm[finite], alpha=0.6, s=25, c='coral')
        z = np.polyfit(sig_frac[finite], epoch_beta_norm[finite], 1)
        xline = np.linspace(np.nanmin(sig_frac[finite]), np.nanmax(sig_frac[finite]), 50)
        ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5)
        ax.set_title(f'r={r:.2f}, p={p:.3f}')
    ax.set_xlabel(f'{epoch} signal frac')
    ax.set_ylabel(f'{epoch} |beta|')

plt.suptitle('Signal fraction (mean activity) vs beta variation', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'signal_vs_noise.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 6: Median split on mean_pairwise_corr — compare beta profiles
# ============================================================================

dp_results = results_by_mode['dot_prod']
mpc = np.array([s['mean_pairwise_corr'] for s in dp_results])
finite = np.isfinite(mpc)
median_mpc = np.nanmedian(mpc)

high_idx = np.where(finite & (mpc >= median_mpc))[0]
low_idx = np.where(finite & (mpc < median_mpc))[0]

beta_high = np.array([dp_results[i]['beta_flat'] for i in high_idx])
beta_low = np.array([dp_results[i]['beta_flat'] for i in low_idx])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Mean beta heatmaps for high/low
for ax, betas, label in [
    (axes[0], beta_low, f'Low corr (n={len(low_idx)})'),
    (axes[1], beta_high, f'High corr (n={len(high_idx)})'),
]:
    mean_beta = np.nanmean(betas, axis=0).reshape(4, len(BEHAVIOR_NAMES))
    sns.heatmap(mean_beta, annot=True, fmt='.3f', ax=ax,
                xticklabels=BEHAVIOR_NAMES, yticklabels=EPOCH_ORDER,
                cmap='coolwarm', center=0, vmin=-0.08, vmax=0.08)
    ax.set_title(label)

# Difference + significance
ax = axes[2]
mean_diff = np.nanmean(beta_high, axis=0) - np.nanmean(beta_low, axis=0)
# Permutation p-values for each beta
n_perm = 5000
rng = np.random.default_rng(42)
all_betas = np.vstack([beta_low, beta_high])
n_low = beta_low.shape[0]
n_total = all_betas.shape[0]
perm_diffs = np.zeros((n_perm, all_betas.shape[1]))
for pi in range(n_perm):
    perm = rng.permutation(n_total)
    perm_diffs[pi] = np.nanmean(all_betas[perm[n_low:]], axis=0) - np.nanmean(all_betas[perm[:n_low]], axis=0)

pvals = np.mean(np.abs(perm_diffs) >= np.abs(mean_diff), axis=0)
diff_reshaped = mean_diff.reshape(4, len(BEHAVIOR_NAMES))
pval_reshaped = pvals.reshape(4, len(BEHAVIOR_NAMES))

# Annotate with diff value and significance star
annot = np.array([[f"{diff_reshaped[i,j]:.3f}{'*' if pval_reshaped[i,j]<0.05 else ''}"
                   for j in range(len(BEHAVIOR_NAMES))] for i in range(4)])

sns.heatmap(diff_reshaped, annot=annot, fmt='', ax=ax,
            xticklabels=BEHAVIOR_NAMES, yticklabels=EPOCH_ORDER,
            cmap='coolwarm', center=0, vmin=-0.06, vmax=0.06)
ax.set_title(f'High - Low (permutation test, * p<0.05)')

plt.suptitle(f'Median split on mean pairwise correlation (threshold={median_mpc:.4f})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'median_split_pairwise_corr.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7a: Rerun all analyses after removing outlier sessions
# ============================================================================
from scipy.stats import spearmanr

# Load both result sets
results_by_mode = np.load(os.path.join(RESULTS_DIR, 'followup_results.npy'), allow_pickle=True).item()
orig_results = np.load(os.path.join(RESULTS_DIR, 'session_results.npy'), allow_pickle=True).tolist()

# Use dot_prod mode for main analysis
dp_results = results_by_mode['dot_prod']
dpz_results = results_by_mode['dot_prod_z']

# Identify outliers via beta PC scores (>3 SD on any of first 3 PCs)
beta_stack_all = np.array([s['beta_flat'] for s in dp_results])
beta_z_all = (beta_stack_all - beta_stack_all.mean(axis=0)) / (beta_stack_all.std(axis=0) + 1e-10)
pca_all = PCA(n_components=3)
scores_all = pca_all.fit_transform(beta_z_all)

outlier_mask = np.zeros(len(dp_results), dtype=bool)
for pc in range(3):
    mu, sd = scores_all[:, pc].mean(), scores_all[:, pc].std()
    outlier_mask |= np.abs(scores_all[:, pc] - mu) > 3 * sd

n_outliers = np.sum(outlier_mask)
outlier_sessions = [(dp_results[i]['mouse'], dp_results[i]['session']) for i in np.where(outlier_mask)[0]]
print(f"Removing {n_outliers} outlier sessions: {outlier_sessions}")

# Clean indices
clean_idx = np.where(~outlier_mask)[0]
dp_clean = [dp_results[i] for i in clean_idx]

# Also clean dot_prod_z — match by mouse+session
outlier_set = set(outlier_sessions)
dpz_clean = [s for s in dpz_results if (s['mouse'], s['session']) not in outlier_set]

# Match orig_results to dp_clean by mouse+session for full covariate set
clean_set = set((s['mouse'], s['session']) for s in dp_clean)
orig_clean = [s for s in orig_results if (s['mouse'], s['session']) in clean_set]

print(f"Clean: dot_prod={len(dp_clean)}, dot_prod_z={len(dpz_clean)}, orig={len(orig_clean)}")

#%% ============================================================================
# CELL 7b: PCA on clean betas
# ============================================================================
beta_stack = np.array([s['beta_flat'] for s in dp_clean])
beta_z = (beta_stack - beta_stack.mean(axis=0)) / (beta_stack.std(axis=0) + 1e-10)
n_sess = beta_stack.shape[0]

pca = PCA()
scores = pca.fit_transform(beta_z)
var_explained = pca.explained_variance_ratio_

mouse_ids = np.array([s['mouse'] for s in dp_clean])
unique_mice = np.unique(mouse_ids)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_mice)))

beta_labels = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(range(1, min(11, len(var_explained)+1)),
            np.cumsum(var_explained[:10]), color='steelblue')
axes[0].set_xlabel('PC')
axes[0].set_ylabel('Cumulative var explained')
axes[0].set_title(f'PCA (n={n_sess}, outliers removed)')
axes[0].axhline(0.8, color='r', ls='--', alpha=0.5)

for mi_idx, m in enumerate(unique_mice):
    mask = mouse_ids == m
    axes[1].scatter(scores[mask, 0], scores[mask, 1], c=[colors[mi_idx]],
                    label=m, alpha=0.7, s=40)
axes[1].set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
axes[1].set_title('Sessions in beta-PC space (clean)')
axes[1].legend(fontsize=8)

pc1_loadings = pca.components_[0]
sort_idx = np.argsort(np.abs(pc1_loadings))[::-1]
axes[2].barh(range(len(pc1_loadings)), pc1_loadings[sort_idx], color='steelblue')
axes[2].set_yticks(range(len(pc1_loadings)))
axes[2].set_yticklabels([beta_labels[i] for i in sort_idx], fontsize=7)
axes[2].set_xlabel('PC1 loading')
axes[2].set_title('Which betas vary most? (clean)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'beta_pca_clean.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7c: Full covariate analysis on clean data (Pearson + Spearman)
# ============================================================================

# Build covariate matrix from orig_clean (has the full covariate set)
cov_names = [
    'hit_rate', 'mean_rt', 'std_rt', 'learning_slope',
    'cn_baseline', 'cn_go_activity', 'cn_pop_corr',
    'n_neurons', 'n_trials',
    'pop_mean_activity', 'pop_std_activity', 'mean_pairwise_corr', 'eff_dim',
    'pop_pre_mean', 'pop_go_mean', 'pop_late_mean', 'pop_rew_mean',
    'pop_pre_var', 'pop_go_var', 'pop_late_var', 'pop_rew_var',
    'n_stim_groups', 'mean_dw', 'mean_amp_pre', 'mean_amp_post', 'mean_thr',
]

# Align orig_clean to dp_clean ordering
orig_lookup = {(s['mouse'], s['session']): s for s in orig_clean}
cov_matrix = np.array([
    [orig_lookup[(s['mouse'], s['session'])][name] for name in cov_names]
    for s in dp_clean
], dtype=float)

n_cov = len(cov_names)
n_beta = beta_stack.shape[1]

# Pearson and Spearman for covariates vs betas
corr_pearson = np.full((n_cov, n_beta), np.nan)
pval_pearson = np.full((n_cov, n_beta), np.nan)
corr_spearman = np.full((n_cov, n_beta), np.nan)
pval_spearman = np.full((n_cov, n_beta), np.nan)

for ci in range(n_cov):
    for bi in range(n_beta):
        finite = np.isfinite(cov_matrix[:, ci]) & np.isfinite(beta_stack[:, bi])
        if np.sum(finite) > 5:
            r, p = pearsonr(cov_matrix[finite, ci], beta_stack[finite, bi])
            corr_pearson[ci, bi] = r
            pval_pearson[ci, bi] = p
            r, p = spearmanr(cov_matrix[finite, ci], beta_stack[finite, bi])
            corr_spearman[ci, bi] = r
            pval_spearman[ci, bi] = p

# Covariates vs PCs (Spearman)
n_pcs = min(5, scores.shape[1])
corr_cov_pc_s = np.full((n_cov, n_pcs), np.nan)
pval_cov_pc_s = np.full((n_cov, n_pcs), np.nan)
corr_cov_pc_p = np.full((n_cov, n_pcs), np.nan)
pval_cov_pc_p = np.full((n_cov, n_pcs), np.nan)

for ci in range(n_cov):
    for pi in range(n_pcs):
        finite = np.isfinite(cov_matrix[:, ci])
        if np.sum(finite) > 5:
            r, p = spearmanr(cov_matrix[finite, ci], scores[finite, pi])
            corr_cov_pc_s[ci, pi] = r
            pval_cov_pc_s[ci, pi] = p
            r, p = pearsonr(cov_matrix[finite, ci], scores[finite, pi])
            corr_cov_pc_p[ci, pi] = r
            pval_cov_pc_p[ci, pi] = p

# Plot: Spearman covariate × beta + covariate × PC
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

im = axes[0].imshow(corr_spearman, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[0].set_xticks(range(n_beta))
axes[0].set_xticklabels(beta_labels, rotation=90, fontsize=7)
axes[0].set_yticks(range(n_cov))
axes[0].set_yticklabels(cov_names, fontsize=8)
axes[0].set_title(f'Spearman covariate-beta (n={n_sess}, clean)')
plt.colorbar(im, ax=axes[0], fraction=0.02)
for ci in range(n_cov):
    for bi in range(n_beta):
        if pval_spearman[ci, bi] < 0.05:
            axes[0].text(bi, ci, '*', ha='center', va='center', fontsize=8, fontweight='bold')

im2 = axes[1].imshow(corr_cov_pc_s, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[1].set_xticks(range(n_pcs))
axes[1].set_xticklabels([f'PC{i+1}' for i in range(n_pcs)], fontsize=9)
axes[1].set_yticks(range(n_cov))
axes[1].set_yticklabels(cov_names, fontsize=8)
axes[1].set_title('Spearman covariate-PC (clean)')
plt.colorbar(im2, ax=axes[1], fraction=0.02)
for ci in range(n_cov):
    for pi in range(n_pcs):
        if pval_cov_pc_s[ci, pi] < 0.05:
            axes[1].text(pi, ci, '*', ha='center', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'covariate_summary_clean.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7d: Top hits — scatter plots for strongest Spearman correlations
# ============================================================================

# Gather all covariate-PC Spearman correlations, find top hits
all_hits = []
for ci in range(n_cov):
    for pi in range(min(3, n_pcs)):
        if np.isfinite(corr_cov_pc_s[ci, pi]):
            all_hits.append((ci, pi, corr_cov_pc_s[ci, pi], pval_cov_pc_s[ci, pi]))
    for bi in range(n_beta):
        if np.isfinite(corr_spearman[ci, bi]):
            all_hits.append((ci, -(bi+1), corr_spearman[ci, bi], pval_spearman[ci, bi]))

# Sort by absolute correlation
all_hits.sort(key=lambda x: abs(x[2]), reverse=True)

# Plot top 9 unique covariates
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.ravel()
plotted_covs = set()
plot_idx = 0

for ci, target_idx, rho, pval in all_hits:
    if plot_idx >= 9:
        break
    if ci in plotted_covs:
        continue
    plotted_covs.add(ci)

    ax = axes[plot_idx]
    x = cov_matrix[:, ci]
    if target_idx >= 0:
        y = scores[:, target_idx]
        ylabel = f'Beta PC{target_idx+1}'
    else:
        bi = -(target_idx + 1)
        y = beta_stack[:, bi]
        ylabel = beta_labels[bi]

    finite = np.isfinite(x) & np.isfinite(y)
    for mi_idx, m in enumerate(unique_mice):
        mask = (mouse_ids == m) & finite
        ax.scatter(x[mask], y[mask], c=[colors[mi_idx]], label=m, alpha=0.7, s=30)

    # Spearman on this pair
    rho_s, p_s = spearmanr(x[finite], y[finite])
    ax.set_xlabel(cov_names[ci], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'rho={rho_s:.2f}, p={p_s:.3f}', fontsize=10)

    if np.sum(finite) > 5:
        z = np.polyfit(x[finite], y[finite], 1)
        xline = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), 50)
        ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5)

    plot_idx += 1

if plot_idx > 0:
    axes[0].legend(fontsize=6, loc='best')
plt.suptitle(f'Top covariate-beta relationships (Spearman, n={n_sess}, outliers removed)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'top_scatters_clean.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7e: Mode comparison on clean data
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Fit quality
ax = axes[0]
for mode, data_clean, c in [('dot_prod', dp_clean, 'steelblue'), ('dot_prod_z', dpz_clean, 'coral')]:
    rs = [s['mean_test_r'] for s in data_clean]
    ax.hist(rs, bins=15, alpha=0.5, label=f"{mode} (med={np.median(rs):.3f})", color=c)
ax.set_xlabel('Test r')
ax.set_ylabel('Count')
ax.set_title(f'Fit quality (outliers removed)')
ax.legend(fontsize=8)
ax.axvline(0, color='k', ls='--', alpha=0.3)

# Beta stability comparison
ax = axes[1]
for mi, (mode, data_clean) in enumerate([('dot_prod', dp_clean), ('dot_prod_z', dpz_clean)]):
    betas = np.array([s['beta_flat'] for s in data_clean])
    beta_std = np.std(betas, axis=0)
    x = np.arange(len(beta_std)) + mi * 0.35
    ax.bar(x, beta_std, width=0.35, alpha=0.7, label=mode)
ax.set_xticks(np.arange(len(beta_labels)) + 0.175)
ax.set_xticklabels(beta_labels, rotation=90, fontsize=6)
ax.set_ylabel('Std of beta across sessions')
ax.set_title('Beta stability (clean)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mode_comparison_clean.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7f: FDR correction and full summary
# ============================================================================

# --- Benjamini-Hochberg FDR correction ---
def benjamini_hochberg(pvals_flat, alpha=0.1):
    """Returns boolean mask of which tests survive FDR at given alpha."""
    valid = np.isfinite(pvals_flat)
    n_valid = np.sum(valid)
    if n_valid == 0:
        return np.zeros_like(pvals_flat, dtype=bool), pvals_flat.copy()

    # Work only on valid p-values
    p_valid = pvals_flat[valid]
    sort_idx = np.argsort(p_valid)
    sorted_p = p_valid[sort_idx]

    # BH threshold: p_(k) <= k/m * alpha
    thresholds = (np.arange(1, n_valid + 1) / n_valid) * alpha

    # Find largest k where p_(k) <= threshold
    below = sorted_p <= thresholds
    if not np.any(below):
        # Nothing survives
        adjusted = np.full_like(pvals_flat, np.nan)
        adjusted[valid] = np.minimum.accumulate(
            (sorted_p * n_valid / np.arange(1, n_valid + 1))[::-1])[::-1]
        # Unsort
        adj_unsorted = np.empty_like(adjusted[valid])
        adj_unsorted[sort_idx] = adjusted[valid]
        adjusted[valid] = adj_unsorted
        return np.zeros_like(pvals_flat, dtype=bool), adjusted

    k_max = np.max(np.where(below)[0])

    # All tests with rank <= k_max are significant
    survives = np.zeros_like(pvals_flat, dtype=bool)
    sig_orig_idx = sort_idx[:k_max + 1]
    valid_indices = np.where(valid)[0]
    survives[valid_indices[sig_orig_idx]] = True

    # Adjusted p-values (BH)
    adjusted = np.full_like(pvals_flat, np.nan)
    raw_adj = np.minimum.accumulate(
        (sorted_p * n_valid / np.arange(1, n_valid + 1))[::-1])[::-1]
    raw_adj = np.minimum(raw_adj, 1.0)
    adj_unsorted = np.empty(n_valid)
    adj_unsorted[sort_idx] = raw_adj
    adjusted[valid] = adj_unsorted

    return survives, adjusted

# Flatten p-values for covariate-beta tests
pvals_flat = pval_spearman.ravel()
survives_beta, padj_beta = benjamini_hochberg(pvals_flat, alpha=0.1)
survives_beta_05, padj_beta_05 = benjamini_hochberg(pvals_flat, alpha=0.05)
survives_beta = survives_beta.reshape(pval_spearman.shape)
survives_beta_05 = survives_beta_05.reshape(pval_spearman.shape)
padj_beta = padj_beta.reshape(pval_spearman.shape)

# Flatten p-values for covariate-PC tests
pvals_pc_flat = pval_cov_pc_s.ravel()
survives_pc, padj_pc = benjamini_hochberg(pvals_pc_flat, alpha=0.1)
survives_pc = survives_pc.reshape(pval_cov_pc_s.shape)
padj_pc = padj_pc.reshape(pval_cov_pc_s.shape)

# --- Print results ---
print("\n" + "="*70)
print(f"CLEAN ANALYSIS SUMMARY (n={n_sess}, {n_outliers} outliers removed)")
print(f"Outliers: {outlier_sessions}")
print("="*70)

# Raw counts
n_tests_beta = np.sum(np.isfinite(pval_spearman))
n_sig_raw = np.sum(pval_spearman < 0.05)
n_sig_fdr10 = np.sum(survives_beta)
n_sig_fdr05 = np.sum(survives_beta_05)

print(f"\nCOVARIATE-BETA TESTS:")
print(f"  Total tests: {n_tests_beta}")
print(f"  Expected by chance (p<0.05): {n_tests_beta * 0.05:.0f}")
print(f"  Observed p<0.05 (uncorrected): {n_sig_raw}")
print(f"  Surviving FDR q<0.10: {n_sig_fdr10}")
print(f"  Surviving FDR q<0.05: {n_sig_fdr05}")

if n_sig_fdr10 > 0:
    print(f"\n  FDR q<0.10 SURVIVORS (Spearman, sorted by |rho|):")
    print(f"  {'Covariate':<25} {'Beta':<20} {'rho':>8} {'p_raw':>10} {'p_adj':>10}")
    print("  " + "-" * 75)
    fdr_pairs = []
    for ci in range(n_cov):
        for bi in range(n_beta):
            if survives_beta[ci, bi]:
                fdr_pairs.append((cov_names[ci], beta_labels[bi],
                                corr_spearman[ci, bi], pval_spearman[ci, bi],
                                padj_beta[ci, bi]))
    fdr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for cov, bname, rho, p_raw, p_adj in fdr_pairs:
        print(f"  {cov:<25} {bname:<20} {rho:>8.3f} {p_raw:>10.4f} {p_adj:>10.4f}")

print(f"\n  ALL p<0.05 UNCORRECTED (for reference, sorted by |rho|):")
print(f"  {'Covariate':<25} {'Beta':<20} {'rho':>8} {'p_raw':>10} {'p_adj':>10} {'FDR':>5}")
print("  " + "-" * 82)
sig_pairs = []
for ci in range(n_cov):
    for bi in range(n_beta):
        if pval_spearman[ci, bi] < 0.05:
            sig_pairs.append((cov_names[ci], beta_labels[bi],
                            corr_spearman[ci, bi], pval_spearman[ci, bi],
                            padj_beta[ci, bi], survives_beta[ci, bi]))
sig_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for cov, bname, rho, p_raw, p_adj, fdr in sig_pairs:
    marker = "**" if fdr else ""
    print(f"  {cov:<25} {bname:<20} {rho:>8.3f} {p_raw:>10.4f} {p_adj:>10.4f} {marker:>5}")

# Covariate-PC
n_tests_pc = np.sum(np.isfinite(pval_cov_pc_s))
n_sig_pc_raw = np.sum(pval_cov_pc_s < 0.05)
n_sig_pc_fdr = np.sum(survives_pc)

print(f"\nCOVARIATE-PC TESTS:")
print(f"  Total tests: {n_tests_pc}")
print(f"  Expected by chance: {n_tests_pc * 0.05:.0f}")
print(f"  Observed p<0.05: {n_sig_pc_raw}")
print(f"  Surviving FDR q<0.10: {n_sig_pc_fdr}")

print("\n  Covariate-PC hits (p<0.05 uncorrected):")
for pi in range(min(3, n_pcs)):
    print(f"\n  PC{pi+1} ({var_explained[pi]*100:.1f}% variance):")
    any_sig = False
    for ci in range(n_cov):
        if pval_cov_pc_s[ci, pi] < 0.05:
            fdr_mark = "**" if survives_pc[ci, pi] else ""
            print(f"    {cov_names[ci]:<25} rho={corr_cov_pc_s[ci, pi]:.3f}"
                  f"  p={pval_cov_pc_s[ci, pi]:.4f}  padj={padj_pc[ci, pi]:.4f} {fdr_mark}")
            any_sig = True
    if not any_sig:
        print("    (none)")

# --- Enrichment test: are there more hits than chance? ---
from scipy.stats import binomtest
result = binomtest(n_sig_raw, n_tests_beta, 0.05, alternative='greater')
p_enrichment = result.pvalue
print(f"\nENRICHMENT TEST (binomial):")
print(f"  {n_sig_raw} hits in {n_tests_beta} tests vs {n_tests_beta*0.05:.0f} expected")
print(f"  Binomial p = {p_enrichment:.4f}")
if p_enrichment < 0.05:
    print(f"  -> More hits than expected by chance (enriched)")
else:
    print(f"  -> NOT more hits than expected by chance")

#%% ============================================================================
# CELL 7: Summary statistics
# ============================================================================

print("\n" + "="*70)
print("MODE COMPARISON SUMMARY")
print("="*70)
for mode in MODES:
    rs = [s['mean_test_r'] for s in results_by_mode[mode]]
    ps = [s['geo_mean_p'] for s in results_by_mode[mode]]
    sig = sum(1 for p in ps if p < 0.05)
    print(f"\n{mode}:")
    print(f"  Sessions: {len(rs)}")
    print(f"  Test r: {np.median(rs):.3f} median, {np.mean(rs):.3f} mean")
    print(f"  Significant (p<0.05): {sig}/{len(ps)}")

    betas = np.array([s['beta_flat'] for s in results_by_mode[mode]])
    print(f"  Beta cross-session std (mean): {np.mean(np.std(betas, axis=0)):.4f}")

    mpc = np.array([s['mean_pairwise_corr'] for s in results_by_mode[mode]])
    betas_z = (betas - betas.mean(axis=0)) / (betas.std(axis=0) + 1e-10)
    pca = PCA(n_components=1)
    sc = pca.fit_transform(betas_z)
    finite = np.isfinite(mpc)
    if np.sum(finite) > 5:
        r, p = pearsonr(mpc[finite], sc[finite, 0])
        print(f"  mean_pairwise_corr vs PC1: r={r:.3f}, p={p:.4f}")

print("\n" + "="*70)
print("SIGNAL FRACTION SUMMARY (dot_prod mode)")
print("="*70)
for epoch in EPOCH_ORDER:
    fracs = [s['signal_frac'].get(epoch, np.nan) for s in dp_results]
    print(f"  {epoch}: {np.nanmean(fracs):.3f} +/- {np.nanstd(fracs):.3f}")

print("\n" + "="*70)
print("MEDIAN SPLIT: WHICH BETAS DIFFER? (p<0.05)")
print("="*70)
beta_labels = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]
for bi in range(len(beta_labels)):
    if pvals[bi] < 0.05:
        print(f"  {beta_labels[bi]}: diff={mean_diff[bi]:.4f}, p={pvals[bi]:.4f}")
if not any(pvals < 0.05):
    print("  None significant at p<0.05")
    # Show top 3
    top3 = np.argsort(pvals)[:3]
    for bi in top3:
        print(f"  (top) {beta_labels[bi]}: diff={mean_diff[bi]:.4f}, p={pvals[bi]:.4f}")
