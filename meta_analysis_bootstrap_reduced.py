#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
# Ensure this directory's modules are imported first (not the main repo copy)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import traceback
import plotting_functions as pf
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import os

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

# Override zscore_mat to match working _2026.py (std-only, no mean removal)
def zscore_mat_std_only(X):
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1
    return X / sd

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

# Pairwise modes to compare:
#   dot_prod  — Hebbian: CC[a,b] = sum_t r_a(t) * r_b(t)
#   pre_only  — Non-Hebbian: CC[a,b] = sum_t r_a(t), only presynaptic activity
#   post_only — Non-Hebbian: CC[a,b] = sum_t r_b(t), only postsynaptic activity
MODES_TO_RUN = ['dot_prod', 'pre_only', 'post_only']

num_bins = 10
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_NAMES = ['hit', 'hit_rpe', '10-rt', 'rpe', '2-factor']
n_epochs = len(EPOCH_ORDER)
n_behav = len(BEHAVIOR_NAMES)
n_full = n_epochs * n_behav

N_BOOT = 200  # bootstrap iterations per session

results_by_mode = {mode: [] for mode in MODES_TO_RUN}

print(f"Bootstrap iterations: {N_BOOT}")
print(f"Full model: {n_epochs} epochs x {n_behav} behaviors = {n_full} features")
print(f"Modes: {MODES_TO_RUN}")

#%% ============================================================================
# CELL 3: Main loop — bootstrap CIs + reduced model fits per session
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
            # Load + preprocess (same as before)
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

            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)
            miss_rpe = compute_rpe((~hit).astype(float), baseline=0.0, tau=tau_elig, fill_value=1.0)

            # ==============================================================
            # Helper: fit pinv and return CV test r
            # ==============================================================
            def cv_fit(X, Y, n_splits=5):
                """5-fold CV with pinv, returns mean test r."""
                if X.shape[1] == 0 or X.shape[0] < 10:
                    return np.nan
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                rs = []
                for train_idx, test_idx in cv.split(X, Y):
                    X_tr, X_te = X[train_idx], X[test_idx]
                    Y_tr, Y_te = Y[train_idx], Y[test_idx]
                    mu, sigma = Y_tr.mean(), Y_tr.std()
                    if sigma == 0: sigma = 1.0
                    Y_tr_z = (Y_tr - mu) / sigma
                    Y_te_z = (Y_te - mu) / sigma
                    beta = np.linalg.pinv(X_tr) @ Y_tr_z
                    pred = X_te @ beta
                    if np.std(pred) > 0:
                        r, _ = pearsonr(pred, Y_te_z)
                    else:
                        r = 0.0
                    rs.append(r)
                return np.mean(rs)

            # ==============================================================
            # Loop over pairwise modes (data loaded once, CC recomputed per mode)
            # ==============================================================
            feature_names = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]

            for pairwise_mode in MODES_TO_RUN:
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
                            cc=cc, centered_dot=centered_dot, zscore_mat=zscore_mat_std_only,
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

                    epoch_X = {
                        'pre': Xpre_T @ Bz,
                        'go_cue': Xts_T @ Bz,
                        'late': Xstep_T @ Bz,
                        'reward': Xrew_T @ Bz,
                    }

                    X_full = np.hstack([epoch_X[ep] for ep in EPOCH_ORDER])
                    assert X_full.shape[0] == Y_T.shape[0]
                    n_pairs = Y_T.shape[0]

                    # A: Full model CV fit
                    full_r = cv_fit(X_full, Y_T)

                    # Full model beta
                    mu_y, sig_y = Y_T.mean(), Y_T.std()
                    if sig_y == 0: sig_y = 1.0
                    Y_z = (Y_T - mu_y) / sig_y
                    beta_full = np.linalg.pinv(X_full) @ Y_z

                    # B: Bootstrap CIs
                    rng = np.random.default_rng(42)
                    boot_betas = np.zeros((N_BOOT, n_full))
                    for bi in range(N_BOOT):
                        idx = rng.choice(n_pairs, n_pairs, replace=True)
                        X_b, Y_b = X_full[idx], Y_T[idx]
                        mu_b, sig_b = Y_b.mean(), Y_b.std()
                        if sig_b == 0: sig_b = 1.0
                        Y_b_z = (Y_b - mu_b) / sig_b
                        boot_betas[bi] = np.linalg.pinv(X_b) @ Y_b_z

                    beta_ci_lo = np.percentile(boot_betas, 2.5, axis=0)
                    beta_ci_hi = np.percentile(boot_betas, 97.5, axis=0)
                    beta_reliable = (beta_ci_lo > 0) | (beta_ci_hi < 0)

                    # C: Reduced models
                    epoch_r = {}
                    for ep in EPOCH_ORDER:
                        epoch_r[ep] = cv_fit(epoch_X[ep], Y_T)

                    behav_r = {}
                    for bii, bname in enumerate(BEHAVIOR_NAMES):
                        X_beh = np.hstack([epoch_X[ep][:, bii:bii+1] for ep in EPOCH_ORDER])
                        behav_r[bname] = cv_fit(X_beh, Y_T)

                    single_r = np.zeros(n_full)
                    for fi in range(n_full):
                        single_r[fi] = cv_fit(X_full[:, fi:fi+1], Y_T)

                    twof_idx = [ei * n_behav + BEHAVIOR_NAMES.index('2-factor') for ei in range(n_epochs)]
                    twofactor_r = cv_fit(X_full[:, twof_idx], Y_T)

                    best_single_idx = np.argmax(single_r)
                    best_single_r = single_r[best_single_idx]

                    # Store
                    results_by_mode[pairwise_mode].append({
                        'mouse': mouse,
                        'session': session,
                        'n_pairs': n_pairs,
                        'hit_rate': np.nanmean(hit),
                        'full_r': full_r,
                        'beta_full': beta_full.ravel(),
                        'beta_reliable': beta_reliable,
                        'beta_ci_lo': beta_ci_lo,
                        'beta_ci_hi': beta_ci_hi,
                        'boot_betas': boot_betas,
                        'epoch_r': epoch_r,
                        'behav_r': behav_r,
                        'single_r': single_r,
                        'twofactor_r': twofactor_r,
                        'best_single_idx': best_single_idx,
                        'best_single_r': best_single_r,
                    })

                    n_rel = np.sum(beta_reliable)
                    print(f"  {pairwise_mode}: r={full_r:.3f} | reliable: {n_rel}/{n_full} | "
                          f"2-factor: {twofactor_r:.3f}")

                except Exception as e:
                    print(f"  {pairwise_mode} FAILED: {e}")
                    continue

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\n{'='*70}")
for mode in MODES_TO_RUN:
    print(f"{mode}: {len(results_by_mode[mode])} sessions")
print(f"{'='*70}")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'bootstrap_reduced_results.npy'),
        results_by_mode, allow_pickle=True)
for mode in MODES_TO_RUN:
    print(f"Saved {mode}: {len(results_by_mode[mode])} sessions")
print("Done.")

#%% ============================================================================
# CELL 5: Load + analyze bootstrap results
# ============================================================================
results_by_mode = np.load(os.path.join(RESULTS_DIR, 'bootstrap_reduced_results.npy'),
                          allow_pickle=True).item()

feature_names = [f"{ep}_{beh}" for ep in EPOCH_ORDER for beh in BEHAVIOR_NAMES]
n_full = len(feature_names)
MODES_TO_RUN = list(results_by_mode.keys())

# Build stacked arrays per mode
stacked = {}
for mode in MODES_TO_RUN:
    sess = results_by_mode[mode]
    n_sess = len(sess)
    stacked[mode] = {
        'beta_stack': np.array([s['beta_full'] for s in sess]),
        'reliable_stack': np.array([s['beta_reliable'] for s in sess]),
        'full_r': np.array([s['full_r'] for s in sess]),
        'twof_r': np.array([s['twofactor_r'] for s in sess]),
        'best_r': np.array([s['best_single_r'] for s in sess]),
        'mouse_ids': np.array([s['mouse'] for s in sess]),
        'n_sess': n_sess,
        'sessions': sess,
    }
    print(f"Loaded {mode}: {n_sess} sessions")

# Default mode for single-mode plots (Cells 6-9)
MODE = 'dot_prod'
sd = stacked[MODE]
beta_stack = sd['beta_stack']
reliable_stack = sd['reliable_stack']
full_r_arr = sd['full_r']
n_sess = sd['n_sess']
print(f"\nDefault mode for detailed plots: {MODE}")

#%% ============================================================================
# CELL 6: How many betas are reliable within session? (uses MODE from Cell 5)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f'Bootstrap reliability — {MODE}', fontsize=12, y=1.02)

# Panel 1: Fraction of sessions where each beta is reliable
frac_reliable = np.mean(reliable_stack, axis=0)
sort_idx = np.argsort(frac_reliable)[::-1]

ax = axes[0]
ax.barh(range(n_full), frac_reliable[sort_idx], color='steelblue')
ax.set_yticks(range(n_full))
ax.set_yticklabels([feature_names[i] for i in sort_idx], fontsize=7)
ax.set_xlabel('Fraction of sessions where 95% CI excludes 0')
ax.set_title('Beta reliability across sessions')
ax.axvline(0.05, color='r', ls='--', alpha=0.5, label='chance (5%)')
ax.invert_yaxis()
ax.legend(fontsize=8)

# Panel 2: Distribution of # reliable betas per session
n_reliable_per_session = np.sum(reliable_stack, axis=1)
ax = axes[1]
ax.hist(n_reliable_per_session, bins=np.arange(-0.5, n_full + 1.5, 1),
        color='steelblue', edgecolor='white')
ax.set_xlabel('# reliable betas (out of 20)')
ax.set_ylabel('# sessions')
ax.set_title(f'Reliable betas per session\nmedian={np.median(n_reliable_per_session):.0f}')
ax.axvline(n_full * 0.05, color='r', ls='--', alpha=0.5, label=f'chance ({n_full*0.05:.1f})')
ax.legend(fontsize=8)

# Panel 3: Reliable beta count vs fit quality
ax = axes[2]
ax.scatter(n_reliable_per_session, full_r_arr, alpha=0.6, s=30, c='steelblue')
if np.std(n_reliable_per_session) > 0:
    r, p = pearsonr(n_reliable_per_session, full_r_arr)
    ax.set_title(f'r={r:.2f}, p={p:.3f}')
    z = np.polyfit(n_reliable_per_session, full_r_arr, 1)
    xline = np.linspace(0, np.max(n_reliable_per_session), 50)
    ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5)
ax.set_xlabel('# reliable betas')
ax.set_ylabel('Full model test r')
ax.set_title('More reliable betas = better fit?')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bootstrap_reliability.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Are reliable betas more consistent across sessions? (uses MODE)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Beta consistency — {MODE}', fontsize=12, y=1.02)

# Panel 1: sign consistency — when a beta is reliable, how often is it positive?
frac_positive_when_reliable = np.zeros(n_full)
for fi in range(n_full):
    rel_mask = reliable_stack[:, fi]
    if np.sum(rel_mask) > 0:
        frac_positive_when_reliable[fi] = np.mean(beta_stack[rel_mask, fi] > 0)
    else:
        frac_positive_when_reliable[fi] = np.nan

ax = axes[0]
colors_bar = ['coral' if f > 0.5 else 'steelblue' for f in frac_positive_when_reliable]
ax.barh(range(n_full), frac_positive_when_reliable - 0.5, left=0.5,
        color=colors_bar, alpha=0.7)
ax.set_yticks(range(n_full))
ax.set_yticklabels(feature_names, fontsize=7)
ax.set_xlabel('Fraction positive (when reliable)')
ax.axvline(0.5, color='k', ls='-', alpha=0.3)
ax.set_title('Sign consistency of reliable betas')
ax.set_xlim(0, 1)
ax.invert_yaxis()

# Panel 2: Mean signed beta across all sessions (with bootstrap SEM)
mean_beta = np.mean(beta_stack, axis=0)
sem_beta = np.std(beta_stack, axis=0) / np.sqrt(n_sess)

ax = axes[1]
sort_idx2 = np.argsort(np.abs(mean_beta))[::-1]
colors_mean = ['coral' if m > 0 else 'steelblue' for m in mean_beta[sort_idx2]]
ax.barh(range(n_full), mean_beta[sort_idx2], xerr=sem_beta[sort_idx2],
        color=colors_mean, alpha=0.7, capsize=2)
ax.set_yticks(range(n_full))
ax.set_yticklabels([feature_names[i] for i in sort_idx2], fontsize=7)
ax.axvline(0, color='k', ls='-', alpha=0.3)
ax.set_xlabel('Mean beta +/- SEM')
ax.set_title('Grand mean betas across sessions')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bootstrap_consistency.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Reduced model comparison (uses MODE)
# ============================================================================

sess_list = sd['sessions']
full_rs = sd['full_r']
twof_rs = sd['twof_r']
best_rs = sd['best_r']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Reduced model comparison — {MODE}', fontsize=12, y=1.02)

# Panel 1: Full vs 2-factor vs best single feature
ax = axes[0, 0]
positions = [1, 2, 3]
bp = ax.boxplot([full_rs, twof_rs, best_rs], positions=positions, widths=0.5,
                patch_artist=True)
colors_box = ['steelblue', 'coral', 'mediumseagreen']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(['Full (20)', '2-factor (4)', 'Best single (1)'], fontsize=9)
ax.set_ylabel('Test r (5-fold CV)')
ax.set_title('Model complexity comparison')
ax.axhline(0, color='k', ls='--', alpha=0.3)

# Panel 2: Per-epoch model performance
ax = axes[0, 1]
epoch_r_matrix = np.array([[s['epoch_r'][ep] for ep in EPOCH_ORDER] for s in sess_list])
bp = ax.boxplot([epoch_r_matrix[:, i] for i in range(n_epochs)],
                labels=EPOCH_ORDER, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.set_ylabel('Test r (single epoch model)')
ax.set_title('Which epoch carries the signal?')
ax.axhline(0, color='k', ls='--', alpha=0.3)

# Panel 3: Per-behavior model performance
ax = axes[1, 0]
behav_r_matrix = np.array([[s['behav_r'][beh] for beh in BEHAVIOR_NAMES] for s in sess_list])
bp = ax.boxplot([behav_r_matrix[:, i] for i in range(n_behav)],
                labels=BEHAVIOR_NAMES, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('coral')
    patch.set_alpha(0.6)
ax.set_ylabel('Test r (single behavior model)')
ax.set_title('Which behavior carries the signal?')
ax.axhline(0, color='k', ls='--', alpha=0.3)

# Panel 4: Which single feature is "best" most often?
ax = axes[1, 1]
best_counts = np.zeros(n_full)
for s in sess_list:
    best_counts[s['best_single_idx']] += 1
sort_best = np.argsort(best_counts)[::-1]
top_n = 10
ax.barh(range(top_n), best_counts[sort_best[:top_n]], color='mediumseagreen', alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in sort_best[:top_n]], fontsize=8)
ax.set_xlabel('# sessions where this is best single predictor')
ax.set_title('Best single predictor frequency')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'reduced_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 9: Stability of 2-factor betas + single-feature ranking (uses MODE)
# ============================================================================

twof_idx = [ei * n_behav + BEHAVIOR_NAMES.index('2-factor') for ei in range(n_epochs)]
twof_betas = np.array([s['beta_full'][twof_idx] for s in sess_list])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Beta stability — {MODE}', fontsize=12, y=1.02)

# 2-factor betas across sessions
ax = axes[0]
for ei, ep in enumerate(EPOCH_ORDER):
    ax.scatter(np.full(n_sess, ei) + np.random.randn(n_sess)*0.05,
               twof_betas[:, ei], alpha=0.4, s=20)
ax.boxplot([twof_betas[:, i] for i in range(n_epochs)],
           positions=range(n_epochs), widths=0.3)
ax.set_xticks(range(n_epochs))
ax.set_xticklabels(EPOCH_ORDER, fontsize=9)
ax.set_ylabel('Beta (2-factor)')
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_title('2-factor betas: are they consistent?')

# Single-feature r distribution by feature
ax = axes[1]
single_r_stack = np.array([s['single_r'] for s in sess_list])
mean_single_r = np.nanmean(single_r_stack, axis=0)
sort_sr = np.argsort(mean_single_r)[::-1]
ax.barh(range(n_full), mean_single_r[sort_sr], color='steelblue', alpha=0.7)
ax.set_yticks(range(n_full))
ax.set_yticklabels([feature_names[i] for i in sort_sr], fontsize=7)
ax.set_xlabel('Mean single-feature test r')
ax.set_title('Which single features predict dW best?')
ax.axvline(0, color='k', ls='--', alpha=0.3)
ax.invert_yaxis()

# Full vs reduced: paired comparison
ax = axes[2]
ax.scatter(twof_rs, full_rs, alpha=0.6, s=30, c='steelblue')
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.set_xlabel('2-factor model r')
ax.set_ylabel('Full model r')
ax.set_title('Full vs 2-factor: how much do behaviors add?')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'reduced_stability.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 10: Cross-mode comparison — Hebbian vs pre_only vs post_only
# ============================================================================

mode_colors = {'dot_prod': 'steelblue', 'pre_only': 'coral', 'post_only': 'mediumseagreen'}
modes_present = [m for m in MODES_TO_RUN if len(results_by_mode[m]) > 0]

if len(modes_present) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hebbian vs Non-Hebbian mode comparison', fontsize=13, y=1.02)

    # Panel 1: Full model test r by mode
    ax = axes[0, 0]
    data_r = [stacked[m]['full_r'] for m in modes_present]
    bp = ax.boxplot(data_r, labels=modes_present, patch_artist=True, widths=0.5)
    for patch, m in zip(bp['boxes'], modes_present):
        patch.set_facecolor(mode_colors.get(m, 'gray'))
        patch.set_alpha(0.6)
    ax.set_ylabel('Full model test r (5-fold CV)')
    ax.set_title('Fit quality by pairwise mode')
    ax.axhline(0, color='k', ls='--', alpha=0.3)

    # Panel 2: 2-factor model test r by mode
    ax = axes[0, 1]
    data_2f = [stacked[m]['twof_r'] for m in modes_present]
    bp = ax.boxplot(data_2f, labels=modes_present, patch_artist=True, widths=0.5)
    for patch, m in zip(bp['boxes'], modes_present):
        patch.set_facecolor(mode_colors.get(m, 'gray'))
        patch.set_alpha(0.6)
    ax.set_ylabel('2-factor model test r')
    ax.set_title('2-factor fit by mode')
    ax.axhline(0, color='k', ls='--', alpha=0.3)

    # Panel 3: # reliable betas per session by mode
    ax = axes[1, 0]
    for mi_idx, m in enumerate(modes_present):
        n_rel = np.sum(stacked[m]['reliable_stack'], axis=1)
        ax.scatter(np.full(len(n_rel), mi_idx) + np.random.randn(len(n_rel))*0.05,
                   n_rel, alpha=0.4, s=20, color=mode_colors.get(m, 'gray'))
    bp = ax.boxplot([np.sum(stacked[m]['reliable_stack'], axis=1) for m in modes_present],
                    positions=range(len(modes_present)), widths=0.3)
    ax.set_xticks(range(len(modes_present)))
    ax.set_xticklabels(modes_present)
    ax.set_ylabel('# reliable betas per session')
    ax.set_title('Bootstrap reliability by mode')
    ax.axhline(n_full * 0.05, color='r', ls='--', alpha=0.5, label='chance')
    ax.legend(fontsize=8)

    # Panel 4: Beta sign consistency by mode
    # For each mode, compute fraction of betas where sign is consistent (>70% same sign when reliable)
    ax = axes[1, 1]
    for mi_idx, m in enumerate(modes_present):
        bs = stacked[m]['beta_stack']
        rs = stacked[m]['reliable_stack']
        sign_consistency = []
        for fi in range(n_full):
            rel_mask = rs[:, fi]
            if np.sum(rel_mask) >= 3:
                frac_pos = np.mean(bs[rel_mask, fi] > 0)
                sign_consistency.append(max(frac_pos, 1 - frac_pos))
        if len(sign_consistency) > 0:
            ax.bar(mi_idx, np.mean(sign_consistency), color=mode_colors.get(m, 'gray'),
                   alpha=0.6, width=0.5)
            ax.scatter(np.full(len(sign_consistency), mi_idx) + np.random.randn(len(sign_consistency))*0.03,
                       sign_consistency, alpha=0.3, s=15, color='k')
    ax.set_xticks(range(len(modes_present)))
    ax.set_xticklabels(modes_present)
    ax.set_ylabel('Sign consistency (max(frac+, frac-))')
    ax.set_title('Do betas have consistent sign across sessions?')
    ax.axhline(0.5, color='r', ls='--', alpha=0.3, label='chance')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'mode_comparison_hebb_vs_nonhebb.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
else:
    print(f"Only 1 mode present ({modes_present}), skipping cross-mode comparison plot.")

#%% ============================================================================
# CELL 11: Print summary for all modes
# ============================================================================

print("\n" + "="*70)
print("BOOTSTRAP + REDUCED MODEL SUMMARY — ALL MODES")
print("="*70)

for mode in modes_present:
    sd_m = stacked[mode]
    n_s = sd_m['n_sess']
    fr = sd_m['full_r']
    tr = sd_m['twof_r']
    br = sd_m['best_r']
    rs = sd_m['reliable_stack']
    bs = sd_m['beta_stack']
    n_rel_per = np.sum(rs, axis=1)
    improvement = fr - tr

    print(f"\n{'─'*50}")
    print(f"  MODE: {mode}  ({n_s} sessions)")
    print(f"{'─'*50}")
    print(f"  Full model (20):   median r = {np.median(fr):.4f},  mean = {np.mean(fr):.4f}")
    print(f"  2-factor (4):      median r = {np.median(tr):.4f},  mean = {np.mean(tr):.4f}")
    print(f"  Best single (1):   median r = {np.median(br):.4f},  mean = {np.mean(br):.4f}")
    print(f"  Full - 2factor:    median = {np.median(improvement):.4f},  "
          f"full wins {np.sum(improvement > 0)}/{n_s}")
    print(f"  Reliable betas/session: median = {np.median(n_rel_per):.0f} (chance = {n_full*0.05:.1f})")

    # Sign consistency
    n_consistent = 0
    for fi in range(n_full):
        rel_mask = rs[:, fi]
        if np.sum(rel_mask) >= 3:
            frac_pos = np.mean(bs[rel_mask, fi] > 0)
            if max(frac_pos, 1 - frac_pos) > 0.7:
                n_consistent += 1
    print(f"  Betas with consistent sign (>70%): {n_consistent}/{n_full}")

print(f"\n{'='*70}")

#%% ============================================================================
# CELL 12: Save comprehensive text report for review
# ============================================================================

report_path = os.path.join(RESULTS_DIR, 'mode_comparison_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("HEBBIAN vs NON-HEBBIAN MODE COMPARISON REPORT\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")

    for mode in modes_present:
        sd_m = stacked[mode]
        n_s = sd_m['n_sess']
        fr = sd_m['full_r']
        tr = sd_m['twof_r']
        br = sd_m['best_r']
        rs = sd_m['reliable_stack']
        bs = sd_m['beta_stack']
        sess = sd_m['sessions']
        n_rel_per = np.sum(rs, axis=1)
        improvement = fr - tr

        f.write(f"{'─'*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'─'*50}\n\n")

        # Fit quality
        f.write("FIT QUALITY (5-fold CV test r)\n")
        f.write(f"  Full model (20 params):  median={np.median(fr):.4f}  mean={np.mean(fr):.4f}  std={np.std(fr):.4f}\n")
        f.write(f"  2-factor (4 params):     median={np.median(tr):.4f}  mean={np.mean(tr):.4f}  std={np.std(tr):.4f}\n")
        f.write(f"  Best single (1 param):   median={np.median(br):.4f}  mean={np.mean(br):.4f}  std={np.std(br):.4f}\n")
        f.write(f"  Full - 2factor:          median={np.median(improvement):.4f}  full wins {np.sum(improvement > 0)}/{n_s}\n\n")

        # Per-epoch model r
        epoch_r_mat = np.array([[s['epoch_r'][ep] for ep in EPOCH_ORDER] for s in sess])
        f.write("PER-EPOCH MODEL (5 behaviors, 1 epoch at a time)\n")
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = epoch_r_mat[:, ei]
            f.write(f"  {ep:10s}: median={np.median(vals):.4f}  mean={np.mean(vals):.4f}\n")
        f.write("\n")

        # Per-behavior model r
        behav_r_mat = np.array([[s['behav_r'][beh] for beh in BEHAVIOR_NAMES] for s in sess])
        f.write("PER-BEHAVIOR MODEL (4 epochs, 1 behavior at a time)\n")
        for bi, beh in enumerate(BEHAVIOR_NAMES):
            vals = behav_r_mat[:, bi]
            f.write(f"  {beh:10s}: median={np.median(vals):.4f}  mean={np.mean(vals):.4f}\n")
        f.write("\n")

        # Bootstrap reliability
        f.write("BOOTSTRAP RELIABILITY\n")
        f.write(f"  Reliable betas per session: median={np.median(n_rel_per):.0f}  mean={np.mean(n_rel_per):.1f}  (chance={n_full*0.05:.1f})\n")
        frac_rel = np.mean(rs, axis=0)
        sort_rel = np.argsort(frac_rel)[::-1]
        f.write("  Top 5 most reliable features:\n")
        for i in sort_rel[:5]:
            f.write(f"    {feature_names[i]:20s}: {frac_rel[i]*100:.0f}% of sessions\n")
        f.write("  Bottom 5:\n")
        for i in sort_rel[-5:]:
            f.write(f"    {feature_names[i]:20s}: {frac_rel[i]*100:.0f}% of sessions\n")
        f.write("\n")

        # Sign consistency
        f.write("SIGN CONSISTENCY (when reliable, fraction positive)\n")
        for fi in range(n_full):
            rel_mask = rs[:, fi]
            n_rel_fi = np.sum(rel_mask)
            if n_rel_fi >= 3:
                frac_pos = np.mean(bs[rel_mask, fi] > 0)
                f.write(f"  {feature_names[fi]:20s}: {frac_pos*100:.0f}% positive  (reliable in {n_rel_fi} sessions)\n")
        f.write("\n")

        # Mean betas across sessions
        mean_b = np.mean(bs, axis=0)
        std_b = np.std(bs, axis=0)
        f.write("GRAND MEAN BETAS (across sessions)\n")
        for fi in range(n_full):
            f.write(f"  {feature_names[fi]:20s}: mean={mean_b[fi]:+.4f}  std={std_b[fi]:.4f}  |mean|/std={abs(mean_b[fi])/(std_b[fi]+1e-10):.2f}\n")
        f.write("\n")

        # Best single feature counts
        best_counts = np.zeros(n_full)
        for s in sess:
            best_counts[s['best_single_idx']] += 1
        sort_bc = np.argsort(best_counts)[::-1]
        f.write("BEST SINGLE PREDICTOR (how often each feature wins)\n")
        for i in sort_bc[:10]:
            if best_counts[i] > 0:
                f.write(f"  {feature_names[i]:20s}: {int(best_counts[i])} sessions\n")
        f.write("\n")

        # Per-session detail
        f.write("PER-SESSION DETAIL\n")
        f.write(f"  {'mouse':8s} {'session':20s} {'n_pairs':>7s} {'full_r':>7s} {'2f_r':>7s} {'n_rel':>5s} {'hit_rate':>8s}\n")
        for s in sess:
            f.write(f"  {s['mouse']:8s} {s['session']:20s} {s['n_pairs']:7d} {s['full_r']:7.3f} {s['twofactor_r']:7.3f} "
                    f"{int(np.sum(s['beta_reliable'])):5d} {s['hit_rate']:8.2f}\n")
        f.write("\n\n")

    # Cross-mode summary
    if len(modes_present) > 1:
        f.write("=" * 70 + "\n")
        f.write("CROSS-MODE COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<35s}")
        for m in modes_present:
            f.write(f"  {m:>12s}")
        f.write("\n" + "-" * (35 + 14 * len(modes_present)) + "\n")

        # Median full r
        f.write(f"{'Median full model r':<35s}")
        for m in modes_present:
            f.write(f"  {np.median(stacked[m]['full_r']):12.4f}")
        f.write("\n")

        # Median 2-factor r
        f.write(f"{'Median 2-factor r':<35s}")
        for m in modes_present:
            f.write(f"  {np.median(stacked[m]['twof_r']):12.4f}")
        f.write("\n")

        # Median reliable betas
        f.write(f"{'Median reliable betas/session':<35s}")
        for m in modes_present:
            f.write(f"  {np.median(np.sum(stacked[m]['reliable_stack'], axis=1)):12.0f}")
        f.write("\n")

        # Sign-consistent betas
        f.write(f"{'Betas with consistent sign (>70%)':<35s}")
        for m in modes_present:
            rs_m = stacked[m]['reliable_stack']
            bs_m = stacked[m]['beta_stack']
            nc = 0
            for fi in range(n_full):
                rel_mask = rs_m[:, fi]
                if np.sum(rel_mask) >= 3:
                    frac_pos = np.mean(bs_m[rel_mask, fi] > 0)
                    if max(frac_pos, 1 - frac_pos) > 0.7:
                        nc += 1
            f.write(f"  {nc:12d}")
        f.write(f"  (out of {n_full})\n")

        # Paired comparison
        if 'dot_prod' in modes_present:
            f.write("\nPAIRED COMPARISON vs dot_prod (full model r)\n")
            dp_r = stacked['dot_prod']['full_r']
            for m in modes_present:
                if m == 'dot_prod':
                    continue
                m_r = stacked[m]['full_r']
                n_compare = min(len(dp_r), len(m_r))
                wins = np.sum(m_r[:n_compare] > dp_r[:n_compare])
                diff = m_r[:n_compare] - dp_r[:n_compare]
                f.write(f"  {m} > dot_prod: {wins}/{n_compare} sessions, "
                        f"mean diff = {np.mean(diff):+.4f}\n")

print(f"Report saved to: {report_path}")
