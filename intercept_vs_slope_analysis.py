#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

def zscore_mat_std_only(X):
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1
    return X / sd

print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

# Use pre_only since it outperforms dot_prod; also run dot_prod for comparison
MODES = ['pre_only', 'dot_prod']

num_bins = 10
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
BEHAVIOR_NAMES = ['hit', 'hit_rpe', '10-rt', 'rpe', '2-factor']
n_epochs = len(EPOCH_ORDER)
n_behav = len(BEHAVIOR_NAMES)
n_full = n_epochs * n_behav

all_results = []

print("Intercept vs slope analysis")
print(f"Modes: {MODES}")

#%% ============================================================================
# CELL 3: Main loop — for each session, test intercept effects
# ============================================================================
#
# For each session & mode, we do three fits:
#   A) STANDARD: Y ~ X_full (no intercept column, current approach)
#   B) WITH INTERCEPTS: Y ~ [X_full, bin_dummies] (per-bin intercept absorbs mean dW per bin)
#   C) DEMEANED: subtract mean(Y) per bin, then fit Y_dm ~ X_full
#
# Also compute per-bin diagnostics:
#   - mean_dW per bin (the intercept)
#   - mean_CC per bin (mean presynaptic activity)
#   - slope of dW vs CC per bin (the "hebbian index")
#   - correlation between bin-level mean_dW and bin-level slope

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

            # Load data (same as bootstrap script)
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time',
                'reward_time', 'BCI_thresholds',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
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

            # ---- Helper: CV fit ----
            def cv_fit(X, Y, n_splits=5):
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

            # ---- Loop over modes ----
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
                    n_pairs = Y_T.shape[0]

                    # =========================================================
                    # Figure out which bin each pair belongs to
                    # (used_bins tells us which bins survived filtering)
                    # =========================================================
                    n_used = len(used_bins)
                    pairs_per_bin = n_pairs // n_used  # each bin contributes same # pairs
                    bin_labels = np.repeat(np.arange(n_used), pairs_per_bin)
                    if len(bin_labels) < n_pairs:
                        # handle rounding
                        bin_labels = np.concatenate([bin_labels,
                            np.full(n_pairs - len(bin_labels), n_used - 1)])
                    bin_labels = bin_labels[:n_pairs]

                    # =========================================================
                    # FIT A: Standard (no intercept)
                    # =========================================================
                    r_standard = cv_fit(X_full, Y_T)

                    mu_y, sig_y = Y_T.mean(), Y_T.std()
                    if sig_y == 0: sig_y = 1.0
                    Y_z = (Y_T - mu_y) / sig_y
                    beta_standard = np.linalg.pinv(X_full) @ Y_z

                    # =========================================================
                    # FIT B: With per-bin intercepts (dummy columns)
                    # =========================================================
                    bin_dummies = np.zeros((n_pairs, n_used))
                    for bi in range(n_used):
                        bin_dummies[bin_labels == bi, bi] = 1.0
                    X_with_intercept = np.hstack([X_full, bin_dummies])
                    r_with_intercept = cv_fit(X_with_intercept, Y_T)

                    beta_with_int = np.linalg.pinv(X_with_intercept) @ Y_z
                    beta_slopes_with_int = beta_with_int[:n_full]  # just the CC*behavior betas
                    beta_intercepts = beta_with_int[n_full:]       # the per-bin intercepts

                    # =========================================================
                    # FIT C: Demean Y per bin, then fit standard model
                    # =========================================================
                    Y_demeaned = Y_T.copy()
                    bin_means_dW = np.zeros(n_used)
                    for bi in range(n_used):
                        mask = bin_labels == bi
                        bin_means_dW[bi] = np.mean(Y_T[mask])
                        Y_demeaned[mask] = Y_T[mask] - bin_means_dW[bi]

                    r_demeaned = cv_fit(X_full, Y_demeaned)
                    mu_yd, sig_yd = Y_demeaned.mean(), Y_demeaned.std()
                    if sig_yd == 0: sig_yd = 1.0
                    Y_dm_z = (Y_demeaned - mu_yd) / sig_yd
                    beta_demeaned = np.linalg.pinv(X_full) @ Y_dm_z

                    # =========================================================
                    # Per-bin diagnostics
                    # =========================================================
                    # For the 2-factor (ones) columns, compute per-bin slope
                    # This is the "hebbian index" per bin
                    twof_idx = [ei * n_behav + BEHAVIOR_NAMES.index('2-factor')
                                for ei in range(n_epochs)]
                    X_2f = X_full[:, twof_idx]  # (n_pairs, 4) = epoch-specific CC*ones

                    bin_slopes = np.zeros((n_used, n_epochs))  # slope of dW vs CC per bin
                    bin_intercepts_fit = np.zeros((n_used, n_epochs))
                    bin_mean_CC = np.zeros((n_used, n_epochs))

                    for bi in range(n_used):
                        mask = bin_labels == bi
                        y_bin = Y_T[mask]
                        for ei in range(n_epochs):
                            x_bin = X_2f[mask, ei]
                            bin_mean_CC[bi, ei] = np.mean(x_bin)
                            if np.std(x_bin) > 0 and len(x_bin) > 5:
                                # fit y = a + b*x
                                A = np.column_stack([np.ones(len(x_bin)), x_bin])
                                coeffs = np.linalg.lstsq(A, y_bin, rcond=None)[0]
                                bin_intercepts_fit[bi, ei] = coeffs[0]
                                bin_slopes[bi, ei] = coeffs[1]

                    # Correlation between intercept and slope across bins
                    intercept_slope_corr = np.zeros(n_epochs)
                    for ei in range(n_epochs):
                        if np.std(bin_intercepts_fit[:, ei]) > 0 and np.std(bin_slopes[:, ei]) > 0:
                            intercept_slope_corr[ei], _ = pearsonr(
                                bin_intercepts_fit[:, ei], bin_slopes[:, ei])

                    # Correlation between bin-mean-dW and bin-mean-CC
                    mean_dW_vs_CC_corr = np.zeros(n_epochs)
                    for ei in range(n_epochs):
                        if np.std(bin_mean_CC[:, ei]) > 0 and np.std(bin_means_dW) > 0:
                            mean_dW_vs_CC_corr[ei], _ = pearsonr(
                                bin_means_dW, bin_mean_CC[:, ei])

                    # How much do betas change between standard and demeaned?
                    # Use 2-factor betas for simplicity
                    beta_2f_standard = beta_standard[twof_idx]
                    beta_2f_demeaned = beta_demeaned[twof_idx]
                    beta_2f_with_int = beta_slopes_with_int[twof_idx]

                    # Do any signs flip?
                    sign_flip_demeaned = np.sum(
                        np.sign(beta_2f_standard) != np.sign(beta_2f_demeaned))
                    sign_flip_with_int = np.sum(
                        np.sign(beta_2f_standard) != np.sign(beta_2f_with_int))

                    # Variance of per-bin intercepts vs variance of per-bin slopes
                    intercept_var = np.var(bin_means_dW)
                    slope_var_per_epoch = np.var(bin_slopes, axis=0)

                    # Per-bin behavioral variables (only kept bins)
                    kept = np.where(kept_bin_mask)[0]
                    bin_hit = hit_bin[kept]
                    bin_rpe = rpe_bin[kept]
                    bin_rt = rt_bin[kept]
                    bin_hit_rpe = hit_rpe_bin[kept]
                    bin_thr = thr_bin[kept] if 'thr_bin' in dir() else np.full(len(kept), np.nan)

                    result = {
                        'mouse': mouse,
                        'session': session,
                        'mode': pairwise_mode,
                        'n_pairs': n_pairs,
                        'n_bins': n_used,
                        'hit_rate': np.nanmean(hit),
                        # Fit quality
                        'r_standard': r_standard,
                        'r_with_intercept': r_with_intercept,
                        'r_demeaned': r_demeaned,
                        # 2-factor betas under 3 conditions
                        'beta_2f_standard': beta_2f_standard,
                        'beta_2f_demeaned': beta_2f_demeaned,
                        'beta_2f_with_int': beta_2f_with_int,
                        # Full betas
                        'beta_standard': beta_standard,
                        'beta_demeaned': beta_demeaned,
                        # Sign flips
                        'sign_flip_demeaned': sign_flip_demeaned,
                        'sign_flip_with_int': sign_flip_with_int,
                        # Per-bin diagnostics
                        'bin_means_dW': bin_means_dW,
                        'bin_mean_CC': bin_mean_CC,
                        'bin_slopes': bin_slopes,
                        'bin_intercepts': bin_intercepts_fit,
                        'beta_intercepts': beta_intercepts,
                        # Per-bin behavioral variables
                        'bin_hit': bin_hit,
                        'bin_rpe': bin_rpe,
                        'bin_rt': bin_rt,
                        'bin_hit_rpe': bin_hit_rpe,
                        # Correlations
                        'intercept_slope_corr': intercept_slope_corr,
                        'mean_dW_vs_CC_corr': mean_dW_vs_CC_corr,
                        # Variance decomposition
                        'intercept_var': intercept_var,
                        'slope_var_per_epoch': slope_var_per_epoch,
                    }
                    all_results.append(result)

                    print(f"  {pairwise_mode}: r_std={r_standard:.3f}  "
                          f"r_intercept={r_with_intercept:.3f}  "
                          f"r_demean={r_demeaned:.3f}  "
                          f"sign_flips={sign_flip_demeaned}/4  "
                          f"int-slope corr={np.mean(intercept_slope_corr):.2f}")

                except Exception as e:
                    print(f"  {pairwise_mode} FAILED: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_results)} session-mode results")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'intercept_analysis_results.npy'),
        all_results, allow_pickle=True)
print(f"Saved {len(all_results)} results.")

#%% ============================================================================
# CELL 5: Load + analyze
# ============================================================================
all_results = np.load(os.path.join(RESULTS_DIR, 'intercept_analysis_results.npy'),
                      allow_pickle=True).tolist()

# Split by mode
by_mode = {}
for r in all_results:
    m = r['mode']
    if m not in by_mode:
        by_mode[m] = []
    by_mode[m].append(r)

for m in by_mode:
    print(f"{m}: {len(by_mode[m])} sessions")

#%% ============================================================================
# CELL 6: Key question — does demeaning change the fit or the betas?
# ============================================================================

for mode in by_mode:
    sess = by_mode[mode]
    n_s = len(sess)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Intercept analysis -- {mode}', fontsize=13, y=1.02)

    # Panel 1: r_standard vs r_demeaned
    ax = axes[0, 0]
    r_std = np.array([s['r_standard'] for s in sess])
    r_dm = np.array([s['r_demeaned'] for s in sess])
    ax.scatter(r_std, r_dm, alpha=0.6, s=30)
    lims = [min(np.nanmin(r_std), np.nanmin(r_dm)) - 0.01,
            max(np.nanmax(r_std), np.nanmax(r_dm)) + 0.01]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('r (standard)')
    ax.set_ylabel('r (demeaned per bin)')
    ax.set_title('Does removing bin means hurt the fit?')

    # Panel 2: r_standard vs r_with_intercept
    ax = axes[0, 1]
    r_int = np.array([s['r_with_intercept'] for s in sess])
    ax.scatter(r_std, r_int, alpha=0.6, s=30)
    lims2 = [min(np.nanmin(r_std), np.nanmin(r_int)) - 0.01,
             max(np.nanmax(r_std), np.nanmax(r_int)) + 0.01]
    ax.plot(lims2, lims2, 'k--', alpha=0.3)
    ax.set_xlabel('r (standard)')
    ax.set_ylabel('r (with bin intercepts)')
    ax.set_title('Does adding bin intercepts help?')

    # Panel 3: 2-factor beta comparison (standard vs demeaned)
    ax = axes[0, 2]
    for si, s in enumerate(sess):
        for ei, ep in enumerate(EPOCH_ORDER):
            ax.scatter(s['beta_2f_standard'][ei], s['beta_2f_demeaned'][ei],
                       alpha=0.3, s=15, c=f'C{ei}',
                       label=ep if si == 0 else None)
    all_betas = np.concatenate([s['beta_2f_standard'] for s in sess])
    all_betas_dm = np.concatenate([s['beta_2f_demeaned'] for s in sess])
    lims3 = [min(np.nanmin(all_betas), np.nanmin(all_betas_dm)),
             max(np.nanmax(all_betas), np.nanmax(all_betas_dm))]
    ax.plot(lims3, lims3, 'k--', alpha=0.3)
    ax.set_xlabel('Beta (standard)')
    ax.set_ylabel('Beta (demeaned)')
    ax.set_title('Do 2-factor betas change?')
    ax.legend(fontsize=7)

    # Panel 4: Intercept-slope correlation per epoch
    ax = axes[1, 0]
    isc = np.array([s['intercept_slope_corr'] for s in sess])
    bp = ax.boxplot([isc[:, ei] for ei in range(n_epochs)],
                    labels=EPOCH_ORDER, patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Pearson r (intercept vs slope across bins)')
    ax.set_title('Are bin intercepts & slopes correlated?\n(strong = slopes absorb intercept)')

    # Panel 5: How many 2-factor betas flip sign after demeaning?
    ax = axes[1, 1]
    flips_dm = np.array([s['sign_flip_demeaned'] for s in sess])
    flips_int = np.array([s['sign_flip_with_int'] for s in sess])
    ax.hist(flips_dm, bins=np.arange(-0.5, 5.5, 1), alpha=0.6,
            label='demeaned', color='steelblue', edgecolor='white')
    ax.hist(flips_int, bins=np.arange(-0.5, 5.5, 1), alpha=0.6,
            label='with intercepts', color='coral', edgecolor='white')
    ax.set_xlabel('# 2-factor betas that flip sign (out of 4)')
    ax.set_ylabel('# sessions')
    ax.set_title('How many betas flip sign\nwhen intercept is handled?')
    ax.legend()

    # Panel 6: Intercept variance vs slope variance
    ax = axes[1, 2]
    int_var = np.array([s['intercept_var'] for s in sess])
    slope_var = np.array([np.mean(s['slope_var_per_epoch']) for s in sess])
    ax.scatter(int_var, slope_var, alpha=0.6, s=30)
    ax.set_xlabel('Variance of bin-mean dW (intercept var)')
    ax.set_ylabel('Mean variance of bin slopes')
    ax.set_title('Intercept vs slope variability')
    if np.std(int_var) > 0 and np.std(slope_var) > 0:
        r, p = pearsonr(int_var, slope_var)
        ax.set_title(f'Intercept vs slope variability\nr={r:.2f}, p={p:.3f}')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'intercept_analysis_{mode}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

#%% ============================================================================
# CELL 7: Save text report
# ============================================================================

report_path = os.path.join(RESULTS_DIR, 'intercept_analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("INTERCEPT VS SLOPE ANALYSIS REPORT\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write("QUESTION: Are the sign flips in betas caused by ignoring per-bin\n")
    f.write("intercepts (mean dW that varies across bins independent of CC)?\n\n")
    f.write("Three fits per session:\n")
    f.write("  A) Standard: Y ~ X (no intercept, current approach)\n")
    f.write("  B) With intercepts: Y ~ [X, bin_dummies]\n")
    f.write("  C) Demeaned: subtract mean(Y) per bin, then fit Y_dm ~ X\n\n")

    for mode in by_mode:
        sess = by_mode[mode]
        n_s = len(sess)

        r_std = np.array([s['r_standard'] for s in sess])
        r_dm = np.array([s['r_demeaned'] for s in sess])
        r_int = np.array([s['r_with_intercept'] for s in sess])
        flips_dm = np.array([s['sign_flip_demeaned'] for s in sess])
        flips_int = np.array([s['sign_flip_with_int'] for s in sess])
        isc = np.array([s['intercept_slope_corr'] for s in sess])

        f.write(f"{'='*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'='*50}\n\n")

        f.write("FIT QUALITY COMPARISON\n")
        f.write(f"  Standard:        median r = {np.nanmedian(r_std):.4f}  mean = {np.nanmean(r_std):.4f}\n")
        f.write(f"  With intercepts: median r = {np.nanmedian(r_int):.4f}  mean = {np.nanmean(r_int):.4f}\n")
        f.write(f"  Demeaned:        median r = {np.nanmedian(r_dm):.4f}  mean = {np.nanmean(r_dm):.4f}\n")
        diff_int = r_int - r_std
        diff_dm = r_dm - r_std
        f.write(f"  Intercept - Standard: mean diff = {np.nanmean(diff_int):+.4f}  "
                f"(intercept wins {np.sum(diff_int > 0)}/{n_s})\n")
        f.write(f"  Demeaned - Standard:  mean diff = {np.nanmean(diff_dm):+.4f}  "
                f"(demeaned wins {np.sum(diff_dm > 0)}/{n_s})\n\n")

        f.write("SIGN FLIPS IN 2-FACTOR BETAS (out of 4 epoch betas)\n")
        f.write(f"  After demeaning:       mean flips = {np.mean(flips_dm):.1f}  "
                f"0 flips in {np.sum(flips_dm == 0)}/{n_s} sessions  "
                f"3-4 flips in {np.sum(flips_dm >= 3)}/{n_s} sessions\n")
        f.write(f"  After adding intercepts: mean flips = {np.mean(flips_int):.1f}  "
                f"0 flips in {np.sum(flips_int == 0)}/{n_s} sessions  "
                f"3-4 flips in {np.sum(flips_int >= 3)}/{n_s} sessions\n\n")

        f.write("INTERCEPT-SLOPE CORRELATION (across bins within session)\n")
        f.write("  (Strong negative = slope absorbs intercept shifts = sign flips are artifact)\n")
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = isc[:, ei]
            f.write(f"  {ep:10s}: mean r = {np.nanmean(vals):+.3f}  "
                    f"median = {np.nanmedian(vals):+.3f}  "
                    f"frac < -0.5: {np.mean(vals < -0.5)*100:.0f}%\n")
        f.write(f"  Overall:    mean r = {np.nanmean(isc):+.3f}\n\n")

        # Per-session detail
        f.write("PER-SESSION DETAIL\n")
        f.write(f"  {'mouse':8s} {'session':12s} {'r_std':>6s} {'r_int':>6s} {'r_dm':>6s} "
                f"{'flips':>5s} {'isc_mean':>8s}\n")
        for s in sess:
            f.write(f"  {s['mouse']:8s} {s['session']:12s} "
                    f"{s['r_standard']:6.3f} {s['r_with_intercept']:6.3f} "
                    f"{s['r_demeaned']:6.3f} {s['sign_flip_demeaned']:5d} "
                    f"{np.mean(s['intercept_slope_corr']):+8.3f}\n")
        f.write("\n\n")

    f.write("INTERPRETATION GUIDE\n")
    f.write("-" * 50 + "\n")
    f.write("If r_demeaned << r_standard: the fit was relying on bin-mean dW,\n")
    f.write("  meaning the 'slope' was partly capturing intercept variation.\n")
    f.write("If r_demeaned ~ r_standard: the slope is real, not driven by intercept.\n")
    f.write("If many sign flips after demeaning: the sign of betas was determined\n")
    f.write("  by intercept, not by genuine activity-dW relationship.\n")
    f.write("If intercept-slope corr is strongly negative: slopes are absorbing\n")
    f.write("  intercept shifts, sign flips are likely artifactual.\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 8: Intercept & slope vs block-level behavioral variables
# ============================================================================
# For each session, we have per-bin:
#   - bin_means_dW (the global plasticity signal = intercept)
#   - bin_slopes (n_bins x 4 epochs, the activity-dependent component)
#   - bin_hit, bin_rpe, bin_rt, bin_hit_rpe (behavioral variables)
#
# Question: what behavioral variable predicts the intercept? the slope?
# And does the intercept-slope tradeoff depend on behavior?

mode = 'pre_only'
sess = by_mode[mode]

# Behavioral variable names and extraction
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']

def get_beh(s, bname):
    if bname == 'hit_rate': return s['bin_hit']
    if bname == 'RPE': return s['bin_rpe']
    if bname == 'RT': return s['bin_rt']
    if bname == 'hit_RPE': return s['bin_hit_rpe']

# Collect within-session correlations: behavior vs intercept, behavior vs slope
# For each session, correlate the bin-level behavioral variable with
# bin_means_dW (intercept) and bin_slopes per epoch (slope)

n_beh = len(beh_names)
corr_beh_intercept = np.full((len(sess), n_beh), np.nan)
corr_beh_slope = np.full((len(sess), n_beh, n_epochs), np.nan)

for si, s in enumerate(sess):
    dw = s['bin_means_dW']
    slopes = s['bin_slopes']  # (n_bins, 4)
    for bi, bname in enumerate(beh_names):
        bvar = get_beh(s, bname)
        if len(bvar) < 4 or np.std(bvar) == 0:
            continue
        # Intercept vs behavior
        if np.std(dw) > 0:
            corr_beh_intercept[si, bi], _ = spearmanr(bvar, dw)
        # Slope vs behavior per epoch
        for ei in range(n_epochs):
            if np.std(slopes[:, ei]) > 0:
                corr_beh_slope[si, bi, ei], _ = spearmanr(bvar, slopes[:, ei])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Block-level behavior vs intercept/slope -- {mode}', fontsize=13, y=1.02)

# Panel 1: Behavior vs intercept (mean dW)
ax = axes[0]
bp = ax.boxplot([corr_beh_intercept[:, bi] for bi in range(n_beh)],
                labels=beh_names, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_ylabel('Spearman r (within session, across bins)')
ax.set_title('What predicts the INTERCEPT\n(global plasticity signal)?')

# Panel 2: Behavior vs slope (averaged across epochs)
ax = axes[1]
mean_corr_slope = np.nanmean(corr_beh_slope, axis=2)  # average over epochs
bp = ax.boxplot([mean_corr_slope[:, bi] for bi in range(n_beh)],
                labels=beh_names, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('coral')
    patch.set_alpha(0.6)
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_ylabel('Spearman r (within session, across bins)')
ax.set_title('What predicts the SLOPE\n(activity-dependent component)?')

# Panel 3: Behavior vs slope broken out by epoch
ax = axes[2]
x_positions = []
x_labels = []
data_list = []
pos = 0
for bi, bname in enumerate(beh_names):
    for ei, ep in enumerate(EPOCH_ORDER):
        data_list.append(corr_beh_slope[:, bi, ei])
        x_positions.append(pos)
        x_labels.append(f'{bname}\n{ep}')
        pos += 1
    pos += 0.5  # gap between behaviors

bp = ax.boxplot(data_list, positions=x_positions, widths=0.6, patch_artist=True)
colors_ep = ['C0', 'C1', 'C2', 'C3'] * n_beh
for patch, c in zip(bp['boxes'], colors_ep):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=5, rotation=45, ha='right')
ax.set_ylabel('Spearman r')
ax.set_title('Slope vs behavior by epoch')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'intercept_slope_vs_behavior.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 9: Pooled analysis — stack all bins across sessions
# ============================================================================
# Pool all bins from all sessions to ask: across the entire dataset,
# what is the relationship between behavior, intercept, and slope?

all_intercepts = []
all_slopes = []
all_beh = {b: [] for b in beh_names}
all_session_ids = []

for si, s in enumerate(sess):
    nb = len(s['bin_means_dW'])
    all_intercepts.append(s['bin_means_dW'])
    all_slopes.append(s['bin_slopes'])  # (nb, 4)
    for bname in beh_names:
        all_beh[bname].append(get_beh(s, bname))
    all_session_ids.append(np.full(nb, si))

all_intercepts = np.concatenate(all_intercepts)
all_slopes = np.vstack(all_slopes)  # (total_bins, 4)
for bname in beh_names:
    all_beh[bname] = np.concatenate(all_beh[bname])
all_session_ids = np.concatenate(all_session_ids)

n_total_bins = len(all_intercepts)
print(f"Total bins pooled: {n_total_bins}")

fig, axes = plt.subplots(2, n_beh, figsize=(4*n_beh, 8))
fig.suptitle(f'Pooled bins: behavior vs intercept & slope -- {mode}', fontsize=13, y=1.02)

for bi, bname in enumerate(beh_names):
    bvar = all_beh[bname]
    valid = np.isfinite(bvar) & np.isfinite(all_intercepts)

    # Top row: behavior vs intercept
    ax = axes[0, bi]
    ax.scatter(bvar[valid], all_intercepts[valid], alpha=0.15, s=5, c='steelblue')
    if np.std(bvar[valid]) > 0 and np.std(all_intercepts[valid]) > 0:
        r, p = spearmanr(bvar[valid], all_intercepts[valid])
        ax.set_title(f'{bname} vs intercept\nrho={r:.3f}, p={p:.1e}')
        # add trend line
        z = np.polyfit(bvar[valid], all_intercepts[valid], 1)
        xline = np.linspace(np.nanmin(bvar[valid]), np.nanmax(bvar[valid]), 50)
        ax.plot(xline, np.polyval(z, xline), 'r-', linewidth=2)
    ax.set_xlabel(bname)
    if bi == 0:
        ax.set_ylabel('Intercept (mean dW)')

    # Bottom row: behavior vs mean slope
    ax = axes[1, bi]
    mean_slope = np.nanmean(all_slopes, axis=1)
    valid2 = np.isfinite(bvar) & np.isfinite(mean_slope)
    ax.scatter(bvar[valid2], mean_slope[valid2], alpha=0.15, s=5, c='coral')
    if np.std(bvar[valid2]) > 0 and np.std(mean_slope[valid2]) > 0:
        r, p = spearmanr(bvar[valid2], mean_slope[valid2])
        ax.set_title(f'{bname} vs slope\nrho={r:.3f}, p={p:.1e}')
        z = np.polyfit(bvar[valid2], mean_slope[valid2], 1)
        xline = np.linspace(np.nanmin(bvar[valid2]), np.nanmax(bvar[valid2]), 50)
        ax.plot(xline, np.polyval(z, xline), 'r-', linewidth=2)
    ax.set_xlabel(bname)
    if bi == 0:
        ax.set_ylabel('Slope (activity-dependent)')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pooled_behavior_vs_intercept_slope.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 10: The key plot — intercept vs slope colored by behavior
# ============================================================================

fig, axes = plt.subplots(1, n_epochs, figsize=(4*n_epochs, 4))
fig.suptitle(f'Intercept vs slope per epoch, colored by hit_RPE -- {mode}',
             fontsize=13, y=1.05)

for ei, ep in enumerate(EPOCH_ORDER):
    ax = axes[ei]
    valid = np.isfinite(all_intercepts) & np.isfinite(all_slopes[:, ei])
    sc = ax.scatter(all_intercepts[valid], all_slopes[valid, ei],
                    c=all_beh['hit_RPE'][valid], cmap='RdBu_r',
                    alpha=0.4, s=8, vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Intercept (mean dW)')
    ax.set_ylabel(f'Slope ({ep})')
    ax.set_title(ep)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    # Fit line
    if np.std(all_intercepts[valid]) > 0:
        z = np.polyfit(all_intercepts[valid], all_slopes[valid, ei], 1)
        xline = np.linspace(np.nanmin(all_intercepts[valid]),
                           np.nanmax(all_intercepts[valid]), 50)
        ax.plot(xline, np.polyval(z, xline), 'k-', linewidth=2)
        r, p = pearsonr(all_intercepts[valid], all_slopes[valid, ei])
        ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes,
                fontsize=9, va='top')

plt.colorbar(sc, ax=axes[-1], label='hit_RPE')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'intercept_vs_slope_by_epoch.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 11: Save text report for intercept/slope vs behavior
# ============================================================================

report_path2 = os.path.join(RESULTS_DIR, 'intercept_slope_behavior_report.txt')
with open(report_path2, 'w', encoding='utf-8') as f:
    f.write("INTERCEPT & SLOPE vs BLOCK-LEVEL BEHAVIOR REPORT\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Mode: {mode}\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL: deltaW = intercept(bin) + slope(bin) * presynaptic_activity\n")
    f.write("  intercept = global plasticity (mean dW across all pairs in bin)\n")
    f.write("  slope = activity-dependent plasticity (dW vs presynaptic activity)\n")
    f.write("  These are strongly anticorrelated (r ~ -0.95)\n\n")

    # Within-session correlations
    f.write("WITHIN-SESSION CORRELATIONS (Spearman, across bins)\n")
    f.write("-" * 50 + "\n\n")

    f.write("Behavior vs INTERCEPT:\n")
    f.write(f"  {'behavior':12s} {'mean_rho':>8s} {'median':>8s} {'%>0':>6s} {'%sig':>6s}\n")
    for bi, bname in enumerate(beh_names):
        vals = corr_beh_intercept[:, bi]
        valid = np.isfinite(vals)
        if np.sum(valid) > 0:
            f.write(f"  {bname:12s} {np.nanmean(vals):+8.3f} {np.nanmedian(vals):+8.3f} "
                    f"{np.nanmean(vals > 0)*100:5.0f}% "
                    f"{np.nanmean(np.abs(vals) > 0.5)*100:5.0f}%\n")
    f.write("\n")

    f.write("Behavior vs SLOPE (averaged across epochs):\n")
    f.write(f"  {'behavior':12s} {'mean_rho':>8s} {'median':>8s} {'%>0':>6s} {'%sig':>6s}\n")
    for bi, bname in enumerate(beh_names):
        vals = np.nanmean(corr_beh_slope[:, bi, :], axis=1)
        valid = np.isfinite(vals)
        if np.sum(valid) > 0:
            f.write(f"  {bname:12s} {np.nanmean(vals):+8.3f} {np.nanmedian(vals):+8.3f} "
                    f"{np.nanmean(vals > 0)*100:5.0f}% "
                    f"{np.nanmean(np.abs(vals) > 0.5)*100:5.0f}%\n")
    f.write("\n")

    f.write("Behavior vs SLOPE (by epoch):\n")
    f.write(f"  {'beh x epoch':20s} {'mean_rho':>8s} {'median':>8s}\n")
    for bi, bname in enumerate(beh_names):
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = corr_beh_slope[:, bi, ei]
            f.write(f"  {bname+'_'+ep:20s} {np.nanmean(vals):+8.3f} {np.nanmedian(vals):+8.3f}\n")
        f.write("\n")

    # Pooled correlations
    f.write("\nPOOLED CORRELATIONS (all bins across all sessions)\n")
    f.write("-" * 50 + "\n\n")

    _n_total = len(all_intercepts) if 'all_intercepts' in dir() else 0
    f.write(f"Total bins: {_n_total}\n\n")

    f.write("Behavior vs INTERCEPT (pooled):\n")
    for bname in beh_names:
        bvar = all_beh[bname]
        valid = np.isfinite(bvar) & np.isfinite(all_intercepts)
        if np.sum(valid) > 3:
            r, p = spearmanr(bvar[valid], all_intercepts[valid])
            f.write(f"  {bname:12s}: rho={r:+.3f}  p={p:.2e}\n")
    f.write("\n")

    f.write("Behavior vs SLOPE (pooled, mean across epochs):\n")
    mean_slope = np.nanmean(all_slopes, axis=1)
    for bname in beh_names:
        bvar = all_beh[bname]
        valid = np.isfinite(bvar) & np.isfinite(mean_slope)
        if np.sum(valid) > 3:
            r, p = spearmanr(bvar[valid], mean_slope[valid])
            f.write(f"  {bname:12s}: rho={r:+.3f}  p={p:.2e}\n")
    f.write("\n")

    f.write("Behavior vs SLOPE (pooled, by epoch):\n")
    for ei, ep in enumerate(EPOCH_ORDER):
        f.write(f"  {ep}:\n")
        for bname in beh_names:
            bvar = all_beh[bname]
            valid = np.isfinite(bvar) & np.isfinite(all_slopes[:, ei])
            if np.sum(valid) > 3:
                r, p = spearmanr(bvar[valid], all_slopes[valid, ei])
                f.write(f"    {bname:12s}: rho={r:+.3f}  p={p:.2e}\n")
        f.write("\n")

    # Intercept vs slope
    f.write("\nINTERCEPT vs SLOPE (pooled, per epoch):\n")
    for ei, ep in enumerate(EPOCH_ORDER):
        valid = np.isfinite(all_intercepts) & np.isfinite(all_slopes[:, ei])
        r, p = pearsonr(all_intercepts[valid], all_slopes[valid, ei])
        rs, ps = spearmanr(all_intercepts[valid], all_slopes[valid, ei])
        f.write(f"  {ep:10s}: Pearson r={r:+.3f} (p={p:.2e})  "
                f"Spearman rho={rs:+.3f} (p={ps:.2e})\n")

print(f"Report saved to: {report_path2}")
