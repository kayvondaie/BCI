#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Plot outlier (putative inhibitory) neuron activity aligned to trial start and
reward delivery.  Outliers are identified by the same mean+2SD non-target
photostim amplitude criterion used in sliding_window_temporal_offset_v3_by_group.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import traceback

import session_counting
import data_dict_create_module_test as ddct
import bci_time_series as bts
from BCI_data_helpers import (
    compute_amp_from_photostim,
    parse_hdf5_array_string,
    get_reward_aligned_df_truncated,
    get_trial_aligned_df_padded,
)

# Environment toggle: 'local' or 'codeocean'
RUN_ENV = 'local'

if RUN_ENV == 'codeocean':
    DATA_ROOT = r'/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal'
    RESULTS_DIR = '/root/capsule/results'
else:
    DATA_ROOT = r'//allen/aind/scratch/BCI/2p-raw'
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

session_counting._BASE_DIR_OVERRIDE = DATA_ROOT
list_of_dirs = session_counting.counter()

os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ["BCI88", "BCI93", "BCI107"]
n_sd_threshold = 2          # outlier = mean + n_sd * SD of non-target amp
REWARD_WINDOW = (-4, 10)    # seconds around reward
TRIAL_WINDOW  = (-2, 4)     # seconds around trial start
MEDFILT_KERNEL = 11         # median-filter kernel for smoothing traces

#%% ============================================================================
# CELL 3: Loop over sessions and collect aligned traces
# ============================================================================
all_rta_outlier = []   # reward-aligned, shape per entry: (time, n_outlier, trials)
all_sta_outlier = []   # trial-start-aligned
all_rta_rest = []
all_sta_rest = []
all_t_reward = []      # actual time vectors from each session
all_t_trial = []
all_dt_si = []
all_df_closedloop = [] # full df_closedloop per session for pairwise correlations
all_is_outlier = []    # boolean mask per session
all_rt = []            # per-trial reaction time
all_hit = []           # per-trial hit boolean
all_F_outlier = []     # F tensor for outlier cells: (frames, n_outlier, trials)
all_reward_vector = [] # for reward-aligned extraction in lasso analysis
session_labels = []

for mi, mouse in enumerate(mice):
    session_inds = np.where(
        (list_of_dirs['Mouse'] == mouse) &
        (list_of_dirs['Has data_main.npy'] == True)
    )[0]

    for sii in range(len(session_inds)):
        try:
            mouse = list_of_dirs['Mouse'][session_inds[sii]]
            session = list_of_dirs['Session'][session_inds[sii]]
            folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
            print(f"\n--- {mouse} {session} ({sii+1}/{len(session_inds)}) ---")

            # Load data
            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time', 'reward_time',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]

            # ---- Classify outlier neurons ----
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            amp_ep0 = AMP[0]
            amp_masked = amp_ep0.copy()
            amp_masked[stimDist < 30] = np.nan
            mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

            mu = np.nanmean(mean_amp_nontarg)
            sd = np.nanstd(mean_amp_nontarg)
            outlier_threshold = mu + n_sd_threshold * sd
            is_outlier = mean_amp_nontarg > outlier_threshold
            is_rest = ~is_outlier & np.isfinite(mean_amp_nontarg)
            print(f"  Outlier: {np.sum(is_outlier)}, Rest: {np.sum(is_rest)} "
                  f"(threshold={outlier_threshold:.4f})")

            if np.sum(is_outlier) == 0:
                print("  No outliers -- skipping.")
                continue

            # ---- Build event vectors ----
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0

            step_vector, reward_vector, trial_start_vector = \
                bts.bci_time_series_fun(folder, data, rt_filled, dt_si)

            # ---- Align activity ----
            df = data['df_closedloop']
            rta, t_reward = get_reward_aligned_df_truncated(
                df, reward_vector, trial_start_vector, dt_si, window=REWARD_WINDOW)
            sta, t_trial = get_trial_aligned_df_padded(
                df, trial_start_vector, reward_vector, dt_si, window=TRIAL_WINDOW)

            # Store per-session
            all_rta_outlier.append(rta[:, is_outlier, :])
            all_sta_outlier.append(sta[:, is_outlier, :])
            all_rta_rest.append(rta[:, is_rest, :])
            all_sta_rest.append(sta[:, is_rest, :])
            all_t_reward.append(t_reward)
            all_t_trial.append(t_trial)
            all_dt_si.append(dt_si)
            all_df_closedloop.append(df)
            all_is_outlier.append(is_outlier)
            all_rt.append(rt)          # NaN for miss trials
            all_hit.append(np.isfinite(rt))
            all_F_outlier.append(F[:, is_outlier, :])
            all_reward_vector.append(reward_vector)
            session_labels.append(f"{mouse} {session}")

        except Exception:
            traceback.print_exc()
            continue

print(f"\nCollected {len(session_labels)} sessions with outliers.")

#%% ============================================================================
# CELL 4: Plot per-session outlier traces (trial-start + reward aligned)
# ============================================================================
for si in range(len(session_labels)):
    rta_o = all_rta_outlier[si]   # (time, n_outlier, trials)
    sta_o = all_sta_outlier[si]
    t_trial = all_t_trial[si]
    t_reward = all_t_reward[si]
    n_outlier = rta_o.shape[1]

    # Trial-averaged traces (no per-cell baseline subtraction; data is already dff)
    rr = np.nanmean(rta_o, axis=2)  # (time, n_outlier)
    ss = np.nanmean(sta_o, axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    ax = axes[0]
    for i in range(n_outlier):
        offset = i * 0.01
        ax.plot(t_trial, medfilt(ss[:, i], MEDFILT_KERNEL) + offset, lw=0.5)
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_title(f"{session_labels[si]}  ({n_outlier} outlier cells)")

    ax = axes[1]
    for i in range(n_outlier):
        offset = i * 0.01
        ax.plot(t_reward, medfilt(rr[:, i], MEDFILT_KERNEL) + offset, lw=0.5)
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time from reward (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'outlier_traces_{session_labels[si].replace(" ", "_")}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

#%% ============================================================================
# CELL 5: Population average -- outlier vs rest
# ============================================================================
def _pool_mean(traces_list, time_vectors):
    """Average across neurons within each session, then across sessions.
    Uses the actual time vector from the first session for plotting.
    No per-cell baseline subtraction -- data is already dff."""
    session_means = []
    for si, arr in enumerate(traces_list):
        # arr shape: (time, neurons, trials)
        m = np.nanmean(np.nanmean(arr, axis=2), axis=1)  # (time,)
        session_means.append(m)
    # Truncate to shortest
    min_len = min(len(m) for m in session_means)
    stacked = np.column_stack([m[:min_len] for m in session_means])
    t_out = time_vectors[0][:min_len]
    return np.nanmean(stacked, axis=1), np.nanstd(stacked, axis=1) / np.sqrt(stacked.shape[1]), t_out


if len(all_rta_outlier) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Trial-start aligned ---
    ax = axes[0]
    for traces_list, label, color in [
        (all_sta_outlier, 'Outlier', 'red'),
        (all_sta_rest, 'Rest', 'black'),
    ]:
        mu, se, t = _pool_mean(traces_list, all_t_trial)
        ax.plot(t, medfilt(mu, MEDFILT_KERNEL), color=color, label=label)
        ax.fill_between(t, medfilt(mu - se, MEDFILT_KERNEL),
                        medfilt(mu + se, MEDFILT_KERNEL),
                        color=color, alpha=0.2)
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('DF/F')
    ax.set_title('Trial-start aligned')
    ax.legend()

    # --- Reward aligned ---
    ax = axes[1]
    for traces_list, label, color in [
        (all_rta_outlier, 'Outlier', 'red'),
        (all_rta_rest, 'Rest', 'black'),
    ]:
        mu, se, t = _pool_mean(traces_list, all_t_reward)
        ax.plot(t, medfilt(mu, MEDFILT_KERNEL), color=color, label=label)
        ax.fill_between(t, medfilt(mu - se, MEDFILT_KERNEL),
                        medfilt(mu + se, MEDFILT_KERNEL),
                        color=color, alpha=0.2)
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('DF/F')
    ax.set_title('Reward aligned')
    ax.legend()

    plt.suptitle(f'Outlier vs Rest neurons across {len(session_labels)} sessions',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'outlier_event_aligned_population.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved population figure.")

#%% ============================================================================
# CELL 6: Heatmap of all outlier cells across all sessions
# ============================================================================
if len(all_rta_outlier) > 0:
    # Collect trial-averaged traces for every outlier cell
    # Trial-start aligned
    sta_cells = []
    for si in range(len(all_sta_outlier)):
        arr = all_sta_outlier[si]  # (time, n_outlier, trials)
        m = np.nanmean(arr, axis=2)  # (time, n_outlier)
        t = all_t_trial[si]
        sta_cells.append((m, t))

    # Reward aligned
    rta_cells = []
    for si in range(len(all_rta_outlier)):
        arr = all_rta_outlier[si]
        m = np.nanmean(arr, axis=2)
        t = all_t_reward[si]
        rta_cells.append((m, t))

    # Interpolate all cells onto a common time axis
    from scipy.interpolate import interp1d

    t_trial_common = np.arange(TRIAL_WINDOW[0], TRIAL_WINDOW[1], 0.034)
    t_reward_common = np.arange(REWARD_WINDOW[0], REWARD_WINDOW[1], 0.034)

    sta_all = []  # each row = one outlier cell
    for m, t in sta_cells:
        for ci in range(m.shape[1]):
            trace = m[:, ci]
            valid = np.isfinite(trace)
            if np.sum(valid) > 10:
                f_interp = interp1d(t[valid], trace[valid], bounds_error=False, fill_value=np.nan)
                sta_all.append(f_interp(t_trial_common))

    rta_all = []
    for m, t in rta_cells:
        for ci in range(m.shape[1]):
            trace = m[:, ci]
            valid = np.isfinite(trace)
            if np.sum(valid) > 10:
                f_interp = interp1d(t[valid], trace[valid], bounds_error=False, fill_value=np.nan)
                rta_all.append(f_interp(t_reward_common))

    sta_mat = np.array(sta_all)  # (n_total_outlier_cells, time)
    rta_mat = np.array(rta_all)

    # Sort by peak time in reward-aligned trace
    peak_idx = np.nanargmax(rta_mat, axis=1)
    sort_order = np.argsort(peak_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    vmax = np.nanpercentile(np.abs(rta_mat), 95)

    ax = axes[0]
    im = ax.imshow(sta_mat[sort_order, :], aspect='auto', cmap='bwr',
                   vmin=-vmax, vmax=vmax, interpolation='none',
                   extent=[t_trial_common[0], t_trial_common[-1], sta_mat.shape[0], 0])
    ax.axvline(0, color='k', lw=0.8, alpha=0.7)
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Outlier cell #')
    ax.set_title('Trial-start aligned')
    plt.colorbar(im, ax=ax, label='DF/F')

    ax = axes[1]
    im = ax.imshow(rta_mat[sort_order, :], aspect='auto', cmap='bwr',
                   vmin=-vmax, vmax=vmax, interpolation='none',
                   extent=[t_reward_common[0], t_reward_common[-1], rta_mat.shape[0], 0])
    ax.axvline(0, color='k', lw=0.8, alpha=0.7)
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('Outlier cell #')
    ax.set_title('Reward aligned')
    plt.colorbar(im, ax=ax, label='DF/F')

    plt.suptitle(f'All outlier cells (n={sta_mat.shape[0]}) across {len(session_labels)} sessions',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'outlier_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Heatmap: {sta_mat.shape[0]} outlier cells, sorted by reward-aligned peak time.")

#%% ============================================================================
# CELL 7: Pairwise correlations -- outlier-outlier, outlier-other, other-other
# ============================================================================
corr_oo = []   # outlier-outlier
corr_or = []   # outlier-other
corr_rr = []   # other-other
corr_session_labels = []

for si in range(len(all_df_closedloop)):
    df_cl = all_df_closedloop[si]   # (nCells, nFrames)
    is_out = all_is_outlier[si]
    n_cells = df_cl.shape[0]

    # Only use cells with valid classification
    valid = np.isfinite(np.nanmean(df_cl, axis=1))
    if np.sum(is_out & valid) < 2:
        continue

    # Compute full pairwise correlation matrix on closed-loop activity
    # Use only finite values -- nanmean-fill for corrcoef
    df_clean = df_cl.copy()
    row_means = np.nanmean(df_clean, axis=1, keepdims=True)
    df_clean[np.isnan(df_clean)] = 0  # zero-fill NaNs for correlation
    C = np.corrcoef(df_clean)  # (nCells, nCells)

    outlier_idx = np.where(is_out)[0]
    rest_idx = np.where(~is_out & valid)[0]

    # Outlier-Outlier pairs
    oo_vals = []
    for i in range(len(outlier_idx)):
        for j in range(i + 1, len(outlier_idx)):
            val = C[outlier_idx[i], outlier_idx[j]]
            if np.isfinite(val):
                oo_vals.append(val)

    # Outlier-Other pairs
    or_vals = []
    for i in outlier_idx:
        for j in rest_idx:
            val = C[i, j]
            if np.isfinite(val):
                or_vals.append(val)

    # Other-Other pairs (subsample if too many)
    rr_idx = rest_idx
    if len(rr_idx) > 100:
        rng = np.random.default_rng(42)
        rr_idx = rng.choice(rest_idx, 100, replace=False)
    rr_vals = []
    for i in range(len(rr_idx)):
        for j in range(i + 1, len(rr_idx)):
            val = C[rr_idx[i], rr_idx[j]]
            if np.isfinite(val):
                rr_vals.append(val)

    corr_oo.append(np.mean(oo_vals) if oo_vals else np.nan)
    corr_or.append(np.mean(or_vals) if or_vals else np.nan)
    corr_rr.append(np.mean(rr_vals) if rr_vals else np.nan)
    corr_session_labels.append(session_labels[si])

corr_oo = np.array(corr_oo)
corr_or = np.array(corr_or)
corr_rr = np.array(corr_rr)

# --- Bar plot: mean +/- SEM across sessions ---
fig, ax = plt.subplots(figsize=(5, 4))
means = [np.nanmean(corr_oo), np.nanmean(corr_or), np.nanmean(corr_rr)]
sems = [np.nanstd(corr_oo) / np.sqrt(np.sum(np.isfinite(corr_oo))),
        np.nanstd(corr_or) / np.sqrt(np.sum(np.isfinite(corr_or))),
        np.nanstd(corr_rr) / np.sqrt(np.sum(np.isfinite(corr_rr)))]
labels = ['Outlier-Outlier', 'Outlier-Other', 'Other-Other']
colors = ['red', 'gray', 'black']

bars = ax.bar(labels, means, yerr=sems, color=colors, alpha=0.7,
              capsize=5, edgecolor='k', linewidth=0.8)
# Overlay individual session points
for xi, (vals, col) in enumerate(zip([corr_oo, corr_or, corr_rr], colors)):
    ax.scatter(np.full(len(vals), xi), vals, color=col, edgecolor='w',
               s=30, zorder=3, alpha=0.7)
ax.set_ylabel('Mean pairwise correlation')
ax.set_title(f'Pairwise correlations (n={len(corr_session_labels)} sessions)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_pairwise_correlations.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Outlier-Outlier: {np.nanmean(corr_oo):.4f} +/- {sems[0]:.4f}")
print(f"Outlier-Other:   {np.nanmean(corr_or):.4f} +/- {sems[1]:.4f}")
print(f"Other-Other:     {np.nanmean(corr_rr):.4f} +/- {sems[2]:.4f}")

#%% ============================================================================
# CELL 8: Reward modulation vs RT and RPE
# ============================================================================
from scipy.stats import pearsonr
from BCI_data_helpers import compute_rpe

all_corr_rt = []    # one r-value per outlier cell
all_corr_rpe = []

# Also collect pooled scatter data for plotting
pooled_rm = []      # reward modulation per trial (all cells, all sessions)
pooled_rt = []
pooled_rpe = []

for si in range(len(all_rta_outlier)):
    rta_o = all_rta_outlier[si]   # (time, n_outlier, n_rewards)
    t_reward = all_t_reward[si]
    rt = all_rt[si]
    hit = all_hit[si]
    n_outlier = rta_o.shape[1]
    n_rewards = rta_o.shape[2]

    # Both windows come from the reward-aligned data (same trial dimension)
    # Reward window: 1s centered on reward
    rew_mask = (t_reward >= -0.5) & (t_reward <= 0.5)
    # Baseline window: early pre-reward period (-4 to -3 s)
    base_mask = (t_reward >= -4) & (t_reward <= -3)

    if np.sum(rew_mask) == 0 or np.sum(base_mask) == 0:
        continue

    # RT and RPE for hit trials only; rta only contains rewarded trials
    # so we need to match: reward-aligned has n_rewards entries = number of hits
    hit_idx = np.where(hit)[0]
    if len(hit_idx) != n_rewards:
        # fallback: use first n_rewards hit trials
        hit_idx = hit_idx[:n_rewards]
    rt_hit = rt[hit_idx]

    # Compute RPE on all trials, then select hit trials
    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 30.0
    rpe_all = -compute_rpe(rt_filled, baseline=3.0, tau=10, fill_value=30.0)
    rpe_hit = rpe_all[hit_idx]

    for ci in range(n_outlier):
        # Per-trial reward modulation (both from reward-aligned data)
        rew_act = np.nanmean(rta_o[rew_mask, ci, :], axis=0)    # (n_rewards,)
        base_act = np.nanmean(rta_o[base_mask, ci, :], axis=0)  # (n_rewards,)
        rm = rew_act - base_act

        # Only use trials with valid data
        valid = np.isfinite(rm) & np.isfinite(rt_hit) & np.isfinite(rpe_hit)
        if np.sum(valid) < 10:
            continue

        r_rt, _ = pearsonr(rm[valid], rt_hit[valid])
        r_rpe, _ = pearsonr(rm[valid], rpe_hit[valid])
        all_corr_rt.append(r_rt)
        all_corr_rpe.append(r_rpe)

        # Pooled scatter data
        pooled_rm.extend(rm[valid])
        pooled_rt.extend(rt_hit[valid])
        pooled_rpe.extend(rpe_hit[valid])

all_corr_rt = np.array(all_corr_rt)
all_corr_rpe = np.array(all_corr_rpe)
pooled_rm = np.array(pooled_rm)
pooled_rt = np.array(pooled_rt)
pooled_rpe = np.array(pooled_rpe)

print(f"Total outlier cells with enough trials: {len(all_corr_rt)}")

# --- Figure: scatter plots + histograms ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: binned RT vs reward modulation
ax = axes[0, 0]
n_bins = 10
valid = np.isfinite(pooled_rt) & np.isfinite(pooled_rm)
rt_bins = np.percentile(pooled_rt[valid], np.linspace(0, 100, n_bins + 1))
bin_centers, bin_means, bin_sems = [], [], []
for bi in range(n_bins):
    mask = (pooled_rt >= rt_bins[bi]) & (pooled_rt < rt_bins[bi + 1]) & valid
    if np.sum(mask) > 5:
        bin_centers.append(np.mean(pooled_rt[mask]))
        bin_means.append(np.mean(pooled_rm[mask]))
        bin_sems.append(np.std(pooled_rm[mask]) / np.sqrt(np.sum(mask)))
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-', color='k', capsize=3)
ax.set_xlabel('Reaction time (s)')
ax.set_ylabel('Reward modulation (DF/F)')
ax.set_title('RT vs reward modulation')

# Top-right: binned RPE vs reward modulation
ax = axes[0, 1]
valid = np.isfinite(pooled_rpe) & np.isfinite(pooled_rm)
rpe_bins = np.percentile(pooled_rpe[valid], np.linspace(0, 100, n_bins + 1))
bin_centers, bin_means, bin_sems = [], [], []
for bi in range(n_bins):
    mask = (pooled_rpe >= rpe_bins[bi]) & (pooled_rpe < rpe_bins[bi + 1]) & valid
    if np.sum(mask) > 5:
        bin_centers.append(np.mean(pooled_rpe[mask]))
        bin_means.append(np.mean(pooled_rm[mask]))
        bin_sems.append(np.std(pooled_rm[mask]) / np.sqrt(np.sum(mask)))
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-', color='k', capsize=3)
ax.set_xlabel('RPE (positive = faster than expected)')
ax.set_ylabel('Reward modulation (DF/F)')
ax.set_title('RPE vs reward modulation')

# Bottom-left: histogram of per-cell correlation with RT
ax = axes[1, 0]
ax.hist(all_corr_rt, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(all_corr_rt), color='red', lw=1.5,
           label=f'mean={np.nanmean(all_corr_rt):.3f}')
ax.set_xlabel('Correlation (reward mod. vs RT)')
ax.set_ylabel('# outlier cells')
ax.set_title('Per-cell corr with RT')
ax.legend()

# Bottom-right: histogram of per-cell correlation with RPE
ax = axes[1, 1]
ax.hist(all_corr_rpe, bins=20, color='coral', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(all_corr_rpe), color='red', lw=1.5,
           label=f'mean={np.nanmean(all_corr_rpe):.3f}')
ax.set_xlabel('Correlation (reward mod. vs RPE)')
ax.set_ylabel('# outlier cells')
ax.set_title('Per-cell corr with RPE')
ax.legend()

plt.suptitle(f'Outlier reward modulation vs behavior (n={len(all_corr_rt)} cells)',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_reward_mod_vs_behavior.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Corr with RT:  mean={np.nanmean(all_corr_rt):.4f}, "
      f"median={np.nanmedian(all_corr_rt):.4f}")
print(f"Corr with RPE: mean={np.nanmean(all_corr_rpe):.4f}, "
      f"median={np.nanmedian(all_corr_rpe):.4f}")

#%% ============================================================================
# CELL 9: Reward modulation vs RT and RPE -- session-averaged outlier activity
# ============================================================================
sess_corr_rt = []    # one r-value per session
sess_corr_rpe = []
sess_pooled_rm = []
sess_pooled_rt = []
sess_pooled_rpe = []

for si in range(len(all_rta_outlier)):
    rta_o = all_rta_outlier[si]   # (time, n_outlier, n_rewards)
    t_reward = all_t_reward[si]
    rt = all_rt[si]
    hit = all_hit[si]
    n_rewards = rta_o.shape[2]

    rew_mask = (t_reward >= -0.5) & (t_reward <= 0.5)
    base_mask = (t_reward >= -4) & (t_reward <= -3)

    if np.sum(rew_mask) == 0 or np.sum(base_mask) == 0:
        continue

    hit_idx = np.where(hit)[0]
    if len(hit_idx) != n_rewards:
        hit_idx = hit_idx[:n_rewards]
    rt_hit = rt[hit_idx]

    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 30.0
    rpe_all = -compute_rpe(rt_filled, baseline=3.0, tau=10, fill_value=30.0)
    rpe_hit = rpe_all[hit_idx]

    # Average across all outlier cells first, then compute per-trial reward mod
    # rta_o shape: (time, n_outlier, n_rewards) -> average over axis 1
    rta_mean = np.nanmean(rta_o, axis=1)  # (time, n_rewards)
    rew_act = np.nanmean(rta_mean[rew_mask, :], axis=0)   # (n_rewards,)
    base_act = np.nanmean(rta_mean[base_mask, :], axis=0)  # (n_rewards,)
    rm = rew_act - base_act

    valid = np.isfinite(rm) & np.isfinite(rt_hit) & np.isfinite(rpe_hit)
    if np.sum(valid) < 10:
        continue

    r_rt, _ = pearsonr(rm[valid], rt_hit[valid])
    r_rpe, _ = pearsonr(rm[valid], rpe_hit[valid])
    sess_corr_rt.append(r_rt)
    sess_corr_rpe.append(r_rpe)

    sess_pooled_rm.extend(rm[valid])
    sess_pooled_rt.extend(rt_hit[valid])
    sess_pooled_rpe.extend(rpe_hit[valid])

sess_corr_rt = np.array(sess_corr_rt)
sess_corr_rpe = np.array(sess_corr_rpe)
sess_pooled_rm = np.array(sess_pooled_rm)
sess_pooled_rt = np.array(sess_pooled_rt)
sess_pooled_rpe = np.array(sess_pooled_rpe)

print(f"Sessions with enough trials: {len(sess_corr_rt)}")

# --- Figure ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: binned RT vs reward modulation
ax = axes[0, 0]
n_bins = 10
valid = np.isfinite(sess_pooled_rt) & np.isfinite(sess_pooled_rm)
rt_bins = np.percentile(sess_pooled_rt[valid], np.linspace(0, 100, n_bins + 1))
bin_centers, bin_means, bin_sems = [], [], []
for bi in range(n_bins):
    mask = (sess_pooled_rt >= rt_bins[bi]) & (sess_pooled_rt < rt_bins[bi + 1]) & valid
    if np.sum(mask) > 5:
        bin_centers.append(np.mean(sess_pooled_rt[mask]))
        bin_means.append(np.mean(sess_pooled_rm[mask]))
        bin_sems.append(np.std(sess_pooled_rm[mask]) / np.sqrt(np.sum(mask)))
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-', color='k', capsize=3)
ax.set_xlabel('Reaction time (s)')
ax.set_ylabel('Reward modulation (DF/F)')
ax.set_title('RT vs mean outlier reward mod.')

# Top-right: binned RPE vs reward modulation
ax = axes[0, 1]
valid = np.isfinite(sess_pooled_rpe) & np.isfinite(sess_pooled_rm)
rpe_bins = np.percentile(sess_pooled_rpe[valid], np.linspace(0, 100, n_bins + 1))
bin_centers, bin_means, bin_sems = [], [], []
for bi in range(n_bins):
    mask = (sess_pooled_rpe >= rpe_bins[bi]) & (sess_pooled_rpe < rpe_bins[bi + 1]) & valid
    if np.sum(mask) > 5:
        bin_centers.append(np.mean(sess_pooled_rpe[mask]))
        bin_means.append(np.mean(sess_pooled_rm[mask]))
        bin_sems.append(np.std(sess_pooled_rm[mask]) / np.sqrt(np.sum(mask)))
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-', color='k', capsize=3)
ax.set_xlabel('RPE (positive = faster than expected)')
ax.set_ylabel('Reward modulation (DF/F)')
ax.set_title('RPE vs mean outlier reward mod.')

# Bottom-left: histogram of per-session correlation with RT
ax = axes[1, 0]
ax.hist(sess_corr_rt, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(sess_corr_rt), color='red', lw=1.5,
           label=f'mean={np.nanmean(sess_corr_rt):.3f}')
ax.set_xlabel('Correlation (mean outlier reward mod. vs RT)')
ax.set_ylabel('# sessions')
ax.set_title('Per-session corr with RT')
ax.legend()

# Bottom-right: histogram of per-session correlation with RPE
ax = axes[1, 1]
ax.hist(sess_corr_rpe, bins=15, color='coral', edgecolor='k', alpha=0.7)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.nanmean(sess_corr_rpe), color='red', lw=1.5,
           label=f'mean={np.nanmean(sess_corr_rpe):.3f}')
ax.set_xlabel('Correlation (mean outlier reward mod. vs RPE)')
ax.set_ylabel('# sessions')
ax.set_title('Per-session corr with RPE')
ax.legend()

plt.suptitle(f'Session-averaged outlier reward mod. vs behavior (n={len(sess_corr_rt)} sessions)',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_reward_mod_session_avg.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Corr with RT:  mean={np.nanmean(sess_corr_rt):.4f}, "
      f"median={np.nanmedian(sess_corr_rt):.4f}")
print(f"Corr with RPE: mean={np.nanmean(sess_corr_rpe):.4f}, "
      f"median={np.nanmedian(sess_corr_rpe):.4f}")

#%% ============================================================================
# CELL 10: Lasso encoding model -- outcome predictors of outlier epoch activity
# ============================================================================
"""
Adapted from axons_vs_outcomes_lasso.py.  For each session, compute the
population-average outlier activity in 4 trial epochs (pre, early, late, reward),
then fit a Lasso regression against behavioral predictors:
  RPE_hit, RPE_speed, |RPE_hit|, |RPE_speed|, Speed
"""
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq, inv
from scipy.stats import t as tdist

# ---------- helpers (from axons_vs_outcomes_lasso.py) ----------
method = "Lasso"
lasso_alphas = np.logspace(-3, 1, 20)

def zscore_1d(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 0 else 1)

def detrend_trials(Y, order=1, lam=1e-6):
    """Remove slow trial-by-trial drift from each row of Y."""
    Y = np.asarray(Y, float)
    trl = np.arange(Y.shape[1])
    trl = (trl - np.nanmean(trl)) / (np.nanstd(trl) if np.nanstd(trl) > 0 else 1.0)
    T = np.column_stack([trl**k for k in range(1, order+1)] + [np.ones_like(trl)])
    Yd = np.full_like(Y, np.nan, float)
    I = np.eye(T.shape[1])
    for r in range(Y.shape[0]):
        y = Y[r, :]
        valid = np.isfinite(y)
        if valid.sum() < (order + 1):
            mu = np.nanmean(y)
            Yd[r, :] = y - (mu if np.isfinite(mu) else 0.0)
            continue
        M = T[valid, :]; yr = y[valid]
        beta = np.linalg.solve(M.T @ M + lam*I, M.T @ yr)
        Yd[r, :] = y - (T @ beta)
    return Yd

def fit_std(X, y):
    """Lasso regression with standardized predictors and response."""
    y = np.ravel(y)
    m = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[m], y[m]
    if len(y) < 20:
        return None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_X.fit_transform(X[:, :-1])  # exclude intercept column
    ys = scaler_y.fit_transform(y[:, None]).ravel()

    lasso = LassoCV(alphas=lasso_alphas, cv=5, fit_intercept=False).fit(Xs, ys)
    bz = lasso.coef_
    yhat = scaler_y.inverse_transform(lasso.predict(Xs).reshape(-1, 1)).ravel()
    R2 = lasso.score(Xs, ys)

    return dict(R2=R2, y=y, yhat=yhat, bz=bz, p=[np.nan]*len(bz), mask=m)

def pstar(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''

# ---------- build per-epoch responses and predictors ----------
from BCI_data_helpers import compute_rpe

pre_list, early_list, late_list, rew_list = [], [], [], []
rpe_list, rt_list_lasso, hit_rpe_list, miss_rpe_list = [], [], [], []

rpe_window = 5

for si in range(len(all_F_outlier)):
    F_o = all_F_outlier[si]       # (frames, n_outlier, trials)
    rta_o = all_rta_outlier[si]   # (time_rew, n_outlier, n_rewards)
    t_reward = all_t_reward[si]
    rt = all_rt[si].copy()
    hit = all_hit[si]
    dt_si = all_dt_si[si]
    n_frames, n_out, n_trials = F_o.shape

    # Behavioural predictors (all trials)
    rt_lasso = rt.copy()
    rt_lasso[~np.isfinite(rt_lasso)] = 20.0

    hit_rpe  = compute_rpe((rt_lasso == 20).astype(float), baseline=1,
                           tau=rpe_window, fill_value=50)
    miss_rpe = compute_rpe((rt_lasso != 20).astype(float), baseline=1,
                           tau=rpe_window, fill_value=50)
    rpe = compute_rpe(rt_lasso, baseline=5, tau=rpe_window, fill_value=50)

    # --- Epoch definitions using time (F time axis: tsta) ---
    tsta = np.arange(0, 12, dt_si)
    tsta = tsta - tsta[int(2 / dt_si)]  # t=0 at photostim/go-cue
    tsta = tsta[:n_frames]  # clip to match F dimension

    # pre:   baseline before go cue (-2 to 0 s)
    pre_mask = (tsta >= -2) & (tsta < 0)
    # early: go-cue response (0 to 1.5 s)
    early_mask = (tsta >= 0) & (tsta < 1.5)
    # other: end of trial (last 1 s of F) -- fallback for miss trials
    other_mask = tsta >= (tsta[-1] - 1.0)

    pre_epoch   = np.nanmean(F_o[pre_mask, :, :], axis=0)    # (n_out, n_trials)
    early_epoch = np.nanmean(F_o[early_mask, :, :], axis=0)

    # late and reward come from reward-aligned data (only hit trials)
    rew_frames = (t_reward >= 0) & (t_reward <= 1.5)       # 0-1.5s post-reward
    late_frames = (t_reward >= -2) & (t_reward <= 0)        # -2 to 0s pre-reward

    later_epoch = np.nanmean(rta_o[late_frames, :, :], axis=0) if np.sum(late_frames) > 0 \
                  else np.zeros((n_out, rta_o.shape[2]))
    rewr_epoch  = np.nanmean(rta_o[rew_frames, :, :], axis=0) if np.sum(rew_frames) > 0 \
                  else np.zeros((n_out, rta_o.shape[2]))
    other_epoch = np.nanmean(F_o[other_mask, :, :], axis=0)

    # Build full-trial late/rew arrays (hit trials get reward-aligned, miss get end-of-trial)
    late_epoch = pre_epoch * 0
    rew_epoch  = pre_epoch * 0

    hit_idx = np.where(hit)[0]
    miss_idx = np.where(~hit)[0]

    # reward-aligned array has n_rewards entries; map to hit trial indices
    n_rewards = rta_o.shape[2]
    hit_for_rta = hit_idx[:n_rewards]

    for ri, ti in enumerate(hit_for_rta):
        if ti < n_trials:
            late_epoch[:, ti] = later_epoch[:, ri]
            rew_epoch[:, ti]  = rewr_epoch[:, ri]
    for ti in miss_idx:
        if ti < n_trials:
            late_epoch[:, ti] = other_epoch[:, ti]
            rew_epoch[:, ti]  = other_epoch[:, ti]

    # Append population-average (across outlier cells) after detrending
    pre_list.append(np.nanmean(detrend_trials(pre_epoch), axis=0))
    early_list.append(np.nanmean(detrend_trials(early_epoch), axis=0))
    late_list.append(np.nanmean(detrend_trials(late_epoch), axis=0))
    rew_list.append(np.nanmean(detrend_trials(rew_epoch), axis=0))

    rpe_list.append(zscore_1d(rpe))
    rt_list_lasso.append(zscore_1d(10 - rt_lasso))  # speed = 10 - RT
    hit_rpe_list.append(zscore_1d(hit_rpe))
    miss_rpe_list.append(zscore_1d(miss_rpe))

# ---------- concatenate across sessions ----------
rpe_cat     = np.concatenate(rpe_list)
rt_cat      = np.concatenate(rt_list_lasso)
hit_rpe_cat = np.concatenate(hit_rpe_list)

rpe_abs     = np.abs(rpe_cat)
hit_rpe_abs = np.abs(hit_rpe_cat)

pre_cat   = np.concatenate(pre_list)
early_cat = np.concatenate(early_list)
late_cat  = np.concatenate(late_list)
rew_cat   = np.concatenate(rew_list)

Y = {"pre": pre_cat, "early": early_cat, "late": late_cat, "rew": rew_cat}

# ---------- design matrix ----------
X = np.c_[hit_rpe_cat, rpe_cat, hit_rpe_abs, rpe_abs, rt_cat, np.ones_like(rpe_cat)]
pred_labels = ["$RPE_{hit}$", "$RPE_{speed}$", "|$RPE_{hit}$|", "|$RPE_{speed}$|", "Speed"]

# ---------- fit ----------
n_preds = len(pred_labels)
coef_mat = np.zeros((len(Y), n_preds))
R2_list_lasso = []

for i, (ep_name, ep_y) in enumerate(Y.items()):
    res = fit_std(X, ep_y)
    if res is not None:
        coef_mat[i, :] = res["bz"]
        R2_list_lasso.append(res["R2"])
    else:
        R2_list_lasso.append(np.nan)

# ---------- plot: split signed vs unsigned ----------
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.titlesize": 10
})

signed_idx   = [0, 1, 4]   # RPE_hit, RPE_speed, Speed
unsigned_idx = [2, 3]      # |RPE_hit|, |RPE_speed|

coef_signed   = coef_mat[:, signed_idx]
coef_unsigned = coef_mat[:, unsigned_idx]
preds_signed   = [pred_labels[i] for i in signed_idx]
preds_unsigned = [pred_labels[i] for i in unsigned_idx]

v = np.nanmax(np.abs(coef_mat)) if np.nanmax(np.abs(coef_mat)) > 0 else 0.1
n_epochs = len(Y)

fig = plt.figure(figsize=(5, 2.5))
gs = fig.add_gridspec(
    1, 2, width_ratios=[len(preds_signed), len(preds_unsigned)],
    left=0.25, right=0.85, top=0.85, bottom=0.2, wspace=0.05
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

im1 = ax1.imshow(coef_signed, cmap='bwr', vmin=-v, vmax=v,
                 aspect='auto', origin='upper')
ax1.set_xticks(range(len(preds_signed)))
ax1.set_xticklabels(preds_signed, rotation=30)
ax1.set_yticks(np.arange(n_epochs))
ax1.set_yticklabels(list(Y.keys()))
ax1.set_ylim(n_epochs - 0.5, -0.5)
ax1.set_title("Signed")

im2 = ax2.imshow(coef_unsigned, cmap='bwr', vmin=-v, vmax=v,
                 aspect='auto', origin='upper')
ax2.set_xticks(range(len(preds_unsigned)))
ax2.set_xticklabels(preds_unsigned, rotation=30)
ax2.set_yticks([])
ax2.set_ylim(n_epochs - 0.5, -0.5)
ax2.set_title("Unsigned")

cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
cbar.set_label('Standardized $\\beta$')

fig.suptitle(f'Outlier neuron encoding ({len(pre_list)} sessions, Lasso)', fontsize=10)
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_lasso_encoding.png'),
            dpi=150, bbox_inches='tight')
plt.show()

for i, ep in enumerate(Y.keys()):
    print(f"  {ep:6s}  R²={R2_list_lasso[i]:.3f}  "
          f"β={coef_mat[i, :]}")

#%% ============================================================================
# CELL 11: Reward-aligned traces split by fast vs slow trials
# ============================================================================
"""
Adapted from rpe_trial_start_traces3.py.  Median-split trials by RT,
plot population-average outlier activity aligned to reward for fast (red)
vs slow (blue) trials.
"""

def detrend_across_trials(data_2d, order=1, lam=1e-6):
    """Remove slow trial-by-trial drift from (time, trials) array."""
    T, N = data_2d.shape
    trial_means = np.nanmean(data_2d, axis=0)
    x = np.arange(N)
    X_dt = np.column_stack([x**k for k in range(1, order+1)] + [np.ones_like(x)])
    I = np.eye(X_dt.shape[1])
    mask = np.isfinite(trial_means)
    if mask.sum() < (order + 1):
        trend = np.full(N, np.nanmean(trial_means))
    else:
        beta = np.linalg.solve(X_dt[mask].T @ X_dt[mask] + lam * I,
                               X_dt[mask].T @ trial_means[mask])
        trend = X_dt @ beta
    return data_2d - trend[None, :]

RTA_fast_all = []
RTA_slow_all = []
t_rew_common = None

for si in range(len(all_rta_outlier)):
    rta_o = all_rta_outlier[si]   # (time, n_outlier, n_rewards)
    t_reward = all_t_reward[si]
    rt = all_rt[si]
    hit = all_hit[si]
    n_rewards = rta_o.shape[2]

    # Match hit trials to reward-aligned array
    hit_idx = np.where(hit)[0]
    if len(hit_idx) != n_rewards:
        hit_idx = hit_idx[:n_rewards]
    rt_hit = rt[hit_idx]

    if np.sum(np.isfinite(rt_hit)) < 10:
        continue

    # Population average across outlier cells -> (time, n_rewards)
    mean_rta = np.nanmean(rta_o, axis=1)
    mean_rta = detrend_across_trials(mean_rta, order=1)

    # Median split on RT
    med_rt = np.nanmedian(rt_hit)
    fast_inds = np.where(rt_hit < med_rt)[0]
    slow_inds = np.where(rt_hit > med_rt)[0]

    if len(fast_inds) < 5 or len(slow_inds) < 5:
        continue

    RTA_fast_all.append(mean_rta[:, fast_inds])
    RTA_slow_all.append(mean_rta[:, slow_inds])

    if t_rew_common is None:
        t_rew_common = t_reward

# --- Plot ---
if len(RTA_fast_all) > 0:
    fig, ax = plt.subplots(figsize=(4, 3))

    for traces_list, color, label in [
        (RTA_slow_all, 'b', 'Slow'),
        (RTA_fast_all, 'r', 'Fast'),
    ]:
        # Truncate to common length
        min_len = min(a.shape[0] for a in traces_list)
        a = np.concatenate([x[:min_len, :] for x in traces_list], axis=1)
        trace = np.nanmean(a, axis=1)
        sem = np.nanstd(a, axis=1) / np.sqrt(a.shape[1])
        t = t_rew_common[:min_len]

        ax.plot(t, trace, color=color, label=label)
        ax.fill_between(t, trace - sem, trace + sem, color=color, alpha=0.3)

    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim(REWARD_WINDOW)
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('Population avg DF/F')
    ax.set_title('Outlier neurons: Fast vs Slow trials')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'outlier_fast_slow_reward_aligned.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Fast/slow plot: {len(RTA_fast_all)} sessions, "
          f"{sum(a.shape[1] for a in RTA_fast_all)} fast trials, "
          f"{sum(a.shape[1] for a in RTA_slow_all)} slow trials")

# --- Trial-start (go-cue) aligned: fast vs slow ---
STA_fast_all = []
STA_slow_all = []
t_trial_common = None

for si in range(len(all_F_outlier)):
    F_o = all_F_outlier[si]   # (frames, n_outlier, trials)
    rt = all_rt[si]
    dt_si = all_dt_si[si]
    n_frames = F_o.shape[0]

    # Build tsta time vector (go-cue at t=0)
    tsta = np.arange(0, 12, dt_si)
    tsta = tsta - tsta[int(2 / dt_si)]
    tsta = tsta[:n_frames]

    # Only hit trials with valid RT
    valid_rt = np.isfinite(rt)
    if np.sum(valid_rt) < 10:
        continue

    # Population average across outlier cells -> (frames, trials)
    mean_F = np.nanmean(F_o, axis=1)
    mean_F = detrend_across_trials(mean_F, order=1)

    # Median split on RT (hit trials only)
    rt_valid = rt.copy()
    rt_valid[~valid_rt] = np.nan
    med_rt = np.nanmedian(rt_valid[valid_rt])
    fast_inds = np.where(valid_rt & (rt < med_rt))[0]
    slow_inds = np.where(valid_rt & (rt > med_rt))[0]

    if len(fast_inds) < 5 or len(slow_inds) < 5:
        continue

    STA_fast_all.append(mean_F[:, fast_inds])
    STA_slow_all.append(mean_F[:, slow_inds])

    if t_trial_common is None:
        t_trial_common = tsta

if len(STA_fast_all) > 0:
    fig, ax = plt.subplots(figsize=(4, 3))

    for traces_list, color, label in [
        (STA_slow_all, 'b', 'Slow'),
        (STA_fast_all, 'r', 'Fast'),
    ]:
        min_len = min(a.shape[0] for a in traces_list)
        a = np.concatenate([x[:min_len, :] for x in traces_list], axis=1)
        trace = np.nanmean(a, axis=1)
        sem = np.nanstd(a, axis=1) / np.sqrt(a.shape[1])
        t = t_trial_common[:min_len]

        ax.plot(t, trace, color=color, label=label)
        ax.fill_between(t, trace - sem, trace + sem, color=color, alpha=0.3)

    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim((-2, 4))
    ax.set_xlabel('Time from go cue (s)')
    ax.set_ylabel('Population avg DF/F')
    ax.set_title('Outlier neurons: Fast vs Slow trials (go-cue aligned)')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'outlier_fast_slow_gocue_aligned.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Go-cue aligned fast/slow: {len(STA_fast_all)} sessions")
