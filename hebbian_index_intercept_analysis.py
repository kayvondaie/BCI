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

# Use pre_only (best mode) and dot_prod for comparison
MODES = ['pre_only', 'dot_prod']

num_bins = 10
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

all_results = []
print("Hebbian index with intercept analysis")

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
# For each session and mode:
#   - Get per-pair deltaW (Y_T) and per-pair per-bin CC (X_ep_T)
#   - For each bin, compute the "hebbian index" three ways:
#     1) slope WITHOUT intercept: b = (CC' CC)^-1 CC' dW
#     2) slope WITH intercept: fit dW = a + b*CC, report b
#     3) correlation: corr(dW, CC) — invariant to centering
#   - Track how these evolve across bins
#   - Also store per-bin behavioral variables

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

            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)
            miss_rpe = compute_rpe((~hit).astype(float), baseline=0.0, tau=tau_elig, fill_value=1.0)

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

                    n_pairs = Y_T.shape[0]
                    n_used = len(used_bins)

                    # Map epoch names to X matrices
                    epoch_X_raw = {
                        'pre': Xpre_T,      # (n_pairs, n_bins)
                        'go_cue': Xts_T,
                        'late': Xstep_T,
                        'reward': Xrew_T,
                    }

                    # Per-bin hebbian index: 3 versions x 4 epochs
                    hi_no_intercept = np.full((n_used, n_epochs), np.nan)
                    hi_with_intercept = np.full((n_used, n_epochs), np.nan)
                    hi_correlation = np.full((n_used, n_epochs), np.nan)
                    hi_intercept_val = np.full((n_used, n_epochs), np.nan)
                    hi_mean_CC = np.full((n_used, n_epochs), np.nan)

                    for bi in range(n_used):
                        for ei, ep in enumerate(EPOCH_ORDER):
                            cc_bin = epoch_X_raw[ep][:, bi]  # CC for this pair in this bin
                            dw = Y_T

                            if np.std(cc_bin) == 0 or np.std(dw) == 0:
                                continue

                            # 1) Slope WITHOUT intercept: b = sum(cc*dw) / sum(cc*cc)
                            hi_no_intercept[bi, ei] = (
                                np.dot(cc_bin, dw) / np.dot(cc_bin, cc_bin))

                            # 2) Slope WITH intercept: fit dw = a + b*cc
                            A = np.column_stack([np.ones(n_pairs), cc_bin])
                            coeffs = np.linalg.lstsq(A, dw, rcond=None)[0]
                            hi_intercept_val[bi, ei] = coeffs[0]
                            hi_with_intercept[bi, ei] = coeffs[1]

                            # 3) Correlation
                            hi_correlation[bi, ei], _ = pearsonr(cc_bin, dw)

                            # Mean CC for this bin
                            hi_mean_CC[bi, ei] = np.mean(cc_bin)

                    # Per-bin behavioral variables
                    kept = np.where(np.isfinite(hit_bin))[0]
                    bin_hit = hit_bin[kept]
                    bin_rpe = rpe_bin[kept]
                    bin_rt = rt_bin[kept]
                    bin_hit_rpe = hit_rpe_bin[kept]

                    # Count sign flips across bins for each method
                    def count_sign_flips(arr):
                        """Count how many times sign changes across rows (bins)."""
                        signs = np.sign(arr)
                        flips = np.sum(signs[1:] != signs[:-1], axis=0)
                        return flips

                    flips_no_int = count_sign_flips(hi_no_intercept)
                    flips_with_int = count_sign_flips(hi_with_intercept)
                    flips_corr = count_sign_flips(hi_correlation)

                    # Intercept-slope correlation across bins
                    isc = np.full(n_epochs, np.nan)
                    for ei in range(n_epochs):
                        v1 = hi_intercept_val[:, ei]
                        v2 = hi_with_intercept[:, ei]
                        ok = np.isfinite(v1) & np.isfinite(v2)
                        if np.sum(ok) > 3 and np.std(v1[ok]) > 0 and np.std(v2[ok]) > 0:
                            isc[ei], _ = pearsonr(v1[ok], v2[ok])

                    result = {
                        'mouse': mouse,
                        'session': session,
                        'mode': pairwise_mode,
                        'n_pairs': n_pairs,
                        'n_bins': n_used,
                        'hit_rate': np.nanmean(hit),
                        # Per-bin hebbian index (3 versions)
                        'hi_no_intercept': hi_no_intercept,
                        'hi_with_intercept': hi_with_intercept,
                        'hi_correlation': hi_correlation,
                        'hi_intercept_val': hi_intercept_val,
                        'hi_mean_CC': hi_mean_CC,
                        # Sign flips
                        'flips_no_int': flips_no_int,
                        'flips_with_int': flips_with_int,
                        'flips_corr': flips_corr,
                        # Intercept-slope correlation
                        'intercept_slope_corr': isc,
                        # Per-bin behavior
                        'bin_hit': bin_hit,
                        'bin_rpe': bin_rpe,
                        'bin_rt': bin_rt,
                        'bin_hit_rpe': bin_hit_rpe,
                    }
                    all_results.append(result)

                    print(f"  {pairwise_mode}: sign flips no_int={flips_no_int} "
                          f"with_int={flips_with_int} corr={flips_corr}")

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
np.save(os.path.join(RESULTS_DIR, 'hebbian_index_intercept_results.npy'),
        all_results, allow_pickle=True)
print(f"Saved {len(all_results)} results.")

#%% ============================================================================
# CELL 5: Load + split by mode
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'hebbian_index_intercept_results.npy'),
    allow_pickle=True).tolist()

by_mode = {}
for r in all_results:
    m = r['mode']
    if m not in by_mode:
        by_mode[m] = []
    by_mode[m].append(r)

for m in by_mode:
    print(f"{m}: {len(by_mode[m])} sessions")

#%% ============================================================================
# CELL 6: Key plot — do sign flips decrease when intercept is included?
# ============================================================================

for mode in by_mode:
    sess = by_mode[mode]
    n_s = len(sess)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Hebbian index: intercept analysis -- {mode}', fontsize=13, y=1.02)

    # Panel 1: Sign flips per epoch — no intercept vs with intercept
    ax = axes[0, 0]
    flips_ni = np.array([s['flips_no_int'] for s in sess])   # (n_sess, 4)
    flips_wi = np.array([s['flips_with_int'] for s in sess])
    flips_cr = np.array([s['flips_corr'] for s in sess])

    x = np.arange(n_epochs)
    w = 0.25
    ax.bar(x - w, np.mean(flips_ni, axis=0), w, label='no intercept',
           color='steelblue', alpha=0.7)
    ax.bar(x, np.mean(flips_wi, axis=0), w, label='with intercept',
           color='coral', alpha=0.7)
    ax.bar(x + w, np.mean(flips_cr, axis=0), w, label='correlation',
           color='mediumseagreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCH_ORDER)
    ax.set_ylabel('Mean # sign flips across bins')
    ax.set_title('Sign flips per epoch\n(fewer = more stable)')
    ax.legend(fontsize=8)

    # Panel 2: Total sign flips (summed across epochs) — paired comparison
    ax = axes[0, 1]
    total_ni = np.sum(flips_ni, axis=1)
    total_wi = np.sum(flips_wi, axis=1)
    total_cr = np.sum(flips_cr, axis=1)
    ax.scatter(total_ni, total_wi, alpha=0.6, s=30, label='with intercept')
    ax.scatter(total_ni, total_cr, alpha=0.6, s=30, marker='x',
               label='correlation')
    lim = max(np.max(total_ni), np.max(total_wi), np.max(total_cr)) + 1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('Sign flips (no intercept)')
    ax.set_ylabel('Sign flips (with intercept / corr)')
    ax.set_title('Does including intercept\nreduce sign flips?')
    ax.legend(fontsize=8)

    # Panel 3: Example session — hebbian index time series
    # Pick session with most sign flips in no-intercept
    best_idx = np.argmax(total_ni)
    s = sess[best_idx]
    ax = axes[0, 2]
    for ei, ep in enumerate(EPOCH_ORDER):
        ax.plot(s['hi_no_intercept'][:, ei], '--', color=f'C{ei}',
                alpha=0.5, label=f'{ep} (no int)')
        ax.plot(s['hi_with_intercept'][:, ei], '-', color=f'C{ei}',
                alpha=0.9, label=f'{ep} (with int)')
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Bin')
    ax.set_ylabel('Hebbian index (slope)')
    ax.set_title(f'Example: {s["mouse"]} {s["session"]}\n'
                 f'(most sign flips: {total_ni[best_idx]})')
    ax.legend(fontsize=5, ncol=2)

    # Panel 4: Intercept-slope correlation
    ax = axes[1, 0]
    isc = np.array([s['intercept_slope_corr'] for s in sess])
    bp = ax.boxplot([isc[:, ei] for ei in range(n_epochs)],
                    labels=EPOCH_ORDER, patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Pearson r (intercept vs slope)')
    ax.set_title('Intercept-slope correlation\nacross bins within session')

    # Panel 5: Mean CC trajectory (does activity level drift?)
    ax = axes[1, 1]
    for ei, ep in enumerate(EPOCH_ORDER):
        mean_cc_traj = np.array([s['hi_mean_CC'][:, ei] for s in sess])
        # Normalize per session for visualization
        norms = np.nanstd(mean_cc_traj, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mean_cc_z = (mean_cc_traj - np.nanmean(mean_cc_traj, axis=1, keepdims=True)) / norms
        ax.plot(np.nanmean(mean_cc_z, axis=0), '-', color=f'C{ei}', label=ep)
        ax.fill_between(range(mean_cc_z.shape[1]),
                        np.nanmean(mean_cc_z, axis=0) - np.nanstd(mean_cc_z, axis=0)/np.sqrt(n_s),
                        np.nanmean(mean_cc_z, axis=0) + np.nanstd(mean_cc_z, axis=0)/np.sqrt(n_s),
                        alpha=0.2, color=f'C{ei}')
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Bin')
    ax.set_ylabel('Mean CC (z-scored within session)')
    ax.set_title('Does mean activity drift across bins?')
    ax.legend(fontsize=8)

    # Panel 6: Intercept trajectory
    ax = axes[1, 2]
    for ei, ep in enumerate(EPOCH_ORDER):
        int_traj = np.array([s['hi_intercept_val'][:, ei] for s in sess])
        ax.plot(np.nanmean(int_traj, axis=0), '-', color=f'C{ei}', label=ep)
        ax.fill_between(range(int_traj.shape[1]),
                        np.nanmean(int_traj, axis=0) - np.nanstd(int_traj, axis=0)/np.sqrt(n_s),
                        np.nanmean(int_traj, axis=0) + np.nanstd(int_traj, axis=0)/np.sqrt(n_s),
                        alpha=0.2, color=f'C{ei}')
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Bin')
    ax.set_ylabel('Intercept (mean dW after removing CC)')
    ax.set_title('Intercept trajectory across bins')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'hebbian_index_intercept_{mode}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

#%% ============================================================================
# CELL 7: Hebbian index vs block-level behavior
# ============================================================================

mode = 'pre_only'
sess = by_mode[mode]

beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']

def get_beh(s, bname):
    if bname == 'hit_rate': return s['bin_hit']
    if bname == 'RPE': return s['bin_rpe']
    if bname == 'RT': return s['bin_rt']
    if bname == 'hit_RPE': return s['bin_hit_rpe']

# Within-session correlation: behavior vs hebbian index (with intercept)
n_beh = len(beh_names)
corr_beh_hi = np.full((len(sess), n_beh, n_epochs), np.nan)
corr_beh_intercept = np.full((len(sess), n_beh), np.nan)

for si, s in enumerate(sess):
    for bi, bname in enumerate(beh_names):
        bvar = get_beh(s, bname)
        if len(bvar) < 4 or np.std(bvar) == 0:
            continue
        # Behavior vs intercept (average intercept across epochs)
        mean_int = np.nanmean(s['hi_intercept_val'], axis=1)
        if np.std(mean_int) > 0:
            corr_beh_intercept[si, bi], _ = spearmanr(bvar, mean_int)
        # Behavior vs slope per epoch
        for ei in range(n_epochs):
            slope = s['hi_with_intercept'][:, ei]
            if np.std(slope) > 0:
                corr_beh_hi[si, bi, ei], _ = spearmanr(bvar, slope)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Block behavior vs hebbian index (with intercept) -- {mode}',
             fontsize=13, y=1.02)

# Panel 1: Behavior vs intercept
ax = axes[0]
bp = ax.boxplot([corr_beh_intercept[:, bi] for bi in range(n_beh)],
                labels=beh_names, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_ylabel('Spearman rho (within session)')
ax.set_title('What predicts the INTERCEPT?')

# Panel 2: Behavior vs slope (with intercept), by epoch
ax = axes[1]
mean_corr = np.nanmean(corr_beh_hi, axis=2)
bp = ax.boxplot([mean_corr[:, bi] for bi in range(n_beh)],
                labels=beh_names, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('coral')
    patch.set_alpha(0.6)
ax.axhline(0, color='k', ls='--', alpha=0.3)
ax.set_ylabel('Spearman rho (within session)')
ax.set_title('What predicts the SLOPE\n(after accounting for intercept)?')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hebbian_index_vs_behavior.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Save text report
# ============================================================================

report_path = os.path.join(RESULTS_DIR, 'hebbian_index_intercept_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("HEBBIAN INDEX: INTERCEPT vs SLOPE ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write("For each bin of trials, compute the hebbian index three ways:\n")
    f.write("  1) No intercept:   b = (CC'CC)^-1 CC'dW\n")
    f.write("  2) With intercept: fit dW = a + b*CC, report b\n")
    f.write("  3) Correlation:    corr(dW, CC)\n\n")
    f.write("Key question: do sign flips in the hebbian index go away\n")
    f.write("when we include an intercept?\n\n")

    for mode in by_mode:
        sess = by_mode[mode]
        n_s = len(sess)

        flips_ni = np.array([s['flips_no_int'] for s in sess])
        flips_wi = np.array([s['flips_with_int'] for s in sess])
        flips_cr = np.array([s['flips_corr'] for s in sess])
        isc = np.array([s['intercept_slope_corr'] for s in sess])

        f.write(f"{'='*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'='*50}\n\n")

        f.write("SIGN FLIPS ACROSS BINS (mean per epoch)\n")
        f.write(f"  {'epoch':10s} {'no_int':>8s} {'with_int':>8s} {'corr':>8s} {'reduction':>10s}\n")
        for ei, ep in enumerate(EPOCH_ORDER):
            ni = np.mean(flips_ni[:, ei])
            wi = np.mean(flips_wi[:, ei])
            cr = np.mean(flips_cr[:, ei])
            red = (ni - wi) / ni * 100 if ni > 0 else 0
            f.write(f"  {ep:10s} {ni:8.2f} {wi:8.2f} {cr:8.2f} {red:9.0f}%\n")

        total_ni = np.sum(flips_ni, axis=1)
        total_wi = np.sum(flips_wi, axis=1)
        total_cr = np.sum(flips_cr, axis=1)
        f.write(f"\n  Total:     {np.mean(total_ni):8.2f} {np.mean(total_wi):8.2f} "
                f"{np.mean(total_cr):8.2f}\n")
        f.write(f"  Sessions with FEWER flips after intercept: "
                f"{np.sum(total_wi < total_ni)}/{n_s}\n")
        f.write(f"  Sessions with MORE flips after intercept:  "
                f"{np.sum(total_wi > total_ni)}/{n_s}\n")
        f.write(f"  Sessions with SAME flips:                  "
                f"{np.sum(total_wi == total_ni)}/{n_s}\n\n")

        f.write("INTERCEPT-SLOPE CORRELATION (across bins within session)\n")
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = isc[:, ei]
            f.write(f"  {ep:10s}: mean r = {np.nanmean(vals):+.3f}  "
                    f"median = {np.nanmedian(vals):+.3f}\n")
        f.write("\n")

        # Per-session detail
        f.write("PER-SESSION DETAIL\n")
        f.write(f"  {'mouse':8s} {'session':12s} {'flips_ni':>8s} {'flips_wi':>8s} "
                f"{'flips_cr':>8s} {'isc_mean':>8s}\n")
        for si, s in enumerate(sess):
            f.write(f"  {s['mouse']:8s} {s['session']:12s} "
                    f"{int(np.sum(s['flips_no_int'])):8d} "
                    f"{int(np.sum(s['flips_with_int'])):8d} "
                    f"{int(np.sum(s['flips_corr'])):8d} "
                    f"{np.nanmean(s['intercept_slope_corr']):+8.3f}\n")
        f.write("\n\n")

    # Behavior correlations (pre_only only)
    if 'pre_only' in by_mode:
        f.write("BEHAVIOR vs HEBBIAN INDEX (pre_only, with intercept)\n")
        f.write("-" * 50 + "\n")
        f.write("Within-session Spearman correlations across bins\n\n")

        f.write("Behavior vs INTERCEPT (mean across epochs):\n")
        for bi, bname in enumerate(beh_names):
            vals = corr_beh_intercept[:, bi]
            f.write(f"  {bname:12s}: mean rho = {np.nanmean(vals):+.3f}  "
                    f"median = {np.nanmedian(vals):+.3f}\n")
        f.write("\n")

        f.write("Behavior vs SLOPE (mean across epochs):\n")
        for bi, bname in enumerate(beh_names):
            vals = np.nanmean(corr_beh_hi[:, bi, :], axis=1)
            f.write(f"  {bname:12s}: mean rho = {np.nanmean(vals):+.3f}  "
                    f"median = {np.nanmedian(vals):+.3f}\n")
        f.write("\n")

        f.write("Behavior vs SLOPE by epoch:\n")
        for bi, bname in enumerate(beh_names):
            for ei, ep in enumerate(EPOCH_ORDER):
                vals = corr_beh_hi[:, bi, ei]
                f.write(f"  {bname+'_'+ep:20s}: mean rho = {np.nanmean(vals):+.3f}  "
                        f"median = {np.nanmedian(vals):+.3f}\n")
            f.write("\n")

print(f"Report saved to: {report_path}")
