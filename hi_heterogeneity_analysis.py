#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Heterogeneity analysis for the Hebbian Index (HI) time series.
#
# Tests whether HI(t) varies more over time than expected from measurement
# noise, using a meta-analytic Q-statistic. Under a constant HI, Q ~ chi2(K-1).
# Significant Q means the two-factor Hebbian rule (constant HI) is rejected.
#
# Uses Fisher-z transformed correlations: z = arctanh(r), Var(z) ≈ 1/(n-3).
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, spearmanr
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import plotting_functions as pf

list_of_dirs = session_counting.counter()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})

print("Setup complete!")

#%% ============================================================================
# CELL 2: Load sliding window results
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded modes: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]

#%% ============================================================================
# CELL 3: Compute Q-statistic per session
# ============================================================================
# Focus on dev2_lag mode, pre epoch (the main result)
MODE = 'dev2_lag'
EPOCH = 'pre'
ei = EPOCH_ORDER.index(EPOCH)

q_results = []

for s in all_results[MODE]:
    r_t = s['hi_corr'][:, ei]      # Pearson r per time window
    n = s['n_pairs']                # number of pairs (same for all windows)

    # Drop NaN windows
    valid = np.isfinite(r_t)
    r_valid = r_t[valid]
    K = len(r_valid)

    if K < 3 or n < 6:
        continue

    # Fisher-z transform
    # Clip r to avoid arctanh(±1) = ±inf
    r_clipped = np.clip(r_valid, -0.9999, 0.9999)
    z_t = np.arctanh(r_clipped)

    # Variance of each z: 1/(n-3)
    var_z = 1.0 / (n - 3)
    w_t = 1.0 / var_z  # all weights are equal since n is constant

    # Weighted mean
    z_bar = np.mean(z_t)  # equivalent to weighted mean when weights are equal

    # Q-statistic
    Q = w_t * np.sum((z_t - z_bar) ** 2)

    # p-value from chi2 with K-1 degrees of freedom
    df = K - 1
    p_val = 1.0 - chi2.cdf(Q, df)

    # Also compute I² = max(0, (Q - df) / Q) * 100
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    q_results.append({
        'mouse': s['mouse'],
        'session': s['session'],
        'n_pairs': n,
        'n_windows': K,
        'z_t': z_t,
        'z_bar': z_bar,
        'r_t': r_valid,
        'se_r': np.sqrt((1 - r_valid**2)**2 / (n - 2)),
        'win_centers': s['win_centers'][valid] if 'win_centers' in s else np.arange(K),
        'Q': Q,
        'df': df,
        'p_val': p_val,
        'I2': I2,
        'win_rpe': s['win_rpe'][valid] if np.sum(valid) == len(s['win_rpe']) else s['win_rpe'][:K],
    })

n_sess = len(q_results)
n_sig = sum(1 for r in q_results if r['p_val'] < 0.05)
print(f"\n{n_sess} sessions analyzed")
print(f"  Significant Q (p < 0.05): {n_sig}/{n_sess} ({100*n_sig/n_sess:.0f}%)")
print(f"  Median I²: {np.median([r['I2'] for r in q_results]):.1f}%")

#%% ============================================================================
# CELL 4: Figure — example sessions + population summary
# ============================================================================
# Sort by Q p-value to find good example sessions
q_sorted = sorted(q_results, key=lambda r: r['p_val'])

# Pick 3 example sessions: most significant, median, least significant
idx_examples = [0, n_sess // 2, n_sess - 1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Top row: Example sessions with HI(t) ± SE and weighted mean ---
for col, idx in enumerate(idx_examples):
    ax = axes[0, col]
    r = q_sorted[idx]
    wc = r['win_centers']
    hi_t = r['r_t']
    se_t = r['se_r']
    hi_mean = np.tanh(r['z_bar'])  # back-transform weighted mean

    ax.fill_between(wc, hi_t - se_t, hi_t + se_t,
                     alpha=0.25, color='#3498db')
    ax.plot(wc, hi_t, 'o-', color='#2c3e50', linewidth=1.5,
            markersize=4, label='HI(t)')
    ax.axhline(hi_mean, color='#e74c3c', ls='--', lw=2,
               label=f'weighted mean = {hi_mean:.3f}')
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Window center (trial)')
    ax.set_ylabel('HI (Pearson r)')

    rank_label = ['Most heterogeneous', 'Median', 'Most homogeneous'][col]
    ax.set_title(f'{rank_label}\n{r["mouse"]} {r["session"]}\n'
                 f'Q={r["Q"]:.1f}, df={r["df"]}, p={r["p_val"]:.1e}, '
                 f'I²={r["I2"]:.0f}%',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')

# --- Bottom left: Histogram of Q p-values ---
ax = axes[1, 0]
pvals = np.array([r['p_val'] for r in q_results])
ax.hist(pvals, bins=20, range=(0, 1), color='#3498db',
        edgecolor='k', alpha=0.8)
ax.axhline(n_sess / 20, color='r', ls='--', lw=1.5,
           label=f'uniform expectation')
ax.axvline(0.05, color='k', ls=':', alpha=0.5)
ax.set_xlabel('Q p-value')
ax.set_ylabel('Sessions')
ax.set_title(f'Heterogeneity test (dev2, pre epoch)\n'
             f'{n_sig}/{n_sess} sessions p < 0.05',
             fontweight='bold')
ax.legend(fontsize=10)

# --- Bottom center: I² distribution ---
ax = axes[1, 1]
I2_vals = np.array([r['I2'] for r in q_results])
ax.hist(I2_vals, bins=20, color='#2ecc71', edgecolor='k', alpha=0.8)
ax.axvline(np.median(I2_vals), color='r', ls='-', lw=2,
           label=f'median = {np.median(I2_vals):.0f}%')
ax.set_xlabel('I² (%)')
ax.set_ylabel('Sessions')
ax.set_title('Excess heterogeneity\n(I² > 0 means more than noise)',
             fontweight='bold')
ax.legend(fontsize=10)

# --- Bottom right: Q/df vs n_pairs (sanity check) ---
ax = axes[1, 2]
q_over_df = np.array([r['Q'] / r['df'] for r in q_results])
n_pairs_arr = np.array([r['n_pairs'] for r in q_results])
ax.scatter(n_pairs_arr, q_over_df, s=30, c='#9b59b6', alpha=0.7,
           edgecolor='k', linewidth=0.5)
ax.axhline(1, color='k', ls='--', alpha=0.5, label='expected under H0')
ax.set_xlabel('Number of pairs')
ax.set_ylabel('Q / df')
ax.set_title('Heterogeneity scales with power',
             fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig21_hi_heterogeneity.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 21 saved.")

#%% ============================================================================
# CELL 5: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'hi_heterogeneity_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("HEBBIAN INDEX HETEROGENEITY ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Mode: {MODE}, Epoch: {EPOCH}\n")
    f.write(f"Q-statistic on Fisher-z transformed HI correlations\n")
    f.write(f"Var(z) = 1/(n_pairs - 3)\n")
    f.write("=" * 70 + "\n\n")

    f.write("POPULATION SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Sessions: {n_sess}\n")
    f.write(f"  Significant Q (p < 0.05): {n_sig}/{n_sess} "
            f"({100*n_sig/n_sess:.0f}%)\n")
    f.write(f"  Median I²: {np.median(I2_vals):.1f}%\n")
    f.write(f"  Mean I²:   {np.mean(I2_vals):.1f}%\n")
    f.write(f"  Median Q/df: {np.median(q_over_df):.2f}\n\n")

    f.write("PER-SESSION DETAIL\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'mouse':8s} {'session':8s} {'pairs':>6s} {'wins':>5s} "
            f"{'Q':>8s} {'df':>4s} {'p':>10s} {'I2':>6s}\n")
    for r in sorted(q_results, key=lambda x: (x['mouse'], x['session'])):
        f.write(f"  {r['mouse']:8s} {r['session']:8s} {r['n_pairs']:6d} "
                f"{r['n_windows']:5d} {r['Q']:8.1f} {r['df']:4d} "
                f"{r['p_val']:10.2e} {r['I2']:5.1f}%\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 6: Pooled dW vs CC scatter, split by RPE (all sessions)
# ============================================================================
# For each session, recompute per-window CC (dev2, pre epoch, lag=0),
# split windows into high/low RPE, average CC within each group,
# z-score CC and dW within session, then pool across sessions.

mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20
RPE_PERCENTILE = 33  # split at median

cc_hi_all, cc_lo_all = [], []
dw_hi_all, dw_lo_all = [], []

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
            folder = (r'//allen/aind/scratch/BCI/2p-raw/'
                      + mouse + r'/' + session + '/pophys/')
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
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # RPE
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)

            # Pair selection
            dw_list, pair_cl_list, pair_nt_list = [], [], []
            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < 10) &
                    (AMP[0][:, gi] > 0.1) &
                    (AMP[1][:, gi] > 0.1)
                )[0]
                if cl.size == 0:
                    continue
                nontarg = np.where(
                    (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000)
                )[0]
                if nontarg.size == 0:
                    continue
                dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            n_pairs = len(Y_T)

            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # Pre-epoch activity
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_pre = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
            epoch_post = epoch_pre  # lag=0, same window

            bl_post_mean = np.nanmean(epoch_post[:, :min(N_BASELINE, trl)], axis=1)

            # Sliding windows
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            cc_per_win = np.full((n_wins, n_pairs), np.nan)
            rpe_per_win = np.full(n_wins, np.nan)
            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                rpe_per_win[wi] = np.nanmean(rt_rpe[trial_idx])
                pre_act = cl_weights @ epoch_pre[:, trial_idx]
                post_dev = epoch_post[all_nt, :][:, trial_idx] - bl_post_mean[all_nt, np.newaxis]
                cc_per_win[wi, :] = np.sum(pre_act * post_dev, axis=1)

            # Split by RPE
            hi_rpe = rpe_per_win >= np.percentile(rpe_per_win, 100 - RPE_PERCENTILE)
            lo_rpe = rpe_per_win < np.percentile(rpe_per_win, RPE_PERCENTILE)

            if np.sum(hi_rpe) < 2 or np.sum(lo_rpe) < 2:
                print("  Not enough windows for RPE split.")
                continue

            cc_hi = np.nanmean(cc_per_win[hi_rpe, :], axis=0)
            cc_lo = np.nanmean(cc_per_win[lo_rpe, :], axis=0)

            # Z-score within session before pooling
            cc_all_sess = np.concatenate([cc_hi, cc_lo])
            dw_all_sess = np.concatenate([Y_T, Y_T])
            mu_cc, sd_cc = np.mean(cc_all_sess), np.std(cc_all_sess)
            mu_dw, sd_dw = np.mean(dw_all_sess), np.std(dw_all_sess)
            if sd_cc > 0 and sd_dw > 0:
                cc_hi_all.append((cc_hi - mu_cc) / sd_cc)
                cc_lo_all.append((cc_lo - mu_cc) / sd_cc)
                dw_hi_all.append((Y_T - mu_dw) / sd_dw)
                dw_lo_all.append((Y_T - mu_dw) / sd_dw)
                
                print(f"  {n_pairs} pairs, {n_wins} wins, "
                      f"{np.sum(hi_rpe)} hi / {np.sum(lo_rpe)} lo RPE wins")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

cc_hi_pool = np.concatenate(cc_hi_all)
cc_lo_pool = np.concatenate(cc_lo_all)
dw_hi_pool = np.concatenate(dw_hi_all)
dw_lo_pool = np.concatenate(dw_lo_all)
print(f"\nPooled: {len(cc_hi_pool)} pairs (hi RPE), {len(cc_lo_pool)} pairs (lo RPE)")

#%% ============================================================================
# CELL 7: Figure — pooled dW vs CC, high vs low RPE
# ============================================================================
n_bins = 3

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

for ax, cc, dw, label, color in [
    (axes[0], cc_hi_pool, dw_hi_pool,
     f'High RPE (top {RPE_PERCENTILE}%)', '#e74c3c'),
    (axes[1], cc_lo_pool, dw_lo_pool,
     f'Low RPE (bottom {RPE_PERCENTILE}%)', '#3498db'),
]:
    ok = np.isfinite(cc) & np.isfinite(dw)
    cc_ok = cc[ok]
    dw_ok = dw[ok]

    edges = np.percentile(cc_ok, np.linspace(0, 100, n_bins + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            mask = (cc_ok >= edges[bi]) & (cc_ok < edges[bi + 1])
        else:
            mask = (cc_ok >= edges[bi]) & (cc_ok <= edges[bi + 1])
        if np.sum(mask) < 3:
            continue
        bx.append(np.mean(cc_ok[mask]))
        by.append(np.mean(dw_ok[mask]))
        be.append(np.std(dw_ok[mask]) / np.sqrt(np.sum(mask)))

    bx, by, be = np.array(bx), np.array(by), np.array(be)
    ax.errorbar(bx, by, yerr=be, fmt='o-', color=color,
                capsize=5, linewidth=2, markersize=7)

    # Fit line
    if np.std(cc_ok) > 0:
        A = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A, dw_ok, rcond=None)[0]
        xr = np.array([bx[0], bx[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, '--', color='k', linewidth=1.5)
        r, p = spearmanr(cc_ok, dw_ok)
        print(p)
        ax.set_title(f'{label}')
#        ax.set_title(f'{label}\nslope={coeffs[1]:.4f}, r={r:.4f}, p={p:.1e}',
#                     fontsize=12, fontweight='bold')

    ax.axhline(0, color='k', ls='-', alpha=0.2)
    ax.axvline(0, color='k', ls='--', alpha=0.2)
    ax.set_xlabel('$\Delta r_{post} x r_{pre}$')
    ax.set_ylabel('$\Delta W_{i,j}$')

n_sess_pooled = len(cc_hi_all)
#fig.suptitle(f'Pooled dW vs CC split by RPE — all {n_sess_pooled} sessions\n'
#             f'(dev2, pre epoch, lag=0s, split at {RPE_PERCENTILE}th percentile)',
#             fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig22_dw_vs_cc_rpe_split_pooled.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 22 saved.")
