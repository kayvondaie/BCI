#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Second-order gradient term controls:
#   1. Shuffle control: permute AMP[0] connectivity to break weight-activity alignment
#   2. Uniform weights: replace net_w with uniform weights (population activity only)
#   3. First-order vs second-order vs combined
#
# All focused on pre epoch, dev2 mode, RPE modulation of HI.
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

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
# CELL 2: Configuration
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]

WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
OFFSET_SEC = 0
N_BASELINE = 20
N_SHUFFLES = 100  # number of connectivity shuffles

# CC modes to compute:
#   'dev2_1st'      — first-order: r_pre * (r_post - baseline)
#   'dev2_2nd'      — second-order: [sum_g w_jg r_g(t)] * (r_post - baseline)
#   'dev2_combined' — first + second: [r_pre + sum_g w_jg r_g(t)] * (r_post - baseline)
#   'dev2_uniform'  — uniform weights: [sum_g r_g(t)] * (r_post - baseline)  (no connectivity)
#   'dev2_shuffled' — shuffled AMP[0]: break pair-specific connectivity, keep marginals
CC_MODES = ['dev2_1st', 'dev2_2nd', 'dev2_combined', 'dev2_uniform']

all_results = {mode: [] for mode in CC_MODES}
# Shuffle results: store per-session distribution of RPE-HI correlations
shuffle_results = []

print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"CC modes: {CC_MODES}")
print(f"Shuffles: {N_SHUFFLES}")

#%% ============================================================================
# CELL 3: Main loop
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

            lag_frames = int(round(OFFSET_SEC / dt_si))

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)

            # ---- Pair selection ----
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.1
            amp1_thr = 0.1

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []
            pair_gi_list = []

            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < dist_target_lt) &
                    (AMP[0][:, gi] > amp0_thr) &
                    (AMP[1][:, gi] > amp1_thr)
                )[0]
                if cl.size == 0:
                    continue
                nontarg = np.where(
                    (stimDist[:, gi] > dist_nontarg_min) &
                    (stimDist[:, gi] < dist_nontarg_max)
                )[0]
                if nontarg.size == 0:
                    continue
                dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)
                pair_gi_list.append(np.full(len(nontarg), gi, dtype=int))

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            all_gi = np.concatenate(pair_gi_list)
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

            # ---- Network connectivity ----
            n_groups = stimDist.shape[1]

            group_targets = []
            for gi in range(n_groups):
                targs = np.where(stimDist[:, gi] < dist_target_lt)[0]
                group_targets.append(targs)

            # True connectivity: net_w[pair, g'] = mean AMP[0][cl, g']
            net_w = np.zeros((n_pairs, n_groups))
            for pi in range(n_pairs):
                cl_neurons = np.where(cl_weights[pi, :] > 0)[0]
                if len(cl_neurons) > 0:
                    net_w[pi, :] = np.mean(AMP[0][cl_neurons, :], axis=0)

            # Uniform weights: same structure but all 1s (just population activity)
            net_w_uniform = np.ones((n_pairs, n_groups))

            # ---- Sliding windows (pre epoch only) ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            t0, t1 = ts_pre[0], ts_pre[-1]
            t0_lag = max(0, min(t0 + lag_frames, n_frames - 1))
            t1_lag = max(0, min(t1 + lag_frames, n_frames - 1))

            epoch_pre_act = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)    # (N, trl)
            epoch_post_act = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            bl_post_mean = np.nanmean(epoch_post_act[:, baseline_trials_arr], axis=1)

            # Group activity: (n_groups, trl)
            group_act = np.zeros((n_groups, trl))
            for gi in range(n_groups):
                targs = group_targets[gi]
                if len(targs) > 0:
                    group_act[gi, :] = np.mean(epoch_pre_act[targs, :], axis=0)

            # ---- Compute CC per window for each mode ----
            win_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            # Store CC arrays: (n_wins, n_pairs) per mode
            cc_arrays = {mode: np.full((n_wins, n_pairs), np.nan) for mode in CC_MODES}

            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                win_center[wi] = (ws + ws + WIN_SIZE) / 2.0
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])

                # Shared quantities
                pre_act = cl_weights @ epoch_pre_act[:, trial_idx]  # (n_pairs, win)
                post_dev = (epoch_post_act[all_nt, :][:, trial_idx]
                            - bl_post_mean[all_nt, np.newaxis])     # (n_pairs, win)
                net_pre = net_w @ group_act[:, trial_idx]           # (n_pairs, win)
                net_pre_uniform = net_w_uniform @ group_act[:, trial_idx]

                # 1st order: r_pre * (r_post - baseline)
                cc_arrays['dev2_1st'][wi, :] = np.sum(pre_act * post_dev, axis=1)

                # 2nd order: net_input * (r_post - baseline)
                cc_arrays['dev2_2nd'][wi, :] = np.sum(net_pre * post_dev, axis=1)

                # Combined: (r_pre + net_input) * (r_post - baseline)
                cc_arrays['dev2_combined'][wi, :] = np.sum(
                    (pre_act + net_pre) * post_dev, axis=1)

                # Uniform: population activity (no connectivity weighting)
                cc_arrays['dev2_uniform'][wi, :] = np.sum(
                    net_pre_uniform * post_dev, axis=1)

            # ---- Compute HI (slope with intercept) and RPE-HI correlation per mode ----
            for mode in CC_MODES:
                hi_slope = np.full(n_wins, np.nan)
                for wi in range(n_wins):
                    cc_pair = cc_arrays[mode][wi, :]
                    if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                        continue
                    A = np.column_stack([np.ones(n_pairs), cc_pair])
                    coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                    hi_slope[wi] = coeffs[1]

                # Within-session RPE-HI correlation
                ok = np.isfinite(win_rpe) & np.isfinite(hi_slope)
                rpe_hi_rho = np.nan
                if np.sum(ok) >= 5 and np.std(hi_slope[ok]) > 0 and np.std(win_rpe[ok]) > 0:
                    rpe_hi_rho, _ = spearmanr(win_rpe[ok], hi_slope[ok])

                all_results[mode].append({
                    'mouse': mouse,
                    'session': session,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
                    'hi_slope': hi_slope,
                    'win_rpe': win_rpe.copy(),
                    'win_center': win_center.copy(),
                    'rpe_hi_rho': rpe_hi_rho,
                })

            # ---- Shuffle control: permute rows of net_w, recompute 2nd-order HI ----
            rng = np.random.default_rng(42)
            shuf_rhos = np.full(N_SHUFFLES, np.nan)

            for si in range(N_SHUFFLES):
                # Shuffle connectivity across pairs (rows of net_w)
                perm = rng.permutation(n_pairs)
                net_w_shuf = net_w[perm, :]

                hi_slope_shuf = np.full(n_wins, np.nan)
                for wi, ws in enumerate(win_starts):
                    trial_idx = np.arange(ws, ws + WIN_SIZE)
                    post_dev = (epoch_post_act[all_nt, :][:, trial_idx]
                                - bl_post_mean[all_nt, np.newaxis])
                    net_pre_shuf = net_w_shuf @ group_act[:, trial_idx]
                    cc_shuf = np.sum(net_pre_shuf * post_dev, axis=1)

                    if np.std(cc_shuf) == 0:
                        continue
                    A = np.column_stack([np.ones(n_pairs), cc_shuf])
                    coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                    hi_slope_shuf[wi] = coeffs[1]

                ok = np.isfinite(win_rpe) & np.isfinite(hi_slope_shuf)
                if np.sum(ok) >= 5 and np.std(hi_slope_shuf[ok]) > 0 and np.std(win_rpe[ok]) > 0:
                    shuf_rhos[si], _ = spearmanr(win_rpe[ok], hi_slope_shuf[ok])

            # Store shuffle results
            real_rho = all_results['dev2_2nd'][-1]['rpe_hi_rho']
            shuf_p = np.nanmean(shuf_rhos >= real_rho) if np.isfinite(real_rho) else np.nan
            shuffle_results.append({
                'mouse': mouse,
                'session': session,
                'real_rho': real_rho,
                'shuf_rhos': shuf_rhos,
                'shuf_mean': np.nanmean(shuf_rhos),
                'shuf_std': np.nanstd(shuf_rhos),
                'shuf_p': shuf_p,
            })

            print(f"  {n_wins} wins, {n_pairs} pairs | "
                  f"1st={all_results['dev2_1st'][-1]['rpe_hi_rho']:.3f}, "
                  f"2nd={real_rho:.3f}, "
                  f"comb={all_results['dev2_combined'][-1]['rpe_hi_rho']:.3f}, "
                  f"unif={all_results['dev2_uniform'][-1]['rpe_hi_rho']:.3f}, "
                  f"shuf_p={shuf_p:.3f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

for mode in CC_MODES:
    print(f"{mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'second_order_controls.npy'),
        {'all_results': all_results, 'shuffle_results': shuffle_results},
        allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
loaded = np.load(os.path.join(RESULTS_DIR, 'second_order_controls.npy'),
                 allow_pickle=True).item()
all_results = loaded['all_results']
shuffle_results = loaded['shuffle_results']
CC_MODES = list(all_results.keys())
print(f"Loaded: {CC_MODES}, {len(shuffle_results)} sessions with shuffles")

#%% ============================================================================
# CELL 6: Summary statistics — RPE-HI correlation per mode
# ============================================================================

print("\n" + "=" * 70)
print("RPE-HI CORRELATION (Spearman) — PRE EPOCH, DEV2")
print("=" * 70)

mode_labels = {
    'dev2_1st': '1st order (r_pre)',
    'dev2_2nd': '2nd order (net input)',
    'dev2_combined': 'Combined (1st + 2nd)',
    'dev2_uniform': 'Uniform (no connectivity)',
}

for mode in CC_MODES:
    rhos = np.array([s['rpe_hi_rho'] for s in all_results[mode]])
    v = rhos[np.isfinite(rhos)]
    n_pos = np.sum(v > 0)
    try:
        _, p = wilcoxon(v)
    except Exception:
        p = 1.0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"\n  {mode_labels.get(mode, mode)}:")
    print(f"    mean={np.mean(v):+.4f}, median={np.median(v):+.4f}, "
          f">{0}={n_pos}/{len(v)}, Wilcoxon p={p:.4f} {sig}")

# Shuffle summary
print(f"\n  Shuffle control (2nd order, {N_SHUFFLES} permutations):")
real_rhos = np.array([s['real_rho'] for s in shuffle_results])
shuf_means = np.array([s['shuf_mean'] for s in shuffle_results])
shuf_ps = np.array([s['shuf_p'] for s in shuffle_results])
v_real = real_rhos[np.isfinite(real_rhos)]
v_shuf = shuf_means[np.isfinite(shuf_means)]
n_sig_shuf = np.sum(shuf_ps[np.isfinite(shuf_ps)] < 0.05)
print(f"    Real mean rho={np.mean(v_real):+.4f}")
print(f"    Shuffle mean rho={np.mean(v_shuf):+.4f}")
print(f"    Sessions where real > 95% of shuffles: {n_sig_shuf}/{len(shuf_ps)}")

# Paired comparison: 2nd order vs uniform (is connectivity helping?)
rhos_2nd = np.array([s['rpe_hi_rho'] for s in all_results['dev2_2nd']])
rhos_unif = np.array([s['rpe_hi_rho'] for s in all_results['dev2_uniform']])
ok = np.isfinite(rhos_2nd) & np.isfinite(rhos_unif)
if np.sum(ok) >= 5:
    diff = rhos_2nd[ok] - rhos_unif[ok]
    try:
        _, p_diff = wilcoxon(diff)
    except Exception:
        p_diff = 1.0
    print(f"\n  2nd order vs Uniform (paired):")
    print(f"    mean diff={np.mean(diff):+.4f}, Wilcoxon p={p_diff:.4f}")

# Paired comparison: combined vs 1st order (does 2nd order add value?)
rhos_1st = np.array([s['rpe_hi_rho'] for s in all_results['dev2_1st']])
rhos_comb = np.array([s['rpe_hi_rho'] for s in all_results['dev2_combined']])
ok2 = np.isfinite(rhos_1st) & np.isfinite(rhos_comb)
if np.sum(ok2) >= 5:
    diff2 = rhos_comb[ok2] - rhos_1st[ok2]
    try:
        _, p_diff2 = wilcoxon(diff2)
    except Exception:
        p_diff2 = 1.0
    print(f"\n  Combined vs 1st order (paired):")
    print(f"    mean diff={np.mean(diff2):+.4f}, Wilcoxon p={p_diff2:.4f}")

#%% ============================================================================
# CELL 7: Figure — RPE-HI correlation across modes
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Panel A: Bar plot of mean RPE-HI rho per mode ---
ax = axes[0]
mode_order = ['dev2_1st', 'dev2_2nd', 'dev2_combined', 'dev2_uniform']
short_labels = ['1st order\n(r_pre)', '2nd order\n(net input)',
                'Combined\n(1st+2nd)', 'Uniform\n(no w)']
colors = ['#3498db', '#e74c3c', '#9b59b6', '#95a5a6']

means = []
sems = []
pvals = []
for mode in mode_order:
    rhos = np.array([s['rpe_hi_rho'] for s in all_results[mode]])
    v = rhos[np.isfinite(rhos)]
    means.append(np.mean(v))
    sems.append(np.std(v) / np.sqrt(len(v)))
    try:
        _, p = wilcoxon(v)
    except Exception:
        p = 1.0
    pvals.append(p)

x = np.arange(len(mode_order))
bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor='k',
              linewidth=1.2, alpha=0.85)

for xi, p in enumerate(pvals):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(xi, means[xi] + sems[xi] + 0.005, sig,
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel('Mean RPE-HI rho (Spearman)')
ax.set_title('RPE modulation by mode', fontweight='bold')

# --- Panel B: Paired scatter 2nd order vs uniform ---
ax2 = axes[1]
rhos_2nd = np.array([s['rpe_hi_rho'] for s in all_results['dev2_2nd']])
rhos_unif = np.array([s['rpe_hi_rho'] for s in all_results['dev2_uniform']])
ok = np.isfinite(rhos_2nd) & np.isfinite(rhos_unif)

ax2.scatter(rhos_unif[ok], rhos_2nd[ok], c='#e74c3c', s=40, alpha=0.7,
            edgecolor='k', linewidth=0.5)
lims = [min(np.min(rhos_unif[ok]), np.min(rhos_2nd[ok])) - 0.05,
        max(np.max(rhos_unif[ok]), np.max(rhos_2nd[ok])) + 0.05]
ax2.plot(lims, lims, 'k--', alpha=0.4)
ax2.set_xlabel('Uniform (no connectivity)')
ax2.set_ylabel('2nd order (true connectivity)')
ax2.set_title('Connectivity vs population activity', fontweight='bold')
n_above = np.sum(rhos_2nd[ok] > rhos_unif[ok])
ax2.text(0.05, 0.95, f'{n_above}/{np.sum(ok)} above unity',
         transform=ax2.transAxes, va='top', fontsize=11)

# --- Panel C: Shuffle distribution (pooled) ---
ax3 = axes[2]
# Pool all shuffle rhos
all_shuf_rhos = np.concatenate([s['shuf_rhos'] for s in shuffle_results])
all_shuf_rhos = all_shuf_rhos[np.isfinite(all_shuf_rhos)]
real_mean = np.nanmean([s['real_rho'] for s in shuffle_results])

ax3.hist(all_shuf_rhos, bins=50, color='#95a5a6', edgecolor='k',
         alpha=0.7, density=True, label='Shuffled')
ax3.axvline(real_mean, color='#e74c3c', linewidth=2.5, ls='-',
            label=f'Real mean={real_mean:.3f}')
ax3.set_xlabel('RPE-HI rho (Spearman)')
ax3.set_ylabel('Density')
ax3.set_title('Shuffle control (2nd order)', fontweight='bold')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig17_second_order_controls.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 17 saved.")

#%% ============================================================================
# CELL 8: Figure — Per-session comparison (1st vs 2nd vs combined)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Panel A: Paired scatter 1st vs 2nd ---
ax = axes[0]
rhos_1st = np.array([s['rpe_hi_rho'] for s in all_results['dev2_1st']])
rhos_2nd = np.array([s['rpe_hi_rho'] for s in all_results['dev2_2nd']])
ok = np.isfinite(rhos_1st) & np.isfinite(rhos_2nd)

ax.scatter(rhos_1st[ok], rhos_2nd[ok], c='#9b59b6', s=40, alpha=0.7,
           edgecolor='k', linewidth=0.5)
lims = [min(np.min(rhos_1st[ok]), np.min(rhos_2nd[ok])) - 0.05,
        max(np.max(rhos_1st[ok]), np.max(rhos_2nd[ok])) + 0.05]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('1st order RPE-HI rho')
ax.set_ylabel('2nd order RPE-HI rho')
ax.set_title('1st vs 2nd order', fontweight='bold')
r_corr, p_corr = spearmanr(rhos_1st[ok], rhos_2nd[ok])
ax.text(0.05, 0.95, f'r={r_corr:.2f}, p={p_corr:.3f}',
        transform=ax.transAxes, va='top', fontsize=11)

# --- Panel B: Combined improvement over 1st order ---
ax2 = axes[1]
rhos_comb = np.array([s['rpe_hi_rho'] for s in all_results['dev2_combined']])
ok2 = np.isfinite(rhos_1st) & np.isfinite(rhos_comb)

ax2.scatter(rhos_1st[ok2], rhos_comb[ok2], c='#2c3e50', s=40, alpha=0.7,
            edgecolor='k', linewidth=0.5)
lims2 = [min(np.min(rhos_1st[ok2]), np.min(rhos_comb[ok2])) - 0.05,
         max(np.max(rhos_1st[ok2]), np.max(rhos_comb[ok2])) + 0.05]
ax2.plot(lims2, lims2, 'k--', alpha=0.4)
ax2.set_xlabel('1st order RPE-HI rho')
ax2.set_ylabel('Combined RPE-HI rho')
ax2.set_title('Does 2nd order add to 1st?', fontweight='bold')
n_above = np.sum(rhos_comb[ok2] > rhos_1st[ok2])
ax2.text(0.05, 0.95, f'{n_above}/{np.sum(ok2)} improved',
         transform=ax2.transAxes, va='top', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig18_first_vs_second_order.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 18 saved.")

#%% ============================================================================
# CELL 9: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'second_order_controls_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("SECOND-ORDER GRADIENT TERM — CONTROLS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Pre epoch, dev2 mode\n")
    f.write(f"Shuffles: {N_SHUFFLES}\n")
    f.write("=" * 70 + "\n\n")

    f.write("QUESTION: Is the 2nd-order RPE-HI effect driven by true connectivity\n")
    f.write("or just correlated population activity?\n\n")

    f.write("MODES:\n")
    f.write("  dev2_1st      : r_pre * (r_post - baseline)                    [1st order]\n")
    f.write("  dev2_2nd      : [sum_g w_jg r_g(t)] * (r_post - baseline)      [2nd order]\n")
    f.write("  dev2_combined : [r_pre + sum_g w_jg r_g(t)] * (r_post - base)  [both]\n")
    f.write("  dev2_uniform  : [sum_g r_g(t)] * (r_post - baseline)           [no weights]\n")
    f.write("  shuffle       : permute net_w rows across pairs                 [break connectivity]\n\n")

    f.write("RPE-HI SPEARMAN CORRELATION (within-session, pre epoch)\n")
    f.write("-" * 60 + "\n")
    f.write(f"  {'Mode':20s} {'mean':>8s} {'median':>8s} {'%>0':>6s} {'Wilcoxon p':>12s} {'sig':>4s}\n")

    for mode in CC_MODES:
        rhos = np.array([s['rpe_hi_rho'] for s in all_results[mode]])
        v = rhos[np.isfinite(rhos)]
        n_pos = np.sum(v > 0)
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        f.write(f"  {mode:20s} {np.mean(v):+8.4f} {np.median(v):+8.4f} "
                f"{n_pos}/{len(v):>3d} {p:12.4f} {sig:>4s}\n")

    # Shuffle
    v_shuf = shuf_means[np.isfinite(shuf_means)]
    f.write(f"  {'shuffle (mean)':20s} {np.mean(v_shuf):+8.4f} {np.median(v_shuf):+8.4f}\n")
    n_sig_shuf = np.sum(shuf_ps[np.isfinite(shuf_ps)] < 0.05)
    f.write(f"  Sessions with real > 95th pct of shuffle: {n_sig_shuf}/{len(shuf_ps)}\n\n")

    f.write("PAIRED COMPARISONS\n")
    f.write("-" * 60 + "\n")

    # 2nd vs uniform
    rhos_2nd = np.array([s['rpe_hi_rho'] for s in all_results['dev2_2nd']])
    rhos_unif = np.array([s['rpe_hi_rho'] for s in all_results['dev2_uniform']])
    ok = np.isfinite(rhos_2nd) & np.isfinite(rhos_unif)
    diff = rhos_2nd[ok] - rhos_unif[ok]
    try:
        _, p = wilcoxon(diff)
    except Exception:
        p = 1.0
    f.write(f"  2nd order vs Uniform:  mean diff={np.mean(diff):+.4f}, "
            f"Wilcoxon p={p:.4f}, {np.sum(diff>0)}/{np.sum(ok)} sessions 2nd>uniform\n")

    # Combined vs 1st
    rhos_1st = np.array([s['rpe_hi_rho'] for s in all_results['dev2_1st']])
    rhos_comb = np.array([s['rpe_hi_rho'] for s in all_results['dev2_combined']])
    ok2 = np.isfinite(rhos_1st) & np.isfinite(rhos_comb)
    diff2 = rhos_comb[ok2] - rhos_1st[ok2]
    try:
        _, p2 = wilcoxon(diff2)
    except Exception:
        p2 = 1.0
    f.write(f"  Combined vs 1st order: mean diff={np.mean(diff2):+.4f}, "
            f"Wilcoxon p={p2:.4f}, {np.sum(diff2>0)}/{np.sum(ok2)} sessions combined>1st\n")

    # 1st vs 2nd correlation
    ok3 = np.isfinite(rhos_1st) & np.isfinite(rhos_2nd)
    r_12, p_12 = spearmanr(rhos_1st[ok3], rhos_2nd[ok3])
    f.write(f"\n  1st-2nd order correlation: r={r_12:.3f}, p={p_12:.4f}\n")

    f.write("\nPER-SESSION DETAIL\n")
    f.write("-" * 90 + "\n")
    f.write(f"  {'mouse':8s} {'session':8s} {'pairs':>6s} {'1st':>8s} {'2nd':>8s} "
            f"{'comb':>8s} {'unif':>8s} {'shuf_p':>8s}\n")
    for i, mode_res in enumerate(all_results['dev2_1st']):
        m = mode_res['mouse']
        s = mode_res['session']
        np_ = mode_res['n_pairs']
        r1 = mode_res['rpe_hi_rho']
        r2 = all_results['dev2_2nd'][i]['rpe_hi_rho']
        rc = all_results['dev2_combined'][i]['rpe_hi_rho']
        ru = all_results['dev2_uniform'][i]['rpe_hi_rho']
        sp = shuffle_results[i]['shuf_p']
        f.write(f"  {m:8s} {s:8s} {np_:6d} {r1:+8.3f} {r2:+8.3f} "
                f"{rc:+8.3f} {ru:+8.3f} {sp:8.3f}\n")

print(f"Report saved to: {report_path}")
