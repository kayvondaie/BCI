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
from scipy.stats import pearsonr, spearmanr, wilcoxon
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

# ---- Global plot style: large, readable fonts ----
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

pairwise_mode = 'dot_prod'

# Sliding window parameters
WIN_SIZE = 25    # trials per window
WIN_STEP = 5     # step between windows
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

all_results = []
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Mode: {pairwise_mode}")

#%% ============================================================================
# CELL 3: Main loop — sliding window hebbian index
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
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Epoch averages per neuron per trial
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

            # Per-trial behavioral variables
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

            # Epoch activity arrays: (n_neurons, n_trials)
            epoch_activity = {
                'pre': kpre, 'go_cue': k, 'late': kstep, 'reward': krewards,
            }

            # ---- Build pair selection (same as build_pairwise_XY_per_bin) ----
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.1
            amp1_thr = 0.1

            # deltaW per pair
            dw_list = []
            pair_cl_list = []    # indices of target neurons
            pair_nt_list = []    # indices of nontarget neurons

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

                dw = AMP[1][nontarg, gi] - AMP[0][nontarg, gi]
                dw_list.append(dw)
                pair_cl_list.append(np.tile(cl, (len(nontarg), 1)))
                pair_nt_list.append(nontarg)

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)  # (n_pairs,)
            n_pairs = len(Y_T)

            # Pre-compute cl averaging weights: (n_pairs, n_neurons)
            # cl_weights[p, n] = 1/|cl| if neuron n is in cl for pair p
            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]  # (n_nt, n_cl)
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # Per-window: hebbian index, intercept, mean CC, behavior
            hi_no_int = np.full((n_wins, n_epochs), np.nan)
            hi_with_int = np.full((n_wins, n_epochs), np.nan)
            hi_intercept = np.full((n_wins, n_epochs), np.nan)
            hi_corr = np.full((n_wins, n_epochs), np.nan)
            win_hit = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            for wi, ws in enumerate(win_starts):
                we = ws + WIN_SIZE
                trial_idx = np.arange(ws, we)
                win_center[wi] = (ws + we) / 2.0

                # Block-level behavior
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])

                for ei, ep in enumerate(EPOCH_ORDER):
                    act = epoch_activity[ep][:, trial_idx]  # (n_neurons, win_size)

                    if pairwise_mode == 'pre_only':
                        # CC[pair] = mean presynaptic activity summed over trials
                        act_sum = np.sum(act, axis=1)  # (n_neurons,)
                        cc_pair = cl_weights @ act_sum  # (n_pairs,)

                    elif pairwise_mode == 'dot_prod':
                        # CC[pair] = sum_t mean(r_pre(t)) * r_post(t)
                        # Pre activity per trial: cl_weights @ act gives (n_pairs, n_trials)
                        pre_act = cl_weights @ act  # (n_pairs, win_size)
                        post_act = act[all_nt, :]   # (n_pairs, win_size)
                        cc_pair = np.sum(pre_act * post_act, axis=1)  # (n_pairs,)

                    elif pairwise_mode == 'post_only':
                        # CC[pair] = sum of postsynaptic activity
                        act_sum = np.sum(act, axis=1)  # (n_neurons,)
                        cc_pair = act_sum[all_nt]  # (n_pairs,)

                    else:
                        raise ValueError(f"Unknown pairwise_mode: {pairwise_mode}")

                    if np.std(cc_pair) == 0:
                        continue

                    # 1) No intercept
                    hi_no_int[wi, ei] = (
                        np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair))

                    # 2) With intercept
                    A = np.column_stack([np.ones(n_pairs), cc_pair])
                    coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                    hi_intercept[wi, ei] = coeffs[0]
                    hi_with_int[wi, ei] = coeffs[1]

                    # 3) Correlation
                    hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

            # Sign flips
            def count_flips(arr):
                signs = np.sign(arr)
                return np.sum(signs[1:] != signs[:-1], axis=0)

            result = {
                'mouse': mouse,
                'session': session,
                'n_pairs': n_pairs,
                'n_trials': trl,
                'n_windows': n_wins,
                'win_centers': win_center,
                'hit_rate': np.nanmean(hit),
                # Hebbian index time series
                'hi_no_int': hi_no_int,
                'hi_with_int': hi_with_int,
                'hi_intercept': hi_intercept,
                'hi_corr': hi_corr,
                # Sign flips
                'flips_no_int': count_flips(hi_no_int),
                'flips_with_int': count_flips(hi_with_int),
                'flips_corr': count_flips(hi_corr),
                # Per-window behavior
                'win_hit': win_hit,
                'win_rpe': win_rpe,
                'win_rt': win_rt,
                'win_hit_rpe': win_hit_rpe,
            }
            all_results.append(result)

            fn = result['flips_no_int']
            fw = result['flips_with_int']
            print(f"  {n_wins} windows | flips no_int={fn} with_int={fw}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_results)} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
_save_name = f'sliding_window_results_{pairwise_mode}.npy'
np.save(os.path.join(RESULTS_DIR, _save_name),
        all_results, allow_pickle=True)
print(f"Saved {len(all_results)} sessions to {_save_name}")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
_save_name = f'sliding_window_results_{pairwise_mode}.npy'
all_results = np.load(
    os.path.join(RESULTS_DIR, _save_name),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_results)} sessions from {_save_name}")

#%% ============================================================================
# CELL 6: Within-session correlation of behavior with hebbian index
# ============================================================================
# For each session, Spearman correlate block behavior with hebbian index
# (with intercept version). Then test across sessions: is the mean
# correlation significantly different from 0?

n_s = len(all_results)
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

# Compute within-session correlations
corr_beh_slope = np.full((n_s, n_beh, n_epochs), np.nan)
corr_beh_intercept = np.full((n_s, n_beh, n_epochs), np.nan)

for si, s in enumerate(all_results):
    for bi, bname in enumerate(beh_names):
        bvar = get_beh(s, bname)
        if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
            continue
        for ei in range(n_epochs):
            slope = s['hi_with_int'][:, ei]
            intercept = s['hi_intercept'][:, ei]
            ok = np.isfinite(bvar) & np.isfinite(slope)
            if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
                corr_beh_slope[si, bi, ei], _ = spearmanr(bvar[ok], slope[ok])
            ok2 = np.isfinite(bvar) & np.isfinite(intercept)
            if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
                corr_beh_intercept[si, bi, ei], _ = spearmanr(bvar[ok2], intercept[ok2])

print("Within-session correlations computed.")

#%% ============================================================================
# CELL 7: Figure 1 — Does behavior predict the hebbian slope?
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Does block-level behavior predict the Hebbian slope?',
             fontweight='bold')

epoch_colors = {'pre': '#1f77b4', 'go_cue': '#ff7f0e',
                'late': '#2ca02c', 'reward': '#d62728'}

for bi, bname in enumerate(beh_names):
    ax = axes[bi // 2, bi % 2]

    means = []
    sems = []
    pvals = []
    colors = []

    for ei, ep in enumerate(EPOCH_ORDER):
        vals = corr_beh_slope[:, bi, ei]
        valid = np.isfinite(vals)
        v = vals[valid]
        m = np.mean(v)
        se = np.std(v) / np.sqrt(len(v))
        # Wilcoxon signed-rank test: is median != 0?
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        means.append(m)
        sems.append(se)
        pvals.append(p)
        colors.append(epoch_colors[ep])

    x = np.arange(n_epochs)
    bars = ax.bar(x, means, yerr=sems, capsize=6,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    # Mark significance
    for xi, p in enumerate(pvals):
        if p < 0.001:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '***', ha='center',
                    fontsize=14, fontweight='bold')
        elif p < 0.01:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '**', ha='center',
                    fontsize=14, fontweight='bold')
        elif p < 0.05:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '*', ha='center',
                    fontsize=14, fontweight='bold')

    ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCH_ORDER)
    ax.set_ylabel('Mean within-session rho')
    ax.set_title(f'{bname}', fontweight='bold')

    # y-axis symmetric
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.3
    ax.set_ylim(-ylim, ylim)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_behavior_vs_slope.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Figure 2 — Does behavior predict the intercept?
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Does block-level behavior predict the intercept (global dW)?',
             fontweight='bold')

for bi, bname in enumerate(beh_names):
    ax = axes[bi // 2, bi % 2]

    means = []
    sems = []
    pvals = []
    colors = []

    for ei, ep in enumerate(EPOCH_ORDER):
        vals = corr_beh_intercept[:, bi, ei]
        valid = np.isfinite(vals)
        v = vals[valid]
        m = np.mean(v)
        se = np.std(v) / np.sqrt(len(v))
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        means.append(m)
        sems.append(se)
        pvals.append(p)
        colors.append(epoch_colors[ep])

    x = np.arange(n_epochs)
    bars = ax.bar(x, means, yerr=sems, capsize=6,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    for xi, p in enumerate(pvals):
        if p < 0.001:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '***', ha='center',
                    fontsize=14, fontweight='bold')
        elif p < 0.01:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '**', ha='center',
                    fontsize=14, fontweight='bold')
        elif p < 0.05:
            ax.text(xi, means[xi] + sems[xi] + 0.005, '*', ha='center',
                    fontsize=14, fontweight='bold')

    ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCH_ORDER)
    ax.set_ylabel('Mean within-session rho')
    ax.set_title(f'{bname}', fontweight='bold')

    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.3
    ax.set_ylim(-ylim, ylim)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_behavior_vs_intercept.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 9: Figure 3 — Sign flips: no intercept vs with intercept
# ============================================================================

flips_ni = np.array([s['flips_no_int'] for s in all_results])
flips_wi = np.array([s['flips_with_int'] for s in all_results])
flips_cr = np.array([s['flips_corr'] for s in all_results])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: mean sign flips per epoch
ax = axes[0]
x = np.arange(n_epochs)
w = 0.25
ax.bar(x - w, np.mean(flips_ni, axis=0), w, label='No intercept',
       color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.8)
ax.bar(x, np.mean(flips_wi, axis=0), w, label='With intercept',
       color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.8)
ax.bar(x + w, np.mean(flips_cr, axis=0), w, label='Correlation',
       color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(EPOCH_ORDER)
ax.set_ylabel('Mean sign flips across windows')
ax.set_title('Sign flips per epoch', fontweight='bold')
ax.legend()

# Panel 2: paired scatter — total flips
ax = axes[1]
total_ni = np.sum(flips_ni, axis=1)
total_wi = np.sum(flips_wi, axis=1)
ax.scatter(total_ni, total_wi, s=50, alpha=0.6, edgecolors='black',
           linewidth=0.5)
lim = max(np.max(total_ni), np.max(total_wi)) + 2
ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, linewidth=1.5)
ax.set_xlabel('Sign flips (no intercept)')
ax.set_ylabel('Sign flips (with intercept)')
ax.set_title('Does intercept reduce sign flips?', fontweight='bold')
ax.set_xlim(-1, lim)
ax.set_ylim(-1, lim)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig3_sign_flips.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 10: Figure 4 — Example sessions: hebbian index trajectory
# ============================================================================

# Pick 4 sessions with interesting patterns (most windows, spread of mice)
n_wins_list = [s['n_windows'] for s in all_results]
sorted_idx = np.argsort(n_wins_list)[::-1]

# Pick from different mice
shown_mice = set()
example_idx = []
for idx in sorted_idx:
    m = all_results[idx]['mouse']
    if m not in shown_mice and len(example_idx) < 4:
        example_idx.append(idx)
        shown_mice.add(m)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hebbian index across trial windows (with intercept)',
             fontweight='bold')

for pi, idx in enumerate(example_idx):
    ax = axes[pi // 2, pi % 2]
    s = all_results[idx]
    wc = s['win_centers']

    for ei, ep in enumerate(EPOCH_ORDER):
        ax.plot(wc, s['hi_with_int'][:, ei], '-o', markersize=3,
                color=list(epoch_colors.values())[ei], label=ep, linewidth=1.5)

    ax.axhline(0, color='k', ls='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Hebbian slope')
    ax.set_title(f"{s['mouse']} {s['session']}  "
                 f"({s['n_trials']} trials, {s['n_pairs']} pairs)",
                 fontweight='bold')
    if pi == 0:
        ax.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_example_trajectories.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 11: Figure 5 — Distribution of within-session correlations
# ============================================================================
# Show the actual distributions, not just means

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# hit_rate vs slope (pre epoch — strongest effect)
ax = axes[0]
vals = corr_beh_slope[:, 0, 0]  # hit_rate, pre epoch
valid = np.isfinite(vals)
v = vals[valid]
ax.hist(v, bins=20, color='#1f77b4', alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', ls='--', alpha=0.4, linewidth=1.5)
ax.axvline(np.mean(v), color='red', ls='-', linewidth=2,
           label=f'mean = {np.mean(v):.3f}')
ax.axvline(np.median(v), color='orange', ls='--', linewidth=2,
           label=f'median = {np.median(v):.3f}')
try:
    _, p = wilcoxon(v)
except Exception:
    p = 1.0
ax.set_xlabel('Within-session Spearman rho')
ax.set_ylabel('# sessions')
ax.set_title(f'hit_rate vs Hebbian slope (pre epoch)\n'
             f'Wilcoxon p = {p:.4f}, n = {len(v)}',
             fontweight='bold')
ax.legend()

# RT vs slope (pre epoch)
ax = axes[1]
vals = corr_beh_slope[:, 2, 0]  # RT, pre epoch
valid = np.isfinite(vals)
v = vals[valid]
ax.hist(v, bins=20, color='#2ca02c', alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', ls='--', alpha=0.4, linewidth=1.5)
ax.axvline(np.mean(v), color='red', ls='-', linewidth=2,
           label=f'mean = {np.mean(v):.3f}')
ax.axvline(np.median(v), color='orange', ls='--', linewidth=2,
           label=f'median = {np.median(v):.3f}')
try:
    _, p = wilcoxon(v)
except Exception:
    p = 1.0
ax.set_xlabel('Within-session Spearman rho')
ax.set_ylabel('# sessions')
ax.set_title(f'RT vs Hebbian slope (pre epoch)\n'
             f'Wilcoxon p = {p:.4f}, n = {len(v)}',
             fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig5_correlation_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 12: Save text report
# ============================================================================

report_path = os.path.join(RESULTS_DIR, 'sliding_window_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("SLIDING WINDOW HEBBIAN INDEX ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Mode: {pairwise_mode}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Sessions: {n_s}\n")
    f.write("=" * 70 + "\n\n")

    # Sign flips
    f.write("SIGN FLIPS ACROSS WINDOWS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  {'epoch':10s} {'no_int':>8s} {'with_int':>8s} {'corr':>8s}\n")
    for ei, ep in enumerate(EPOCH_ORDER):
        f.write(f"  {ep:10s} {np.mean(flips_ni[:, ei]):8.2f} "
                f"{np.mean(flips_wi[:, ei]):8.2f} "
                f"{np.mean(flips_cr[:, ei]):8.2f}\n")
    f.write(f"\n  Total:     {np.mean(total_ni):8.2f} {np.mean(total_wi):8.2f} "
            f"{np.mean(np.sum(flips_cr, axis=1)):8.2f}\n")
    f.write(f"  Fewer flips with intercept: {np.sum(total_wi < total_ni)}/{n_s}\n")
    f.write(f"  More flips with intercept:  {np.sum(total_wi > total_ni)}/{n_s}\n\n")

    # Behavior vs slope
    f.write("BEHAVIOR vs HEBBIAN SLOPE (with intercept)\n")
    f.write("-" * 40 + "\n")
    f.write("Within-session Spearman rho, tested with Wilcoxon signed-rank\n\n")

    f.write(f"  {'beh x epoch':20s} {'mean':>7s} {'median':>7s} {'%>0':>5s} "
            f"{'Wilcoxon p':>10s} {'sig':>4s}\n")
    for bi, bname in enumerate(beh_names):
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = corr_beh_slope[:, bi, ei]
            v = vals[np.isfinite(vals)]
            m = np.mean(v) if len(v) > 0 else np.nan
            md = np.median(v) if len(v) > 0 else np.nan
            fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            f.write(f"  {bname+'_'+ep:20s} {m:+7.3f} {md:+7.3f} {fpos:4.0f}% "
                    f"{p:10.4f} {sig:>4s}\n")
        f.write("\n")

    # Behavior vs intercept
    f.write("\nBEHAVIOR vs INTERCEPT (global dW)\n")
    f.write("-" * 40 + "\n\n")

    f.write(f"  {'beh x epoch':20s} {'mean':>7s} {'median':>7s} {'%>0':>5s} "
            f"{'Wilcoxon p':>10s} {'sig':>4s}\n")
    for bi, bname in enumerate(beh_names):
        for ei, ep in enumerate(EPOCH_ORDER):
            vals = corr_beh_intercept[:, bi, ei]
            v = vals[np.isfinite(vals)]
            m = np.mean(v) if len(v) > 0 else np.nan
            md = np.median(v) if len(v) > 0 else np.nan
            fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            f.write(f"  {bname+'_'+ep:20s} {m:+7.3f} {md:+7.3f} {fpos:4.0f}% "
                    f"{p:10.4f} {sig:>4s}\n")
        f.write("\n")

    # Per-session summary
    f.write("\nPER-SESSION SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"  {'mouse':8s} {'session':12s} {'trials':>6s} {'wins':>5s} "
            f"{'flips_ni':>8s} {'flips_wi':>8s}\n")
    for s in all_results:
        f.write(f"  {s['mouse']:8s} {s['session']:12s} {s['n_trials']:6d} "
                f"{s['n_windows']:5d} "
                f"{int(np.sum(s['flips_no_int'])):8d} "
                f"{int(np.sum(s['flips_with_int'])):8d}\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 13: Summary matrix figures (coefficient-matrix style)
# ============================================================================
import matplotlib.colors as mcolors

n_s = len(all_results)
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
EPOCH_ORDER = ['pre', 'go_cue', 'late', 'reward']
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']
n_beh = len(beh_names)
n_epochs = len(EPOCH_ORDER)

# Build mean rho and p-value matrices: (n_beh, n_epochs)
mean_slope = np.full((n_beh, n_epochs), np.nan)
mean_intercept = np.full((n_beh, n_epochs), np.nan)
pval_slope = np.full((n_beh, n_epochs), 1.0)
pval_intercept = np.full((n_beh, n_epochs), 1.0)

for bi in range(n_beh):
    for ei in range(n_epochs):
        vs = corr_beh_slope[:, bi, ei]
        vs = vs[np.isfinite(vs)]
        vi = corr_beh_intercept[:, bi, ei]
        vi = vi[np.isfinite(vi)]
        if len(vs) >= 5:
            mean_slope[bi, ei] = np.mean(vs)
            try:
                _, pval_slope[bi, ei] = wilcoxon(vs)
            except Exception:
                pass
        if len(vi) >= 5:
            mean_intercept[bi, ei] = np.mean(vi)
            try:
                _, pval_intercept[bi, ei] = wilcoxon(vi)
            except Exception:
                pass

# ---- Figure: side-by-side coefficient matrices ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
vlim = 0.25

for ax, mat, pmat, title in [
    (axes[0], mean_slope, pval_slope, 'Behavior vs Slope\n(activity-dependent)'),
    (axes[1], mean_intercept, pval_intercept, 'Behavior vs Intercept\n(global plasticity)'),
]:
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-vlim, vmax=vlim)

    # Annotate values and significance
    for bi in range(n_beh):
        for ei in range(n_epochs):
            val = mat[bi, ei]
            p = pmat[bi, ei]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            txt = f'{val:+.2f}\n{sig}' if sig else f'{val:+.2f}'
            # White text on dark cells, black on light
            textcolor = 'white' if abs(val) > vlim * 0.6 else 'black'
            ax.text(ei, bi, txt, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=textcolor)

    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels(epoch_labels, fontsize=13)
    ax.set_yticks(range(n_beh))
    ax.set_yticklabels(beh_labels, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Mean within-session rho', fontsize=11)

fig.suptitle(f'Behavioral modulation of plasticity components (n={n_s} sessions)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig6_coefficient_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 6 saved.")

#%% ============================================================================
# CELL 14: Per-mouse coefficient matrices
# ============================================================================

unique_mice = sorted(set(s['mouse'] for s in all_results))
n_mice = len(unique_mice)

fig, axes = plt.subplots(2, n_mice, figsize=(4 * n_mice, 8))
if n_mice == 1:
    axes = axes[:, np.newaxis]

for mi, mouse in enumerate(unique_mice):
    # Get session indices for this mouse
    mouse_idx = [i for i, s in enumerate(all_results) if s['mouse'] == mouse]
    n_sess = len(mouse_idx)

    for row, (corr_arr, label) in enumerate([
        (corr_beh_slope, 'Slope'),
        (corr_beh_intercept, 'Intercept'),
    ]):
        ax = axes[row, mi]
        mat = np.full((n_beh, n_epochs), np.nan)
        for bi in range(n_beh):
            for ei in range(n_epochs):
                vals = corr_arr[mouse_idx, bi, ei]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    mat[bi, ei] = np.mean(vals)

        im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-vlim, vmax=vlim)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                val = mat[bi, ei]
                if np.isfinite(val):
                    textcolor = 'white' if abs(val) > vlim * 0.6 else 'black'
                    ax.text(ei, bi, f'{val:+.2f}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=textcolor)

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, fontsize=10)
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels if mi == 0 else [], fontsize=10)

        if row == 0:
            ax.set_title(f'{mouse}\n(n={n_sess})', fontsize=13, fontweight='bold')
        if mi == 0:
            ax.set_ylabel(label, fontsize=13, fontweight='bold')

fig.suptitle('Behavioral modulation by mouse', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig7_coefficient_matrices_by_mouse.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 7 saved.")

#%% ============================================================================
# CELL 15: Distributions of raw intercept and slope values
# ============================================================================

# Collect all intercept and slope values across all sessions and windows
# Two versions: raw and z-scored within session (for binned plots)
all_intercepts_raw = []
all_slopes_raw = []
all_rpe_raw = []
all_intercepts_z = []
all_slopes_z = []
all_rpe_z = []

def _zscore(x):
    x = x.copy()
    ok = np.isfinite(x)
    if np.sum(ok) < 2 or np.std(x[ok]) == 0:
        return x * np.nan
    x[ok] = (x[ok] - np.mean(x[ok])) / np.std(x[ok])
    return x

for s in all_results:
    rpe = s['win_rpe']
    rpe_z = _zscore(rpe)
    for ei in range(n_epochs):
        intc = s['hi_intercept'][:, ei]
        slp = s['hi_with_int'][:, ei]
        ok = np.isfinite(intc) & np.isfinite(slp) & np.isfinite(rpe)
        all_intercepts_raw.append(intc[ok])
        all_slopes_raw.append(slp[ok])
        all_rpe_raw.append(rpe[ok])
        # Z-scored within session
        intc_z = _zscore(intc)
        slp_z = _zscore(slp)
        ok_z = np.isfinite(intc_z) & np.isfinite(slp_z) & np.isfinite(rpe_z)
        all_intercepts_z.append(intc_z[ok_z])
        all_slopes_z.append(slp_z[ok_z])
        all_rpe_z.append(rpe_z[ok_z])

all_intercepts_z = np.concatenate(all_intercepts_z)
all_slopes_z = np.concatenate(all_slopes_z)
all_rpe_z = np.concatenate(all_rpe_z)

all_intercepts_raw = np.concatenate(all_intercepts_raw)
all_slopes_raw = np.concatenate(all_slopes_raw)
all_rpe_raw = np.concatenate(all_rpe_raw)

# Per-session means (one intercept and slope per session per epoch)
sess_intercepts = np.full((n_s, n_epochs), np.nan)
sess_slopes = np.full((n_s, n_epochs), np.nan)
for si, s in enumerate(all_results):
    for ei in range(n_epochs):
        sess_intercepts[si, ei] = np.nanmean(s['hi_intercept'][:, ei])
        sess_slopes[si, ei] = np.nanmean(s['hi_with_int'][:, ei])

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
n_rpe_bins = 15

def _binned_plot(ax, x, y, color, xlabel, ylabel, title):
    """Helper for binned mean +/- SEM plot."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    bin_edges = np.percentile(x, np.linspace(0, 100, n_rpe_bins + 1))
    centers, means, sems = [], [], []
    for bi_idx in range(n_rpe_bins):
        if bi_idx < n_rpe_bins - 1:
            mask = (x >= bin_edges[bi_idx]) & (x < bin_edges[bi_idx + 1])
        else:
            mask = (x >= bin_edges[bi_idx]) & (x <= bin_edges[bi_idx + 1])
        if np.sum(mask) > 0:
            centers.append(np.mean(x[mask]))
            means.append(np.mean(y[mask]))
            sems.append(np.std(y[mask]) / np.sqrt(np.sum(mask)))
    ax.errorbar(centers, means, yerr=sems,
                fmt='o-', color=color, capsize=4, markersize=6, linewidth=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')

# Row 1: Intercept
ax = axes[0, 0]
ax.hist(all_intercepts_raw, bins=60, color='#1f77b4', alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', ls='--', linewidth=1.5)
ax.axvline(np.mean(all_intercepts_raw), color='red', ls='-', linewidth=2,
           label=f'mean = {np.mean(all_intercepts_raw):.4f}')
ax.axvline(np.median(all_intercepts_raw), color='orange', ls='--', linewidth=2,
           label=f'median = {np.median(all_intercepts_raw):.4f}')
pct_pos = np.mean(all_intercepts_raw > 0) * 100
ax.set_title(f'Intercept (all windows)\n{pct_pos:.0f}% positive', fontweight='bold')
ax.set_xlabel('Intercept (mean ΔW)')
ax.set_ylabel('# windows')
ax.legend(fontsize=10)

ax = axes[0, 1]
for ei, ep in enumerate(EPOCH_ORDER):
    vals = sess_intercepts[:, ei]
    ax.hist(vals[np.isfinite(vals)], bins=20, alpha=0.5,
            label=f'{ep} ({np.nanmean(vals > 0)*100:.0f}%+ )')
ax.axvline(0, color='k', ls='--', linewidth=1.5)
ax.set_title('Intercept: session means by epoch', fontweight='bold')
ax.set_xlabel('Mean intercept')
ax.set_ylabel('# sessions')
ax.legend(fontsize=10)

_binned_plot(axes[0, 2], all_rpe_raw, all_intercepts_raw, '#1f77b4',
             'RPE (raw)', 'Mean intercept',
             'Intercept vs RPE\n(raw pooled)')

_binned_plot(axes[0, 3], all_rpe_z, all_intercepts_z, '#1f77b4',
             'RPE (z-scored within session)', 'Mean intercept (z)',
             'Intercept vs RPE\n(within-session)')

# Row 2: Slope
ax = axes[1, 0]
ax.hist(all_slopes_raw, bins=60, color='#2ca02c', alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', ls='--', linewidth=1.5)
ax.axvline(np.mean(all_slopes_raw), color='red', ls='-', linewidth=2,
           label=f'mean = {np.mean(all_slopes_raw):.4f}')
ax.axvline(np.median(all_slopes_raw), color='orange', ls='--', linewidth=2,
           label=f'median = {np.median(all_slopes_raw):.4f}')
pct_pos = np.mean(all_slopes_raw > 0) * 100
ax.set_title(f'Slope (all windows)\n{pct_pos:.0f}% positive', fontweight='bold')
ax.set_xlabel('Slope (activity-dependent ΔW)')
ax.set_ylabel('# windows')
ax.legend(fontsize=10)

ax = axes[1, 1]
for ei, ep in enumerate(EPOCH_ORDER):
    vals = sess_slopes[:, ei]
    ax.hist(vals[np.isfinite(vals)], bins=20, alpha=0.5,
            label=f'{ep} ({np.nanmean(vals > 0)*100:.0f}%+ )')
ax.axvline(0, color='k', ls='--', linewidth=1.5)
ax.set_title('Slope: session means by epoch', fontweight='bold')
ax.set_xlabel('Mean slope')
ax.set_ylabel('# sessions')
ax.legend(fontsize=10)

_binned_plot(axes[1, 2], all_rpe_raw, all_slopes_raw, '#2ca02c',
             'RPE (raw)', 'Mean slope',
             'Slope vs RPE\n(raw pooled)')

_binned_plot(axes[1, 3], all_rpe_z, all_slopes_z, '#2ca02c',
             'RPE (z-scored within session)', 'Mean slope (z)',
             'Slope vs RPE\n(within-session)')

plt.suptitle('Raw distributions of intercept and slope', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig8_intercept_slope_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("\n=== INTERCEPT DISTRIBUTION ===")
print(f"  All windows: mean={np.mean(all_intercepts_raw):.5f}, "
      f"median={np.median(all_intercepts_raw):.5f}, "
      f"{np.mean(all_intercepts_raw > 0)*100:.1f}% positive")
for ei, ep in enumerate(EPOCH_ORDER):
    vals = sess_intercepts[:, ei]
    vals = vals[np.isfinite(vals)]
    print(f"  {ep:8s} session means: mean={np.mean(vals):.5f}, "
          f"median={np.median(vals):.5f}, {np.mean(vals > 0)*100:.0f}% positive")

print("\n=== SLOPE DISTRIBUTION ===")
print(f"  All windows: mean={np.mean(all_slopes_raw):.5f}, "
      f"median={np.median(all_slopes_raw):.5f}, "
      f"{np.mean(all_slopes_raw > 0)*100:.1f}% positive")
for ei, ep in enumerate(EPOCH_ORDER):
    vals = sess_slopes[:, ei]
    vals = vals[np.isfinite(vals)]
    print(f"  {ep:8s} session means: mean={np.mean(vals):.5f}, "
          f"median={np.median(vals):.5f}, {np.mean(vals > 0)*100:.0f}% positive")

print(f"\n=== INTERCEPT-SLOPE CORRELATION (pooled) ===")
ok = np.isfinite(all_intercepts_raw) & np.isfinite(all_slopes_raw)
r, p = pearsonr(all_intercepts_raw[ok], all_slopes_raw[ok])
print(f"  Pearson r = {r:.3f}, p = {p:.2e}")

#%% ============================================================================
# CELL 16: Between-session RPE vs slope/intercept, colored by mouse
# ============================================================================

# Session-level means
sess_mean_rpe = np.array([np.nanmean(s['win_rpe']) for s in all_results])
sess_mean_slope = np.array([np.nanmean(s['hi_with_int']) for s in all_results])
sess_mean_intercept = np.array([np.nanmean(s['hi_intercept']) for s in all_results])
sess_mouse = np.array([s['mouse'] for s in all_results])

unique_mice = sorted(set(sess_mouse))
mouse_colors = {m: c for m, c in zip(unique_mice,
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
     '#e377c2', '#7f7f7f'][:len(unique_mice)])}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, yvals, ylabel, title in [
    (axes[0], sess_mean_slope, 'Mean slope', 'Between-session: RPE vs Slope'),
    (axes[1], sess_mean_intercept, 'Mean intercept', 'Between-session: RPE vs Intercept'),
]:
    for mouse in unique_mice:
        mask = sess_mouse == mouse
        ax.scatter(sess_mean_rpe[mask], yvals[mask], s=60, alpha=0.7,
                   color=mouse_colors[mouse], label=mouse, edgecolors='black',
                   linewidth=0.5)
    # Label every point with mouse + session
    for si in range(len(all_results)):
        ax.annotate(f"{all_results[si]['mouse'][-3:]}-{all_results[si]['session']}",
                    (sess_mean_rpe[si], yvals[si]),
                    fontsize=6, alpha=0.7, ha='left', va='bottom',
                    xytext=(2, 2), textcoords='offset points')

    # Overall regression line
    ok = np.isfinite(sess_mean_rpe) & np.isfinite(yvals)
    if np.sum(ok) > 2:
        coeffs = np.polyfit(sess_mean_rpe[ok], yvals[ok], 1)
        xline = np.linspace(np.min(sess_mean_rpe[ok]), np.max(sess_mean_rpe[ok]), 50)
        ax.plot(xline, np.polyval(coeffs, xline), 'k--', alpha=0.5, linewidth=1.5)
        r_val, p_val = pearsonr(sess_mean_rpe[ok], yvals[ok])
        ax.text(0.05, 0.95, f'r = {r_val:.2f}, p = {p_val:.3f}',
                transform=ax.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axhline(0, color='k', ls='-', alpha=0.2)
    ax.set_xlabel('Session mean RPE')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig9_between_session_rpe.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# Print per-mouse correlations
print("\nPer-mouse between-session correlations (RPE vs slope):")
for mouse in unique_mice:
    mask = sess_mouse == mouse
    ok = np.isfinite(sess_mean_rpe[mask]) & np.isfinite(sess_mean_slope[mask])
    n = np.sum(ok)
    if n >= 4:
        r_val, p_val = pearsonr(sess_mean_rpe[mask][ok], sess_mean_slope[mask][ok])
        print(f"  {mouse}: r = {r_val:+.2f}, p = {p_val:.3f} (n={n})")
    else:
        print(f"  {mouse}: n={n}, too few sessions")

print("\nPer-mouse between-session correlations (RPE vs intercept):")
for mouse in unique_mice:
    mask = sess_mouse == mouse
    ok = np.isfinite(sess_mean_rpe[mask]) & np.isfinite(sess_mean_intercept[mask])
    n = np.sum(ok)
    if n >= 4:
        r_val, p_val = pearsonr(sess_mean_rpe[mask][ok], sess_mean_intercept[mask][ok])
        print(f"  {mouse}: r = {r_val:+.2f}, p = {p_val:.3f} (n={n})")
    else:
        print(f"  {mouse}: n={n}, too few sessions")
