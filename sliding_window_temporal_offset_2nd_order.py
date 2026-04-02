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

# ---- Global plot style ----
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

# Sliding window parameters
WIN_SIZE = 10    # trials per window
WIN_STEP = 5     # step between windows
tau_elig = 10

# Temporal offset in seconds (pre leads post)
OFFSET_SEC = 0

# Baseline trials for dev2 mode
N_BASELINE = 20

# CC modes:
#   'dot_prod_lag'  — sum_t pre(t) * post(t + lag), using full trial F data
#   'dev2_lag'      — sum_t pre(t) * (post(t + lag) - mean_post_baseline)
#   'dev2_network'  — sum_t net_input_pre(t) * (post(t) - baseline)
#                     where net_input_pre(t) = sum_g AMP0[cl,g] * r_g(t)
#                     This is the second-order gradient term.
CC_MODES = ['dot_prod_lag', 'dev2_lag', 'dev2_network']

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s (pre leads post)")
print(f"CC modes: {CC_MODES}")

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
            F = data['F']  # (timepoints, neurons, trials)
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            # Temporal offset in frames
            lag_frames = int(round(OFFSET_SEC / dt_si))
            print(f"  dt_si={dt_si:.4f}s, lag={lag_frames} frames ({lag_frames*dt_si:.2f}s)")

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

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

            # ---- Build pair selection ----
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.1
            amp1_thr = 0.1

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []
            pair_gi_list = []  # stim group index per pair block

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
                pair_gi_list.append(np.full(len(nontarg), gi, dtype=int))

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            all_gi = np.concatenate(pair_gi_list)
            n_pairs = len(Y_T)

            # cl averaging weights: (n_pairs, n_neurons)
            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # ---- Network input weights for second-order term ----
            # For each pair, the "presynaptic" term in the 2nd-order gradient is
            # sum_k w_{jk} r_k(t), where j is the cl neuron.
            # w_{jk} = how cl neuron j responds to stim of group g' targeting neuron k.
            # We approximate: for each pair's cl neurons, average their
            # pre-session evoked responses to ALL other stim groups.
            # net_input_weights[pair, :] = mean over cl neurons of AMP[0][cl, :]
            # Then net_input_pre(t) = net_input_weights @ r_all(t)
            # But we want the input from other neurons, not self-stimulation.
            # So for each pair, we use AMP[0][cl, g'] for all g' != pair's own group,
            # weighted by the activity of group g's target neurons.
            #
            # Simpler approach: AMP[0] gives us w_{neuron, stim_group}.
            # For cl neuron j, AMP[0][j, :] is how j responds to each stim group.
            # net_input to j on trial t = sum_{g'} AMP[0][j, g'] * mean(r_{targets_of_g'}(t))
            #
            # We precompute: for each stim group g', identify its target neurons,
            # then net_input_weights is (n_pairs, n_stim_groups) and we multiply
            # by (n_stim_groups, trl) target activities.

            n_groups = stimDist.shape[1]

            # Target neuron indices per stim group (close neurons)
            group_targets = []
            for gi in range(n_groups):
                targs = np.where(stimDist[:, gi] < dist_target_lt)[0]
                group_targets.append(targs)

            # For each pair: connectivity from cl neurons to each stim group
            # net_w[pair, g'] = mean over cl neurons of AMP[0][cl, g']
            net_w = np.zeros((n_pairs, n_groups))
            for pi in range(n_pairs):
                cl_neurons = np.where(cl_weights[pi, :] > 0)[0]
                if len(cl_neurons) > 0:
                    net_w[pi, :] = np.mean(AMP[0][cl_neurons, :], axis=0)

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # ---- Prepare F for epoch computation ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            # ---- Epoch time indices ----
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            # ---- Compute lagged epoch averages ----
            # Same epoch structure as before, but pre and post use offset time windows.
            # Pre: average F over [epoch_start, epoch_end]
            # Post: average F over [epoch_start + lag, epoch_end + lag]
            # This gives one scalar per neuron per trial per epoch, same as before.

            EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
            n_epochs = len(EPOCH_ORDER)

            # Compute lagged epoch activity: pre neuron uses original window,
            # post neuron uses window shifted by lag_frames
            # For simplicity: average pre over epoch, average post over epoch+lag
            # Shape: (n_neurons, trl) for each

            epoch_pre_act = {}   # presynaptic: original epoch window
            epoch_post_act = {}  # postsynaptic: epoch window + lag

            for ep in ['pre', 'go_cue']:
                if ep == 'pre':
                    t0, t1 = ts_pre[0], ts_pre[-1]
                else:
                    t0, t1 = ts_go[0], ts_go[-1]
                t0_lag = max(0, min(t0 + lag_frames, n_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, n_frames - 1))
                epoch_pre_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)  # (N, trl)
                epoch_post_act[ep] = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

            # Late and reward: per-trial epoch windows
            epoch_pre_act['late'] = np.zeros((n_neurons, trl))
            epoch_post_act['late'] = np.zeros((n_neurons, trl))
            epoch_pre_act['reward'] = np.zeros((n_neurons, trl))
            epoch_post_act['reward'] = np.zeros((n_neurons, trl))

            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    # Late (pre-reward)
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_pre_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        indices_lag = indices + lag_frames
                        indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                        if len(indices_lag) > 0:
                            epoch_post_act['late'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

                    # Reward (post-reward)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_pre_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        indices_lag = indices + lag_frames
                        indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                        if len(indices_lag) > 0:
                            epoch_post_act['reward'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

            # Baseline post mean per epoch (for dev2)
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            baseline_post_mean_ep = {}
            for ep in EPOCH_ORDER:
                baseline_post_mean_ep[ep] = np.nanmean(
                    epoch_post_act[ep][:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # ---- Compute per-group target activity for network input mode ----
            # group_act[ep]: (n_groups, trl) = mean activity of each group's targets
            group_act_ep = {}
            for ep in EPOCH_ORDER:
                ga = np.zeros((n_groups, trl))
                for gi in range(n_groups):
                    targs = group_targets[gi]
                    if len(targs) > 0:
                        ga[gi, :] = np.mean(epoch_pre_act[ep][targs, :], axis=0)
                group_act_ep[ep] = ga

            # Baseline for network input (dev2 on the network-input pre term)
            baseline_net_mean_ep = {}
            for ep in EPOCH_ORDER:
                # net_input per pair per trial: net_w @ group_act  (n_pairs, trl)
                net_input_all = net_w @ group_act_ep[ep]
                baseline_net_mean_ep[ep] = np.nanmean(
                    net_input_all[:, baseline_trials_arr], axis=1)  # (n_pairs,)

            # ---- Compute CC per window per epoch (same structure as v2) ----
            raw_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            dev2_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            network_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)

            win_hit = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            win_center = np.full(n_wins, np.nan)

            for wi, ws in enumerate(win_starts):
                we = ws + WIN_SIZE
                trial_idx = np.arange(ws, we)
                win_center[wi] = (ws + we) / 2.0
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])

                for ei, ep in enumerate(EPOCH_ORDER):
                    # Pre activity: original epoch window; Post: shifted by lag
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]   # (n_pairs, win_size)
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]   # (n_pairs, win_size)
                    cc_raw = np.sum(pre_act * post_act, axis=1)              # (n_pairs,)
                    raw_cc[wi, :, ei] = cc_raw

                    # dev2: pre(t) * (post(t+lag) - mean_post_baseline)
                    post_dev = epoch_post_act[ep][all_nt, :][:, trial_idx] - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    cc_dev2 = np.sum(pre_act * post_dev, axis=1)
                    dev2_cc[wi, :, ei] = cc_dev2

                    # network: net_input_pre(t) * (post(t) - baseline)
                    # net_input_pre = net_w @ group_act for this epoch/trials
                    net_pre = net_w @ group_act_ep[ep][:, trial_idx]  # (n_pairs, win_size)
                    cc_net = np.sum(net_pre * post_dev, axis=1)
                    network_cc[wi, :, ei] = cc_net

            # ---- Fit slope/intercept for each mode per epoch ----
            for mode in CC_MODES:
                hi_no_int = np.full((n_wins, n_epochs), np.nan)
                hi_with_int = np.full((n_wins, n_epochs), np.nan)
                hi_intercept = np.full((n_wins, n_epochs), np.nan)
                hi_corr = np.full((n_wins, n_epochs), np.nan)

                for ei, ep in enumerate(EPOCH_ORDER):
                    if mode == 'dot_prod_lag':
                        cc_all = raw_cc[:, :, ei]
                    elif mode == 'dev2_lag':
                        cc_all = dev2_cc[:, :, ei]
                    elif mode == 'dev2_network':
                        cc_all = network_cc[:, :, ei]

                    for wi in range(n_wins):
                        cc_pair = cc_all[wi, :]
                        if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                            continue

                        hi_no_int[wi, ei] = np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair)

                        A = np.column_stack([np.ones(n_pairs), cc_pair])
                        coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                        hi_intercept[wi, ei] = coeffs[0]
                        hi_with_int[wi, ei] = coeffs[1]

                        hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

                def count_flips(arr):
                    signs = np.sign(arr)
                    valid = np.isfinite(signs)
                    s = signs[valid]
                    if len(s) < 2:
                        return 0
                    return int(np.sum(s[1:] != s[:-1]))

                result = {
                    'mouse': mouse,
                    'session': session,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
                    'n_frames': n_frames,
                    'lag_frames': lag_frames,
                    'lag_sec': lag_frames * dt_si,
                    'dt_si': dt_si,
                    'win_centers': win_center,
                    'hit_rate': np.nanmean(hit),
                    'hi_no_int': hi_no_int,
                    'hi_with_int': hi_with_int,
                    'hi_intercept': hi_intercept,
                    'hi_corr': hi_corr,
                    'win_hit': win_hit,
                    'win_rpe': win_rpe,
                    'win_rt': win_rt,
                    'win_hit_rpe': win_hit_rpe,
                }
                all_results[mode].append(result)

            print(f"  {n_wins} windows, {n_pairs} pairs")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

for mode in CC_MODES:
    print(f"{mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded modes: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 6: Compute within-session correlations
# ============================================================================
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

corr_slope = {}
corr_intercept = {}

for mode in CC_MODES:
    results = all_results[mode]
    n_s = len(results)
    cs = np.full((n_s, n_beh, n_epochs), np.nan)
    ci = np.full((n_s, n_beh, n_epochs), np.nan)

    for si, s in enumerate(results):
        for bi, bname in enumerate(beh_names):
            bvar = get_beh(s, bname)
            if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
                continue
            for ei in range(n_epochs):
                slope = s['hi_with_int'][:, ei]
                intercept = s['hi_intercept'][:, ei]
                ok = np.isfinite(bvar) & np.isfinite(slope)
                if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
                    cs[si, bi, ei], _ = spearmanr(bvar[ok], slope[ok])
                ok2 = np.isfinite(bvar) & np.isfinite(intercept)
                if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
                    ci[si, bi, ei], _ = spearmanr(bvar[ok2], intercept[ok2])

    corr_slope[mode] = cs
    corr_intercept[mode] = ci

print("Within-session correlations computed.")

#%% ============================================================================
# CELL 7: Coefficient matrices — behavior x epoch (same style as v2)
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(5 * len(CC_MODES), 6),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    n_s = len(all_results[mode])

    for row, (corr_arr, row_label) in enumerate([
        (corr_slope[mode], 'Slope'),
        (corr_intercept[mode], 'Intercept'),
    ]):
        ax = axes[row, col]
        mat_mean = np.full((n_beh, n_epochs), np.nan)
        mat_p = np.full((n_beh, n_epochs), np.nan)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                vals = corr_arr[:, bi, ei]
                v = vals[np.isfinite(vals)]
                if len(v) < 3:
                    continue
                mat_mean[bi, ei] = np.mean(v)
                try:
                    _, p = wilcoxon(v)
                except Exception:
                    p = 1.0
                mat_p[bi, ei] = p

        vmax = np.nanmax(np.abs(mat_mean)) if np.any(np.isfinite(mat_mean)) else 0.2
        vmax = max(vmax, 0.05)
        im = ax.imshow(mat_mean, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        # Annotate with values and significance stars
        for bi in range(n_beh):
            for ei in range(n_epochs):
                val = mat_mean[bi, ei]
                p = mat_p[bi, ei]
                if np.isnan(val):
                    continue
                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                txt = f'{val:+.3f}'
                if sig:
                    txt += f'\n{sig}'
                ax.text(ei, bi, txt, ha='center', va='center',
                        fontsize=9, fontweight='bold' if sig else 'normal')

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean rho')

        if row == 0:
            if mode == 'dev2_network':
                disp = 'dev2_network (2nd-order)'
            else:
                disp = mode.replace('_lag', f' (lag={OFFSET_SEC}s)')
            ax.set_title(f'{disp}\n({row_label})', fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=12)

lag_str = f"{OFFSET_SEC}s"
fig.suptitle(f'Temporal offset: pre leads post by {lag_str} (n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig12_temporal_offset_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 12 saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'temporal_offset_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("TEMPORAL OFFSET COACTIVITY ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s (pre leads post)\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write(f"  dot_prod_lag  : sum_t pre(t) * post(t + lag), epoch-averaged\n")
    f.write(f"  dev2_lag      : sum_t pre(t) * (post(t+lag) - mean_post_baseline)\n")
    f.write(f"  dev2_network  : sum_t [sum_g w_jg * r_g(t)] * (post(t) - baseline)\n")
    f.write(f"                  2nd-order gradient term: network input to pre neuron\n\n")

    epoch_labels_rpt = ['pre', 'go_cue', 'late', 'reward']

    for mode in CC_MODES:
        n_s = len(all_results[mode])
        f.write(f"\n{'='*50}\n")
        f.write(f"MODE: {mode}  ({n_s} sessions)\n")
        f.write(f"{'='*50}\n\n")

        for target, corr_arr, label in [
            ('slope', corr_slope[mode], 'BEHAVIOR vs SLOPE'),
            ('intercept', corr_intercept[mode], 'BEHAVIOR vs INTERCEPT'),
        ]:
            f.write(f"{label}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  {'beh x epoch':25s} {'mean':>7s} {'median':>7s} "
                    f"{'%>0':>5s} {'Wilcoxon p':>10s} {'sig':>4s}\n")

            for bi, bname in enumerate(beh_names):
                for ei, ep in enumerate(epoch_labels_rpt):
                    vals = corr_arr[:, bi, ei]
                    v = vals[np.isfinite(vals)]
                    m = np.mean(v) if len(v) > 0 else np.nan
                    md = np.median(v) if len(v) > 0 else np.nan
                    fpos = np.mean(v > 0) * 100 if len(v) > 0 else np.nan
                    try:
                        _, p = wilcoxon(v)
                    except Exception:
                        p = 1.0
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    row_name = f"{bname}_{ep}"
                    f.write(f"  {row_name:25s} {m:+7.3f} {md:+7.3f} "
                            f"{fpos:4.0f}% {p:10.4f} {sig:>4s}\n")
                f.write("\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 9: Binned scatter — within-session z-scored RPE vs Slope (dev2, pre epoch)
# ============================================================================
mode_plot = 'dev2_lag'
ei_plot = 0  # pre epoch

# Collect within-session z-scored RPE and slope
all_rpe_z = []
all_slope_z = []

for s in all_results[mode_plot]:
    rpe = s['win_rpe']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rpe) & np.isfinite(slope)
    if np.sum(ok) < 5:
        continue
    rpe_ok = rpe[ok]
    slope_ok = slope[ok]
    if np.std(rpe_ok) == 0 or np.std(slope_ok) == 0:
        continue
    all_rpe_z.append((rpe_ok - np.mean(rpe_ok)) / np.std(rpe_ok))
    all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))

all_rpe_z = np.concatenate(all_rpe_z)
all_slope_z = np.concatenate(all_slope_z)

# Bin by z-scored RPE
n_bins = 3
bin_edges = np.percentile(all_rpe_z, np.linspace(0, 100, n_bins + 1))
bin_centers = []
bin_means = []
bin_sems = []

for bi in range(n_bins):
    if bi < n_bins - 1:
        mask = (all_rpe_z >= bin_edges[bi]) & (all_rpe_z < bin_edges[bi + 1])
    else:
        mask = (all_rpe_z >= bin_edges[bi]) & (all_rpe_z <= bin_edges[bi + 1])
    if np.sum(mask) < 3:
        continue
    bin_centers.append(np.mean(all_rpe_z[mask]))
    bin_means.append(np.mean(all_slope_z[mask]))
    bin_sems.append(np.std(all_slope_z[mask]) / np.sqrt(np.sum(mask)))

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_sems = np.array(bin_sems)

# Find session ranked by RPE-slope correlation (1 = best, 2 = second best, etc.)
RANK = 5

all_corrs = []
for si, s in enumerate(all_results[mode_plot]):
    rpe = s['win_rpe']
    slope = s['hi_with_int'][:, ei_plot]
    ok = np.isfinite(rpe) & np.isfinite(slope)
    if np.sum(ok) >= 5 and np.std(slope[ok]) > 0 and np.std(rpe[ok]) > 0:
        r, _ = spearmanr(rpe[ok], slope[ok])
        all_corrs.append((r, si))
    else:
        all_corrs.append((np.nan, si))

all_corrs.sort(key=lambda x: -x[0] if np.isfinite(x[0]) else np.inf)
best_corr, best_idx = all_corrs[RANK - 1]
best_s = all_results[mode_plot][best_idx]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: binned scatter
ax = axes[0]
ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-',
            color='#2c3e50', capsize=5, linewidth=2, markersize=7)
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.axvline(0, color='k', ls='--', alpha=0.3)
ax.set_xlabel('RPE (within-session z-score)')
ax.set_ylabel('Slope (within-session z-score)')
ax.set_title(f'dev2 lag={OFFSET_SEC}s — Pre epoch\nRPE vs HI slope (n={len(all_results[mode_plot])} sessions)',
             fontsize=13, fontweight='bold')

# Right: time series for best session
ax2 = axes[1]
wc = best_s['win_centers']
rpe_ts = best_s['win_rpe']
slope_ts = best_s['hi_with_int'][:, ei_plot]

ax2.plot(wc, (rpe_ts - np.nanmean(rpe_ts)) / np.nanstd(rpe_ts),
         'o-', color='#e74c3c', label='RPE', linewidth=2, markersize=4)
ax2.plot(wc, (slope_ts - np.nanmean(slope_ts)) / np.nanstd(slope_ts),
         'o-', color='#2c3e50', label='HI slope', linewidth=2, markersize=4)
ax2.axhline(0, color='k', ls='-', alpha=0.3)
ax2.set_xlabel('Trial (window center)')
ax2.set_ylabel('z-score')
ax2.legend(loc='best')
ax2.set_title(f'{best_s["mouse"]} {best_s["session"]}\nrho={best_corr:.3f}',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig14_rpe_vs_slope_binned.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 14 saved.")

#%% ============================================================================
# CELL 10: Single-session dW vs CC scatter, split by RPE (re-computes CC)
# ============================================================================
# Uses best_s from Cell 9 (the RANK-th best session)
ex_mouse = best_s['mouse']
ex_session = best_s['session']
ex_lag_sec = best_s['lag_sec']
ei_plot_10 = 0  # pre epoch

print(f"Re-computing CC for {ex_mouse} {ex_session}, lag={ex_lag_sec}s ...")

# --- Reload session data ---
folder = (r'//allen/aind/scratch/BCI/2p-raw/'
          + ex_mouse + r'/' + ex_session + '/pophys/')
photostim_keys = ['stimDist', 'favg_raw']
bci_keys = [
    'df_closedloop', 'F', 'mouse', 'session',
    'conditioned_neuron', 'dt_si', 'step_time',
    'reward_time', 'BCI_thresholds',
]
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)

BCI_thresholds = np.asarray(data['BCI_thresholds'], dtype=float)
thr = BCI_thresholds[1, :]
for i in range(1, thr.size):
    if np.isnan(thr[i]):
        thr[i] = thr[i - 1]
if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
    thr[0] = thr[np.isfinite(thr)][0]
BCI_thresholds[1, :] = thr

AMP, stimDist = compute_amp_from_photostim(ex_mouse, data, folder)
dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
n_neurons = F.shape[1]
n_frames = F.shape[0]
tsta = np.arange(0, 12, dt_si)
tsta = tsta - tsta[int(2 / dt_si)]
lag_frames = int(round(ex_lag_sec / dt_si))

data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
hit = np.isfinite(rt)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)

# Pair selection
dw_list = []
pair_cl_list = []
pair_nt_list = []
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

# Compute lagged epoch activity for pre epoch only
F_nan = F.copy()
F_nan[np.isnan(F_nan)] = 0
ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
t0e, t1e = ts_pre[0], ts_pre[-1]
t0_lag = max(0, min(t0e + lag_frames, n_frames - 1))
t1_lag = max(0, min(t1e + lag_frames, n_frames - 1))
epoch_pre = np.nanmean(F_nan[t0e:t1e+1, :, :], axis=0)   # (N, trl)
epoch_post = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

# Baseline for dev2
baseline_trials_arr = np.arange(min(N_BASELINE, trl))
bl_post_mean = np.nanmean(epoch_post[:, baseline_trials_arr], axis=1)  # (N,)

# Compute CC per window
win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
n_wins = len(win_starts)

cc_per_win = np.full((n_wins, n_pairs), np.nan)
rpe_per_win = np.full(n_wins, np.nan)

for wi, ws in enumerate(win_starts):
    trial_idx = np.arange(ws, ws + WIN_SIZE)
    rpe_per_win[wi] = np.nanmean(rt_rpe[trial_idx])
    pre_act = cl_weights @ epoch_pre[:, trial_idx]
    post_dev = epoch_post[all_nt, :][:, trial_idx] - bl_post_mean[all_nt, np.newaxis]
    cc_per_win[wi, :] = np.sum(pre_act * post_dev, axis=1)

# Split windows by RPE sign (within-session, relative to median)
med_rpe = np.nanmedian(rpe_per_win)
hi_rpe = rpe_per_win >= np.percentile(rpe_per_win,90)
lo_rpe = rpe_per_win < np.percentile(rpe_per_win,10)

# Average CC across windows in each group
cc_hi = np.nanmean(cc_per_win[hi_rpe, :], axis=0)
cc_lo = np.nanmean(cc_per_win[lo_rpe, :], axis=0)

n_bins_10 = 5

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

for ax, cc, label, color in [
    (axes[0], cc_hi, f'RPE > median ({np.sum(hi_rpe)} wins)', '#e74c3c'),
    (axes[1], cc_lo, f'RPE < median ({np.sum(lo_rpe)} wins)', '#3498db'),
]:
    ok = np.isfinite(cc) & np.isfinite(Y_T)
    cc_ok = cc[ok]
    dw_ok = Y_T[ok]

    if len(cc_ok) < n_bins_10:
        continue

    edges = np.percentile(cc_ok, np.linspace(0, 100, n_bins_10 + 1))
    bx, by, be = [], [], []
    for bi in range(n_bins_10):
        if bi < n_bins_10 - 1:
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

    # Fit line on raw data for stats
    if np.std(cc_ok) > 0:
        A = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A, dw_ok, rcond=None)[0]
        xr = np.array([bx[0], bx[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, '--', color='k', linewidth=1.5)
        r, p = spearmanr(cc_ok, dw_ok)
        ax.set_title(f'{label}\nslope={coeffs[1]:.4f}, r={r:.3f}, p={p:.3f}',
                     fontsize=12, fontweight='bold')

    ax.axhline(0, color='k', ls='-', alpha=0.2)
    ax.axvline(0, color='k', ls='--', alpha=0.2)
    ax.set_xlabel('CC (dev2, pre epoch)')
    ax.set_ylabel('dW')

fig.suptitle(f'{ex_mouse} {ex_session} — dW vs CC split by RPE\n(dev2, pre epoch, lag={ex_lag_sec}s)',
             fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig15_dw_vs_cc_rpe_split.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 15 saved.")
