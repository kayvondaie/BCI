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
N_BASELINE = 20

# Range of lags to test (seconds). Positive = pre leads post.
LAG_RANGE = np.arange(-5, 6, 1)

CC_MODES = ['dot_prod_lag', 'dev2_lag']
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Lags to test: {LAG_RANGE}")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop — sweep over lags
# ============================================================================
# For each lag, run the full analysis and store per-session within-session
# correlations (RPE vs slope, RPE vs intercept) for the pre epoch.

# Pre-load all session data once, then sweep lags
print("Loading all sessions...")

session_data = []
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
            print(f"  Loading {mouse} {session} ({sii+1}/{len(session_inds)})")

            photostim_keys = ['stimDist', 'favg_raw']
            bci_keys = [
                'df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si', 'step_time',
                'reward_time', 'BCI_thresholds',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"    Skipping -- file not found.")
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

            # Behavioral variables
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

            # Pair selection
            dist_target_lt = 10
            dist_nontarg_min = 30
            dist_nontarg_max = 1000
            amp0_thr = 0.1
            amp1_thr = 0.1

            dw_list = []
            pair_cl_list = []
            pair_nt_list = []

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
                print("    No valid pairs.")
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

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"    Only {n_wins} windows, skipping.")
                continue

            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            # Window behavioral variables (computed once)
            win_hit = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])

            session_data.append({
                'mouse': mouse, 'session': session,
                'F_nan': F_nan, 'n_frames': n_frames, 'n_neurons': n_neurons,
                'trl': trl, 'dt_si': dt_si, 'tsta': tsta,
                'ts_pre': ts_pre, 'ts_go': ts_go,
                'data_reward_time': data['reward_time'],
                'Y_T': Y_T, 'all_nt': all_nt, 'cl_weights': cl_weights,
                'n_pairs': n_pairs, 'win_starts': win_starts, 'n_wins': n_wins,
                'win_rpe': win_rpe, 'win_hit': win_hit,
                'win_rt': win_rt, 'win_hit_rpe': win_hit_rpe,
                'hit': hit,
            })

        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nLoaded {len(session_data)} sessions.")

#%% ============================================================================
# CELL 4: Sweep lags
# ============================================================================
import time

n_lags = len(LAG_RANGE)
n_sessions = len(session_data)

# Store per-session within-session correlations for each lag, mode, epoch
# corr arrays: (n_lags, n_sessions, n_epochs) for slope and intercept vs RPE
corr_slope_rpe = {mode: np.full((n_lags, n_sessions, n_epochs), np.nan)
                  for mode in CC_MODES}
corr_int_rpe = {mode: np.full((n_lags, n_sessions, n_epochs), np.nan)
                for mode in CC_MODES}

for li, lag_sec in enumerate(LAG_RANGE):
    t0 = time.time()
    print(f"\nLag {lag_sec:+.1f}s ({li+1}/{n_lags})")

    for si, sd in enumerate(session_data):
        F_nan = sd['F_nan']
        n_frames = sd['n_frames']
        n_neurons = sd['n_neurons']
        trl = sd['trl']
        dt_si = sd['dt_si']
        tsta = sd['tsta']
        ts_pre = sd['ts_pre']
        ts_go = sd['ts_go']
        Y_T = sd['Y_T']
        all_nt = sd['all_nt']
        cl_weights = sd['cl_weights']
        n_pairs = sd['n_pairs']
        win_starts = sd['win_starts']
        n_wins = sd['n_wins']
        win_rpe = sd['win_rpe']

        lag_frames = int(round(lag_sec / dt_si))

        # Compute lagged epoch averages
        epoch_pre_act = {}
        epoch_post_act = {}

        for ep in ['pre', 'go_cue']:
            if ep == 'pre':
                t0e, t1e = ts_pre[0], ts_pre[-1]
            else:
                t0e, t1e = ts_go[0], ts_go[-1]
            t0_lag = max(0, min(t0e + lag_frames, n_frames - 1))
            t1_lag = max(0, min(t1e + lag_frames, n_frames - 1))
            epoch_pre_act[ep] = np.nanmean(F_nan[t0e:t1e+1, :, :], axis=0)
            epoch_post_act[ep] = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

        epoch_pre_act['late'] = np.zeros((n_neurons, trl))
        epoch_post_act['late'] = np.zeros((n_neurons, trl))
        epoch_pre_act['reward'] = np.zeros((n_neurons, trl))
        epoch_post_act['reward'] = np.zeros((n_neurons, trl))

        for ti in range(trl):
            rewards = sd['data_reward_time'][ti]
            if len(rewards) > 0:
                indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                indices = indices[indices < n_frames]
                if len(indices) > 0:
                    epoch_pre_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                    indices_lag = indices + lag_frames
                    indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                    if len(indices_lag) > 0:
                        epoch_post_act['late'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

                indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                indices = indices[indices < n_frames]
                if len(indices) > 0:
                    epoch_pre_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                    indices_lag = indices + lag_frames
                    indices_lag = indices_lag[(indices_lag >= 0) & (indices_lag < n_frames)]
                    if len(indices_lag) > 0:
                        epoch_post_act['reward'][:, ti] = np.nanmean(F_nan[indices_lag, :, ti], axis=0)

        # Baseline for dev2
        baseline_trials_arr = np.arange(min(N_BASELINE, trl))
        baseline_post_mean_ep = {}
        for ep in EPOCH_ORDER:
            baseline_post_mean_ep[ep] = np.nanmean(
                epoch_post_act[ep][:, baseline_trials_arr], axis=1)

        # CC per window per epoch
        raw_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
        dev2_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)

        for wi, ws in enumerate(win_starts):
            trial_idx = np.arange(ws, ws + WIN_SIZE)
            for ei, ep in enumerate(EPOCH_ORDER):
                pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]
                post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]
                raw_cc[wi, :, ei] = np.sum(pre_act * post_act, axis=1)

                post_dev = epoch_post_act[ep][all_nt, :][:, trial_idx] - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                dev2_cc[wi, :, ei] = np.sum(pre_act * post_dev, axis=1)

        # Slope/intercept per mode per epoch, then correlate with RPE
        for mode in CC_MODES:
            for ei, ep in enumerate(EPOCH_ORDER):
                if mode == 'dot_prod_lag':
                    cc_all = raw_cc[:, :, ei]
                else:
                    cc_all = dev2_cc[:, :, ei]

                slope_arr = np.full(n_wins, np.nan)
                int_arr = np.full(n_wins, np.nan)
                for wi in range(n_wins):
                    cc_pair = cc_all[wi, :]
                    if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                        continue
                    A = np.column_stack([np.ones(n_pairs), cc_pair])
                    coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                    int_arr[wi] = coeffs[0]
                    slope_arr[wi] = coeffs[1]

                # Within-session correlation: RPE vs slope/intercept
                ok = np.isfinite(win_rpe) & np.isfinite(slope_arr)
                if np.sum(ok) >= 5 and np.std(slope_arr[ok]) > 0:
                    corr_slope_rpe[mode][li, si, ei], _ = spearmanr(win_rpe[ok], slope_arr[ok])
                ok2 = np.isfinite(win_rpe) & np.isfinite(int_arr)
                if np.sum(ok2) >= 5 and np.std(int_arr[ok2]) > 0:
                    corr_int_rpe[mode][li, si, ei], _ = spearmanr(win_rpe[ok2], int_arr[ok2])

    print(f"  done in {time.time()-t0:.1f}s")

print("\nLag sweep complete.")

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'lag_sweep_results.npy'),
        {'LAG_RANGE': LAG_RANGE, 'CC_MODES': CC_MODES,
         'EPOCH_ORDER': EPOCH_ORDER,
         'corr_slope_rpe': corr_slope_rpe,
         'corr_int_rpe': corr_int_rpe,
         'n_sessions': n_sessions},
        allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 6: Load
# ============================================================================
saved = np.load(os.path.join(RESULTS_DIR, 'lag_sweep_results.npy'),
                allow_pickle=True).item()
LAG_RANGE = saved['LAG_RANGE']
CC_MODES = saved['CC_MODES']
EPOCH_ORDER = saved['EPOCH_ORDER']
corr_slope_rpe = saved['corr_slope_rpe']
corr_int_rpe = saved['corr_int_rpe']
n_sessions = saved['n_sessions']
n_epochs = len(EPOCH_ORDER)
print(f"Loaded: {len(LAG_RANGE)} lags, {n_sessions} sessions")

#%% ============================================================================
# CELL 7: Plot — RPE correlation vs lag for all epochs
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']
epoch_colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']

fig, axes = plt.subplots(1, 1, figsize=(5,5),
                         squeeze=False)

for col, mode in enumerate([CC_MODES[1]]):
    for row, (corr_arr, row_label) in enumerate([
        (corr_slope_rpe[mode], 'RPE vs Slope'),
    ]):
        ax = axes[row, col]

        for ei in range(1):
            vals = corr_arr[:, :, ei]  # (n_lags, n_sessions)
            means = np.nanmean(vals, axis=1)
            sems = np.nanstd(vals, axis=1) / np.sqrt(np.sum(np.isfinite(vals), axis=1))

            ax.plot(LAG_RANGE, means, 'o-', color=epoch_colors[ei],
                    label=epoch_labels[ei], linewidth=2, markersize=5)
            ax.fill_between(LAG_RANGE, means - sems, means + sems,
                            color=epoch_colors[ei], alpha=0.15)

            # Mark significant lags (Wilcoxon p < 0.05)
            for li in range(len(LAG_RANGE)):
                v = vals[li, :]
                v = v[np.isfinite(v)]
                if len(v) >= 5:
                    try:
                        _, p = wilcoxon(v)
                        if p < 0.05:
                            ax.plot(LAG_RANGE[li], means[li], '*',
                                    color=epoch_colors[ei], markersize=12)
                    except Exception:
                        pass

        ax.axhline(0, color='k', ls='-', alpha=0.3, linewidth=0.8)
        ax.axvline(0, color='k', ls='--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Lag (s)  [positive = pre leads post]')
        ax.set_ylabel('Mean within-session rho')
        ax.legend(loc='best', framealpha=0.9)

        disp = mode.replace('_lag', '')
        if row == 0:
            ax.set_title(f'{disp}\n{row_label}', fontsize=14, fontweight='bold')
        else:
            ax.set_title(row_label, fontsize=13)

fig.suptitle(f'RPE correlation vs temporal offset (n={n_sessions} sessions)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig13_lag_sweep_rpe.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 13 saved.")
