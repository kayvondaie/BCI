#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import bci_time_series as bts
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

WIN_SIZE = 15
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20

LAG_RANGE = np.arange(-5, 6, 1)

CC_MODES = ['dot_prod_lag', 'dev2_lag']
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Lags to test: {LAG_RANGE}")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Load each session and sweep all lags (memory-efficient)
# ============================================================================
# Instead of loading all sessions into memory, process one at a time:
# for each session, load data, sweep all lags, store only the small
# correlation results, then free the session data before the next one.
import time

n_lags = len(LAG_RANGE)

pre_sec = 10.0
go_sec = 2.0
late_pre_frames = 20
reward_post_frames = 10

corr_slope_list = {mode: [] for mode in CC_MODES}
corr_int_list = {mode: [] for mode in CC_MODES}
session_count = 0

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
            trl = data['F'].shape[2]
            n_neurons = data['F'].shape[1]
            del data['F']  # free trial-segmented F; we use df_closedloop instead

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)

            # Pair selection (before freeing AMP/stimDist)
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

            # Window behavioral variables
            win_rpe = np.full(n_wins, np.nan)
            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])

            # Continuous time series
            df = data['df_closedloop']
            total_frames = df.shape[1]
            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
                folder, data, rt_filled, dt_si)

            trial_starts = np.where(trial_start_vector == 1)[0]
            reward_frames_arr = np.where(reward_vector == 1)[0]

            trial_reward_frame = np.full(trl, -1, dtype=int)
            for ri in reward_frames_arr:
                tidx = np.searchsorted(trial_starts, ri, side='right') - 1
                if 0 <= tidx < trl:
                    trial_reward_frame[tidx] = ri

            df_clean = df.copy()
            df_clean[np.isnan(df_clean)] = 0

            # Free large objects no longer needed
            del data, df, AMP, stimDist
            del step_vector, reward_vector, trial_start_vector
            del dw_list, pair_cl_list, pair_nt_list

            # --- Sweep all lags for this session ---
            sess_corr_slope = {mode: np.full((n_lags, n_epochs), np.nan)
                               for mode in CC_MODES}
            sess_corr_int = {mode: np.full((n_lags, n_epochs), np.nan)
                             for mode in CC_MODES}

            t0_sess = time.time()

            for li, lag_sec in enumerate(LAG_RANGE):
                lag_frames = int(round(lag_sec / dt_si))

                # Compute lagged epoch averages from continuous df
                epoch_pre_act = {ep: np.zeros((n_neurons, trl)) for ep in EPOCH_ORDER}
                epoch_post_act = {ep: np.zeros((n_neurons, trl)) for ep in EPOCH_ORDER}

                for ti in range(trl):
                    ts = trial_starts[ti]

                    # Pre epoch
                    t0 = max(0, ts - int(round(pre_sec / dt_si)))
                    t1 = ts
                    t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                    t1_lag = max(0, min(t1 + lag_frames, total_frames - 1))
                    if t1 > t0:
                        epoch_pre_act['pre'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                    if t1_lag > t0_lag:
                        epoch_post_act['pre'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                    # Go cue epoch
                    t0 = ts
                    t1 = min(ts + int(round(go_sec / dt_si)), total_frames)
                    t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                    t1_lag = max(0, min(t1 + lag_frames, total_frames))
                    if t1 > t0:
                        epoch_pre_act['go_cue'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                    if t1_lag > t0_lag:
                        epoch_post_act['go_cue'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                    # Late and reward
                    rf = trial_reward_frame[ti]
                    if rf > 0:
                        t0 = max(0, rf - late_pre_frames)
                        t1 = max(0, rf - 1)
                        t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                        t1_lag = max(0, min(t1 + lag_frames, total_frames - 1))
                        if t1 > t0:
                            epoch_pre_act['late'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                        if t1_lag > t0_lag:
                            epoch_post_act['late'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                        t0 = max(0, rf - 1)
                        t1 = min(rf + reward_post_frames, total_frames)
                        t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                        t1_lag = max(0, min(t1 + lag_frames, total_frames))
                        if t1 > t0:
                            epoch_pre_act['reward'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                        if t1_lag > t0_lag:
                            epoch_post_act['reward'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                # Baseline for dev2
                baseline_trials_arr = np.arange(min(N_BASELINE, trl))
                baseline_post_mean_ep = {}
                for ep in EPOCH_ORDER:
                    baseline_post_mean_ep[ep] = np.nanmean(
                        epoch_post_act[ep][:, baseline_trials_arr], axis=1)

                # CC per window per epoch — process one epoch at a time
                for mode in CC_MODES:
                    for ei, ep in enumerate(EPOCH_ORDER):
                        slope_arr = np.full(n_wins, np.nan)
                        int_arr = np.full(n_wins, np.nan)

                        for wi, ws_val in enumerate(win_starts):
                            trial_idx = np.arange(ws_val, ws_val + WIN_SIZE)
                            pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]
                            post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]

                            if mode == 'dot_prod_lag':
                                cc_pair = np.sum(pre_act * post_act, axis=1)
                            else:
                                post_dev = post_act - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                                cc_pair = np.sum(pre_act * post_dev, axis=1)

                            if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                                continue
                            A = np.column_stack([np.ones(n_pairs), cc_pair])
                            coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                            int_arr[wi] = coeffs[0]
                            slope_arr[wi] = coeffs[1]

                        ok = np.isfinite(win_rpe) & np.isfinite(slope_arr)
                        if np.sum(ok) >= 5 and np.std(slope_arr[ok]) > 0:
                            sess_corr_slope[mode][li, ei], _ = spearmanr(
                                win_rpe[ok], slope_arr[ok])
                        ok2 = np.isfinite(win_rpe) & np.isfinite(int_arr)
                        if np.sum(ok2) >= 5 and np.std(int_arr[ok2]) > 0:
                            sess_corr_int[mode][li, ei], _ = spearmanr(
                                win_rpe[ok2], int_arr[ok2])

            # Store this session's results
            for mode in CC_MODES:
                corr_slope_list[mode].append(sess_corr_slope[mode])
                corr_int_list[mode].append(sess_corr_int[mode])
            session_count += 1

            # Free session-specific large arrays
            del df_clean, cl_weights, epoch_pre_act, epoch_post_act

            print(f"  {n_wins} windows, {n_pairs} pairs, "
                  f"{n_lags} lags in {time.time()-t0_sess:.1f}s")

        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            continue

# Stack into arrays: (n_lags, n_sessions, n_epochs)
n_sessions = session_count
corr_slope_rpe = {}
corr_int_rpe = {}
for mode in CC_MODES:
    # Each list entry is (n_lags, n_epochs); stack along axis 1
    corr_slope_rpe[mode] = np.stack(corr_slope_list[mode], axis=1)
    corr_int_rpe[mode] = np.stack(corr_int_list[mode], axis=1)

print(f"\nLag sweep complete: {n_sessions} sessions, {n_lags} lags.")

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'lag_sweep_v2_results.npy'),
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
saved = np.load(os.path.join(RESULTS_DIR, 'lag_sweep_v2_results.npy'),
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

fig, axes = plt.subplots(2, len(CC_MODES), figsize=(6 * len(CC_MODES), 8),
                         squeeze=False)

for col, mode in enumerate(CC_MODES):
    for row, (corr_arr, row_label) in enumerate([
        (corr_slope_rpe[mode], 'RPE vs Slope'),
        (corr_int_rpe[mode], 'RPE vs Intercept'),
    ]):
        ax = axes[row, col]

        for ei in range(n_epochs):
            vals = corr_arr[:, :, ei]
            means = np.nanmean(vals, axis=1)
            sems = np.nanstd(vals, axis=1) / np.sqrt(np.sum(np.isfinite(vals), axis=1))

            ax.plot(LAG_RANGE, means, 'o-', color=epoch_colors[ei],
                    label=epoch_labels[ei], linewidth=2, markersize=5)
            ax.fill_between(LAG_RANGE, means - sems, means + sems,
                            color=epoch_colors[ei], alpha=0.15)

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

fig.suptitle(f'RPE correlation vs temporal offset — v2 continuous (n={n_sessions} sessions)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig13_lag_sweep_v2.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")
