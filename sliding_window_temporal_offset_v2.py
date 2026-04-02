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

WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20

# Temporal offset in seconds (positive = pre leads post)
OFFSET_SEC = 0

# CC modes
CC_MODES = ['dot_prod_lag', 'dev2_lag']

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

# Epoch definitions relative to events (in seconds):
#   pre:     trial_start - 10s  to  trial_start
#   go_cue:  trial_start        to  trial_start + 2s
#   late:    reward - 20*dt_si  to  reward - 1*dt_si  (pre-reward)
#   reward:  reward - 1*dt_si   to  reward + 10*dt_si (post-reward)

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC}s (pre leads post)")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop — uses df_closedloop (continuous time series)
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

            # ---- Get continuous time series info ----
            df = data['df_closedloop']  # (n_neurons, total_frames)
            total_frames = df.shape[1]
            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
                folder, data, rt_filled, dt_si)

            # Trial start and reward frame indices
            trial_starts = np.where(trial_start_vector == 1)[0]
            reward_frames = np.where(reward_vector == 1)[0]

            # Map each trial to its reward frame (NaN trials have no reward)
            trial_reward_frame = np.full(trl, -1, dtype=int)
            for ri in reward_frames:
                # Find which trial this reward belongs to
                trial_idx = np.searchsorted(trial_starts, ri, side='right') - 1
                if 0 <= trial_idx < trl:
                    trial_reward_frame[trial_idx] = ri

            lag_frames = int(round(OFFSET_SEC / dt_si))
            print(f"  dt_si={dt_si:.4f}s, lag={lag_frames} frames ({lag_frames*dt_si:.2f}s)")
            print(f"  df shape: {df.shape}, {trl} trials, {len(trial_starts)} trial starts")

            # Replace NaN with 0 in df
            df_clean = df.copy()
            df_clean[np.isnan(df_clean)] = 0

            # ---- Compute epoch-averaged activity per trial (continuous time) ----
            # For each trial, extract pre/post activity from continuous df
            # Pre neuron: original epoch window
            # Post neuron: epoch window shifted by lag_frames

            epoch_pre_act = {ep: np.zeros((n_neurons, trl)) for ep in EPOCH_ORDER}
            epoch_post_act = {ep: np.zeros((n_neurons, trl)) for ep in EPOCH_ORDER}

            pre_sec = 10.0   # pre epoch: trial_start - 10s to trial_start
            go_sec = 2.0     # go_cue epoch: trial_start to trial_start + 2s
            late_pre_frames = 20   # late epoch: reward - 20 frames to reward - 1 frame
            reward_post_frames = 10  # reward epoch: reward - 1 frame to reward + 10 frames

            for ti in range(trl):
                ts = trial_starts[ti]

                # --- Pre epoch: [ts - pre_sec, ts) ---
                t0 = max(0, ts - int(round(pre_sec / dt_si)))
                t1 = ts
                t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, total_frames - 1))
                if t1 > t0:
                    epoch_pre_act['pre'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                if t1_lag > t0_lag:
                    epoch_post_act['pre'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                # --- Go cue epoch: [ts, ts + go_sec) ---
                t0 = ts
                t1 = min(ts + int(round(go_sec / dt_si)), total_frames)
                t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, total_frames))
                if t1 > t0:
                    epoch_pre_act['go_cue'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                if t1_lag > t0_lag:
                    epoch_post_act['go_cue'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                # --- Late and reward epochs (only for hit trials) ---
                rf = trial_reward_frame[ti]
                if rf > 0:
                    # Late: [reward - 20 frames, reward - 1 frame]
                    t0 = max(0, rf - late_pre_frames)
                    t1 = max(0, rf - 1)
                    t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                    t1_lag = max(0, min(t1 + lag_frames, total_frames - 1))
                    if t1 > t0:
                        epoch_pre_act['late'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                    if t1_lag > t0_lag:
                        epoch_post_act['late'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

                    # Reward: [reward - 1 frame, reward + 10 frames]
                    t0 = max(0, rf - 1)
                    t1 = min(rf + reward_post_frames, total_frames)
                    t0_lag = max(0, min(t0 + lag_frames, total_frames - 1))
                    t1_lag = max(0, min(t1 + lag_frames, total_frames))
                    if t1 > t0:
                        epoch_pre_act['reward'][:, ti] = np.mean(df_clean[:, t0:t1], axis=1)
                    if t1_lag > t0_lag:
                        epoch_post_act['reward'][:, ti] = np.mean(df_clean[:, t0_lag:t1_lag], axis=1)

            # ---- Build pair selection ----
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

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # ---- Baseline for dev2 ----
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            baseline_post_mean_ep = {}
            for ep in EPOCH_ORDER:
                baseline_post_mean_ep[ep] = np.nanmean(
                    epoch_post_act[ep][:, baseline_trials_arr], axis=1)

            # ---- CC per window per epoch ----
            raw_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)
            dev2_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)

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
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]
                    cc_raw = np.sum(pre_act * post_act, axis=1)
                    raw_cc[wi, :, ei] = cc_raw

                    post_dev = epoch_post_act[ep][all_nt, :][:, trial_idx] - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    cc_dev2 = np.sum(pre_act * post_dev, axis=1)
                    dev2_cc[wi, :, ei] = cc_dev2

            # ---- Fit slope/intercept per mode per epoch ----
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

                result = {
                    'mouse': mouse,
                    'session': session,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
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
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2.npy'),
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
# CELL 7: Coefficient matrices — behavior x epoch
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
            disp = mode.replace('_lag', f' (lag={OFFSET_SEC}s)')
            ax.set_title(f'{disp}\n({row_label})', fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=12)

lag_str = f"{OFFSET_SEC}s"
fig.suptitle(f'Temporal offset v2 (continuous): pre leads post by {lag_str} (n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig12_temporal_offset_v2_matrices.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'temporal_offset_v2_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("TEMPORAL OFFSET COACTIVITY ANALYSIS (v2 — continuous df_closedloop)\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s (pre leads post)\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write(f"  dot_prod_lag : sum_t pre(t) * post(t + lag), epoch-averaged\n")
    f.write(f"  dev2_lag     : sum_t pre(t) * (post(t+lag) - mean_post_baseline)\n\n")

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
