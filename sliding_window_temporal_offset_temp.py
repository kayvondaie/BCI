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
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_1samp
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
CC_MODES = ['dot_prod_lag', 'dev2_lag']

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s (pre leads post)")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
import csv
_qc_csv = os.path.join(RESULTS_DIR, 'qc', 'qc_summary.csv')
_qc_fail = set()
if os.path.exists(_qc_csv):
    with open(_qc_csv) as _f:
        for _r in csv.DictReader(_f):
            if _r['pass_qc'] != 'True':
                _qc_fail.add((_r['mouse'], _r['session']))
    print(f"QC filter: {len(_qc_fail)} sessions excluded")
else:
    print("WARNING: qc_summary.csv not found, no sessions excluded")

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
            if (mouse, session) in _qc_fail:
                print(f"  Skipping {mouse} {session} -- failed QC")
                continue
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

            # ---- Compute CC per window per epoch (same structure as v2) ----
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
                    # Pre activity: original epoch window; Post: shifted by lag
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]   # (n_pairs, win_size)
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]   # (n_pairs, win_size)
                    cc_raw = np.sum(pre_act * post_act, axis=1)              # (n_pairs,)
                    raw_cc[wi, :, ei] = cc_raw

                    # dev2: pre(t) * (post(t+lag) - mean_post_baseline)
                    post_dev = epoch_post_act[ep][all_nt, :][:, trial_idx] - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    cc_dev2 = np.sum(pre_act * post_dev, axis=1)
                    dev2_cc[wi, :, ei] = cc_dev2


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
RANK = 3

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
            dpi=300, bbox_inches='tight')
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

#%% ============================================================================
# CELL 11: HI aligned to threshold changes
# ============================================================================
# Mirror of threshold_analysis.py CELL 8 (CN aligned to threshold changes), but
# with HI (the slope from each sliding window) as the aligned variable.
# Reloads only BCI_thresholds per session (cached) and interpolates the
# windowed HI onto a per-trial axis around each threshold switch.

from scipy.stats import ttest_1samp, wilcoxon

PRE_TRIALS  = 10   # trials before threshold change
POST_TRIALS = 10    # trials after threshold change
MAX_TRIAL   = 400    # restrict analysis to first MAX_TRIAL trials of each session
                    # (set to None to disable)
trial_axis_sw = np.arange(-PRE_TRIALS, POST_TRIALS)

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']
n_epochs = len(EPOCH_ORDER)

# Cache: (mouse, session) -> (switch_trial_inds, thr_upper)
sess_switch_cache = {}

def _get_switches(mouse, session):
    key = (mouse, session)
    if key in sess_switch_cache:
        return sess_switch_cache[key]
    folder = (r'//allen/aind/scratch/BCI/2p-raw/'
              + mouse + r'/' + session + '/pophys/')
    try:
        td = ddct.load_hdf5(folder, ['BCI_thresholds'], [])
    except Exception:
        sess_switch_cache[key] = (np.array([], dtype=int), None)
        return sess_switch_cache[key]
    thr = np.asarray(td['BCI_thresholds'], dtype=float)[1, :].copy()
    for i in range(1, thr.size):
        if np.isnan(thr[i]):
            thr[i] = thr[i - 1]
    if np.isnan(thr[0]) and np.any(np.isfinite(thr)):
        thr[0] = thr[np.isfinite(thr)][0]
    d_thr = np.diff(thr)
    sw = np.where((d_thr != 0) & np.isfinite(d_thr))[0] + 1
    sess_switch_cache[key] = (sw.astype(int), thr)
    return sess_switch_cache[key]

# Accumulators per mode and epoch
aligned_hi = {mode: [[] for _ in range(n_epochs)] for mode in CC_MODES}

for mode in CC_MODES:
    for s in all_results[mode]:
        mouse, session = s['mouse'], s['session']
        wc        = s['win_centers']
        n_trials  = s['n_trials']
        hi_mat    = s['hi_with_int']           # (n_wins, n_epochs)

        sw, thr = _get_switches(mouse, session)
        if len(sw) == 0 or thr is None:
            continue

        # Restrict analysis to first MAX_TRIAL trials of the session
        n_eff = n_trials if MAX_TRIAL is None else min(n_trials, MAX_TRIAL)

        for si_idx, sw_trial in enumerate(sw):
            if sw_trial < 1 or sw_trial >= n_eff:
                continue
            other_sw = np.delete(sw, si_idx)

            target_trials = sw_trial + trial_axis_sw
            valid = (target_trials >= 0) & (target_trials < n_eff)
            for osw in other_sw:
                if osw < sw_trial:
                    valid &= (target_trials >= osw)
                else:
                    valid &= (target_trials < osw)

            for ei in range(n_epochs):
                hi_vals = hi_mat[:, ei]
                ok_w = np.isfinite(hi_vals)
                if np.sum(ok_w) < 2:
                    continue
                # Store RAW HI values; per-transition z-scoring (against the
                # pre-switch window) is applied in the plotting/stats code below.
                wc_ok = wc[ok_w]
                hi_interp = np.interp(target_trials, wc_ok, hi_vals[ok_w],
                                      left=np.nan, right=np.nan)
                hi_interp[~valid] = np.nan
                aligned_hi[mode][ei].append(hi_interp)

# Stack to arrays
for mode in CC_MODES:
    for ei in range(n_epochs):
        if len(aligned_hi[mode][ei]) > 0:
            aligned_hi[mode][ei] = np.vstack(aligned_hi[mode][ei])
        else:
            aligned_hi[mode][ei] = np.zeros((0, PRE_TRIALS + POST_TRIALS))

print("Switch-aligned HI accumulated.")
for mode in CC_MODES:
    print(f"  {mode}: {aligned_hi[mode][0].shape[0]} transitions")

# --- Plot: aligned HI for each (mode, epoch), all transitions pooled ---
def _mean_sem_local(arr, axis=0):
    m = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis)
    s = np.nanstd(arr, axis=axis) / np.sqrt(np.clip(n, 1, None))
    return m, s

print(f"\n{'='*88}")
print(f"Diagnostic: per-panel HI stats (per-transition z-scored to pre-switch window)")
print(f"For each transition i: z_i(t) = (HI_i(t) - mean(HI_i[pre])) / std(HI_i[pre])")
print(f"  pre window = trials -{PRE_TRIALS}..-1 relative to switch")
print(f"  post window = trials 0..{POST_TRIALS - 1} relative to switch")
print(f"{'='*88}")
print(f"{'mode':14s} {'epoch':8s} {'n':>4s} "
      f"{'z_mean':>9s} {'z_sem':>9s} {'z_t':>7s} {'z_p':>9s} "
      f"{'min_p_t':>9s} {'sig_pts':>8s}")

for mode in CC_MODES:
    fig, axes = plt.subplots(2, n_epochs, figsize=(4 * n_epochs, 6.5),
                              sharex=True, squeeze=False)

    for ei in range(n_epochs):
        arr = aligned_hi[mode][ei]   # raw HI values aligned to switch
        n_tr = arr.shape[0]

        # ----- Per-transition z-score using the pre-switch window -----
        # Each row of arr corresponds to one transition. We z-score within
        # row using the transition's own pre-switch trials (col 0 .. PRE_TRIALS-1)
        # so that the pre-switch portion has mean 0, std 1 by construction.
        if n_tr > 0:
            pre_mean = np.nanmean(arr[:, :PRE_TRIALS], axis=1, keepdims=True)
            pre_std  = np.nanstd (arr[:, :PRE_TRIALS], axis=1, keepdims=True)
            arr_z = (arr - pre_mean) / np.where(pre_std > 1e-9, pre_std, np.nan)
        else:
            arr_z = arr.copy()

        # Row 0: raw HI (pooled across transitions)
        ax = axes[0, ei]
        if n_tr >= 2:
            m, sem = _mean_sem_local(arr)
            ax.fill_between(trial_axis_sw, m - sem, m + sem,
                            color='k', alpha=0.2)
            ax.plot(trial_axis_sw, m, color='k', linewidth=1.6,
                    label=f'n={n_tr}')
            ax.legend(loc='best', fontsize=8, frameon=False)
        ax.axvline(0, color='r', ls='--', linewidth=0.8)
        ax.axhline(0, color='gray', ls=':', linewidth=0.5)
        ax.set_title(epoch_labels[ei], fontsize=12)
        if ei == 0:
            ax.set_ylabel('HI (raw)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Row 1: per-transition z-scored HI
        # ---------- Statistics applied here ----------
        # 1) Summary t-test:   one-sample t-test of per-transition post-window
        #    means vs 0  (post window = trials 0..POST_TRIALS-1).
        #    Vector tested:  [mean(z_i[post]) for each transition i].
        # 2) Per-timepoint t:  at each trial offset t, one-sample t-test of
        #    z values across transitions vs 0. Tick marks at panel bottom
        #    show offsets with p<0.05 (uncorrected).
        # 3) Red horizontal bar = mean of post-window z values across transitions,
        #    band = ±SEM of that mean.
        # ---------------------------------------------
        ax2 = axes[1, ei]
        sig_count = 0
        min_p_t = np.nan
        if n_tr >= 2:
            m, sem = _mean_sem_local(arr_z)
            ax2.fill_between(trial_axis_sw, m - sem, m + sem,
                             color='k', alpha=0.2)
            ax2.plot(trial_axis_sw, m, color='k', linewidth=1.6)

            # --- (1) Summary t-test on post-window mean vs 0 ---
            post_means_z = np.nanmean(arr_z[:, PRE_TRIALS:], axis=1)
            ok = np.isfinite(post_means_z)
            n_ok = int(np.sum(ok))
            pval_z = np.nan
            if n_ok >= 5 and np.std(post_means_z[ok]) > 0:
                _, pval_z = ttest_1samp(post_means_z[ok], 0.0)
            if n_ok >= 2:
                pm_mean = np.nanmean(post_means_z[ok])
                pm_sem  = np.nanstd(post_means_z[ok]) / np.sqrt(n_ok)
                ax2.hlines(pm_mean, 0, POST_TRIALS - 1,
                           colors='C3', linewidth=2, alpha=0.8)
                ax2.fill_between([0, POST_TRIALS - 1],
                                 pm_mean - pm_sem, pm_mean + pm_sem,
                                 color='C3', alpha=0.15)

            # --- (2) Per-timepoint t-test vs 0 ---
            n_t = arr_z.shape[1]
            pvals_t = np.full(n_t, np.nan)
            for ti in range(n_t):
                col = arr_z[:, ti]
                col = col[np.isfinite(col)]
                if len(col) >= 5 and np.std(col) > 0:
                    _, pvals_t[ti] = ttest_1samp(col, 0.0)
            sig_mask = pvals_t < 0.05
            sig_count = int(np.sum(sig_mask))
            min_p_t = np.nanmin(pvals_t) if np.any(np.isfinite(pvals_t)) else np.nan
            if sig_count > 0:
                ymin, ymax = ax2.get_ylim()
                y_marker = ymin + 0.03 * (ymax - ymin)
                ax2.plot(trial_axis_sw[sig_mask],
                         np.full(sig_count, y_marker),
                         marker='|', color='C3', linestyle='none',
                         markersize=8, markeredgewidth=1.5)

            sig_lbl = ''
            if np.isfinite(pval_z):
                if pval_z < 0.001:
                    sig_lbl = '***'
                elif pval_z < 0.01:
                    sig_lbl = '**'
                elif pval_z < 0.05:
                    sig_lbl = '*'
                ax2.set_title(f'post-window mean p={pval_z:.3g} {sig_lbl}',
                              fontsize=9)
        ax2.axvline(0, color='r', ls='--', linewidth=0.8)
        ax2.axhline(0, color='gray', ls=':', linewidth=0.5)
        ax2.set_xlabel('Trials from threshold change')
        if ei == 0:
            ax2.set_ylabel('HI (per-transition z, pre-switch baseline)')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # --- Diagnostic print for this panel ---
        if n_tr >= 2:
            pm = np.nanmean(arr_z[:, PRE_TRIALS:], axis=1)

            def _stats(v):
                v = v[np.isfinite(v)]
                n_ = len(v)
                if n_ < 2:
                    return n_, np.nan, np.nan, np.nan, np.nan
                mu = np.mean(v)
                sem = np.std(v) / np.sqrt(n_)
                if n_ >= 5 and np.std(v) > 0:
                    t_, p_ = ttest_1samp(v, 0.0)
                else:
                    t_, p_ = np.nan, np.nan
                return n_, mu, sem, t_, p_

            n_z, mu_z, sem_z, t_z, p_z = _stats(pm)

            print(f"{mode:14s} {EPOCH_ORDER[ei]:8s} {n_z:>4d} "
                  f"{mu_z:+9.4f} {sem_z:>9.4f} {t_z:>+7.2f} {p_z:>9.4g} "
                  f"{min_p_t:>9.4g} {sig_count:>8d}")

    fig.suptitle(f'HI aligned to threshold change — {mode} (lag={OFFSET_SEC}s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'switch_aligned_HI_{mode}.png'
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")

#%% ============================================================================
# CELL 12: Overlay HI with RPE aligned to threshold change (session-z-scored)
# ============================================================================
# Plots the SAME session-z-scored HI traces as CELL 11 row 0, with RPE
# overlaid (also session-z-scored, NO pre-switch subtraction).

mode_overlay = 'dev2_lag'

# Build aligned RPE (session-z-scored, no per-transition subtraction)
aligned_rpe = []
for s in all_results[mode_overlay]:
    mouse, session = s['mouse'], s['session']
    wc       = s['win_centers']
    n_trials = s['n_trials']
    rpe_vals = s['win_rpe']

    sw, thr = _get_switches(mouse, session)
    if len(sw) == 0 or thr is None:
        continue

    # Restrict analysis to first MAX_TRIAL trials of the session
    n_eff = n_trials if MAX_TRIAL is None else min(n_trials, MAX_TRIAL)

    for si_idx, sw_trial in enumerate(sw):
        if sw_trial < 1 or sw_trial >= n_eff:
            continue
        other_sw = np.delete(sw, si_idx)
        target_trials = sw_trial + trial_axis_sw
        valid = (target_trials >= 0) & (target_trials < n_eff)
        for osw in other_sw:
            if osw < sw_trial:
                valid &= (target_trials >= osw)
            else:
                valid &= (target_trials < osw)

        ok_r = np.isfinite(rpe_vals)
        if np.sum(ok_r) < 2:
            continue
        rpe_interp = np.interp(target_trials, wc[ok_r], rpe_vals[ok_r],
                               left=np.nan, right=np.nan)
        rpe_interp[~valid] = np.nan
        aligned_rpe.append(rpe_interp)

aligned_rpe = np.vstack(aligned_rpe) if aligned_rpe else \
              np.zeros((0, len(trial_axis_sw)))
print(f"\nAligned RPE: {aligned_rpe.shape[0]} transitions ({mode_overlay})")

# --- Per-transition z-score (pre-switch baseline) ---
# For each row (transition), z-score using only its pre-switch window:
#     z_i(t) = (x_i(t) - mean(x_i[pre])) / std(x_i[pre])
# This forces the pre-switch portion of every row to have mean 0 and std 1.
def _to_pre_z(arr, pre_n):
    if arr.shape[0] == 0:
        return arr
    pre_mean = np.nanmean(arr[:, :pre_n], axis=1, keepdims=True)
    pre_std  = np.nanstd (arr[:, :pre_n], axis=1, keepdims=True)
    return (arr - pre_mean) / np.where(pre_std > 1e-9, pre_std, np.nan)

aligned_rpe_z = _to_pre_z(aligned_rpe, PRE_TRIALS)

# --- Plot: per-transition z-scored HI overlaid with per-transition z-scored RPE ---
fig, axes = plt.subplots(1, n_epochs, figsize=(4 * n_epochs, 3.5),
                          sharex=True, squeeze=False)

full_x = trial_axis_sw.astype(float)   # -PRE..POST-1, full window

post_x = trial_axis_sw[PRE_TRIALS:].astype(float)   # 0..POST_TRIALS-1

def _per_transition_pre_to_post_test(arr_z):
    """For each transition:
       (a) Slope test: fit a linear slope to ONLY the post-switch portion
           (trials 0..POST_TRIALS-1), then one-sample t-test of those
           slopes vs 0 across transitions.
       (b) Pre→post mean test: per-transition mean of post-window minus
           per-transition mean of pre-window (here pre is forced to ~0
           because of the per-transition z-score, so this reduces to a
           one-sample t-test on the post-window mean across transitions).
       Returns (n, mean_slope, p_slope, mean_diff, p_diff).
    """
    if arr_z.shape[0] < 2:
        return 0, np.nan, np.nan, np.nan, np.nan
    slopes = []
    diffs  = []
    for ti in range(arr_z.shape[0]):
        y_post = arr_z[ti, PRE_TRIALS:]
        ok = np.isfinite(y_post)
        if np.sum(ok) >= 5 and np.std(post_x[ok]) > 0:
            slopes.append(np.polyfit(post_x[ok], y_post[ok], 1)[0])
        pre  = np.nanmean(arr_z[ti, :PRE_TRIALS])
        post = np.nanmean(arr_z[ti, PRE_TRIALS:])
        if np.isfinite(pre) and np.isfinite(post):
            diffs.append(post - pre)
    slopes = np.array(slopes)
    diffs  = np.array(diffs)
    n_s = len(slopes)
    if n_s >= 5 and np.std(slopes) > 0:
        _, p_s = ttest_1samp(slopes, 0.0)
        m_s = float(np.mean(slopes))
    else:
        p_s, m_s = np.nan, np.nan
    n_d = len(diffs)
    if n_d >= 5 and np.std(diffs) > 0:
        _, p_d = ttest_1samp(diffs, 0.0)
        m_d = float(np.mean(diffs))
    else:
        p_d, m_d = np.nan, np.nan
    return n_s, m_s, p_s, m_d, p_d

print(f"\n{'='*78}\nCELL 12: HI pre→post change tests "
      f"(mode={mode_overlay}, per-transition pre-switch z-scored)\n{'='*78}")
print(f"{'epoch':8s} {'n':>4s} {'slope_full':>11s} {'p_slope':>9s} "
      f"{'post-pre':>10s} {'p_diff':>9s}")

for ei in range(n_epochs):
    ax = axes[0, ei]
    arr_hi_z = _to_pre_z(aligned_hi[mode_overlay][ei], PRE_TRIALS)
    n_h = arr_hi_z.shape[0]
    n_r = aligned_rpe_z.shape[0]

    # HI trace (per-transition pre-switch z-scored)
    if n_h >= 2:
        m_hi, s_hi = _mean_sem_local(arr_hi_z)
        ax.fill_between(trial_axis_sw, m_hi - s_hi, m_hi + s_hi,
                        color='k', alpha=0.18)
        ax.plot(trial_axis_sw, m_hi, color='k', linewidth=1.6,
                label=f'HI (n={n_h})')

  
    # --- Significance tests on HI: full-window slope and paired pre vs post ---
    n_s, m_s, p_s, m_d, p_d = _per_transition_pre_to_post_test(arr_hi_z)

    # Overlay post-window fitted line on the panel using mean slope.
    if np.isfinite(m_s):
        post_y = arr_hi_z[:, PRE_TRIALS:]
        intercept = np.nanmean(post_y) - m_s * np.nanmean(post_x)
        ax.plot(post_x, intercept + m_s * post_x,
                color='C3', linewidth=1.4, linestyle='--', alpha=0.9,
                label=f'post-fit slope={m_s:+.3f}')

    def _star(p):
        if not np.isfinite(p):
            return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''
    ax.set_title(f'{epoch_labels[ei]}\n'
                 f'slope p={p_s:.2g}{_star(p_s)}, '
                 f'post-pre p={p_d:.2g}{_star(p_d)}',
                 fontsize=10)

    print(f"{EPOCH_ORDER[ei]:8s} {n_s:>4d} "
          f"{m_s:+11.4f} {p_s:>9.4g} "
          f"{m_d:+10.4f} {p_d:>9.4g}")

    ax.axvline(0, color='r', ls='--', linewidth=0.8)
    ax.axhline(0, color='gray', ls=':', linewidth=0.5)
    ax.set_xlabel('Trials from threshold change')
    if ei == 0:
        ax.set_ylabel('Per-transition z-score (pre-switch baseline)')
    ax.legend(loc='best', fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'HI vs RPE aligned to threshold change — {mode_overlay} '
             f'(per-transition z-scored, lag={OFFSET_SEC}s)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fname = f'switch_aligned_HI_RPE_overlay_{mode_overlay}.png'
plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 13: Distributions of post-switch HI (dev2 pre) — diagnose big p-values
# ============================================================================
# Plots histograms of two per-transition statistics for dev2_lag, pre epoch:
#   (1) mean of HI z-scores over the post-switch window (trials 0..POST_TRIALS-1)
#   (2) HI z-score at the last post-switch trial (trial POST_TRIALS - 1)
# Both are per-transition z-scored to the pre-switch baseline.

mode_diag  = 'dev2_lag'
ei_diag    = 0          # pre epoch
last_offset = POST_TRIALS - 1   # last trial relative to switch

arr_diag = aligned_hi[mode_diag][ei_diag]
arr_z_diag = _to_pre_z(arr_diag, PRE_TRIALS)

post_means = np.nanmean(arr_z_diag[:, PRE_TRIALS:], axis=1)
last_pts   = arr_z_diag[:, PRE_TRIALS + last_offset]

post_means_ok = post_means[np.isfinite(post_means)]
last_pts_ok   = last_pts[np.isfinite(last_pts)]

# --- Stats for each distribution ---
def _summarize(v, name):
    n = len(v)
    if n < 2:
        print(f"  {name}: n={n} (too few)")
        return None
    mu  = np.mean(v)
    md  = np.median(v)
    sd  = np.std(v)
    sem = sd / np.sqrt(n)
    if n >= 5 and sd > 0:
        t_, p_ = ttest_1samp(v, 0.0)
    else:
        t_, p_ = np.nan, np.nan
    frac_neg = np.mean(v < 0)
    print(f"  {name}: n={n}  mean={mu:+.4f}  median={md:+.4f}  "
          f"std={sd:.4f}  sem={sem:.4f}  t={t_:+.3f}  p={p_:.4g}  "
          f"frac<0={frac_neg:.2%}  range=[{np.min(v):+.3f}, {np.max(v):+.3f}]")
    return mu, sem, t_, p_

print(f"\n{'='*70}")
print(f"CELL 13: Diagnostic distributions  ({mode_diag}, epoch={EPOCH_ORDER[ei_diag]})")
print(f"{'='*70}")
s1 = _summarize(post_means_ok, "post-window mean")
s2 = _summarize(last_pts_ok,   f"last point (t=+{last_offset})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, v, label, stats in [
    (axes[0], post_means_ok, 'Post-window mean HI (z)', s1),
    (axes[1], last_pts_ok,   f'HI at t=+{last_offset} (z)', s2),
]:
    if len(v) < 2:
        continue
    n_bins = max(15, int(np.sqrt(len(v))))
    ax.hist(v, bins=n_bins, color='#bbbbbb', edgecolor='k', linewidth=0.5)
    ax.axvline(0, color='k', linestyle=':', linewidth=1.2, label='0')
    mu = np.mean(v)
    sem = np.std(v) / np.sqrt(len(v))
    ax.axvline(mu, color='C3', linewidth=2.0, label=f'mean={mu:+.3f}')
    ax.axvspan(mu - sem, mu + sem, color='C3', alpha=0.18)
    ax.set_xlabel(label)
    ax.set_ylabel('# transitions')
    if stats is not None:
        _, _, t_, p_ = stats
        ax.set_title(f'n={len(v)}  t={t_:+.2f}  p={p_:.3g}', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'Distributions of per-transition z-scored HI '
             f'({mode_diag}, {EPOCH_ORDER[ei_diag]} epoch)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fname = f'switch_aligned_HI_distributions_{mode_diag}_{EPOCH_ORDER[ei_diag]}.png'
plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 14: Bootstrap 95% CI versions of the trajectory plots
# ============================================================================
# Same alignment as CELL 11 (per-transition z-scored to pre-switch) and CELL 12
# (HI vs RPE overlay), but the shaded band is a percentile bootstrap 95% CI on
# the across-transition mean instead of SEM. Uses N_BOOT bootstrap resamples
# of the rows (transitions), with replacement.

N_BOOT = 2000
CI_LO, CI_HI = 2.5, 97.5
_rng_boot = np.random.default_rng(0)

def _bootstrap_mean_ci(arr, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Across-row bootstrap of the mean. Returns (mean, lo_ci, hi_ci).
    arr: (n_transitions, n_timepoints), with NaNs allowed.
    """
    n_t = arr.shape[1]
    if arr.shape[0] < 2:
        nan_v = np.full(n_t, np.nan)
        return nan_v, nan_v, nan_v
    n = arr.shape[0]
    boot_means = np.full((n_boot, n_t), np.nan)
    for b in range(n_boot):
        idx = _rng_boot.integers(0, n, size=n)
        boot_means[b] = np.nanmean(arr[idx], axis=0)
    m  = np.nanmean(arr, axis=0)
    cl = np.nanpercentile(boot_means, lo, axis=0)
    ch = np.nanpercentile(boot_means, hi, axis=0)
    return m, cl, ch

# ---------- Plot 1: per-transition z-scored HI for each mode/epoch ----------
for mode in CC_MODES:
    fig, axes = plt.subplots(1, n_epochs, figsize=(4 * n_epochs, 3.8),
                              sharex=True, squeeze=False)
    for ei in range(n_epochs):
        ax = axes[0, ei]
        arr_z = _to_pre_z(aligned_hi[mode][ei], PRE_TRIALS)
        n_tr = arr_z.shape[0]
        if n_tr < 2:
            ax.set_title(f'{epoch_labels[ei]} (n<2)')
            continue

        m, cl, ch = _bootstrap_mean_ci(arr_z)
        ax.fill_between(trial_axis_sw, cl, ch, color='k', alpha=0.18,
                        label=f'{int(CI_HI - CI_LO)}% CI')
        ax.plot(trial_axis_sw, m, color='k', linewidth=1.6, label=f'mean (n={n_tr})')

        ax.axvline(0, color='r', ls='--', linewidth=0.8)
        ax.axhline(0, color='gray', ls=':', linewidth=0.5)
        ax.set_xlabel('Trials from threshold change')
        if ei == 0:
            ax.set_ylabel('HI (per-transition z, pre-switch baseline)')
        ax.set_title(epoch_labels[ei], fontsize=11)
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'HI aligned to threshold change — bootstrap 95% CI '
                 f'({mode}, lag={OFFSET_SEC}s, {N_BOOT} boots)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'switch_aligned_HI_bootCI_{mode}.png'
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")

# ---------- Plot 2: HI vs RPE overlay (dev2_lag) ----------
fig, axes = plt.subplots(1, n_epochs, figsize=(4 * n_epochs, 3.8),
                          sharex=True, squeeze=False)

for ei in range(n_epochs):
    ax = axes[0, ei]
    arr_hi_z  = _to_pre_z(aligned_hi[mode_overlay][ei], PRE_TRIALS)
    arr_rpe_z = aligned_rpe_z   # already per-transition z-scored

    if arr_hi_z.shape[0] >= 2:
        m_hi, cl_hi, ch_hi = _bootstrap_mean_ci(arr_hi_z)
        ax.fill_between(trial_axis_sw, cl_hi, ch_hi, color='k', alpha=0.18)
        ax.plot(trial_axis_sw, m_hi, color='k', linewidth=1.6,
                label=f'HI (n={arr_hi_z.shape[0]})')

    if arr_rpe_z.shape[0] >= 2:
        m_r, cl_r, ch_r = _bootstrap_mean_ci(arr_rpe_z)
        ax.fill_between(trial_axis_sw, cl_r, ch_r, color='C1', alpha=0.18)
        ax.plot(trial_axis_sw, m_r, color='C1', linewidth=1.6,
                label=f'RPE (n={arr_rpe_z.shape[0]})')

    ax.axvline(0, color='r', ls='--', linewidth=0.8)
    ax.axhline(0, color='gray', ls=':', linewidth=0.5)
    ax.set_xlabel('Trials from threshold change')
    if ei == 0:
        ax.set_ylabel('Per-transition z-score (pre-switch baseline)')
    ax.set_title(epoch_labels[ei], fontsize=11)
    ax.legend(loc='best', fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'HI vs RPE aligned to threshold change — bootstrap 95% CI '
             f'({mode_overlay}, {N_BOOT} boots)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fname = f'switch_aligned_HI_RPE_bootCI_{mode_overlay}.png'
plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved {fname}")

#%% ============================================================================
# CELL 15: Cluster-based permutation test on post-switch HI traces
# ============================================================================
# Rationale: independent per-timepoint t-tests ignore the consistency of sign
# across timepoints. A cluster-based permutation test (Maris & Oostenveld 2007)
# accounts for temporal correlation and tests "is there a sustained cluster of
# negative deviation post-switch?".
#
# Procedure:
#   1) Per-transition z-score HI to its pre-switch baseline.
#   2) Restrict to post-switch timepoints (trials 0..POST_TRIALS-1).
#   3) For each timepoint, compute the t-statistic across transitions vs 0.
#   4) Threshold at |t| > t_crit (uncorrected p<0.05).
#   5) Find contiguous clusters of suprathreshold points; cluster mass =
#      signed sum of t-stats in the cluster.
#   6) Build null distribution by sign-flipping each transition (under H0,
#      sign of each transition is random) and recomputing the largest
#      |cluster mass| over many permutations.
#   7) p-value of an observed cluster = fraction of permutations with
#      |max cluster mass| >= |observed cluster mass|.

from scipy.stats import t as _t_dist

N_PERM   = 2000
ALPHA_CT = 0.05   # cluster-forming threshold (uncorrected)
_rng_perm = np.random.default_rng(1)

def _post_t_stats(arr_post):
    n_per_t = np.sum(np.isfinite(arr_post), axis=0)
    m = np.nanmean(arr_post, axis=0)
    s = np.nanstd(arr_post, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sem = s / np.sqrt(np.maximum(n_per_t, 1))
        t_  = np.where(sem > 0, m / sem, np.nan)
    return t_, m, n_per_t

def _find_clusters(t_arr, t_thresh):
    out = []
    above = np.abs(t_arr) > t_thresh
    above = np.where(np.isfinite(t_arr), above, False)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            mass = np.nansum(t_arr[i:j])
            out.append((i, j, float(mass)))
            i = j
        else:
            i += 1
    return out

def cluster_perm_test(arr_z, n_perm=N_PERM, alpha=ALPHA_CT):
    """arr_z: (n_transitions, PRE+POST) per-transition z-scored aligned HI.
    Run the permutation test on the post-switch portion."""
    arr_post = arr_z[:, PRE_TRIALS:]
    n = arr_post.shape[0]
    if n < 5:
        return None
    t_thresh = _t_dist.ppf(1 - alpha / 2, n - 1)
    obs_t, obs_m, _ = _post_t_stats(arr_post)
    obs_clusters = _find_clusters(obs_t, t_thresh)

    # Null: sign-flip each transition, recompute largest |cluster mass|
    null_max = np.zeros(n_perm)
    for p in range(n_perm):
        signs = _rng_perm.choice([-1, 1], size=(n, 1))
        perm_t, _, _ = _post_t_stats(arr_post * signs)
        perm_clusters = _find_clusters(perm_t, t_thresh)
        if perm_clusters:
            null_max[p] = max(abs(c[2]) for c in perm_clusters)
        # else: null_max[p] stays 0

    # p-value per observed cluster
    out = []
    for (a, b, mass) in obs_clusters:
        p_clust = np.mean(null_max >= abs(mass))
        out.append({
            'start_offset': int(trial_axis_sw[PRE_TRIALS + a]),
            'end_offset':   int(trial_axis_sw[PRE_TRIALS + b - 1]),
            'mass':         mass,
            'p':            float(p_clust),
            'sign':         '−' if mass < 0 else '+',
        })
    return {
        'clusters':  out,
        't_thresh':  float(t_thresh),
        'null_max':  null_max,
        'obs_t':     obs_t,
        'obs_mean':  obs_m,
    }

# --- Run on dev2_lag, all epochs ---
print(f"\n{'='*88}")
print(f"CELL 15: Cluster-based permutation test on post-switch HI "
      f"(per-transition z, post window only)")
print(f"  Cluster threshold: |t| > t_crit at alpha={ALPHA_CT} (two-tailed)")
print(f"  Permutations: {N_PERM} sign-flips of transitions")
print(f"{'='*88}")
print(f"{'mode':14s} {'epoch':8s} {'n':>4s} {'t_crit':>7s} "
      f"{'cluster':>22s} {'mass':>9s} {'p_perm':>8s}")

for mode in CC_MODES:
    for ei in range(n_epochs):
        arr_z = _to_pre_z(aligned_hi[mode][ei], PRE_TRIALS)
        n_tr = arr_z.shape[0]
        result = cluster_perm_test(arr_z)
        if result is None:
            print(f"{mode:14s} {EPOCH_ORDER[ei]:8s} {n_tr:>4d} "
                  f"  (too few transitions)")
            continue
        if not result['clusters']:
            print(f"{mode:14s} {EPOCH_ORDER[ei]:8s} {n_tr:>4d} "
                  f"{result['t_thresh']:>7.3f}   (no suprathreshold cluster)")
            continue
        for c in result['clusters']:
            sig = '*' if c['p'] < 0.05 else ' '
            cluster_lbl = f"[{c['start_offset']:+d}..{c['end_offset']:+d}] {c['sign']}"
            print(f"{mode:14s} {EPOCH_ORDER[ei]:8s} {n_tr:>4d} "
                  f"{result['t_thresh']:>7.3f} "
                  f"{cluster_lbl:>22s} {c['mass']:>+9.2f} {c['p']:>8.4g}{sig}")

# --- Visualize the observed t-trace and clusters for dev2_lag, pre epoch ---
mode_v = mode_overlay   # 'dev2_lag'
ei_v   = 0              # 'pre'
arr_z_v = _to_pre_z(aligned_hi[mode_v][ei_v], PRE_TRIALS)
result_v = cluster_perm_test(arr_z_v)

if result_v is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: mean trace with shaded clusters
    ax = axes[0]
    m, cl, ch = _bootstrap_mean_ci(arr_z_v)
    ax.fill_between(trial_axis_sw, cl, ch, color='k', alpha=0.18,
                    label='95% bootstrap CI')
    ax.plot(trial_axis_sw, m, color='k', linewidth=1.6, label='mean')
    for c in result_v['clusters']:
        a = PRE_TRIALS + (c['start_offset'] - trial_axis_sw[PRE_TRIALS])
        b = PRE_TRIALS + (c['end_offset']   - trial_axis_sw[PRE_TRIALS]) + 1
        ax.axvspan(c['start_offset'] - 0.5, c['end_offset'] + 0.5,
                   color='C3' if c['p'] < 0.05 else 'C0', alpha=0.18)
        ax.text((c['start_offset'] + c['end_offset']) / 2,
                ax.get_ylim()[0] * 0.95,
                f"p={c['p']:.3g}", ha='center', fontsize=8,
                color='C3' if c['p'] < 0.05 else 'C0')
    ax.axvline(0, color='r', ls='--', linewidth=0.8)
    ax.axhline(0, color='gray', ls=':', linewidth=0.5)
    ax.set_xlabel('Trials from threshold change')
    ax.set_ylabel('HI (per-transition z)')
    ax.set_title(f'{mode_v}, {EPOCH_ORDER[ei_v]}: mean trace + cluster shading')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: null distribution of max |cluster mass| with observed
    ax = axes[1]
    ax.hist(result_v['null_max'], bins=40, color='#bbbbbb', edgecolor='k',
            linewidth=0.5, label=f'null (sign-flip, {N_PERM})')
    if result_v['clusters']:
        max_obs = max(abs(c['mass']) for c in result_v['clusters'])
        ax.axvline(max_obs, color='C3', linewidth=2,
                   label=f'observed max |mass|={max_obs:.2f}')
    ax.set_xlabel('|cluster mass| (sum of t)')
    ax.set_ylabel('# permutations')
    ax.set_title('Null distribution of max |cluster mass|')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'Cluster permutation test — {mode_v}, '
                 f'{EPOCH_ORDER[ei_v]} epoch (post-switch only)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'switch_aligned_HI_clusterperm_{mode_v}_{EPOCH_ORDER[ei_v]}.png'
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
