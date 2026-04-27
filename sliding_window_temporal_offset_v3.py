#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Variant of sliding_window_temporal_offset.py with 5 CC modes:
  1. dot_prod     — sum_t pre(t) * post(t+lag)
  2. dev2         — sum_t pre(t) * (post(t+lag) - baseline_post)
  3. pre_dev_only — sum_t (pre(t) - baseline_pre) * post(t+lag)
  4. pre_dev      — sum_t (pre(t) - baseline_pre) * (post(t+lag) - baseline_post)
  5. phi_prime_dev2 — sum_t pre(t) * (post(t+lag) - baseline_post) * 1(post(t+lag) > 20th pctl)
"""
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

CC_MODES = ['dot_prod', 'dev2', 'pre_dev_only', 'pre_dev', 'phi_prime_dev2']

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"Temporal offset: {OFFSET_SEC} s")
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
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

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

            # ---- Pair selection ----
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

            # ---- Epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
            n_epochs = len(EPOCH_ORDER)

            epoch_pre_act = {}
            epoch_post_act = {}

            for ep in ['pre', 'go_cue']:
                if ep == 'pre':
                    t0, t1 = ts_pre[0], ts_pre[-1]
                else:
                    t0, t1 = ts_go[0], ts_go[-1]
                t0_lag = max(0, min(t0 + lag_frames, n_frames - 1))
                t1_lag = max(0, min(t1 + lag_frames, n_frames - 1))
                epoch_pre_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)
                epoch_post_act[ep] = np.nanmean(F_nan[t0_lag:t1_lag+1, :, :], axis=0)

            epoch_pre_act['late'] = np.zeros((n_neurons, trl))
            epoch_post_act['late'] = np.zeros((n_neurons, trl))
            epoch_pre_act['reward'] = np.zeros((n_neurons, trl))
            epoch_post_act['reward'] = np.zeros((n_neurons, trl))

            for ti in range(trl):
                rewards = data['reward_time'][ti]
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

            # ---- Baselines ----
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))

            # Post-synaptic baseline per epoch (for dev2, pre_dev, phi_prime_dev2)
            baseline_post_mean_ep = {}
            for ep in EPOCH_ORDER:
                baseline_post_mean_ep[ep] = np.nanmean(
                    epoch_post_act[ep][:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # Pre-synaptic baseline per epoch (for pre_dev_only, pre_dev)
            baseline_pre_mean_ep = {}
            for ep in EPOCH_ORDER:
                baseline_pre_mean_ep[ep] = np.nanmean(
                    epoch_pre_act[ep][:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # 20th percentile of post-synaptic activity across ALL trials (for phi_prime_dev2)
            pctl20_post_ep = {}
            for ep in EPOCH_ORDER:
                pctl20_post_ep[ep] = np.percentile(
                    epoch_post_act[ep], 20, axis=1)  # (n_neurons,)

            # Pre-synaptic baseline projected through cl_weights (for pre_dev_only, pre_dev)
            # baseline_pre_pair[ep] = cl_weights @ baseline_pre_mean_ep[ep]  -> (n_pairs,)
            baseline_pre_pair = {}
            for ep in EPOCH_ORDER:
                baseline_pre_pair[ep] = cl_weights @ baseline_pre_mean_ep[ep]

            # ---- CC arrays ----
            cc_arrays = {mode: np.full((n_wins, n_pairs, n_epochs), np.nan)
                         for mode in CC_MODES}

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
                    pre_act = cl_weights @ epoch_pre_act[ep][:, trial_idx]   # (n_pairs, win_size)
                    post_act = epoch_post_act[ep][all_nt, :][:, trial_idx]   # (n_pairs, win_size)

                    # dot_prod: raw
                    cc_arrays['dot_prod'][wi, :, ei] = np.sum(pre_act * post_act, axis=1)

                    # dev2: subtract post baseline
                    post_dev = post_act - baseline_post_mean_ep[ep][all_nt, np.newaxis]
                    cc_arrays['dev2'][wi, :, ei] = np.sum(pre_act * post_dev, axis=1)

                    # pre_dev_only: subtract pre baseline only
                    pre_dev = pre_act - baseline_pre_pair[ep][:, np.newaxis]
                    cc_arrays['pre_dev_only'][wi, :, ei] = np.sum(pre_dev * post_act, axis=1)

                    # pre_dev: subtract both baselines
                    cc_arrays['pre_dev'][wi, :, ei] = np.sum(pre_dev * post_dev, axis=1)

                    # phi_prime_dev2: dev2 gated by post > 20th percentile
                    gate = (post_act > pctl20_post_ep[ep][all_nt, np.newaxis]).astype(float)
                    cc_arrays['phi_prime_dev2'][wi, :, ei] = np.sum(
                        pre_act * post_dev * gate, axis=1)

            # ---- Fit slope/intercept for each mode ----
            for mode in CC_MODES:
                hi_no_int = np.full((n_wins, n_epochs), np.nan)
                hi_with_int = np.full((n_wins, n_epochs), np.nan)
                hi_intercept = np.full((n_wins, n_epochs), np.nan)
                hi_corr = np.full((n_wins, n_epochs), np.nan)

                for ei in range(n_epochs):
                    cc_all = cc_arrays[mode][:, :, ei]

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
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v3.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v3.npy'),
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
                if p < 0.001: sig = '***'
                elif p < 0.01: sig = '**'
                elif p < 0.05: sig = '*'
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
            ax.set_title(f'{mode}\n({row_label})', fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=11)

n_s = len(all_results[CC_MODES[0]])
fig.suptitle(f'CC mode comparison (n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'sliding_window_v3_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("SLIDING WINDOW CC MODE COMPARISON (v3)\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Temporal offset: {OFFSET_SEC}s\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write("  dot_prod       : sum_t pre(t) * post(t+lag)\n")
    f.write("  dev2           : sum_t pre(t) * (post(t+lag) - baseline_post)\n")
    f.write("  pre_dev_only   : sum_t (pre(t) - baseline_pre) * post(t+lag)\n")
    f.write("  pre_dev        : sum_t (pre(t) - baseline_pre) * (post(t+lag) - baseline_post)\n")
    f.write("  phi_prime_dev2 : sum_t pre(t) * (post(t+lag) - baseline_post) * 1(post > 20th pctl)\n\n")

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
                for ei, ep in enumerate(EPOCH_ORDER):
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
# CELL 9: Bar graph — RPE × Pre p-value per CC mode
# ============================================================================
"""
For each CC mode, extract the Wilcoxon p-value for RPE × Slope in the pre epoch.
Plot as -log10(p) bar graph for easy comparison.
"""
bi_rpe = beh_names.index('RPE')
ei_pre = EPOCH_ORDER.index('pre')

mode_pvals = []
mode_means = []
for mode in CC_MODES:
    vals = corr_slope[mode][:, bi_rpe, ei_pre]
    v = vals[np.isfinite(vals)]
    if len(v) >= 3:
        _, p = wilcoxon(v)
        m = np.mean(v)
    else:
        p, m = 1.0, 0.0
    mode_pvals.append(p)
    mode_means.append(m)

mode_pvals = np.array(mode_pvals)
mode_means = np.array(mode_means)
neg_log_p = -np.log10(mode_pvals)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

# Panel A: -log10(p)
colors = ['k', 'k', 'k', 'k', 'k']
bars = ax.bar(range(len(CC_MODES)), neg_log_p, color=colors, edgecolor='k',
              linewidth=0.8)
ax.axhline(-np.log10(0.05), color='r', ls='--', linewidth=1.5, label='p = 0.05')
ax.set_xticks(range(len(CC_MODES)))
ax.set_xticklabels(CC_MODES, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('-log10(p)')
ax.set_title('RPE × Slope (Pre epoch)\nWilcoxon p-value per CC mode', fontweight='bold')
ax.set_ylim((0,4))
ax.legend(fontsize=9)

# Annotate bars with actual p-values


plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_rpe_pre_bar.png'),
            dpi=200, bbox_inches='tight')
plt.show()
