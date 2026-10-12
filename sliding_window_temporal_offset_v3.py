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

from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

# Colored math labels: pre-side blue, post-side red, operators black/gray.
PRE_COLOR  = (0.0, 0.0, 1.0)
POST_COLOR = (1.0, 0.0, 0.0)
OP_COLOR   = '0.15'
LABEL_SIZE = 13

LABEL_SEGMENTS = {
    'dot_prod':       [(r'$r_{\mathrm{pre}}$',         PRE_COLOR),
                       (r'$\times$',                   OP_COLOR),
                       (r'$r_{\mathrm{post}}$',        POST_COLOR)],
    'dev2':           [(r'$r_{\mathrm{pre}}$',         PRE_COLOR),
                       (r'$\times$',                   OP_COLOR),
                       (r'$\Delta r_{\mathrm{post}}$', POST_COLOR)],
    'pre_dev_only':   [(r'$\Delta r_{\mathrm{pre}}$',  PRE_COLOR),
                       (r'$\times$',                   OP_COLOR),
                       (r'$r_{\mathrm{post}}$',        POST_COLOR)],
    'pre_dev':        [(r'$\Delta r_{\mathrm{pre}}$',  PRE_COLOR),
                       (r'$\times$',                   OP_COLOR),
                       (r'$\Delta r_{\mathrm{post}}$', POST_COLOR)],
    'phi_prime_dev2': [(r'$r_{\mathrm{pre}}$',         PRE_COLOR),
                       (r'$\times$',                   OP_COLOR),
                       (r'$\Delta r_{\mathrm{post}}$', POST_COLOR),
                       [(r"$\cdot$",                   OP_COLOR),
                        (r"$\phi'($",                  POST_COLOR),
                        (r'$r_{\mathrm{post}}$',       POST_COLOR),
                        (r'$)$',                       POST_COLOR)]],
}

# Sort ascending so the most significant bar ends up at the top of the y axis
order        = np.argsort(neg_log_p)
modes_sorted = [CC_MODES[i] for i in order]
nlp_sorted   = neg_log_p[order]
p_sorted     = mode_pvals[order]

SIG_COLOR    = '#2C7FB8'   # muted blue
NONSIG_COLOR = '#C8CDD2'   # light gray
bar_colors   = [SIG_COLOR if p < 0.05 else NONSIG_COLOR for p in p_sorted]

fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
ax.barh(range(len(CC_MODES)), nlp_sorted, color=bar_colors,
        edgecolor='none', height=0.65)

# p = 0.05 reference line
thr = -np.log10(0.05)
ax.axvline(thr, color='0.25', ls=(0, (2, 2)), linewidth=1.0, alpha=0.7)
ax.text(thr, len(CC_MODES) - 0.35, ' p = 0.05',
        ha='left', va='top', fontsize=10, color='0.25')

# Per-bar p-value annotations
xmax = max(np.nanmax(neg_log_p) * 1.28, 4.0)
for i, (nlp, p) in enumerate(zip(nlp_sorted, p_sorted)):
    txt = f'p = {p:.3f}' if p >= 1e-3 else f'p = {p:.1e}'
    ax.text(nlp + xmax * 0.012, i, txt,
            va='center', ha='left', fontsize=11, color='0.2')

# Hide default y-tick labels; render colored math labels via HPacker
ax.set_yticks(range(len(CC_MODES)))
ax.set_yticklabels([''] * len(CC_MODES))
for i, mode in enumerate(modes_sorted):
    segs = LABEL_SEGMENTS.get(mode, [(mode, OP_COLOR)])
    children = []
    for s in segs:
        if isinstance(s, list):
            sub = [TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE))
                   for (t, c) in s]
            children.append(HPacker(children=sub, align='center', pad=0, sep=0))
        else:
            t, c = s
            children.append(TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE)))
    packer = HPacker(children=children, align='center', pad=0, sep=3)
    ab = AnnotationBbox(packer, xy=(0, i), xybox=(-8, 0),
                        xycoords=('axes fraction', 'data'),
                        boxcoords='offset points',
                        box_alignment=(1.0, 0.5),
                        frameon=False, pad=0)
    ax.add_artist(ab)

ax.set_xlabel(r'$-\log_{10}(p)$', fontsize=12, labelpad=6)
ax.set_title('RPE x slope, pre-epoch', fontsize=14, pad=10, loc='left')

ax.set_xlim(0, xmax)
ax.tick_params(axis='x', labelsize=11, color='0.4')
ax.tick_params(axis='y', length=0)

for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color('0.4')

ax.xaxis.grid(True, color='0.92', linewidth=0.8)
ax.set_axisbelow(True)

fig.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.14)
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_rpe_pre_bar.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_rpe_pre_bar.svg'),
            bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 10: Pre-epoch matrix — eligibility (rows) x 3rd factor (cols)
# ============================================================================
# Reuses LABEL_SEGMENTS / colors / offsetbox imports from CELL 9.
# Each cell = signed -log10(p): sign(mean rho across sessions) * -log10(Wilcoxon p)
# for the pre epoch only. RT column is sign-flipped because RT is anticorrelated
# with performance (low RT = good).

# Display order pairs Hit/ΔHit and Speed/ΔSpeed.
# Mapping back to beh_names:
#   hit_rate -> Hit       (raw hit rate)
#   hit_RPE  -> ΔHit      (RPE on hit signal)
#   RT       -> Speed     (raw RT, sign-flipped: low RT = high speed)
#   RPE      -> ΔSpeed    (RPE on reaction time)
DISPLAY_ORDER  = ['hit_rate', 'hit_RPE', 'RT', 'RPE']
DISPLAY_LABELS = {'hit_rate': 'Hit',
                  'hit_RPE':  r'$\Delta$Hit',
                  'RT':       'Speed',
                  'RPE':      'RPE'}
SIGN_FLIP      = {'RT': -1}

ei_pre = EPOCH_ORDER.index('pre')
beh_idx_disp = [beh_names.index(b) for b in DISPLAY_ORDER]
n_disp = len(DISPLAY_ORDER)

mat_mean = np.full((len(CC_MODES), n_disp), np.nan)
mat_p    = np.full((len(CC_MODES), n_disp), np.nan)
for mi, mode in enumerate(CC_MODES):
    for dj, bi in enumerate(beh_idx_disp):
        vals = corr_slope[mode][:, bi, ei_pre]
        v = vals[np.isfinite(vals)]
        if len(v) < 3:
            continue
        mat_mean[mi, dj] = np.mean(v)
        try:
            _, p = wilcoxon(v)
        except Exception:
            p = 1.0
        mat_p[mi, dj] = p

# Signed -log10(p), with optional per-behavior sign flips
beh_sign   = np.array([SIGN_FLIP.get(b, 1) for b in DISPLAY_ORDER], dtype=float)
mat_signed = np.sign(mat_mean) * (-np.log10(np.clip(mat_p, 1e-300, 1.0)))
mat_signed = mat_signed * beh_sign[np.newaxis, :]

fig_m, ax_m = plt.subplots(1, 1, figsize=(7.0, 5.0))
vmax = max(np.log10(1 / 0.05), np.nanmax(np.abs(mat_signed)))   # at least to p=0.05
im = ax_m.imshow(mat_signed, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                 aspect='auto', interpolation='nearest')

# Cell annotations: show signed -log10(p) with significance stars
for mi in range(len(CC_MODES)):
    for bj in range(n_disp):
        v = mat_signed[mi, bj]
        p = mat_p[mi, bj]
        if np.isnan(v):
            continue
        if   p < 0.001: sig = '***'
        elif p < 0.01:  sig = '**'
        elif p < 0.05:  sig = '*'
        else:           sig = ''
        txt = f'{v:+.2f}'
        if sig:
            txt += f'\n{sig}'
        ax_m.text(bj, mi, txt, ha='center', va='center', fontsize=11,
                  fontweight='bold' if sig else 'normal', color='0.05')

# Column labels: 3rd factors
X_LABEL_COLOR = '#F99D20'
ax_m.set_xticks(range(n_disp))
ax_m.set_xticklabels([DISPLAY_LABELS[b] for b in DISPLAY_ORDER],
                     fontsize=13, color=X_LABEL_COLOR, fontweight='bold')
ax_m.set_xlabel('3rd factor', fontsize=14, labelpad=8,
                color=X_LABEL_COLOR, fontweight='bold')

# Row labels: colored math via HPacker (matches CELL 9)
ax_m.set_yticks(range(len(CC_MODES)))
ax_m.set_yticklabels([''] * len(CC_MODES))
for i, mode in enumerate(CC_MODES):
    segs = LABEL_SEGMENTS.get(mode, [(mode, OP_COLOR)])
    children = []
    for s in segs:
        if isinstance(s, list):
            sub = [TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE))
                   for (t, c) in s]
            children.append(HPacker(children=sub, align='center', pad=0, sep=0))
        else:
            t, c = s
            children.append(TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE)))
    packer = HPacker(children=children, align='center', pad=0, sep=3)
    ab = AnnotationBbox(packer, xy=(0, i), xybox=(-10, 0),
                        xycoords=('axes fraction', 'data'),
                        boxcoords='offset points',
                        box_alignment=(1.0, 0.5),
                        frameon=False, pad=0)
    ax_m.add_artist(ab)

ax_m.tick_params(axis='y', length=0)
ax_m.tick_params(axis='x', length=0)
for s in ('top', 'right', 'left', 'bottom'):
    ax_m.spines[s].set_visible(False)

cbar = plt.colorbar(im, ax=ax_m, shrink=0.75, pad=0.03)
cbar.set_label(r'sign$(\rho)\cdot -\log_{10}(p)$', fontsize=11)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_visible(False)
# Mark p=0.05 thresholds on the colorbar
for thr in (-np.log10(0.05), np.log10(0.05)):
    cbar.ax.axhline(thr, color='k', linewidth=0.8, alpha=0.7)

ax_m.set_title('Pre epoch: eligibility x 3rd factor (RT sign-flipped)',
               fontsize=14, pad=12, loc='left')

fig_m.subplots_adjust(left=0.28, right=0.92, top=0.86, bottom=0.14)
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_pre_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_pre_matrix.svg'),
            bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 11: Nested matrix — eligibility x (3rd factor, epoch)
# ============================================================================
# Same outer layout as CELL 10, but each outer cell expands into a 1xN_epochs
# strip so the epoch dimension is shown inline. Reuses DISPLAY_ORDER / SIGN_FLIP
# / LABEL_SEGMENTS from CELL 10.

EPOCH_DISP_LABELS = ['pre', 'go', 'late', 'rew']  # short forms for compactness
n_ep_disp = len(EPOCH_ORDER)
n_total_cols = n_disp * n_ep_disp

mat3_signed = np.full((len(CC_MODES), n_total_cols), np.nan)
mat3_p      = np.full((len(CC_MODES), n_total_cols), np.nan)
for mi, mode in enumerate(CC_MODES):
    for dj, bi in enumerate(beh_idx_disp):
        sign = SIGN_FLIP.get(DISPLAY_ORDER[dj], 1)
        for ek in range(n_ep_disp):
            vals = corr_slope[mode][:, bi, ek]
            v = vals[np.isfinite(vals)]
            if len(v) < 3:
                continue
            m = np.mean(v)
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            col = dj * n_ep_disp + ek
            mat3_signed[mi, col] = (np.sign(m) *
                                    -np.log10(max(p, 1e-300)) * sign)
            mat3_p[mi, col] = p

vmax = max(-np.log10(0.05), np.nanmax(np.abs(mat3_signed)))

fig_n, ax_n = plt.subplots(1, 1, figsize=(8, 2))
im = ax_n.imshow(mat3_signed, cmap='bwr', vmin=-vmax, vmax=vmax,
                 aspect='auto', interpolation='nearest')

# Star-only annotations (numbers would be too cramped at 16 columns)
for mi in range(len(CC_MODES)):
    for col in range(n_total_cols):
        p = mat3_p[mi, col]
        if np.isnan(p):
            continue
        if   p < 0.001: sig = '***'
        elif p < 0.01:  sig = '**'
        elif p < 0.05:  sig = '*'
        else:           sig = ''
        if sig:
            ax_n.text(col, mi, sig, ha='center', va='center',
                      fontsize=11, fontweight='bold', color='0.05')

# Black borders between 3rd-factor groups
for k in range(1, n_disp):
    ax_n.axvline(k * n_ep_disp - 0.5, color='k', linewidth=1.5)

# Bottom: epoch labels under each column
ax_n.set_xticks(np.arange(n_total_cols))
ax_n.set_xticklabels(EPOCH_DISP_LABELS * n_disp, fontsize=9)

# Top: 3rd-factor labels centered over each group of 4 epochs
ax_top = ax_n.secondary_xaxis('top')
group_centers = [dj * n_ep_disp + (n_ep_disp - 1) / 2 for dj in range(n_disp)]
ax_top.set_xticks(group_centers)
ax_top.set_xticklabels([DISPLAY_LABELS[b] for b in DISPLAY_ORDER],
                       fontsize=13, color='#F99D20', fontweight='bold')
ax_top.tick_params(axis='x', length=0)
for s in ('top', 'right', 'left', 'bottom'):
    ax_top.spines[s].set_visible(False)

# Row labels: colored math (matches CELL 9 / 10)
ax_n.set_yticks(range(len(CC_MODES)))
ax_n.set_yticklabels([''] * len(CC_MODES))
for i, mode in enumerate(CC_MODES):
    segs = LABEL_SEGMENTS.get(mode, [(mode, OP_COLOR)])
    children = []
    for s in segs:
        if isinstance(s, list):
            sub = [TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE))
                   for (t, c) in s]
            children.append(HPacker(children=sub, align='center', pad=0, sep=0))
        else:
            t, c = s
            children.append(TextArea(t, textprops=dict(color=c, fontsize=LABEL_SIZE)))
    packer = HPacker(children=children, align='center', pad=0, sep=3)
    ab = AnnotationBbox(packer, xy=(0, i), xybox=(-10, 0),
                        xycoords=('axes fraction', 'data'),
                        boxcoords='offset points',
                        box_alignment=(1.0, 0.5),
                        frameon=False, pad=0)
    ax_n.add_artist(ab)

ax_n.tick_params(axis='y', length=0)
ax_n.tick_params(axis='x', length=0)
for s in ('top', 'right', 'left', 'bottom'):
    ax_n.spines[s].set_visible(False)

cbar = plt.colorbar(im, ax=ax_n, shrink=0.75, pad=0.03)
cbar.set_label(r'sign$(\rho)\cdot -\log_{10}(p)$', fontsize=11)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_visible(False)
for thr in (-np.log10(0.05), np.log10(0.05)):
    cbar.ax.axhline(thr, color='k', linewidth=0.8, alpha=0.7)

fig_n.subplots_adjust(left=0.22, right=0.88, top=0.86, bottom=0.14)
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_nested_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_nested_matrix.svg'),
            bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 12: Nested matrix — (eligibility outer, 3rd factor inner) x epoch
# ============================================================================
# y = eligibility (outer, 5 groups of 4 rows) with the four 3rd factors expanded
# within each group; x = epoch. Eligibility colored-math labels on the far left,
# 3rd factor (orange bold) labels repeated to the immediate left of each row.

EPOCH_FULL_LABELS = ['Pre', 'Go cue', 'Late', 'Reward']

n_modes_t = len(CC_MODES)
n_cols_t  = len(EPOCH_ORDER)
n_total_rows_t = n_modes_t * n_disp   # 5 eligibilities x 4 factors = 20

mat_t_signed = np.full((n_total_rows_t, n_cols_t), np.nan)
mat_t_p      = np.full((n_total_rows_t, n_cols_t), np.nan)
for mi, mode in enumerate(CC_MODES):
    for dj, bi in enumerate(beh_idx_disp):
        sign = SIGN_FLIP.get(DISPLAY_ORDER[dj], 1)
        for ek in range(n_cols_t):
            vals = corr_slope[mode][:, bi, ek]
            v = vals[np.isfinite(vals)]
            if len(v) < 3:
                continue
            m = np.mean(v)
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            row = mi * n_disp + dj
            mat_t_signed[row, ek] = (np.sign(m) *
                                     -np.log10(max(p, 1e-300)) * sign)
            mat_t_p[row, ek] = p

vmax_t = max(-np.log10(0.05), np.nanmax(np.abs(mat_t_signed)))

fig_t, ax_t = plt.subplots(1, 1, figsize=(5.0, 10.5))
im_t = ax_t.imshow(mat_t_signed, cmap='coolwarm', vmin=-vmax_t, vmax=vmax_t,
                   aspect='auto', interpolation='nearest')

# Stars
for row in range(n_total_rows_t):
    for ek in range(n_cols_t):
        p = mat_t_p[row, ek]
        if np.isnan(p):
            continue
        if   p < 0.001: sig = '***'
        elif p < 0.01:  sig = '**'
        elif p < 0.05:  sig = '*'
        else:           sig = ''
        if sig:
            ax_t.text(ek, row, sig, ha='center', va='center',
                      fontsize=11, fontweight='bold', color='0.05')

# Black borders between eligibility groups
for k in range(1, n_modes_t):
    ax_t.axhline(k * n_disp - 0.5, color='k', linewidth=1.5)

# Far left: colored-math eligibility labels, centered over each group of 4 rows
group_centers_t = [mi * n_disp + (n_disp - 1) / 2 for mi in range(n_modes_t)]
for mi, mode in enumerate(CC_MODES):
    segs = LABEL_SEGMENTS.get(mode, [(mode, OP_COLOR)])
    children = []
    for s in segs:
        if isinstance(s, list):
            sub = [TextArea(t, textprops=dict(color=c, fontsize=12))
                   for (t, c) in s]
            children.append(HPacker(children=sub, align='center', pad=0, sep=0))
        else:
            t, c = s
            children.append(TextArea(t, textprops=dict(color=c, fontsize=12)))
    packer = HPacker(children=children, align='center', pad=0, sep=3)
    ab = AnnotationBbox(packer, xy=(0, group_centers_t[mi]), xybox=(-58, 0),
                        xycoords=('axes fraction', 'data'),
                        boxcoords='offset points',
                        box_alignment=(1.0, 0.5),
                        frameon=False, pad=0)
    ax_t.add_artist(ab)

# Left ticks: 3rd factor labels (orange bold) repeating, one per row
ax_t.set_yticks(np.arange(n_total_rows_t))
ax_t.set_yticklabels([DISPLAY_LABELS[DISPLAY_ORDER[r % n_disp]]
                      for r in range(n_total_rows_t)],
                     fontsize=10, color='#F99D20', fontweight='bold')

# Bottom: epoch labels
ax_t.set_xticks(range(n_cols_t))
ax_t.set_xticklabels(EPOCH_FULL_LABELS, fontsize=13)

ax_t.tick_params(axis='y', length=0)
ax_t.tick_params(axis='x', length=0)
for s in ('top', 'right', 'left', 'bottom'):
    ax_t.spines[s].set_visible(False)

cbar_t = plt.colorbar(im_t, ax=ax_t, shrink=0.6, pad=0.03)
cbar_t.set_label(r'sign$(\rho)\cdot -\log_{10}(p)$', fontsize=11)
cbar_t.ax.tick_params(labelsize=10)
cbar_t.outline.set_visible(False)
for thr in (-np.log10(0.05), np.log10(0.05)):
    cbar_t.ax.axhline(thr, color='k', linewidth=0.8, alpha=0.7)

fig_t.subplots_adjust(left=0.42, right=0.86, top=0.96, bottom=0.06)
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_nested_matrix_t.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_nested_matrix_t.svg'),
            bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 13: Cleaned 1x5 panel row — behavior x epoch per eligibility
# ============================================================================
# Cleaner version of the original 2x5 figure (CELL 7): drops the intercept row,
# shows signed -log10(p) instead of mean rho, only the first panel keeps the
# 3rd-factor y-tick labels, only the last panel keeps the colorbar.

EPOCH_AXIS_LABELS = ['Pre', 'Go cue', 'Late', 'Reward']

# Per-mode (n_disp x n_epochs) matrices of signed -log10(p) and p
sig_mats = {}
for mode in CC_MODES:
    mat_signed = np.full((n_disp, n_epochs), np.nan)
    mat_p_loc  = np.full((n_disp, n_epochs), np.nan)
    for dj, bi in enumerate(beh_idx_disp):
        sign = SIGN_FLIP.get(DISPLAY_ORDER[dj], 1)
        for ei in range(n_epochs):
            vals = corr_slope[mode][:, bi, ei]
            v = vals[np.isfinite(vals)]
            if len(v) < 3:
                continue
            m = np.mean(v)
            try:
                _, p = wilcoxon(v)
            except Exception:
                p = 1.0
            mat_signed[dj, ei] = (np.sign(m) *
                                  -np.log10(max(p, 1e-300)) * sign)
            mat_p_loc[dj, ei] = p
    sig_mats[mode] = (mat_signed, mat_p_loc)

# Shared color scale
all_vals = np.concatenate([m[0].flatten() for m in sig_mats.values()])
vmax_p   = max(-np.log10(0.05), np.nanmax(np.abs(all_vals)))

fig13, axes13 = plt.subplots(1, len(CC_MODES),
                             figsize=(2.4 * len(CC_MODES), 3.3),
                             squeeze=False)
axes13 = axes13[0]

im13 = None
for col, mode in enumerate(CC_MODES):
    ax = axes13[col]
    mat_signed, mat_p_loc = sig_mats[mode]
    im13 = ax.imshow(mat_signed, cmap='coolwarm',
                     vmin=-vmax_p, vmax=vmax_p,
                     aspect='auto', interpolation='nearest')

    # Stars
    for dj in range(n_disp):
        for ei in range(n_epochs):
            p = mat_p_loc[dj, ei]
            if np.isnan(p):
                continue
            if   p < 0.001: sig = '***'
            elif p < 0.01:  sig = '**'
            elif p < 0.05:  sig = '*'
            else:           sig = ''
            if sig:
                ax.text(ei, dj, sig, ha='center', va='center', fontsize=11,
                        fontweight='bold', color='0.05')

    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels(EPOCH_AXIS_LABELS, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(n_disp))
    if col == 0:
        ax.set_yticklabels([DISPLAY_LABELS[b] for b in DISPLAY_ORDER],
                           fontsize=12, color='#F99D20', fontweight='bold')
    else:
        ax.set_yticklabels([''] * n_disp)
    ax.tick_params(axis='both', length=0)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)

    # Title: eligibility colored math
    segs = LABEL_SEGMENTS.get(mode, [(mode, OP_COLOR)])
    children = []
    for s in segs:
        if isinstance(s, list):
            sub = [TextArea(t, textprops=dict(color=c, fontsize=11))
                   for (t, c) in s]
            children.append(HPacker(children=sub, align='center', pad=0, sep=0))
        else:
            t, c = s
            children.append(TextArea(t, textprops=dict(color=c, fontsize=11)))
    packer = HPacker(children=children, align='center', pad=0, sep=3)
    ab = AnnotationBbox(packer, xy=(0.5, 1), xybox=(0, 10),
                        xycoords='axes fraction',
                        boxcoords='offset points',
                        box_alignment=(0.5, 0.0),
                        frameon=False, pad=0)
    ax.add_artist(ab)

# Colorbar on the last panel only
cbar13 = fig13.colorbar(im13, ax=axes13[-1], shrink=0.85, pad=0.05)
cbar13.set_label(r'sign$(\rho)\cdot -\log_{10}(p)$', fontsize=10)
cbar13.ax.tick_params(labelsize=9)
cbar13.outline.set_visible(False)
for thr in (-np.log10(0.05), np.log10(0.05)):
    cbar13.ax.axhline(thr, color='k', linewidth=0.7, alpha=0.7)

fig13.subplots_adjust(left=0.10, right=0.92, top=0.80, bottom=0.20, wspace=0.15)
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_panel_row.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, 'sliding_window_v3_panel_row.svg'),
            bbox_inches='tight')
plt.show()
