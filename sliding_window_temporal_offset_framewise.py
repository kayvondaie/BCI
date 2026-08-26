#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Framewise variant of sliding_window_temporal_offset_v2.py.

For each (pair, window, epoch) the coactivity is

    CC = sum over (frames in epoch) x (trials in window) of
         pre(t, ti) * (post(t, ti) - bl_post)

where bl_post is the per-neuron, full-trial-mean baseline averaged across
the first N_BASELINE trials (same baseline as v2's dev2_fulltrial_baseline).

The only CC mode is zero-lag dev2 with frame-by-frame multiplication.

Difference from v2: v2 averages each (neuron, trial) within the epoch first,
then forms the dot product across trials.  This script multiplies pre(t) and
post(t)-bl pointwise at each frame, then sums over both frames and trials.
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
N_BASELINE = 20

CC_MODE = 'dev2_framewise'   # zero-lag, framewise pre(t) * (post(t) - bl)

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

all_results = []
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"CC mode: {CC_MODE}")

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

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Per-trial behavior
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

            # ---- Pair selection (identical to v2) ----
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

            # ---- Sliding windows ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # ---- Prepare F (NaN -> 0) ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            # ---- Baseline: per-neuron full-trial mean over first N_BASELINE trials ----
            baseline_trials_arr = np.arange(min(N_BASELINE, trl))
            fulltrial_post_act = np.nanmean(F_nan, axis=0)        # (n_neurons, trl)
            bl_mean = np.nanmean(
                fulltrial_post_act[:, baseline_trials_arr], axis=1)  # (n_neurons,)

            # ---- Epoch frame indices ----
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]

            # ---- Per-trial accumulated product (n_pairs, n_trials) per epoch ----
            # We compute by group: pairs in the same group share their cl set,
            # so we average pre activity once per group rather than per pair.
            cc_per_trial = {ep: np.zeros((n_pairs, trl)) for ep in EPOCH_ORDER}

            pair_offset = 0
            for gi_idx in range(len(dw_list)):
                cl = pair_cl_list[gi_idx][0]
                nontarg = pair_nt_list[gi_idx]
                n_nt = len(nontarg)
                bl_g = bl_mean[nontarg]                          # (n_nt,)

                # Pre and go_cue: fixed contiguous frame ranges
                for ep, idx in (('pre', ts_pre), ('go_cue', ts_go)):
                    if len(idx) == 0:
                        continue
                    f0, f1 = idx[0], idx[-1] + 1
                    F_ep = F_nan[f0:f1, :, :]                    # view
                    pre_g = F_ep[:, cl, :].mean(axis=1)          # (f_ep, trl)
                    post_g = F_ep[:, nontarg, :]                 # (f_ep, n_nt, trl)
                    raw_prod = np.einsum('ft,fnt->nt', pre_g, post_g)
                    sum_pre = pre_g.sum(axis=0)                  # (trl,)
                    prod_ep = raw_prod - bl_g[:, None] * sum_pre[None, :]
                    cc_per_trial[ep][pair_offset:pair_offset + n_nt, :] = prod_ep

                # Late and reward: per-trial reward-aligned windows
                for ti in range(trl):
                    rewards = data['reward_time'][ti]
                    if len(rewards) == 0:
                        continue
                    for ep, pre_n, post_n in (('late', 20, 1), ('reward', 1, 10)):
                        indices = get_indices_around_steps(
                            tsta, rewards, pre=pre_n, post=post_n)
                        indices = indices[(indices >= 0) & (indices < n_frames)]
                        if len(indices) == 0:
                            continue
                        F_t = F_nan[indices, :, ti]              # (f, n_neurons)
                        pre_t = F_t[:, cl].mean(axis=1)          # (f,)
                        post_t = F_t[:, nontarg]                 # (f, n_nt)
                        raw_prod_t = pre_t @ post_t              # (n_nt,)
                        sum_pre_t = pre_t.sum()
                        prod_t = raw_prod_t - bl_g * sum_pre_t
                        cc_per_trial[ep][
                            pair_offset:pair_offset + n_nt, ti] = prod_t

                pair_offset += n_nt

            # ---- Window-level CC: sum per-trial product over trials in window ----
            cc_arr = np.full((n_wins, n_pairs, n_epochs), np.nan)
            for ei, ep in enumerate(EPOCH_ORDER):
                cct = cc_per_trial[ep]
                for wi, ws in enumerate(win_starts):
                    cc_arr[wi, :, ei] = cct[:, ws:ws + WIN_SIZE].sum(axis=1)

            # ---- Behavioral aggregates per window ----
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

            # ---- HI fits per (window, epoch) ----
            hi_no_int = np.full((n_wins, n_epochs), np.nan)
            hi_with_int = np.full((n_wins, n_epochs), np.nan)
            hi_intercept = np.full((n_wins, n_epochs), np.nan)
            hi_corr = np.full((n_wins, n_epochs), np.nan)

            for ei in range(n_epochs):
                cc_all = cc_arr[:, :, ei]
                for wi in range(n_wins):
                    cc_pair = cc_all[wi, :]
                    if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                        continue
                    hi_no_int[wi, ei] = (
                        np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair))
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
                # Per-pair pre-epoch CC for downstream flip analysis
                'cc_pre_pair': cc_arr[:, :, 0],
                'Y_T': Y_T,
            }
            all_results.append(result)
            print(f"  {n_wins} windows, {n_pairs} pairs")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nTotal sessions: {len(all_results)}")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_framewise.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_framewise.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_results)} sessions")

#%% ============================================================================
# CELL 6: Within-session correlations
# ============================================================================
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

n_s = len(all_results)
corr_slope = np.full((n_s, n_beh, n_epochs), np.nan)
corr_intercept = np.full((n_s, n_beh, n_epochs), np.nan)

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
                corr_slope[si, bi, ei], _ = spearmanr(bvar[ok], slope[ok])
            ok2 = np.isfinite(bvar) & np.isfinite(intercept)
            if np.sum(ok2) >= 5 and np.std(intercept[ok2]) > 0:
                corr_intercept[si, bi, ei], _ = spearmanr(bvar[ok2], intercept[ok2])

print("Within-session correlations computed.")

#%% ============================================================================
# CELL 7: Coefficient matrix (behavior x epoch)
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for col, (corr_arr, row_label) in enumerate([
    (corr_slope, 'Slope'),
    (corr_intercept, 'Intercept'),
]):
    ax = axes[col]
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
            sig = ('***' if p < 0.001 else
                   '**'  if p < 0.01  else
                   '*'   if p < 0.05  else '')
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
    ax.set_title(f'{row_label}', fontsize=13, fontweight='bold')

fig.suptitle(f'Framewise dev2 (zero lag, n={n_s} sessions)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
            'fig_temporal_offset_framewise_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 8: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'temporal_offset_framewise_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("FRAMEWISE COACTIVITY (zero lag, dev2)\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write("=" * 70 + "\n\n")
    f.write("CC = sum over (frames in epoch) x (trials in window) of\n")
    f.write("     pre(t) * (post(t) - bl_post)\n")
    f.write("bl_post = per-neuron full-trial mean over first N_BASELINE trials\n")
    f.write("(matches v2 dev2_fulltrial_baseline).\n\n")

    epoch_labels_rpt = ['pre', 'go_cue', 'late', 'reward']

    for target, corr_arr, label in [
        ('slope', corr_slope, 'BEHAVIOR vs SLOPE'),
        ('intercept', corr_intercept, 'BEHAVIOR vs INTERCEPT'),
    ]:
        f.write(f"\n{label}\n")
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
                sig = ('***' if p < 0.001 else
                       '**'  if p < 0.01  else
                       '*'   if p < 0.05  else '')
                row_name = f"{bname}_{ep}"
                f.write(f"  {row_name:25s} {m:+7.3f} {md:+7.3f} "
                        f"{fpos:4.0f}% {p:10.4f} {sig:>4s}\n")
            f.write("\n")
print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 9: Binned HI slope vs behavior
# ============================================================================
n_bins = 3
beh_plot = [('win_rpe', 'RPE'), ('win_hit', 'Hit rate')]
ep_indices = list(range(n_epochs))
ep_names = epoch_labels
colors = ['#c0392b', '#e67e22', '#27ae60', '#2980b9']

fig, axes = plt.subplots(1, len(beh_plot), figsize=(5 * len(beh_plot), 4),
                         squeeze=False)

for col, (beh_key, beh_label) in enumerate(beh_plot):
    ax = axes[0, col]

    for ei, ep_name, clr in zip(ep_indices, ep_names, colors):
        all_beh_z = []
        all_slope_z = []
        for s in all_results:
            bvar = s[beh_key]
            slope = s['hi_with_int'][:, ei]
            ok = np.isfinite(bvar) & np.isfinite(slope)
            if np.sum(ok) < 5:
                continue
            bvar_ok = bvar[ok]
            slope_ok = slope[ok]
            if np.std(bvar_ok) == 0 or np.std(slope_ok) == 0:
                continue
            all_beh_z.append((bvar_ok - np.mean(bvar_ok)) / np.std(bvar_ok))
            all_slope_z.append((slope_ok - np.mean(slope_ok)) / np.std(slope_ok))
        if len(all_beh_z) == 0:
            continue
        all_beh_z = np.concatenate(all_beh_z)
        all_slope_z = np.concatenate(all_slope_z)

        bin_edges = np.percentile(all_beh_z, np.linspace(0, 100, n_bins + 1))
        bc, bm, bs = [], [], []
        for bi in range(n_bins):
            if bi < n_bins - 1:
                mask = (all_beh_z >= bin_edges[bi]) & (all_beh_z < bin_edges[bi + 1])
            else:
                mask = (all_beh_z >= bin_edges[bi]) & (all_beh_z <= bin_edges[bi + 1])
            if np.sum(mask) < 3:
                continue
            bc.append(np.mean(all_beh_z[mask]))
            bm.append(np.mean(all_slope_z[mask]))
            bs.append(np.std(all_slope_z[mask]) / np.sqrt(np.sum(mask)))

        ax.errorbar(bc, bm, yerr=bs, fmt='o-', color=clr, capsize=5,
                    linewidth=2, markersize=7, label=ep_name)

    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel(f'{beh_label} (within-session z)')
    ax.set_ylabel('Slope (within-session z)')
    ax.legend(fontsize=9, loc='best')

fig.suptitle(f'Framewise dev2 (zero lag) — binned HI slope vs behavior '
             f'(n={n_s} sessions, {n_bins} bins)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
            'fig_temporal_offset_framewise_binned.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Binned figure saved.")
