#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Direct test of DWij = sum_t(RPE_t * ri_t * rj_t).

No free parameters. For each pair (i,j):
  predicted_dW_ij = sum_t RPE_t * CC_ij_t

Then correlate predicted vs actual dW across pairs within each session.
Compare RPE-weighted vs unweighted (cumulative CC alone) to isolate RPE's
contribution.

Two CC modes: dot_prod and dev2.
Four epochs: pre, go_cue, late, reward.
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

tau_elig = 10
N_BASELINE = 20

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
CC_MODES = ['dot_prod', 'dev2']

# Restrict sum to first N trials (set None to use all)
MAX_TRIALS = 40

# Pair selection
dist_target_lt = 10
dist_nontarg_min = 30
dist_nontarg_max = 1000
amp0_thr = 0.1
amp1_thr = 0.1

print(f"Epochs: {EPOCH_ORDER}")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop — compute per-pair RPE-weighted CC and dW
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

all_sessions = []

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

            # ---- Build pair selection ----
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

            # ---- Compute epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0

            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]

            epoch_pre_act = {}
            epoch_post_act = {}

            for ep in ['pre', 'go_cue']:
                if ep == 'pre':
                    t0, t1 = ts_pre[0], ts_pre[-1]
                else:
                    t0, t1 = ts_go[0], ts_go[-1]
                epoch_pre_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)
                epoch_post_act[ep] = np.nanmean(F_nan[t0:t1+1, :, :], axis=0)

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
                        epoch_post_act['late'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < n_frames]
                    if len(indices) > 0:
                        epoch_pre_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)
                        epoch_post_act['reward'][:, ti] = np.nanmean(F_nan[indices, :, ti], axis=0)

            # Baseline for dev2 (mean post activity over first N_BASELINE trials)
            baseline_trials = np.arange(min(N_BASELINE, trl))

            # ---- Per-pair, per-trial CC and RPE-weighted CC ----
            # For each pair and epoch, compute:
            #   CC_ij_t = r_pre_t * r_post_t   (one scalar per trial)
            #   RPE_CC_ij = sum_t(RPE_t * CC_ij_t)
            #   cumCC_ij  = sum_t(CC_ij_t)

            sess_result = {
                'mouse': mouse,
                'session': session,
                'n_pairs': n_pairs,
                'n_trials': trl,
                'hit_rate': np.nanmean(hit),
            }

            for cc_mode in CC_MODES:
                for ei, ep in enumerate(EPOCH_ORDER):
                    pre_act = cl_weights @ epoch_pre_act[ep]  # (n_pairs, trl)
                    post_act = epoch_post_act[ep][all_nt, :]  # (n_pairs, trl)

                    if cc_mode == 'dev2':
                        baseline_ep = np.nanmean(epoch_post_act[ep][:, baseline_trials], axis=1)
                        post_act = post_act - baseline_ep[all_nt, np.newaxis]

                    # CC_ij_t: per-pair, per-trial coactivity
                    cc_per_trial = pre_act * post_act  # (n_pairs, trl)

                    # Restrict to first MAX_TRIALS
                    t_max = min(trl, MAX_TRIALS) if MAX_TRIALS is not None else trl
                    cc_trunc = cc_per_trial[:, :t_max]
                    rpe_trunc = rt_rpe[:t_max]

                    # RPE-weighted sum across trials
                    rpe_weighted = cc_trunc @ rpe_trunc  # (n_pairs,)
                    # Unweighted (cumulative) sum across trials
                    cumulative = np.nansum(cc_trunc, axis=1)  # (n_pairs,)

                    # Correlate with dW
                    mask = np.isfinite(rpe_weighted) & np.isfinite(Y_T)
                    if np.sum(mask) > 5 and np.std(rpe_weighted[mask]) > 0:
                        r_rpe, p_rpe = spearmanr(rpe_weighted[mask], Y_T[mask])
                    else:
                        r_rpe, p_rpe = np.nan, np.nan

                    mask2 = np.isfinite(cumulative) & np.isfinite(Y_T)
                    if np.sum(mask2) > 5 and np.std(cumulative[mask2]) > 0:
                        r_cum, p_cum = spearmanr(cumulative[mask2], Y_T[mask2])
                    else:
                        r_cum, p_cum = np.nan, np.nan

                    key = f"{cc_mode}_{ep}"
                    sess_result[f'{key}_r_rpe'] = r_rpe
                    sess_result[f'{key}_p_rpe'] = p_rpe
                    sess_result[f'{key}_r_cum'] = r_cum
                    sess_result[f'{key}_p_cum'] = p_cum

            all_sessions.append(sess_result)
            print(f"  {n_pairs} pairs, {trl} trials")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_sessions)} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'direct_three_factor_test.npy'),
        all_sessions, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_sessions = np.load(
    os.path.join(RESULTS_DIR, 'direct_three_factor_test.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_sessions)} sessions")

#%% ============================================================================
# CELL 6: Summary — RPE-weighted vs cumulative CC predicting dW
# ============================================================================
"""
For each epoch × CC mode, plot per-session Spearman rho distributions.
Two columns: RPE-weighted (sum RPE*CC) vs Cumulative (sum CC).
Wilcoxon test for each.
"""
EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
CC_MODES = ['dot_prod', 'dev2']
n_epochs = len(EPOCH_ORDER)
n_modes = len(CC_MODES)

fig, axes = plt.subplots(n_epochs, 2 * n_modes, figsize=(16, 12))

for ei, ep in enumerate(EPOCH_ORDER):
    for mi, cc_mode in enumerate(CC_MODES):
        for ci, (corr_type, label) in enumerate([('r_rpe', 'RPE × CC'),
                                                  ('r_cum', 'Σ CC')]):
            ax_idx = mi * 2 + ci
            ax = axes[ei, ax_idx]

            key = f"{cc_mode}_{ep}_{corr_type}"
            vals = np.array([s[key] for s in all_sessions])
            v = vals[np.isfinite(vals)]

            if len(v) >= 3:
                stat, p = wilcoxon(v)
            else:
                stat, p = np.nan, np.nan

            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(v))
            ax.scatter(jitter, v, s=25, alpha=0.6, edgecolors='k',
                       linewidths=0.3, zorder=3,
                       color='steelblue' if ci == 0 else 'coral')
            m = np.mean(v)
            sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar(0, m, yerr=sem, fmt='D', color='k', markersize=6,
                        capsize=4, zorder=4)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            # Tight y-axis based on data
            if len(v) > 0:
                ypad = max(np.max(np.abs(v)) * 1.3, 0.01)
                ax.set_ylim(-ypad, ypad)

            sig = ''
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'

            ax.set_title(f'{cc_mode} | {label}\n{ep} {sig} p={p:.3f}',
                         fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel(f'{ep}\nSpearman ρ(pred, dW)')

_trial_label = f'first {MAX_TRIALS} trials' if MAX_TRIALS else 'all trials'
fig.suptitle(f'Direct three-factor test ({_trial_label}): corr(predicted dW, actual dW)\n'
             'RPE-weighted vs cumulative CC across pairs',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'direct_three_factor_summary.png'),
            dpi=200, bbox_inches='tight')
plt.show()

#%% ============================================================================
# CELL 7: Head-to-head — does RPE weighting improve prediction?
# ============================================================================
"""
For each session, plot r(RPE*CC, dW) vs r(CC, dW).
If RPE weighting helps, points should fall above the diagonal.
One panel per epoch, dot_prod vs dev2 overlaid.
"""
fig, axes = plt.subplots(1, n_epochs, figsize=(16, 4))

for ei, ep in enumerate(EPOCH_ORDER):
    ax = axes[ei]
    for mi, (cc_mode, color, marker) in enumerate(
            [('dot_prod', 'steelblue', 'o'), ('dev2', 'coral', 's')]):
        r_rpe_vals = np.array([s[f'{cc_mode}_{ep}_r_rpe'] for s in all_sessions])
        r_cum_vals = np.array([s[f'{cc_mode}_{ep}_r_cum'] for s in all_sessions])
        mask = np.isfinite(r_rpe_vals) & np.isfinite(r_cum_vals)
        ax.scatter(r_cum_vals[mask], r_rpe_vals[mask],
                   s=30, alpha=0.6, color=color, marker=marker,
                   edgecolors='k', linewidths=0.3, label=cc_mode)

    # Auto-scale axis limits to data
    all_vals = []
    for cc_mode in CC_MODES:
        rv = np.array([s[f'{cc_mode}_{ep}_r_rpe'] for s in all_sessions])
        cv = np.array([s[f'{cc_mode}_{ep}_r_cum'] for s in all_sessions])
        m = np.isfinite(rv) & np.isfinite(cv)
        all_vals.extend(rv[m].tolist() + cv[m].tolist())
    if len(all_vals) > 0:
        pad = max(np.max(np.abs(all_vals)) * 1.3, 0.01)
    else:
        pad = 0.1
    lims = [-pad, pad]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('r(Σ CC, dW)')
    ax.set_ylabel('r(Σ RPE×CC, dW)')
    ax.set_title(ep, fontweight='bold')
    ax.set_aspect('equal')
    if ei == 0:
        ax.legend(fontsize=9)

fig.suptitle('Does RPE weighting improve dW prediction?\n'
             'Above diagonal = RPE helps',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'direct_three_factor_rpe_vs_cum.png'),
            dpi=200, bbox_inches='tight')
plt.show()

# Print summary: paired Wilcoxon comparing |r_rpe| vs |r_cum|
print("\nDoes RPE weighting improve |r| over cumulative CC?")
print(f"{'Epoch':<10} {'Mode':<12} {'mean |r_rpe|':>12} {'mean |r_cum|':>12} {'p (paired)':>12}")
print("-" * 60)
for ep in EPOCH_ORDER:
    for cc_mode in CC_MODES:
        r_rpe = np.array([s[f'{cc_mode}_{ep}_r_rpe'] for s in all_sessions])
        r_cum = np.array([s[f'{cc_mode}_{ep}_r_cum'] for s in all_sessions])
        mask = np.isfinite(r_rpe) & np.isfinite(r_cum)
        if np.sum(mask) >= 3:
            abs_diff = np.abs(r_rpe[mask]) - np.abs(r_cum[mask])
            _, p_paired = wilcoxon(abs_diff)
            print(f"{ep:<10} {cc_mode:<12} {np.mean(np.abs(r_rpe[mask])):>12.4f} "
                  f"{np.mean(np.abs(r_cum[mask])):>12.4f} {p_paired:>12.4f}")
