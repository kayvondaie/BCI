#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Frame-by-frame version of 2nd-order variance explained analysis.
#
# Instead of trial-averaged epoch activity, this uses within-trial
# frame-level fluorescence F(t) and step-based RPE to build predictors:
#
#   Local:     Σ_t RPE(t) * r_pre(t) * (r_post(t) - r̄_post)
#   Non-local: Σ_t RPE(t) * r_pre(t) * Σ_k w_{post→k} * (r_k(t) - r̄_k)
#
# Then correlates these predictors with ΔW = AMP[1] - AMP[0].
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

_BCI_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'BCI')
if _BCI_DIR not in sys.path:
    sys.path.insert(0, _BCI_DIR)

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

N_BASELINE = 20       # number of early trials for computing baseline rates
TAU_RPE = 20          # exponential smoothing constant for RPE baseline (in frames)
N_SHUFFLES = 50       # number of connectivity shuffles

CC_MODES = ['1st_order', '2nd_order', 'shuffled', 'random_m']

all_results = {mode: [] for mode in CC_MODES}
print(f"Config: frame-by-frame, TAU_RPE={TAU_RPE}, N_SHUFFLES={N_SHUFFLES}")

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


def compute_framewise_predictors(F, step_time, dt_si, tsta,
                                 cl_weights, all_nt,
                                 post_out_w, has_outgoing,
                                 bl_mean, n_neurons, tau_rpe=TAU_RPE):
    """
    Compute frame-by-frame local and non-local predictors across all trials.

    Parameters
    ----------
    F : (n_frames, n_neurons, n_trials)
    step_time : array of arrays, step times per trial
    dt_si : float, frame period
    tsta : (n_frames,) time axis
    cl_weights : (n_pairs, n_neurons) — presynaptic weights
    all_nt : (n_pairs,) — postsynaptic neuron indices
    post_out_w : (n_pairs, n_neurons) — outgoing connectivity from post
    has_outgoing : (n_pairs,) bool
    bl_mean : (n_neurons,) — baseline activity (mean over first N_BASELINE trials)
    n_neurons : int
    tau_rpe : float — RPE baseline smoothing constant (frames)

    Returns
    -------
    pred_local : (n_pairs,) — local predictor accumulated over all frames/trials
    pred_nonlocal : (n_pairs,) — non-local predictor for pairs with outgoing
    """
    n_frames, _, n_trials = F.shape
    n_pairs = len(all_nt)

    # Clean connectivity matrix
    post_out_w_clean = post_out_w.copy()
    post_out_w_clean[np.isnan(post_out_w_clean)] = 0.0

    # Accumulate predictors
    pred_local = np.zeros(n_pairs)
    pred_nonlocal = np.zeros(n_pairs)

    # Running baseline for RPE (step rate)
    rpe_baseline = 0.0

    # Time indices for the closed-loop period (t >= 0)
    cl_frames = np.where(tsta >= 0)[0]

    for ti in range(n_trials):
        # Get step times for this trial
        steps = step_time[ti] if step_time[ti] is not None else np.array([])
        if isinstance(steps, (list, np.ndarray)):
            steps = np.asarray(steps, dtype=float)
        else:
            steps = np.array([])
        steps = steps[np.isfinite(steps)]

        # Convert step times to frame indices within the trial
        step_frames_in_trial = set()
        for st in steps:
            fi = np.searchsorted(tsta, st)
            if 0 <= fi < n_frames:
                step_frames_in_trial.add(fi)

        # Activity for this trial: (n_frames, n_neurons)
        F_trial = F[:, :, ti].copy()
        F_trial[np.isnan(F_trial)] = 0.0

        # Frame-by-frame accumulation during closed-loop period
        for fi in cl_frames:
            if fi >= n_frames:
                break

            # RPE: step(t) - baseline
            step_now = 1.0 if fi in step_frames_in_trial else 0.0
            rpe = step_now - rpe_baseline
            # Update RPE baseline
            alpha_rpe = 1.0 / tau_rpe
            rpe_baseline = (1 - alpha_rpe) * rpe_baseline + alpha_rpe * step_now

            if abs(rpe) < 1e-10:
                continue  # skip frames with ~zero RPE for efficiency

            r = F_trial[fi, :]  # (n_neurons,)

            # Presynaptic activity: weighted sum of close neurons
            r_pre = cl_weights @ r         # (n_pairs,)

            # Postsynaptic deviation from baseline
            r_post_dev = r[all_nt] - bl_mean[all_nt]  # (n_pairs,)

            # Local: RPE * r_pre * (r_post - bl)
            pred_local += rpe * r_pre * r_post_dev

            # Non-local: RPE * r_pre * Σ_k w_{post→k} * (r_k - bl_k)
            pop_dev = r - bl_mean            # (n_neurons,)
            downstream = post_out_w_clean @ pop_dev  # (n_pairs,)
            downstream[~has_outgoing] = 0.0
            pred_nonlocal += rpe * r_pre * downstream

    return pred_local, pred_nonlocal


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
            tsta = np.arange(0, n_frames * dt_si, dt_si)
            tsta = tsta[:n_frames]
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Pair selection (same as existing script) ----
            dist_target_lt = 10
            dw_list = []
            pair_cl_list = []
            pair_nt_list = []
            pair_gi_list = []

            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < dist_target_lt) &
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
                pair_gi_list.append(np.full(len(nontarg), gi, dtype=int))

            if len(dw_list) == 0:
                print("  No valid pairs.")
                continue

            Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
            all_gi = np.concatenate(pair_gi_list)
            n_pairs = len(Y)

            cl_weights = np.zeros((n_pairs, n_neurons))
            offset = 0
            for gi_idx in range(len(dw_list)):
                n_nt = len(dw_list[gi_idx])
                cl_arr = pair_cl_list[gi_idx]
                for qi in range(n_nt):
                    cl_neurons = cl_arr[qi]
                    cl_weights[offset + qi, cl_neurons] = 1.0 / len(cl_neurons)
                offset += n_nt

            # ---- Baseline activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
            bl_trials = np.arange(min(N_BASELINE, trl))
            bl_mean = np.nanmean(epoch_act[:, bl_trials], axis=1)

            # ---- Outgoing connectivity from post neurons ----
            n_groups = stimDist.shape[1]
            dist_nontarg_min = 30

            post_stim_group = np.full(n_pairs, -1, dtype=int)
            for pi in range(n_pairs):
                nt_neuron = all_nt[pi]
                pair_group = all_gi[pi]
                for gi in range(n_groups):
                    if gi == pair_group:
                        continue
                    if stimDist[nt_neuron, gi] < dist_target_lt:
                        post_stim_group[pi] = gi
                        break

            has_outgoing = post_stim_group >= 0
            n_with_outgoing = np.sum(has_outgoing)
            print(f"  Pairs with outgoing connectivity: "
                  f"{n_with_outgoing}/{n_pairs} "
                  f"({100*n_with_outgoing/n_pairs:.1f}%)")

            post_out_w = np.full((n_pairs, n_neurons), np.nan)
            for pi in range(n_pairs):
                gj = post_stim_group[pi]
                if gj < 0:
                    continue
                w_col = AMP[0][:, gj].copy()
                too_close = stimDist[:, gj] < dist_nontarg_min
                w_col[too_close] = np.nan
                post_out_w[pi, :] = w_col

            # ---- Compute frame-by-frame predictors ----
            pred_local, pred_nonlocal = compute_framewise_predictors(
                F_nan, data['step_time'], dt_si, tsta,
                cl_weights, all_nt,
                post_out_w, has_outgoing,
                bl_mean, n_neurons)

            # ---- Filter to pairs with valid outgoing connectivity ----
            valid_idx = np.where(has_outgoing)[0]
            n_valid = len(valid_idx)
            print(f"  Using {n_valid} pairs with outgoing connectivity")

            if n_valid < 10:
                print(f"  Too few valid pairs, skipping.")
                continue

            Y_valid = Y[valid_idx]
            local_valid = pred_local[valid_idx]
            nonlocal_valid = pred_nonlocal[valid_idx]

            # ---- Partial correlation: non-local | local ----
            def partial_corr(Y, X_remove, X_test):
                if np.std(X_remove) < 1e-15 or np.std(X_test) < 1e-15:
                    return 0.0, 1.0
                b = np.dot(Y - Y.mean(), X_remove - X_remove.mean()) / \
                    (np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean()) + 1e-30)
                Y_resid = Y - (Y.mean() + b * (X_remove - X_remove.mean()))
                b2 = np.dot(X_test - X_test.mean(), X_remove - X_remove.mean()) / \
                     (np.dot(X_remove - X_remove.mean(), X_remove - X_remove.mean()) + 1e-30)
                X_resid = X_test - (X_test.mean() + b2 * (X_remove - X_remove.mean()))
                if np.std(Y_resid) < 1e-15 or np.std(X_resid) < 1e-15:
                    return 0.0, 1.0
                return pearsonr(Y_resid, X_resid)

            # 1st order: raw correlation of local predictor with ΔW
            r_1st, p_1st = pearsonr(Y_valid, local_valid) if np.std(local_valid) > 1e-15 else (0.0, 1.0)
            all_results['1st_order'].append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_valid, 'n_pairs_total': n_pairs,
                'r_test': r_1st, 'p_test': p_1st,
            })

            # 2nd order: partial correlation of non-local | local
            r_2nd, p_2nd = partial_corr(Y_valid, local_valid, nonlocal_valid)
            all_results['2nd_order'].append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_valid, 'n_pairs_total': n_pairs,
                'r_test': r_2nd, 'p_test': p_2nd,
            })

            # ---- Shuffle: permute outgoing weights ----
            rng = np.random.default_rng(42)
            shuf_r_tests = np.zeros(N_SHUFFLES)
            post_out_w_valid = post_out_w[valid_idx, :]
            for shi in range(N_SHUFFLES):
                perm = rng.permutation(n_valid)
                post_out_shuf = post_out_w_valid.copy()
                post_out_shuf_clean = post_out_shuf[perm, :]
                post_out_shuf_clean[np.isnan(post_out_shuf_clean)] = 0.0

                # Recompute non-local predictor with shuffled weights
                pred_nl_shuf = np.zeros(n_valid)
                # Re-run frame-by-frame with shuffled W
                for ti in range(trl):
                    steps = data['step_time'][ti] if data['step_time'][ti] is not None else np.array([])
                    if isinstance(steps, (list, np.ndarray)):
                        steps = np.asarray(steps, dtype=float)
                    else:
                        steps = np.array([])
                    steps = steps[np.isfinite(steps)]

                    step_frames_in_trial = set()
                    for st in steps:
                        fi = np.searchsorted(tsta, st)
                        if 0 <= fi < n_frames:
                            step_frames_in_trial.add(fi)

                    F_trial = F_nan[:, :, ti]
                    cl_frames = np.where(tsta >= 0)[0]
                    rpe_baseline_sh = 0.0

                    for fi in cl_frames:
                        if fi >= n_frames:
                            break
                        step_now = 1.0 if fi in step_frames_in_trial else 0.0
                        rpe = step_now - rpe_baseline_sh
                        alpha_rpe = 1.0 / TAU_RPE
                        rpe_baseline_sh = (1 - alpha_rpe) * rpe_baseline_sh + alpha_rpe * step_now
                        if abs(rpe) < 1e-10:
                            continue
                        r = F_trial[fi, :]
                        r_pre = cl_weights[valid_idx, :] @ r
                        pop_dev = r - bl_mean
                        downstream = post_out_shuf_clean @ pop_dev
                        pred_nl_shuf += rpe * r_pre * downstream

                r_sh, _ = partial_corr(Y_valid, local_valid, pred_nl_shuf)
                shuf_r_tests[shi] = r_sh

            shuf_pval = np.mean(shuf_r_tests >= r_2nd) if np.isfinite(r_2nd) else 1.0
            all_results['shuffled'].append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_valid, 'n_pairs_total': n_pairs,
                'r_test': np.median(shuf_r_tests),
                'p_test': shuf_pval,
                'shuf_distribution': shuf_r_tests,
            })

            # ---- Random-m control ----
            all_groups = np.arange(n_groups)
            random_m_w = np.full((n_valid, n_neurons), 0.0)
            for vi, pi in enumerate(valid_idx):
                pair_group = all_gi[pi]
                candidates = all_groups[all_groups != pair_group]
                if len(candidates) == 0:
                    continue
                m = rng.choice(candidates)
                w_col = AMP[0][:, m].copy()
                too_close = stimDist[:, m] < dist_nontarg_min
                w_col[too_close] = np.nan
                w_col = np.nan_to_num(w_col, nan=0.0)
                random_m_w[vi, :] = w_col

            # Recompute non-local with random-m weights
            pred_nl_rand = np.zeros(n_valid)
            for ti in range(trl):
                steps = data['step_time'][ti] if data['step_time'][ti] is not None else np.array([])
                if isinstance(steps, (list, np.ndarray)):
                    steps = np.asarray(steps, dtype=float)
                else:
                    steps = np.array([])
                steps = steps[np.isfinite(steps)]

                step_frames_in_trial = set()
                for st in steps:
                    fi = np.searchsorted(tsta, st)
                    if 0 <= fi < n_frames:
                        step_frames_in_trial.add(fi)

                F_trial = F_nan[:, :, ti]
                cl_frames = np.where(tsta >= 0)[0]
                rpe_baseline_rm = 0.0

                for fi in cl_frames:
                    if fi >= n_frames:
                        break
                    step_now = 1.0 if fi in step_frames_in_trial else 0.0
                    rpe = step_now - rpe_baseline_rm
                    alpha_rpe = 1.0 / TAU_RPE
                    rpe_baseline_rm = (1 - alpha_rpe) * rpe_baseline_rm + alpha_rpe * step_now
                    if abs(rpe) < 1e-10:
                        continue
                    r = F_trial[fi, :]
                    r_pre = cl_weights[valid_idx, :] @ r
                    pop_dev = r - bl_mean
                    downstream = random_m_w @ pop_dev
                    pred_nl_rand += rpe * r_pre * downstream

            r_rand, p_rand = partial_corr(Y_valid, local_valid, pred_nl_rand)
            all_results['random_m'].append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_valid, 'n_pairs_total': n_pairs,
                'r_test': r_rand, 'p_test': p_rand,
            })

            # Print summary
            parts = []
            for mode in CC_MODES:
                r = all_results[mode][-1]
                sig = '*' if r['p_test'] < 0.05 else ''
                parts.append(f"{mode}={r['r_test']:.3f}{sig}")
            print(f"  {n_valid} pairs | " + ", ".join(parts))

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

for mode in CC_MODES:
    print(f"{mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(os.path.join(RESULTS_DIR, 'variance_explained_framewise.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'variance_explained_framewise.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
print(f"Loaded: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 6: Summary figure
# ============================================================================
mode_labels = {
    '1st_order': '1st order\n(local)',
    '2nd_order': '2nd order\n(non-local)',
    'shuffled': 'Shuffled\n(scrambled w)',
    'random_m': 'Random m\n(wrong neuron w)',
}
mode_colors = {
    '1st_order': '#3498db',
    '2nd_order': '#e74c3c',
    'shuffled': '#f39c12',
    'random_m': '#2ecc71',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Frame-by-frame 2nd-order analysis\n"
             "Local: Σ RPE·r_pre·(r_post−bl)  |  "
             "Non-local: Σ RPE·r_pre·Σ_k w_{post→k}·(r_k−bl_k)",
             fontsize=12)

# --- Panel A: Bar plot ---
ax = axes[0]
mode_order = ['1st_order', '2nd_order', 'shuffled', 'random_m']
medians = []
sems = []

for mode in mode_order:
    rt = np.array([s['r_test'] for s in all_results[mode]])
    medians.append(np.median(rt))
    sems.append(np.std(rt) / np.sqrt(len(rt)))

x = np.arange(len(mode_order))
ax.bar(x, medians, yerr=sems, capsize=5,
       color=[mode_colors[m] for m in mode_order],
       edgecolor='k', linewidth=1.2, alpha=0.85)

# Add significance
for xi, mode in enumerate(mode_order):
    rt = np.array([s['r_test'] for s in all_results[mode]])
    try:
        _, p = wilcoxon(rt)
    except:
        p = 1.0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(xi, medians[xi] + sems[xi] + 0.002, sig,
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels([mode_labels[m] for m in mode_order], fontsize=10)
ax.set_ylabel('Median r (or partial r)')
ax.set_title('Framewise dW prediction', fontweight='bold')

# --- Panel B: 2nd order vs shuffled ---
ax = axes[1]
rt_2nd = np.array([s['r_test'] for s in all_results['2nd_order']])
rt_shuf = np.array([s['r_test'] for s in all_results['shuffled']])

ax.scatter(rt_shuf, rt_2nd, c='#f39c12', s=40, alpha=0.7,
           edgecolor='k', linewidth=0.5)
lims = [min(np.min(rt_shuf), np.min(rt_2nd)) - 0.02,
        max(np.max(rt_shuf), np.max(rt_2nd)) + 0.02]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('Shuffled (scrambled connectivity)')
ax.set_ylabel('2nd order (true connectivity)')
ax.set_title('True vs scrambled', fontweight='bold')
n_above = np.sum(rt_2nd > rt_shuf)
ax.text(0.05, 0.95, f'{n_above}/{len(rt_2nd)} true > shuf',
        transform=ax.transAxes, va='top', fontsize=11)

# --- Panel C: 2nd order vs random-m ---
ax = axes[2]
rt_rand = np.array([s['r_test'] for s in all_results['random_m']])

ax.scatter(rt_rand, rt_2nd, c='#2ecc71', s=40, alpha=0.7,
           edgecolor='k', linewidth=0.5)
lims = [min(np.min(rt_rand), np.min(rt_2nd)) - 0.02,
        max(np.max(rt_rand), np.max(rt_2nd)) + 0.02]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('Random-m (wrong neuron weights)')
ax.set_ylabel('2nd order (true connectivity)')
ax.set_title('True vs random neuron', fontweight='bold')
n_above2 = np.sum(rt_2nd > rt_rand)
ax.text(0.05, 0.95, f'{n_above2}/{len(rt_2nd)} true > random',
        transform=ax.transAxes, va='top', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_framewise_2nd_order.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")
