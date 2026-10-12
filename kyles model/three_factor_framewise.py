#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Frame-by-frame version of 2nd-order variance explained analysis.
#
# Like three_factor_variance_explained_2nd_order.py but computes per-trial
# coactivity features by summing frame-by-frame within each trial:
#
#   1st order:  CC_trial = Σ_t r_pre(t) * (r_post(t) - bl)
#   2nd order:  CC_trial = Σ_t r_pre(t) * Σ_k w_{post→k} * (r_k(t) - bl_k)
#
# Then uses sliding-window MLR + CV to predict ΔW (same as original script).
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
from sklearn.model_selection import KFold
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

N_BASELINE = 20
WIN_SIZE = 10
WIN_STEP = 5
n_cv_folds = 5
N_SHUFFLES = 50

PAIRWISE_MODE = 'dev2'  # 'dev2' = r_pre * (r_post - bl)

CC_MODES = ['1st_order', '2nd_order', 'combined', 'shuffled', 'random_m']

all_results = {mode: [] for mode in CC_MODES}
print(f"Config: framewise CC, win={WIN_SIZE}, step={WIN_STEP}, "
      f"pinv, {n_cv_folds}-fold CV")
print(f"Pairwise mode: {PAIRWISE_MODE}")

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
            tsta = np.arange(0, n_frames * dt_si, dt_si)
            tsta = tsta[:n_frames]
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Pair selection ----
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

            # ---- Baseline activity (from pre-epoch of first N_BASELINE trials) ----
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

            # ---- Compute per-trial framewise CC (vectorized) ----
            # Closed-loop frames (t >= 0)
            cl_frames = np.where(tsta >= 0)[0]
            post_out_w_clean = post_out_w.copy()
            post_out_w_clean[np.isnan(post_out_w_clean)] = 0.0

            # cc_1st[pair, trial] = Σ_t r_pre(t) * (r_post(t) - bl)
            # cc_2nd[pair, trial] = Σ_t r_pre(t) * Σ_k w_{post→k} * (r_k(t) - bl_k)
            cc_1st = np.zeros((n_pairs, trl))
            cc_2nd = np.zeros((n_pairs, trl))

            for ti in range(trl):
                # Activity during closed-loop: (n_cl, n_neurons)
                F_cl = F_nan[cl_frames, :, ti]

                # Presynaptic activity: (n_cl, n_pairs)
                r_pre = F_cl @ cl_weights.T

                if PAIRWISE_MODE == 'dev2':
                    # Postsynaptic deviation: (n_cl, n_pairs)
                    r_post_dev = F_cl[:, all_nt] - bl_mean[all_nt][None, :]
                    # 1st order: sum over frames
                    cc_1st[:, ti] = np.sum(r_pre * r_post_dev, axis=0)
                    # Population deviation for 2nd order
                    pop_dev = F_cl - bl_mean[None, :]
                else:  # dot_prod
                    r_post = F_cl[:, all_nt]
                    cc_1st[:, ti] = np.sum(r_pre * r_post, axis=0)
                    pop_dev = F_cl

                # Downstream sum: (n_cl, n_pairs)
                downstream = pop_dev @ post_out_w_clean.T
                downstream[:, ~has_outgoing] = 0.0
                cc_2nd[:, ti] = np.sum(r_pre * downstream, axis=0)

            # ---- Filter to pairs with valid outgoing connectivity ----
            valid_idx = np.where(has_outgoing)[0]
            n_valid = len(valid_idx)
            print(f"  Using {n_valid} pairs for 2nd-order modes")

            # ---- Sliding window design matrices ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 3:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            X_1st_v = np.zeros((n_valid, n_wins))
            X_2nd_v = np.zeros((n_valid, n_wins))

            for wi, ws in enumerate(win_starts):
                X_1st_v[:, wi] = np.sum(cc_1st[valid_idx, ws:ws+WIN_SIZE], axis=1)
                X_2nd_v[:, wi] = np.sum(cc_2nd[valid_idx, ws:ws+WIN_SIZE], axis=1)

            X_comb_v = np.hstack([X_1st_v, X_2nd_v])

            def zscore_cols(X):
                mu = X.mean(axis=0)
                sig = X.std(axis=0)
                sig[sig == 0] = 1.0
                return (X - mu) / sig

            # ---- Shuffle control: permute outgoing weights, recompute cc_2nd ----
            rng = np.random.default_rng(42)
            X_shuf_list = []
            for shi in range(N_SHUFFLES):
                perm = rng.permutation(n_valid)
                post_out_shuf = post_out_w_clean[valid_idx, :][perm, :]
                # Recompute cc_2nd with shuffled weights
                cc_2nd_shuf = np.zeros((n_valid, trl))
                for ti in range(trl):
                    F_cl = F_nan[cl_frames, :, ti]
                    r_pre = F_cl @ cl_weights[valid_idx, :].T
                    pop_dev = F_cl - bl_mean[None, :] if PAIRWISE_MODE == 'dev2' else F_cl
                    downstream = pop_dev @ post_out_shuf.T
                    cc_2nd_shuf[:, ti] = np.sum(r_pre * downstream, axis=0)

                X_sh = np.zeros((n_valid, n_wins))
                for wi, ws in enumerate(win_starts):
                    X_sh[:, wi] = np.sum(cc_2nd_shuf[:, ws:ws+WIN_SIZE], axis=1)
                X_shuf_list.append(X_sh)

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

            cc_2nd_rand = np.zeros((n_valid, trl))
            for ti in range(trl):
                F_cl = F_nan[cl_frames, :, ti]
                r_pre = F_cl @ cl_weights[valid_idx, :].T
                pop_dev = F_cl - bl_mean[None, :] if PAIRWISE_MODE == 'dev2' else F_cl
                downstream = pop_dev @ random_m_w.T
                cc_2nd_rand[:, ti] = np.sum(r_pre * downstream, axis=0)

            X_random = np.zeros((n_valid, n_wins))
            for wi, ws in enumerate(win_starts):
                X_random[:, wi] = np.sum(cc_2nd_rand[:, ws:ws+WIN_SIZE], axis=1)

            # ---- Build X_dict ----
            Y_valid = Y[valid_idx]
            X_dict = {
                '1st_order': (zscore_cols(X_1st_v), Y_valid, n_valid),
                '2nd_order': (zscore_cols(X_2nd_v), Y_valid, n_valid),
                'combined': (zscore_cols(X_comb_v), Y_valid, n_valid),
                'random_m': (zscore_cols(X_random), Y_valid, n_valid),
            }

            # ---- CV pipeline ----
            def run_cv(X_mode, Y_mode, np_mode):
                if np_mode < n_cv_folds * 2:
                    return np.nan, np.nan, 1.0, np.array([]), np.array([])
                cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                Y_test_all = []
                Y_pred_all = []
                r_train_folds = []
                for train_idx, test_idx in cv.split(X_mode):
                    X_train, X_test = X_mode[train_idx], X_mode[test_idx]
                    Y_train, Y_test = Y_mode[train_idx], Y_mode[test_idx]
                    mu_y, sig_y = Y_train.mean(), Y_train.std()
                    if sig_y == 0 or not np.isfinite(sig_y):
                        sig_y = 1.0
                    Y_train_z = (Y_train - mu_y) / sig_y
                    Y_test_z = (Y_test - mu_y) / sig_y
                    beta = np.linalg.pinv(X_train) @ Y_train_z
                    Y_train_pred = X_train @ beta
                    Y_test_pred = X_test @ beta
                    r_tr = pearsonr(Y_train_pred, Y_train_z)[0] if np.std(Y_train_pred) > 0 else 0.0
                    r_train_folds.append(r_tr)
                    Y_test_all.append(Y_test_z)
                    Y_pred_all.append(Y_test_pred)
                Yt = np.concatenate(Y_test_all)
                Yp = np.concatenate(Y_pred_all)
                if np.std(Yp) > 0:
                    r_test, p_test = spearmanr(Yp, Yt)
                else:
                    r_test, p_test = 0.0, 1.0
                r_train = np.mean(r_train_folds)
                return r_test, r_train, p_test, Yt, Yp

            # ---- Run CV for non-shuffle modes ----
            for mode in CC_MODES:
                if mode == 'shuffled':
                    continue
                if mode in X_dict:
                    X_mode, Y_mode, np_mode = X_dict[mode]
                else:
                    continue
                r_test, r_train, p_test, Yt, Yp = run_cv(X_mode, Y_mode, np_mode)
                all_results[mode].append({
                    'mouse': mouse, 'session': session,
                    'n_pairs': np_mode, 'n_pairs_total': n_pairs,
                    'n_trials': trl, 'n_windows': n_wins,
                    'r_test': r_test, 'r_train': r_train,
                    'p_test': p_test,
                    'Y_test_all': Yt, 'Y_pred_all': Yp,
                })

            # ---- Shuffle: run each permutation through CV ----
            shuf_r_tests = np.zeros(N_SHUFFLES)
            shuf_Yt = None
            shuf_Yp = None
            for shi in range(N_SHUFFLES):
                X_sh_z = zscore_cols(X_shuf_list[shi])
                r_te_sh, _, _, Yt_sh, Yp_sh = run_cv(X_sh_z, Y_valid, n_valid)
                shuf_r_tests[shi] = r_te_sh
                if shi == 0:
                    shuf_Yt = Yt_sh
                    shuf_Yp = Yp_sh

            real_2nd_r = all_results['2nd_order'][-1]['r_test']
            shuf_pval = np.mean(shuf_r_tests >= real_2nd_r) if np.isfinite(real_2nd_r) else 1.0

            all_results['shuffled'].append({
                'mouse': mouse, 'session': session,
                'n_pairs': n_valid, 'n_pairs_total': n_pairs,
                'n_trials': trl, 'n_windows': n_wins,
                'r_test': shuf_r_tests[0],
                'r_train': np.nan,
                'p_test': shuf_pval,
                'shuf_distribution': shuf_r_tests,
                'Y_test_all': shuf_Yt,
                'Y_pred_all': shuf_Yp,
            })

            # Print summary
            parts = []
            for mode in CC_MODES:
                r = all_results[mode][-1]
                sig = '*' if r['p_test'] < 0.05 else ''
                parts.append(f"{mode}={r['r_test']:.3f}{sig}")
            print(f"  {n_pairs} pairs, {n_wins} wins | " + ", ".join(parts))

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
    '1st_order': '1st order',
    '2nd_order': '2nd order\n(outgoing w)',
    'combined': 'Combined\n(1st+2nd)',
    'shuffled': 'Shuffled\n(scrambled w)',
    'random_m': 'Random m\n(wrong neuron w)',
}
mode_colors = {
    '1st_order': '#3498db',
    '2nd_order': '#e74c3c',
    'combined': '#9b59b6',
    'shuffled': '#f39c12',
    'random_m': '#2ecc71',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel A: Bar plot of median test r per mode ---
ax = axes[0]
mode_order = ['1st_order', '2nd_order', 'combined', 'shuffled', 'random_m']
means = []
sems = []
medians = []
pvals = []
n_sigs = []

for mode in mode_order:
    rt = np.array([s['r_test'] for s in all_results[mode]])
    means.append(np.mean(rt))
    medians.append(np.median(rt))
    sems.append(np.std(rt) / np.sqrt(len(rt)))
    n_sigs.append(np.sum(np.array([s['p_test'] for s in all_results[mode]]) < 0.05))
    try:
        _, p = wilcoxon(rt)
    except Exception:
        p = 1.0
    pvals.append(p)

x = np.arange(len(mode_order))
bars = ax.bar(x, medians, yerr=sems, capsize=5,
              color=[mode_colors[m] for m in mode_order],
              edgecolor='k', linewidth=1.2, alpha=0.85)

for xi, (p, ns) in enumerate(zip(pvals, n_sigs)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    n_s = len(all_results[mode_order[xi]])
    ax.text(xi, medians[xi] + sems[xi] + 0.001,
            f'{sig}\n{ns}/{n_s}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels([mode_labels[m] for m in mode_order], fontsize=10)
ax.set_ylabel('Median test r (5-fold CV)')
ax.set_title('Cross-validated dW prediction\n(framewise CC)', fontweight='bold')

# --- Panel B: Paired scatter — 1st vs 2nd order test r ---
ax = axes[1]
rt_1st = np.array([s['r_test'] for s in all_results['1st_order']])
rt_2nd = np.array([s['r_test'] for s in all_results['2nd_order']])

ax.scatter(rt_1st, rt_2nd, c='#9b59b6', s=40, alpha=0.7,
           edgecolor='k', linewidth=0.5)
lims = [min(np.min(rt_1st), np.min(rt_2nd)) - 0.01,
        max(np.max(rt_1st), np.max(rt_2nd)) + 0.01]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('1st order test r (matched pairs)')
ax.set_ylabel('2nd order test r')
ax.set_title('1st vs 2nd order (matched)', fontweight='bold')
n_above = np.sum(rt_2nd > rt_1st)
ax.text(0.05, 0.95, f'{n_above}/{len(rt_1st)} 2nd > 1st',
        transform=ax.transAxes, va='top', fontsize=11)

# --- Panel C: Paired scatter — 2nd order vs shuffled ---
ax = axes[2]
rt_shuf = np.array([s['r_test'] for s in all_results['shuffled']])

ax.scatter(rt_shuf, rt_2nd, c='#f39c12', s=40, alpha=0.7,
           edgecolor='k', linewidth=0.5)
lims2 = [min(np.min(rt_shuf), np.min(rt_2nd)) - 0.01,
         max(np.max(rt_shuf), np.max(rt_2nd)) + 0.01]
ax.plot(lims2, lims2, 'k--', alpha=0.4)
ax.set_xlabel('Shuffled test r (scrambled connectivity)')
ax.set_ylabel('2nd order test r (true connectivity)')
ax.set_title('True vs scrambled connectivity', fontweight='bold')
n_above2 = np.sum(rt_2nd > rt_shuf)
ax.text(0.05, 0.95, f'{n_above2}/{len(rt_2nd)} 2nd > shuffled',
        transform=ax.transAxes, va='top', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_framewise_2nd_order.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 7: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'variance_explained_framewise_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("FRAMEWISE COACTIVITY PREDICTS PLASTICITY — 1ST vs 2ND ORDER\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Fit: pinv, {n_cv_folds}-fold CV across pairs\n")
    f.write(f"CC computed frame-by-frame within closed-loop period (t >= 0)\n")
    f.write(f"Pairwise mode: {PAIRWISE_MODE}\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODELS:\n")
    f.write("  1st_order : CC_trial = Σ_t r_pre(t) * (r_post(t) - bl)           [framewise]\n")
    f.write("  2nd_order : CC_trial = Σ_t r_pre(t) * Σ_k w_{post->k} * (r_k(t) - bl_k)  [framewise]\n")
    f.write("  combined  : 1st + 2nd order features\n")
    f.write(f"  shuffled  : 2nd order with outgoing weights permuted ({N_SHUFFLES}x)\n")
    f.write("  random_m  : 2nd order using outgoing weights from random neuron m != j\n\n")

    f.write("POPULATION RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"  {'Mode':15s} {'median r':>9s} {'mean r':>8s} {'%>0':>5s} "
            f"{'Wilcoxon p':>11s} {'sig':>4s} {'n_sig':>6s}\n")

    for mode in mode_order:
        rt = np.array([s['r_test'] for s in all_results[mode]])
        pt = np.array([s['p_test'] for s in all_results[mode]])
        n_s = len(rt)
        n_pos = np.sum(rt > 0)
        n_sig = np.sum(pt < 0.05)
        try:
            _, p = wilcoxon(rt)
        except Exception:
            p = 1.0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        f.write(f"  {mode:15s} {np.median(rt):+9.4f} {np.mean(rt):+8.4f} "
                f"{n_pos}/{n_s:>3d} {p:11.6f} {sig:>4s} {n_sig}/{n_s:>3d}\n")

    # Paired comparisons
    f.write("\nPAIRED COMPARISONS\n")
    f.write("-" * 70 + "\n")

    rt_1st = np.array([s['r_test'] for s in all_results['1st_order']])
    rt_2nd = np.array([s['r_test'] for s in all_results['2nd_order']])
    rt_comb = np.array([s['r_test'] for s in all_results['combined']])
    rt_shuf = np.array([s['r_test'] for s in all_results['shuffled']])
    rt_rand = np.array([s['r_test'] for s in all_results['random_m']])

    comparisons = [
        ('2nd vs 1st', rt_2nd - rt_1st),
        ('2nd vs shuffled', rt_2nd - rt_shuf),
        ('2nd vs random_m', rt_2nd - rt_rand),
        ('combined vs 1st', rt_comb - rt_1st),
        ('combined vs 2nd', rt_comb - rt_2nd),
    ]
    for label, diff in comparisons:
        try:
            _, p = wilcoxon(diff)
        except Exception:
            p = 1.0
        n_pos = np.sum(diff > 0)
        f.write(f"  {label:22s}: mean diff={np.mean(diff):+.4f}, "
                f"Wilcoxon p={p:.4f}, {n_pos}/{len(diff)} improved\n")

    # Per-session detail
    f.write("\nPER-SESSION DETAIL\n")
    f.write("-" * 100 + "\n")
    f.write(f"  {'mouse':8s} {'session':8s} {'pairs':>6s} {'wins':>5s} "
            f"{'1st_r':>8s} {'2nd_r':>8s} {'comb_r':>8s} "
            f"{'shuf_r':>8s} {'rand_r':>8s} {'shuf_p':>7s}\n")
    for i in range(len(all_results['1st_order'])):
        r1 = all_results['1st_order'][i]
        r2 = all_results['2nd_order'][i]
        rc = all_results['combined'][i]
        rs = all_results['shuffled'][i]
        rr = all_results['random_m'][i]
        shuf_p = rs['p_test']
        f.write(f"  {r1['mouse']:8s} {r1['session']:8s} {r1['n_pairs']:6d} "
                f"{r1['n_windows']:5d} "
                f"{r1['r_test']:+8.3f} "
                f"{r2['r_test']:+8.3f} {rc['r_test']:+8.3f} "
                f"{rs['r_test']:+8.3f} {rr['r_test']:+8.3f} "
                f"{shuf_p:7.3f}\n")

print(f"Report saved → {report_path}")
