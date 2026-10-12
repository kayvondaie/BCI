#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Model-based learning rule: dW ~ d_hat_post * sum(r_pre)
# where d_hat_post = cumulative corr(neuron_activity, lickport_speed) using trials 0..t_w
# lickport_speed = total steps per trial (from zaber step_time)
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

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
tau_elig = 10
N_BASELINE = 20

# Sliding window parameters
WIN_SIZE = 10
WIN_STEP = 5

# Fitting
fit_type = 'pinv'
n_cv_folds = 5

all_results = []
print(f"Config: win={WIN_SIZE}, step={WIN_STEP}, {fit_type}, {n_cv_folds}-fold CV")

#%% ============================================================================
# CELL 3: Main loop — model-based: dW ~ d_hat_post * sum(r_pre)
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
                'dt_si', 'step_time', 'reward_time',
            ]
            try:
                data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
            except FileNotFoundError:
                print(f"  Skipping -- file not found.")
                continue

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']
            trl = F.shape[2]
            n_neurons = F.shape[1]
            n_frames = F.shape[0]
            tsta = np.arange(n_frames) * dt_si
            t0_idx = min(int(round(2.0 / dt_si)), n_frames - 1)
            tsta = tsta - tsta[t0_idx]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # ---- Per-trial behavioral variables ----
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0,
                                  tau=tau_elig, fill_value=0.0)

            # ---- Pair selection (same as three_factor) ----
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
                print("  No valid pairs.")
                continue

            Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
            all_nt = np.concatenate(pair_nt_list)
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

            # ---- Pre-epoch activity ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            if len(ts_pre) == 0:
                print("  No pre-epoch frames, skipping.")
                continue
            epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)

            # ---- Scalar lickport speed per trial: total steps ----
            speed = np.zeros(trl)
            for t in range(trl):
                steps = data['step_time'][t]
                if steps is not None and len(steps) > 0:
                    speed[t] = float(len(steps))

            # ---- Trial-level activity per neuron (same epoch as r_pre) ----
            # epoch_act is (n_neurons, trl) from pre-epoch mean
            # We use this for the cumulative correlation with speed

            # ---- r_pre per pair per trial ----
            r_pre = cl_weights @ epoch_act  # (n_pairs, trl)

            # ---- Sliding window with cumulative d_hat ----
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)

            if n_wins < 3:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            X = np.zeros((n_pairs, n_wins))
            unique_nt = np.unique(all_nt)

            for wi, ws in enumerate(win_starts):
                t_end = ws + WIN_SIZE  # last trial in window (exclusive)
                trial_idx = np.arange(ws, t_end)
                r_pre_sum = np.sum(r_pre[:, trial_idx], axis=1)  # (n_pairs,)

                # Cumulative d_hat: corr(neuron_act[0:t_end], speed[0:t_end])
                speed_cum = speed[:t_end]
                d_hat_neuron = {}
                if t_end >= 5 and np.std(speed_cum) > 0:
                    for ni in unique_nt:
                        act_cum = epoch_act[ni, :t_end]
                        if np.std(act_cum) > 0:
                            d_hat_neuron[ni] = pearsonr(act_cum, speed_cum)[0]

                for pi in range(n_pairs):
                    d_hat = d_hat_neuron.get(all_nt[pi], 0.0)
                    X[pi, wi] = d_hat * r_pre_sum[pi]

            # ---- Hebbian index per window: slope of dW vs X ----
            HI = np.full(n_wins, np.nan)
            for wi in range(n_wins):
                xw = X[:, wi]
                if np.std(xw) > 0:
                    HI[wi] = np.dot(xw - xw.mean(), Y - Y.mean()) / np.dot(xw - xw.mean(), xw - xw.mean())

            # ---- Per-window behavioral variables ----
            win_hit = np.full(n_wins, np.nan)
            win_rt = np.full(n_wins, np.nan)
            win_rpe = np.full(n_wins, np.nan)
            win_hit_rpe = np.full(n_wins, np.nan)
            for wi, ws in enumerate(win_starts):
                trial_idx = np.arange(ws, ws + WIN_SIZE)
                win_hit[wi] = np.nanmean(hit[trial_idx])
                win_rt[wi] = np.nanmean(rt_filled[trial_idx])
                win_rpe[wi] = np.nanmean(rt_rpe[trial_idx])
                win_hit_rpe[wi] = np.nanmean(hit_rpe[trial_idx])

            # Z-score columns
            mu_x = X.mean(axis=0)
            sig_x = X.std(axis=0)
            sig_x[sig_x == 0] = 1.0
            X = (X - mu_x) / sig_x

            # ---- Cross-validated fitting ----
            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            Y_test_all = []
            Y_pred_all = []
            r_train_folds = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

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

            # Pooled out-of-fold r and p
            Yt = np.concatenate(Y_test_all)
            Yp = np.concatenate(Y_pred_all)
            if np.std(Yp) > 0:
                r_test, p_test = spearmanr(Yp, Yt)
            else:
                r_test, p_test = 0.0, 1.0
            r_train = np.mean(r_train_folds)

            result = {
                'mouse': mouse,
                'session': session,
                'n_pairs': n_pairs,
                'n_trials': trl,
                'n_windows': n_wins,
                'r_test': r_test,
                'r_train': r_train,
                'p_test': p_test,
                'Y_test_all': Yt,
                'Y_pred_all': Yp,
                'HI': HI,
                'win_hit': win_hit,
                'win_rt': win_rt,
                'win_rpe': win_rpe,
                'win_hit_rpe': win_hit_rpe,
            }
            all_results.append(result)

            sig = '*' if p_test < 0.05 else ''
            print(f"  {n_pairs} pairs, {n_wins} wins | "
                  f"train r={r_train:.3f}, test r={r_test:.3f}, "
                  f"p={p_test:.4f} {sig}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nDone: {len(all_results)} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'model_based_variance_explained.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'model_based_variance_explained.npy'),
    allow_pickle=True).tolist()
print(f"Loaded {len(all_results)} sessions")

#%% ============================================================================
# CELL 6: Summary figure
# ============================================================================
n_s = len(all_results)
r_test = np.array([s['r_test'] for s in all_results])
r_train = np.array([s['r_train'] for s in all_results])
p_test = np.array([s['p_test'] for s in all_results])
mice_arr = np.array([s['mouse'] for s in all_results])

n_sig = np.sum(p_test < 0.05)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel A: Test r per session ---
ax = axes[0]
mouse_list = sorted(set(mice_arr))
cmap = plt.cm.Set2
mouse_colors = {m: cmap(i) for i, m in enumerate(mouse_list)}

for i in range(n_s):
    ec = 'k' if p_test[i] < 0.05 else 'none'
    lw = 1.5 if p_test[i] < 0.05 else 0
    ax.bar(i, r_test[i], color=mouse_colors[mice_arr[i]], alpha=0.7,
           edgecolor=ec, linewidth=lw)
ax.axhline(0, color='k', ls='-', alpha=0.3)
ax.axhline(np.median(r_test), color='k', ls='--', alpha=0.5)
ax.set_xlabel('Session')
ax.set_ylabel('Test r (5-fold CV)')
ax.set_title(f'Model-based: dW ~ d_hat * sum(r_pre)\n{n_sig}/{n_s} sessions p<0.05',
             fontweight='bold')
for m in mouse_list:
    ax.scatter([], [], color=mouse_colors[m], label=m, s=40)
ax.legend(loc='best', fontsize=7, ncol=2)

try:
    _, p_wilcox = wilcoxon(r_test)
except Exception:
    p_wilcox = 1.0
print(f"Test r: median={np.median(r_test):.4f}, mean={np.mean(r_test):.4f}, "
      f"Wilcoxon p={p_wilcox:.4f}, >0 in {np.sum(r_test>0)}/{n_s}")
print(f"Significant sessions: {n_sig}/{n_s}")

# --- Panel B: Pooled binned predicted vs actual dW ---
ax = axes[1]
Y_pred_pool = np.concatenate([s['Y_pred_all'] for s in all_results])
Y_test_pool = np.concatenate([s['Y_test_all'] for s in all_results])

n_bins_plot = 5
edges = np.percentile(Y_pred_pool, np.linspace(0, 100, n_bins_plot + 1))
bx, by, be = [], [], []
for bi in range(n_bins_plot):
    if bi < n_bins_plot - 1:
        mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool < edges[bi + 1])
    else:
        mask = (Y_pred_pool >= edges[bi]) & (Y_pred_pool <= edges[bi + 1])
    if np.sum(mask) < 3:
        continue
    bx.append(np.mean(Y_pred_pool[mask]))
    by.append(np.mean(Y_test_pool[mask]))
    be.append(np.std(Y_test_pool[mask]) / np.sqrt(np.sum(mask)))

ax.errorbar(bx, by, yerr=be, fmt='o-', color='#2c3e50', capsize=5,
            linewidth=2, markersize=7)
ax.axhline(0, color='k', ls='-', alpha=0.2)
r_pool, p_pool = spearmanr(Y_pred_pool, Y_test_pool)
ax.set_xlabel('Predicted dW (z)')
ax.set_ylabel('Actual dW (z)')
ax.set_title(f'Pooled prediction\nr={r_pool:.3f}, p={p_pool:.2e}',
             fontweight='bold')

# --- Panel C: Train vs test r ---
ax = axes[2]
ax.scatter(r_train, r_test, c='#2c3e50', s=25, alpha=0.6, zorder=3)
lim = max(np.max(np.abs(r_train)), np.max(np.abs(r_test))) * 1.1
lim = max(lim, 0.1)
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
ax.set_xlabel('Train r')
ax.set_ylabel('Test r')
ax.set_title('Overfitting check', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_based_variance_explained.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")

#%% ============================================================================
# CELL 7: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'model_based_variance_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("MODEL-BASED LEARNING RULE — VARIANCE EXPLAINED\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write(f"Fit: {fit_type}, {n_cv_folds}-fold CV across pairs\n")
    f.write(f"Pooled out-of-fold Spearman r for per-session stats\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL:\n")
    f.write("  dW(pair) = sum_w beta_w * X_w(pair)\n")
    f.write("  X_w(pair) = d_hat_post(w) * sum_{t in w} r_pre(t)\n")
    f.write("  d_hat_post(w) = pearsonr(neuron_act[0:t_w], speed[0:t_w])  [cumulative]\n")
    f.write("  speed = total lickport steps per trial (from zaber step_time)\n")
    f.write("  Cumulative: early windows use few trials, late windows use many\n\n")

    f.write("POPULATION RESULTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Sessions: {n_s}\n")
    f.write(f"  Test r: median={np.median(r_test):.4f}, mean={np.mean(r_test):.4f}\n")
    f.write(f"  >0 in {np.sum(r_test>0)}/{n_s} sessions\n")
    f.write(f"  Wilcoxon p = {p_wilcox:.6f}\n")
    f.write(f"  Significant sessions (p<0.05): {n_sig}/{n_s}\n")
    f.write(f"  Pooled r = {r_pool:.4f} (p={p_pool:.2e})\n\n")

    f.write(f"  Train r: median={np.median(r_train):.4f}, "
            f"mean={np.mean(r_train):.4f}\n")
    f.write(f"  Train-test gap: {np.mean(r_train) - np.mean(r_test):.4f}\n\n")

    f.write("PER-SESSION DETAIL\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'mouse':8s} {'session':10s} {'pairs':>6s} {'trials':>6s} "
            f"{'wins':>5s} {'r_train':>8s} {'r_test':>7s} {'p_test':>9s} {'sig':>4s}\n")
    for s in all_results:
        sig = '*' if s['p_test'] < 0.05 else ''
        f.write(f"  {s['mouse']:6s} {s['session']:10s} {s['n_pairs']:6d} "
                f"{s['n_trials']:6d} {s['n_windows']:5d} "
                f"{s['r_train']:8.3f} {s['r_test']:7.3f} "
                f"{s['p_test']:9.4f} {sig:>4s}\n")

print(f"Report saved to: {report_path}")

#%% ============================================================================
# CELL 8: HI vs behavioral variables
# ============================================================================
beh_names = ['Hit rate', 'RPE', 'Speed', 'Hit RPE']
beh_keys  = ['win_hit', 'win_rpe', 'win_rt', 'win_hit_rpe']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for bi, (bname, bkey) in enumerate(zip(beh_names, beh_keys)):
    ax = axes[bi]

    # Pool across sessions
    hi_pool = []
    bv_pool = []
    for s in all_results:
        hi = s['HI']
        bv = s[bkey]
        # For RT, flip sign → Speed
        if bkey == 'win_rt':
            bv = -bv
        ok = np.isfinite(hi) & np.isfinite(bv)
        hi_pool.append(hi[ok])
        bv_pool.append(bv[ok])

    hi_pool = np.concatenate(hi_pool)
    bv_pool = np.concatenate(bv_pool)

    ax.scatter(bv_pool, hi_pool, s=4, alpha=0.15, color='0.4', rasterized=True)
    ax.axhline(0, color='k', ls='-', alpha=0.2)

    if len(hi_pool) > 5 and np.std(bv_pool) > 0 and np.std(hi_pool) > 0:
        rho, pval = spearmanr(bv_pool, hi_pool)
        ax.set_title(f'{bname}\nrho={rho:.3f}, p={pval:.2e}', fontsize=10)
    else:
        ax.set_title(bname, fontsize=10)

    ax.set_xlabel(bname)
    if bi == 0:
        ax.set_ylabel('Hebbian index (slope)')

    # Per-session correlations
    rs = []
    for s in all_results:
        hi = s['HI']
        bv = -s[bkey] if bkey == 'win_rt' else s[bkey]
        ok = np.isfinite(hi) & np.isfinite(bv)
        if np.sum(ok) >= 5 and np.std(hi[ok]) > 0 and np.std(bv[ok]) > 0:
            rs.append(spearmanr(hi[ok], bv[ok])[0])
    rs = np.array(rs)
    if len(rs) > 0:
        try:
            _, pw = wilcoxon(rs)
        except Exception:
            pw = 1.0
        print(f"  {bname:12s}: pooled rho={rho:.4f} (p={pval:.2e}), "
              f"per-session median r={np.median(rs):.4f}, "
              f">{0} in {np.sum(rs>0)}/{len(rs)}, Wilcoxon p={pw:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_based_HI_vs_behavior.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("HI vs behavior figure saved.")
