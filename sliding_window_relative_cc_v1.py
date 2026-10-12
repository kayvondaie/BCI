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
from scipy.stats import pearsonr, spearmanr, wilcoxon
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

def zscore_mat_std_only(X):
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1
    return X / sd

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
WIN_SIZE = 25    # trials per window
WIN_STEP = 5     # step between windows
tau_elig = 10

EPOCH_ORDER = ["pre", "go_cue", "late", "reward"]
n_epochs = len(EPOCH_ORDER)

# CC modes to compare:
#   'dot_prod'     — raw coactivity: sum_t r_pre(t) * r_post(t)
#   'dot_prod_dev' — deviation from early-session baseline (first 20 trials)
#   'dot_prod_rel' — z-scored across windows (is this pair more correlated than usual?)
#   'dot_prod_rolling_X' — deviation from rolling baseline of X preceding trials
CC_MODES = ['dot_prod', 'dot_prod_dev', 'dot_prod_rel']

# Rolling baseline windows to test (empty = none)
ROLLING_BASELINES = []
for rb in ROLLING_BASELINES:
    CC_MODES.append(f'dot_prod_rolling_{rb}')

# Number of baseline trials for dot_prod_dev
N_BASELINE = 20

all_results = {mode: [] for mode in CC_MODES}
print(f"Sliding window: {WIN_SIZE} trials, step {WIN_STEP}")
print(f"CC modes: {CC_MODES}")

#%% ============================================================================
# CELL 3: Main loop — compute CC in all modes per window
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
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # Epoch averages per neuron per trial
            kstep = np.zeros((n_neurons, trl))
            krewards = np.zeros((n_neurons, trl))
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    indices = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    indices = indices[indices < F.shape[0]]
                    kstep[:, ti] = np.nanmean(F[indices, :, ti], axis=0)
                    indices = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    indices = indices[indices < F.shape[0]]
                    krewards[:, ti] = np.nanmean(F[indices, :, ti], axis=0)

            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            k = np.nanmean(F[ts_go[0]:ts_go[-1], :, :], axis=0)
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            kpre = np.nanmean(F[ts_pre[0]:ts_pre[-1], :, :], axis=0)

            kstep[np.isnan(kstep)] = 0
            krewards[np.isnan(krewards)] = 0
            k[np.isnan(k)] = 0
            kpre[np.isnan(kpre)] = 0

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

            epoch_activity = {
                'pre': kpre, 'go_cue': k, 'late': kstep, 'reward': krewards,
            }

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

            # ---- Compute CC per window for all modes ----
            # Store raw dot_prod CC per pair per window per epoch for z-scoring later
            raw_cc = np.full((n_wins, n_pairs, n_epochs), np.nan)

            # Also store rolling baseline CC: for each window, CC of the
            # preceding X trials (not overlapping with the window itself)
            rolling_cc = {rb: np.full((n_wins, n_pairs, n_epochs), np.nan)
                          for rb in ROLLING_BASELINES}

            # Baseline for dot_prod_dev: product of mean activities (not mean of products)
            # This is intentionally different from actual baseline coactivity —
            # it captures expected coactivity if pre and post were independent,
            # so the deviation = actual CC - independence prediction
            baseline_trials = np.arange(min(N_BASELINE, trl))
            baseline_pre = {}
            baseline_post = {}
            for ei, ep in enumerate(EPOCH_ORDER):
                act_bl = epoch_activity[ep][:, baseline_trials]
                bl_sum = np.sum(act_bl, axis=1)
                baseline_pre[ep] = cl_weights @ bl_sum / max(len(baseline_trials), 1)
                baseline_post[ep] = bl_sum[all_nt] / max(len(baseline_trials), 1)

            # Per-window behavior
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
                    act = epoch_activity[ep][:, trial_idx]  # (n_neurons, win_size)
                    pre_act = cl_weights @ act  # (n_pairs, win_size)
                    post_act = act[all_nt, :]   # (n_pairs, win_size)
                    cc_raw = np.sum(pre_act * post_act, axis=1)  # (n_pairs,)
                    raw_cc[wi, :, ei] = cc_raw

                    # Rolling baselines: CC of preceding X trials
                    for rb in ROLLING_BASELINES:
                        bl_start = max(0, ws - rb)
                        bl_end = ws  # non-overlapping: baseline ends where window starts
                        if bl_end - bl_start < 5:  # too few baseline trials
                            continue
                        bl_idx = np.arange(bl_start, bl_end)
                        act_bl = epoch_activity[ep][:, bl_idx]
                        pre_bl = cl_weights @ act_bl
                        post_bl = act_bl[all_nt, :]
                        # Store per-trial CC for rolling baseline
                        cc_bl = np.sum(pre_bl * post_bl, axis=1) / len(bl_idx)
                        rolling_cc[rb][wi, :, ei] = cc_bl

            # ---- Now compute slope/intercept for each mode ----
            for mode in CC_MODES:
                hi_no_int = np.full((n_wins, n_epochs), np.nan)
                hi_with_int = np.full((n_wins, n_epochs), np.nan)
                hi_intercept = np.full((n_wins, n_epochs), np.nan)
                hi_corr = np.full((n_wins, n_epochs), np.nan)

                for ei, ep in enumerate(EPOCH_ORDER):
                    if mode == 'dot_prod':
                        cc_all = raw_cc[:, :, ei]  # (n_wins, n_pairs)

                    elif mode == 'dot_prod_dev':
                        # Subtract product-of-means baseline (independence prediction)
                        bl_cc = baseline_pre[ep] * baseline_post[ep] * len(baseline_trials)
                        cc_all = raw_cc[:, :, ei] - bl_cc[np.newaxis, :]

                    elif mode == 'dot_prod_rel':
                        # Z-score each pair's CC across windows
                        cc_raw_ep = raw_cc[:, :, ei]
                        pair_mean = np.nanmean(cc_raw_ep, axis=0, keepdims=True)
                        pair_std = np.nanstd(cc_raw_ep, axis=0, keepdims=True)
                        pair_std[pair_std == 0] = 1
                        cc_all = (cc_raw_ep - pair_mean) / pair_std

                    elif mode.startswith('dot_prod_rolling_'):
                        rb = int(mode.split('_')[-1])
                        # Both normalized to per-trial
                        cc_all = raw_cc[:, :, ei] / WIN_SIZE - rolling_cc[rb][:, :, ei]

                    for wi in range(n_wins):
                        cc_pair = cc_all[wi, :]

                        # Skip if any NaN (e.g. rolling baseline not available)
                        # or zero variance
                        if np.any(np.isnan(cc_pair)) or np.std(cc_pair) == 0:
                            continue

                        # No intercept
                        hi_no_int[wi, ei] = (
                            np.dot(cc_pair, Y_T) / np.dot(cc_pair, cc_pair))

                        # With intercept
                        A = np.column_stack([np.ones(n_pairs), cc_pair])
                        coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
                        hi_intercept[wi, ei] = coeffs[0]
                        hi_with_int[wi, ei] = coeffs[1]

                        # Correlation
                        hi_corr[wi, ei], _ = pearsonr(cc_pair, Y_T)

                def count_flips(arr):
                    signs = np.sign(arr)
                    return np.sum(signs[1:] != signs[:-1], axis=0)

                result = {
                    'mouse': mouse,
                    'session': session,
                    'n_pairs': n_pairs,
                    'n_trials': trl,
                    'n_windows': n_wins,
                    'win_centers': win_center,
                    'hit_rate': np.nanmean(hit),
                    'hi_no_int': hi_no_int,
                    'hi_with_int': hi_with_int,
                    'hi_intercept': hi_intercept,
                    'hi_corr': hi_corr,
                    'flips_no_int': count_flips(hi_no_int),
                    'flips_with_int': count_flips(hi_with_int),
                    'flips_corr': count_flips(hi_corr),
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
np.save(os.path.join(RESULTS_DIR, 'sliding_window_relative_cc.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = np.load(
    os.path.join(RESULTS_DIR, 'sliding_window_relative_cc.npy'),
    allow_pickle=True).item()
CC_MODES = list(all_results.keys())
EPOCH_ORDER = ['pre', 'go_cue', 'late', 'reward']
n_epochs = len(EPOCH_ORDER)
print(f"Loaded modes: {CC_MODES}")
for mode in CC_MODES:
    print(f"  {mode}: {len(all_results[mode])} sessions")

#%% ============================================================================
# CELL 6: Compute within-session correlations for all modes
# ============================================================================
beh_names = ['hit_rate', 'RPE', 'RT', 'hit_RPE']
beh_labels = ['Hit rate', 'RPE', 'Reaction time', 'Hit RPE']
n_beh = len(beh_names)

def get_beh(s, bname):
    if bname == 'hit_rate': return s['win_hit']
    if bname == 'RPE': return s['win_rpe']
    if bname == 'RT': return s['win_rt']
    if bname == 'hit_RPE': return s['win_hit_rpe']

# Store correlations per mode
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

print("Within-session correlations computed for all modes.")

#%% ============================================================================
# CELL 7: Coefficient matrices — all three modes side by side
# ============================================================================
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']

n_modes = len(CC_MODES)
fig, axes = plt.subplots(2, max(n_modes, 2), figsize=(4.5 * max(n_modes, 2), 8))
if n_modes == 1:
    axes = axes[:, :1]  # keep 2D but only use first column
vlim = 0.25

for col, mode in enumerate(CC_MODES):
    results = all_results[mode]
    n_s = len(results)

    for row, (corr_arr, row_label) in enumerate([
        (corr_slope[mode], 'Slope'),
        (corr_intercept[mode], 'Intercept'),
    ]):
        ax = axes[row, col]
        mat = np.full((n_beh, n_epochs), np.nan)
        pmat = np.full((n_beh, n_epochs), 1.0)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                vals = corr_arr[:, bi, ei]
                v = vals[np.isfinite(vals)]
                if len(v) >= 5:
                    mat[bi, ei] = np.mean(v)
                    try:
                        _, pmat[bi, ei] = wilcoxon(v)
                    except Exception:
                        pass

        im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-vlim, vmax=vlim)

        for bi in range(n_beh):
            for ei in range(n_epochs):
                val = mat[bi, ei]
                p = pmat[bi, ei]
                if not np.isfinite(val):
                    continue
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                txt = f'{val:+.2f}\n{sig}' if sig else f'{val:+.2f}'
                textcolor = 'white' if abs(val) > vlim * 0.6 else 'black'
                ax.text(ei, bi, txt, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=textcolor)

        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, fontsize=10, rotation=45, ha='right')
        ax.set_yticks(range(n_beh))
        ax.set_yticklabels(beh_labels if col == 0 else [], fontsize=10)

        if row == 0:
            # Shorter display name for rolling modes
            disp = mode.replace('dot_prod_', '').replace('dot_prod', 'raw')
            ax.set_title(f'{disp}\n({row_label})', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'({row_label})', fontsize=11)

        if col == 0:
            ax.set_ylabel(row_label, fontsize=12, fontweight='bold')

n_s = len(all_results[CC_MODES[0]])
fig.suptitle(f'CC mode comparison (n={n_s} sessions)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig10_cc_mode_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 10 saved.")

#%% ============================================================================
# CELL 8: Fit quality comparison — which CC mode best predicts deltaW?
# ============================================================================
# For each session, compare median R^2 across windows

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Collect median |correlation| per session per mode
mode_colors = {'dot_prod': '#1f77b4', 'dot_prod_dev': '#ff7f0e', 'dot_prod_rel': '#2ca02c'}

# Panel 1: median absolute correlation by epoch
ax = axes[0]
x = np.arange(n_epochs)
w = 0.25
for mi, mode in enumerate(CC_MODES):
    results = all_results[mode]
    # Mean across sessions of median |corr| per epoch
    med_corrs = np.full((len(results), n_epochs), np.nan)
    for si, s in enumerate(results):
        for ei in range(n_epochs):
            vals = s['hi_corr'][:, ei]
            v = vals[np.isfinite(vals)]
            if len(v) > 0:
                med_corrs[si, ei] = np.median(np.abs(v))

    means = np.nanmean(med_corrs, axis=0)
    sems = np.nanstd(med_corrs, axis=0) / np.sqrt(np.sum(np.isfinite(med_corrs), axis=0))
    ax.bar(x + (mi - 1) * w, means, w, yerr=sems, capsize=4,
           color=mode_colors[mode], alpha=0.8, edgecolor='black', linewidth=0.5,
           label=mode)

ax.set_xticks(x)
ax.set_xticklabels(epoch_labels)
ax.set_ylabel('Mean of median |r| per session')
ax.set_title('Fit quality by CC mode', fontweight='bold')
ax.legend()

# Panel 2: paired comparison (dot_prod vs dot_prod_rel)
ax = axes[1]
n_s = len(all_results['dot_prod'])
r_raw = np.full(n_s, np.nan)
r_rel = np.full(n_s, np.nan)
r_dev = np.full(n_s, np.nan)
for si in range(n_s):
    for mode, arr in [('dot_prod', r_raw), ('dot_prod_dev', r_dev), ('dot_prod_rel', r_rel)]:
        vals = all_results[mode][si]['hi_corr']
        v = vals[np.isfinite(vals)]
        if len(v) > 0:
            arr[si] = np.median(np.abs(v))

ax.scatter(r_raw, r_rel, s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
           color='#2ca02c', label='relative')
ax.scatter(r_raw, r_dev, s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
           color='#ff7f0e', label='deviation', marker='s')
lim = max(np.nanmax(r_raw), np.nanmax(r_rel), np.nanmax(r_dev)) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.4)
ax.set_xlabel('dot_prod (median |r|)')
ax.set_ylabel('Alternative mode (median |r|)')
ax.set_title('Per-session fit quality', fontweight='bold')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig11_cc_mode_fit_quality.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 11 saved.")

#%% ============================================================================
# CELL 9: Text report
# ============================================================================
report_path = os.path.join(RESULTS_DIR, 'relative_cc_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("RELATIVE COACTIVITY ANALYSIS\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Window: {WIN_SIZE} trials, step {WIN_STEP}\n")
    f.write("=" * 70 + "\n\n")

    f.write("CC MODES:\n")
    f.write("  dot_prod     : raw sum_t r_pre(t) * r_post(t)\n")
    f.write(f"  dot_prod_dev : subtract baseline (first {N_BASELINE} trials) coactivity\n")
    f.write("  dot_prod_rel : z-score each pair's CC across windows within session\n\n")

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
            f.write(f"  {'beh x epoch':20s} {'mean':>7s} {'median':>7s} "
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
                    f.write(f"  {bname+'_'+ep:20s} {m:+7.3f} {md:+7.3f} "
                            f"{fpos:4.0f}% {p:10.4f} {sig:>4s}\n")
                f.write("\n")
            f.write("\n")

print(f"Report saved to: {report_path}")
