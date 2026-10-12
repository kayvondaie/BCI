#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')

print("Setup complete!")

#%% ============================================================================
# CELL 2: Search all sessions for best example pairs
# ============================================================================
mice = ["BCI102", "BCI103", "BCI104", "BCI105", "BCI106", "BCI109"]
WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20

candidates = []

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
            n_frames = F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

            # RPE
            rt = np.array([x[0] if len(x) > 0 else np.nan
                           for x in data['reward_time']], dtype=float)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                                  tau=tau_elig, fill_value=10.0)

            # Pre-epoch activity
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
            bl_mean = np.nanmean(epoch_act[:, :min(N_BASELINE, trl)], axis=1)

            # Sliding windows
            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                continue

            rpe_per_win = np.array([np.nanmean(rt_rpe[ws:ws+WIN_SIZE])
                                    for ws in win_starts])

            # Score each (group, nontarget) pair
            for gi in range(stimDist.shape[1]):
                cl = np.where(
                    (stimDist[:, gi] < 10) &
                    (AMP[0][:, gi] > 0.1) &
                    (AMP[1][:, gi] > 0.1)
                )[0]
                if cl.size == 0:
                    continue

                nontargs = np.where(
                    (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000)
                )[0]
                if len(nontargs) == 0:
                    continue

                # Pre activity of target (mean of close neurons)
                r_pre_trials = np.mean(epoch_act[cl, :], axis=0)  # (trl,)

                # Pre-filter: top 20 nontargets by |dW|, minimum |dW| > 0.1
                dw_all = AMP[1][nontargs, gi] - AMP[0][nontargs, gi]
                keep = np.where(np.abs(dw_all) > 0.1)[0]
                if len(keep) == 0:
                    continue
                # Keep at most 20 per group
                if len(keep) > 20:
                    top20 = np.argsort(np.abs(dw_all[keep]))[-20:]
                    keep = keep[top20]
                nontargs = nontargs[keep]

                for ni in nontargs:
                    dw = AMP[1][ni, gi] - AMP[0][ni, gi]
                    r_post_trials = epoch_act[ni, :]  # (trl,)
                    r_post_dev = r_post_trials - bl_mean[ni]

                    # Overall correlation between pre and post
                    if np.std(r_pre_trials) == 0 or np.std(r_post_dev) == 0:
                        continue
                    overall_corr, _ = pearsonr(r_pre_trials, r_post_dev)

                    # Only keep pairs with meaningful correlation
                    if abs(overall_corr) < 0.2:
                        continue

                    # Windowed CC (dev2)
                    cc_per_win = np.array([
                        np.sum(r_pre_trials[ws:ws+WIN_SIZE] *
                               r_post_dev[ws:ws+WIN_SIZE])
                        for ws in win_starts
                    ])

                    # Correlation of windowed CC with RPE
                    if np.std(cc_per_win) == 0:
                        continue
                    rho_cc_rpe, _ = spearmanr(cc_per_win, rpe_per_win)

                    candidates.append({
                        'mouse': mouse,
                        'session': session,
                        'folder': folder,
                        'gi': gi,
                        'cl': cl,
                        'ni': ni,
                        'dw': dw,
                        'amp_pre': AMP[0][ni, gi],
                        'amp_post': AMP[1][ni, gi],
                        'overall_corr': overall_corr,
                        'rho_cc_rpe': rho_cc_rpe,
                        'n_wins': n_wins,
                        'n_trials': trl,
                    })

            print(f"  {len(candidates)} candidates so far")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

print(f"\nTotal candidates: {len(candidates)}")

#%% ============================================================================
# CELL 3: Select best positive and negative examples
# ============================================================================
import pandas as pd

df = pd.DataFrame(candidates)

# Positive example: large dW, neurons are correlated, CC tracks RPE
# Score = dw * overall_corr * rho_cc_rpe (all should be positive)
df['pos_score'] = df['dw'] * df['overall_corr'].clip(lower=0) * df['rho_cc_rpe'].clip(lower=0)

# Negative example: neurons are correlated (overall_corr > 0),
# but CC does NOT track RPE (rho_cc_rpe ~ 0 or negative),
# and dW <= 0
neg_mask = (df['overall_corr'] > 0.05) & (df['rho_cc_rpe'] < 0.1) & (df['dw'] < -0.02)
df['neg_score'] = 0.0
df.loc[neg_mask, 'neg_score'] = df.loc[neg_mask, 'overall_corr'] * (-df.loc[neg_mask, 'dw'])

# Print top candidates
print("\n=== TOP POSITIVE EXAMPLES ===")
top_pos = df.nlargest(10, 'pos_score')
for _, r in top_pos.iterrows():
    print(f"  {r['mouse']} {r['session']} gi={r['gi']} ni={r['ni']} "
          f"dW={r['dw']:.3f} corr={r['overall_corr']:.3f} "
          f"rho_RPE={r['rho_cc_rpe']:.3f} score={r['pos_score']:.4f}")

print("\n=== TOP NEGATIVE EXAMPLES ===")
top_neg = df.nlargest(10, 'neg_score')
for _, r in top_neg.iterrows():
    print(f"  {r['mouse']} {r['session']} gi={r['gi']} ni={r['ni']} "
          f"dW={r['dw']:.3f} corr={r['overall_corr']:.3f} "
          f"rho_RPE={r['rho_cc_rpe']:.3f} score={r['neg_score']:.4f}")

#%% ============================================================================
# CELL 4: Load data for chosen examples and make figure
# ============================================================================
# Pick the top positive and top negative
pos_ex = top_pos.iloc[0]
neg_ex = top_neg.iloc[0]

examples = [
    (pos_ex, 'Potentiated pair', '#c0392b'),
    (neg_ex, 'Depressed pair', '#2980b9'),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 8),
                         gridspec_kw={'width_ratios': [1, 1.5, 1.5, 1]})

for row, (ex, row_title, color) in enumerate(examples):
    # Reload session data
    folder = ex['folder']
    mouse = ex['mouse']
    session = ex['session']

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

    rt = np.array([x[0] if len(x) > 0 else np.nan
                   for x in data['reward_time']], dtype=float)
    rt_filled = rt.copy()
    rt_filled[~np.isfinite(rt_filled)] = 30.0
    rt_rpe = -compute_rpe(rt_filled, baseline=2.0,
                          tau=tau_elig, fill_value=10.0)

    gi = int(ex['gi'])
    ni = int(ex['ni'])
    cl = np.where(
        (stimDist[:, gi] < 10) &
        (AMP[0][:, gi] > 0.1) &
        (AMP[1][:, gi] > 0.1)
    )[0]

    # Pre-epoch activity
    F_nan = F.copy()
    F_nan[np.isnan(F_nan)] = 0
    ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
    epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
    bl_mean = np.nanmean(epoch_act[:, :min(N_BASELINE, trl)], axis=1)

    r_pre_trials = np.mean(epoch_act[cl, :], axis=0)
    r_post_trials = epoch_act[ni, :]
    r_post_dev = r_post_trials - bl_mean[ni]

    # Photostim traces — extract favg for pre and post epochs
    siHeader_path = folder + r'/suite2p_BCI/plane0/siHeader.npy'
    try:
        siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
    except:
        siHeader_path = folder + r'/suite2p_photostim_single/plane0/siHeader.npy'
        siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

    favg_list = []
    for epoch_i in range(2):
        ps_key = 'photostim' if epoch_i == 0 else 'photostim2'
        favg_raw = data[ps_key]['favg_raw']
        favg = np.zeros_like(favg_raw)
        for ii in range(favg.shape[1]):
            bl = np.nanmean(favg_raw[0:3, ii])
            favg[:, ii] = (favg_raw[:, ii] - bl) / bl if bl != 0 else 0
        # Detect and mask artifact
        artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > 0.5)[0]
        artifact = artifact[artifact < 40]
        if artifact.size > 0:
            favg[artifact, :, :] = np.nan
            favg[0:30, :] = np.apply_along_axis(
                lambda m: np.interp(
                    np.arange(len(m)),
                    np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                    m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
                ), axis=0, arr=favg[0:30, :])
        favg_list.append(favg)
    favg1, favg2 = favg_list[0], favg_list[1]

    # ------- Panel A: Photostim evoked response (pre vs post) -------
    ax = axes[row, 0]
    t_ps = np.arange(favg1.shape[0]) * dt_si
    t_ps = t_ps - t_ps[int(len(t_ps) * 0.15)]

    trace_pre = favg1[:, ni, gi]
    trace_post = favg2[:, ni, gi]

    plot_range = slice(5, min(50, favg1.shape[0]))
    t_plot = t_ps[plot_range]

    ax.plot(t_plot, trace_pre[plot_range], color='0.5', linewidth=1.5,
            label='Pre')
    ax.plot(t_plot, trace_post[plot_range], color=color, linewidth=1.5,
            label='Post')
    ax.set_xlabel('Time from stim (s)')
    ax.set_ylabel('dF/F')
    ax.legend(fontsize=9, frameon=False)
    dw_val = ex['dw']
    ax.set_title(f'{row_title}\n$\\Delta W$ = {dw_val:+.3f}',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------- Panel B: Trial-by-trial activity -------
    ax = axes[row, 1]
    trials = np.arange(trl)

    # Smooth for visual clarity
    from scipy.ndimage import uniform_filter1d
    smooth_k = 5
    pre_smooth = uniform_filter1d(r_pre_trials, smooth_k)
    post_smooth = uniform_filter1d(r_post_dev, smooth_k)

    # Z-score for display
    pre_z = (pre_smooth - np.mean(pre_smooth)) / (np.std(pre_smooth) + 1e-10)
    post_z = (post_smooth - np.mean(post_smooth)) / (np.std(post_smooth) + 1e-10)

    ax.plot(trials, pre_z, color='#e67e22', linewidth=1, alpha=0.8,
            label='Pre (target)')
    ax.plot(trials, post_z, color=color, linewidth=1, alpha=0.8,
            label='Post (nontarget)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Activity (z-scored)')
    r_corr = ex['overall_corr']
    ax.set_title(f'Pre-epoch activity\nr = {r_corr:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------- Panel C: Windowed CC and RPE time series -------
    ax = axes[row, 2]
    win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
    n_wins = len(win_starts)
    win_centers = win_starts + WIN_SIZE // 2

    cc_per_win = np.array([
        np.sum(r_pre_trials[ws:ws+WIN_SIZE] * r_post_dev[ws:ws+WIN_SIZE])
        for ws in win_starts
    ])
    rpe_per_win = np.array([np.nanmean(rt_rpe[ws:ws+WIN_SIZE])
                            for ws in win_starts])

    # Z-score both for overlay
    cc_z = (cc_per_win - np.mean(cc_per_win)) / (np.std(cc_per_win) + 1e-10)
    rpe_z = (rpe_per_win - np.mean(rpe_per_win)) / (np.std(rpe_per_win) + 1e-10)

    ax.plot(win_centers, cc_z, 'o-', color=color, linewidth=1.5,
            markersize=3, label='Coactivity')
    ax.plot(win_centers, rpe_z, 's-', color='#27ae60', linewidth=1.5,
            markersize=3, alpha=0.7, label='RPE')
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('z-score')
    rho_val = ex['rho_cc_rpe']
    ax.set_title(f'CC vs RPE over time\n$\\rho$ = {rho_val:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------- Panel D: CC vs RPE scatter -------
    ax = axes[row, 3]
    ax.scatter(rpe_z, cc_z, s=25, color=color, alpha=0.6,
               edgecolor='none')
    # Fit line
    if np.std(rpe_z) > 0:
        coeffs = np.polyfit(rpe_z, cc_z, 1)
        xr = np.array([rpe_z.min(), rpe_z.max()])
        ax.plot(xr, coeffs[0] * xr + coeffs[1], 'k--', linewidth=1.5)
    ax.set_xlabel('RPE (z)')
    ax.set_ylabel('Coactivity (z)')
    ax.set_title(f'$\\rho$ = {rho_val:.3f}',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(h_pad=3)
plt.savefig(os.path.join(RESULTS_DIR, 'fig23_example_pairs.png'),
            dpi=200, bbox_inches='tight')
plt.show()
print("Figure 23 saved.")

#%% ============================================================================
# CELL 5: Print summary of chosen examples
# ============================================================================
for label, ex in [('POSITIVE', pos_ex), ('NEGATIVE', neg_ex)]:
    print(f"\n{label} EXAMPLE:")
    print(f"  Mouse: {ex['mouse']}, Session: {ex['session']}")
    print(f"  Stim group: {ex['gi']}, Nontarget neuron: {ex['ni']}")
    print(f"  dW = {ex['dw']:.4f}")
    print(f"  AMP pre = {ex['amp_pre']:.4f}, AMP post = {ex['amp_post']:.4f}")
    print(f"  Overall correlation: {ex['overall_corr']:.4f}")
    print(f"  CC-RPE correlation (rho): {ex['rho_cc_rpe']:.4f}")
