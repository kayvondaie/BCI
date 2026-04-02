#%% ============================================================================
# CELL 1: Setup — load BCI102 session 6
# ============================================================================
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import session_counting
import data_dict_create_module_test as ddct
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
#%%
mouse = "BCI102"
session_inds = np.where(
    (list_of_dirs['Mouse'] == mouse) &
    (list_of_dirs['Has data_main.npy'] == True)
)[0]
si = session_inds[11]  # session 6 (0-indexed)
session = list_of_dirs['Session'][si]
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + '/' + session + '/pophys/'
print(f"Loading {mouse} {session}")

photostim_keys = ['stimDist', 'favg_raw']
bci_keys = ['df_closedloop', 'F', 'mouse', 'session',
            'conditioned_neuron', 'dt_si', 'step_time',
            'reward_time', 'BCI_thresholds']
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
tsta = np.arange(0, 12, dt_si)
tsta = tsta - tsta[int(2 / dt_si)]

# Pre-epoch activity and RPE for windowed correlations
from scipy.stats import spearmanr
F_nan = F.copy()
F_nan[np.isnan(F_nan)] = 0
ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)
N_BASELINE = 20
bl_mean = np.nanmean(epoch_act[:, :min(N_BASELINE, trl)], axis=1)

# RPE
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=10, fill_value=10.0)

WIN_SIZE = 10
WIN_STEP = 5
win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
rpe_per_win = np.array([np.nanmean(rt_rpe[ws:ws+WIN_SIZE]) for ws in win_starts])
plt.plot(rpe_per_win,'k');plt.plot(plt.xlim()*0,'k:')
print(f"  {n_neurons} neurons, {trl} trials, {stimDist.shape[1]} stim groups")
print(f"  AMP shapes: {AMP[0].shape}, {AMP[1].shape}")

#%% ============================================================================
# CELL 2: Rank nontargets by |dW|, with dev2 corr and RPE corr
# ============================================================================
ranked = []
from scipy.signal import medfilt
for gi in range(stimDist.shape[1]):
    cl = np.where(
        (stimDist[:, gi] < 10) &
        (AMP[0][:, gi] > 0.1) &
        (AMP[1][:, gi] > 0.1)
    )[0]
    if cl.size == 0:
        continue

    r_pre_trials = np.mean(epoch_act[cl, :], axis=0)

    nontargs = np.where(
        (stimDist[:, gi] > 30) & (stimDist[:, gi] < 1000) &
        ((AMP[0][:, gi] > 0.15) | (AMP[1][:, gi] > 0.15))
    )[0]

    for ni in nontargs:
        dw = AMP[1][ni, gi] - AMP[0][ni, gi]
        r_post_dev = epoch_act[ni, :] - bl_mean[ni]

        # dev2 windowed CC
        cc_per_win = np.array([
            np.sum(r_pre_trials[ws:ws+WIN_SIZE] * r_post_dev[ws:ws+WIN_SIZE])
            for ws in win_starts
        ])

        # Correlation of CC with RPE
        if np.std(cc_per_win) > 0 and np.std(rpe_per_win) > 0:
            rho_rpe, _ = spearmanr(cc_per_win, rpe_per_win)
        else:
            rho_rpe = 0.0

        # Overall dev2 correlation
        if np.std(r_pre_trials) > 0 and np.std(r_post_dev) > 0:
            dev2_corr, _ = pearsonr(r_pre_trials, r_post_dev)
        else:
            dev2_corr = 0.0

        ranked.append({'gi': gi, 'ni': ni, 'dw': dw,
                       'dist': stimDist[ni, gi],
                       'amp_pre': AMP[0][ni, gi],
                       'amp_post': AMP[1][ni, gi],
                       'dev2_corr': dev2_corr,
                       'rho_rpe': rho_rpe})

# Filter: positive dW, positive dev2 corr, positive RPE correlation
ranked = [r for r in ranked if r['dw'] > 0 and r['dev2_corr'] > 0 and r['rho_rpe'] > 0]
# ranked = [r for r in ranked if r['dw'] < 0 and r['dev2_corr'] > 0 and r['rho_rpe'] < 0]
ranked.sort(key=lambda x: x['dw'] * x['dev2_corr'] * x['rho_rpe'], reverse=True)
print(f"\n{len(ranked)} nontargets after filtering (dW>0, dev2_corr>0, rho_RPE>0)")
for i, r in enumerate(ranked[:20]):
    print(f"  rank {i}: gi={r['gi']} ni={r['ni']} dW={r['dw']:+.4f} "
          f"dist={r['dist']:.0f}um dev2_corr={r['dev2_corr']:+.3f} "
          f"rho_RPE={r['rho_rpe']:+.3f}")

# Precompute favg (done once)
after = int(np.floor(0.4 / dt_si))
before = int(np.floor(0.2 / dt_si))
if mouse == "BCI103":
    after = int(np.floor(0.5 / dt_si))

favg_list = []
pre_win_list = []
post_win_list = []
artifact_list = []

for epoch_i in range(2):
    ps_key = 'photostim' if epoch_i == 0 else 'photostim2'
    favg_raw = data[ps_key]['favg_raw']

    favg = np.zeros_like(favg_raw)
    for ii in range(favg.shape[1]):
        bl = np.nanmean(favg_raw[0:3, ii])
        if bl != 0:
            favg[:, ii] = (favg_raw[:, ii] - bl) / bl
        else:
            favg[:, ii] = 0

    artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
    artifact = artifact - np.nanmean(artifact[0:4])
    artifact = np.where(artifact > 0.5)[0]
    artifact = artifact[artifact < 40]

    if artifact.size > 0:
        pre_win = (int(artifact[0] - before), int(artifact[0] - 2))
        post_win = (int(artifact[-1] + 2), int(artifact[-1] + after))
        favg[artifact, :, :] = np.nan
        favg[0:30, :] = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
            ), axis=0, arr=favg[0:30, :])
    else:
        pre_win = (0, 0)
        post_win = (0, 0)

    favg_list.append(favg)
    pre_win_list.append(pre_win)
    post_win_list.append(post_win)
    artifact_list.append(artifact)

t_favg = np.arange(favg_list[0].shape[0]) * dt_si
win_centers = win_starts + WIN_SIZE // 2
print("favg precomputed — change RANK and re-run Cell 3 only.")


#%% ============================================================================
# CELL 3: Average the top N pairs
# ============================================================================
from matplotlib.gridspec import GridSpec

pw0, pw1 = pre_win_list[0], pre_win_list[1]
qw0, qw1 = post_win_list[0], post_win_list[1]
art0 = artifact_list[0]

# Time axis: zero at end of photostim artifact
t_zero = art0[-1] * dt_si if art0.size > 0 else 0
t_plot = t_favg - t_zero
stim_start = art0[0] * dt_si - t_zero if art0.size > 0 else 0
stim_end = 0.0
xlims = (-0.2, 0.5)

# Color scheme: pre=blue, post=red, 3rd factor=orange
C_PRE_EARLY  = '#93c5fd'  # light blue
C_PRE_LATE   = '#1d4ed8'  # blue
C_POST_EARLY = '#fca5a5'  # light red
C_POST_LATE  = '#dc2626'  # red
C_3RD        = '#ea580c'  # orange

# Figure dimensions and fonts — set once, import directly into Inkscape
FIG_W_MM = 100   # full-width figure (mm) — change to 89 for single column
FIG_H_MM = 100   # height (mm)
FIG_W = FIG_W_MM / 25.4  # inches
FIG_H = FIG_H_MM / 25.4

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',  # keep text as text in SVG (editable in Inkscape)
})

FS_TITLE = 8
FS_LABEL = 8
FS_TICK = 8
FS_LEGEND = 8

PAIR_DIR = os.path.join(RESULTS_DIR, 'example_pairs')
os.makedirs(PAIR_DIR, exist_ok=True)

N_AVG = min(10, len(ranked))
N_AVG = len(ranked)

# Collect normalized traces for each pair
post_traces_early = []
post_traces_late = []
pre_traces_early = []
pre_traces_late = []
act_pre_all = []
act_post_all = []
rpe_all = []
cc_scatter_all = []   # r_pre * delta_r_post per window
rpe_scatter_all = []  # RPE per window

t_common = np.linspace(xlims[0], xlims[1], 100)  # common time grid for photostim
n_act_bins = 20  # common bin count for activity/RPE

for ri in range(N_AVG):
    gi_r = ranked[ri]['gi']
    ni_r = ranked[ri]['ni']

    cl_r = np.where(
        (stimDist[:, gi_r] < 10) &
        (AMP[0][:, gi_r] > 0.1) &
        (AMP[1][:, gi_r] > 0.1)
    )[0]

    # Postsynaptic photostim traces (baseline-subtracted, normalized)
    tr0 = favg_list[0][:, ni_r, gi_r].copy()
    tr1 = favg_list[1][:, ni_r, gi_r].copy()
    tr0 -= np.nanmean(tr0[pw0[0]:pw0[1]])
    tr1 -= np.nanmean(tr1[pw1[0]:pw1[1]])
    tr0 = medfilt(tr0, 3)
    tr1 = medfilt(tr1, 3)
    # Normalize by max absolute response across both epochs
    peak = max(np.nanmax(np.abs(tr0)), np.nanmax(np.abs(tr1)), 1e-10)
    post_traces_early.append(np.interp(t_common, t_plot, tr0 / peak))
    post_traces_late.append(np.interp(t_common, t_plot, tr1 / peak))

    # Presynaptic photostim traces (baseline-subtracted, normalized)
    tr0 = np.nanmean(favg_list[0][:, cl_r, gi_r], axis=1)
    tr1 = np.nanmean(favg_list[1][:, cl_r, gi_r], axis=1)
    tr0 -= np.nanmean(tr0[pw0[0]:pw0[1]])
    tr1 -= np.nanmean(tr1[pw1[0]:pw1[1]])
    peak = max(np.nanmax(np.abs(tr0)), np.nanmax(np.abs(tr1)), 1e-10)
    pre_traces_early.append(np.interp(t_common, t_plot, tr0 / peak))
    pre_traces_late.append(np.interp(t_common, t_plot, tr1 / peak))

    # Binned activity (z-scored, interpolated to common bin count)
    r_pre_t = np.mean(epoch_act[cl_r, :], axis=0)
    r_post_d = epoch_act[ni_r, :] - bl_mean[ni_r]
    pre_b = np.array([np.mean(r_pre_t[ws:ws+WIN_SIZE]) for ws in win_starts])
    post_b = np.array([np.mean(r_post_d[ws:ws+WIN_SIZE]) for ws in win_starts])
    pre_bz = (pre_b - np.mean(pre_b)) / (np.std(pre_b) + 1e-10)
    post_bz = (post_b - np.mean(post_b)) / (np.std(post_b) + 1e-10)
    x_orig = np.linspace(0, 1, len(pre_bz))
    x_common = np.linspace(0, 1, n_act_bins)
    act_pre_all.append(np.interp(x_common, x_orig, pre_bz))
    act_post_all.append(np.interp(x_common, x_orig, post_bz))

    # RPE (z-scored per pair)
    rpe_z = (rpe_per_win - np.mean(rpe_per_win)) / (np.std(rpe_per_win) + 1e-10)
    rpe_all.append(np.interp(x_common, x_orig, rpe_z))

    # CC and RPE for scatter (z-score CC within pair for pooling)
    cc_win = np.array([
        np.sum(r_pre_t[ws:ws+WIN_SIZE] * r_post_d[ws:ws+WIN_SIZE])
        for ws in win_starts
    ])
    cc_z = (cc_win - np.mean(cc_win)) / (np.std(cc_win) + 1e-10)
    cc_scatter_all.append(cc_z)
    rpe_scatter_all.append(rpe_z)

# Stack and compute mean +/- SEM
def mean_sem(arr):
    arr = np.array(arr)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs4 = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.4,
               left=0.10, right=0.95, top=0.90, bottom=0.10)

# --- (1,1): avg postsynaptic photostim ---
ax = fig.add_subplot(gs4[0, 0])
m0, s0 = mean_sem(post_traces_early)
m1, s1 = mean_sem(post_traces_late)
ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD)
ax.fill_between(t_common, m0 - s0, m0 + s0, color=C_POST_EARLY, alpha=0.3)
ax.fill_between(t_common, m1 - s1, m1 + s1, color=C_POST_LATE, alpha=0.3)
ax.plot(t_common, m0, color=C_POST_EARLY, linewidth=1, label='Early')
ax.plot(t_common, m1, color=C_POST_LATE, linewidth=1, label='Late')
ax.set_xlim(xlims)
ax.set_xlabel('Time from stim offset (s)', fontsize=FS_LABEL)
ax.set_ylabel('dF/F', fontsize=FS_LABEL)
ax.set_title(f'Postsynaptic (n={N_AVG} pairs)', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (1,2): avg presynaptic photostim ---
ax = fig.add_subplot(gs4[0, 1])
m0, s0 = mean_sem(pre_traces_early)
m1, s1 = mean_sem(pre_traces_late)
ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD)
ax.fill_between(t_common, m0 - s0, m0 + s0, color=C_PRE_EARLY, alpha=0.3)
ax.fill_between(t_common, m1 - s1, m1 + s1, color=C_PRE_LATE, alpha=0.3)
ax.plot(t_common, m0, color=C_PRE_EARLY, linewidth=1, label='Early')
ax.plot(t_common, m1, color=C_PRE_LATE, linewidth=1, label='Late')
ax.set_xlim(xlims)
ax.set_xlabel('Time from stim offset (s)', fontsize=FS_LABEL)
ax.set_ylabel('dF/F', fontsize=FS_LABEL)
ax.set_title(f'Presynaptic (n={N_AVG} pairs)', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (2,1): split into activity (top) and RPE (bottom) ---
gs4_inner = gs4[1, 0].subgridspec(2, 1, hspace=0.4)
x_plot = np.linspace(0, 100, n_act_bins)

# Top: avg binned activity
ax = fig.add_subplot(gs4_inner[0])
m_pre, s_pre = mean_sem(act_pre_all)
m_post, s_post = mean_sem(act_post_all)
ax.fill_between(x_plot, m_pre - s_pre, m_pre + s_pre, color=C_PRE_LATE, alpha=0.2)
ax.fill_between(x_plot, m_post - s_post, m_post + s_post, color=C_POST_LATE, alpha=0.2)
ax.plot(x_plot, m_pre, 'o-', color=C_PRE_LATE, linewidth=1.5, markersize=3, label='Presynaptic')
ax.plot(x_plot, m_post, 's-', color=C_POST_LATE, linewidth=1.5, markersize=3, label='Postsynaptic')
ax.set_ylabel('Activity (z)', fontsize=FS_LABEL)
ax.set_title('Avg binned activity', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom: avg binned RPE
ax = fig.add_subplot(gs4_inner[1])
m_rpe, s_rpe = mean_sem(rpe_all)
ax.fill_between(x_plot, m_rpe - s_rpe, m_rpe + s_rpe, color=C_3RD, alpha=0.2)
ax.plot(x_plot, m_rpe, 'o-', color=C_3RD, linewidth=1.5, markersize=3)
ax.set_xlabel('Trial (% session)', fontsize=FS_LABEL)
ax.set_ylabel('RPE (z)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (2,2): pooled scatter of r_pre * delta_r_post vs RPE ---
ax = fig.add_subplot(gs4[1, 1])
cc_pool = np.concatenate(cc_scatter_all)
rpe_pool = np.concatenate(rpe_scatter_all)
ax.scatter(rpe_pool, cc_pool, s=15, color='k', alpha=0.3, edgecolor='none')
if np.std(rpe_pool) > 0:
    coeffs = np.polyfit(rpe_pool, cc_pool, 1)
    xr = np.array([np.min(rpe_pool), np.max(rpe_pool)])
    ax.plot(xr, coeffs[0] * xr + coeffs[1], 'k--', linewidth=1.5)
rho_pool, _ = spearmanr(rpe_pool, cc_pool)
ax.set_xlabel('RPE (z)', fontsize=FS_LABEL)
ax.set_ylabel('$r_{pre} \\times \\Delta r_{post}$ (z)', fontsize=FS_LABEL)
ax.set_title(f'Pooled CC vs RPE\n$\\rho$ = {rho_pool:.3f}, n={len(cc_pool)}',
             fontsize=FS_TITLE, fontweight='bold')
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle(f'{mouse} {session} — avg of top {N_AVG} pairs', fontsize=FS_TITLE, fontweight='bold')

PAIR_DIR = os.path.join(RESULTS_DIR, 'example_pairs')
os.makedirs(PAIR_DIR, exist_ok=True)
fname_avg = f'{mouse}_{session}_avg_top{N_AVG}'
fig.savefig(os.path.join(PAIR_DIR, f'{fname_avg}.png'), dpi=300)
fig.savefig(os.path.join(PAIR_DIR, f'{fname_avg}.svg'))
plt.show()
print(f"Saved avg figure: {PAIR_DIR}/{fname_avg}.png/.svg")

#%% ============================================================================
# CELL 4: Plot — just change RANK and re-run this cell
# ============================================================================
RANK = 0

gi = ranked[RANK]['gi']
ni = ranked[RANK]['ni']
best_dw = ranked[RANK]['dw']

cl = np.where(
    (stimDist[:, gi] < 10) &
    (AMP[0][:, gi] > 0.1) &
    (AMP[1][:, gi] > 0.1)
)[0]

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.4,
              left=0.10, right=0.95, top=0.90, bottom=0.10)

# --- (1,1): postsynaptic photostim ---
ax = fig.add_subplot(gs[0, 0])
tr0 = favg_list[0][:, ni, gi].copy()
tr1 = favg_list[1][:, ni, gi].copy()
tr0 -= np.nanmean(tr0[pw0[0]:pw0[1]])
tr1 -= np.nanmean(tr1[pw1[0]:pw1[1]])
tr0_filt = medfilt(tr0, 3)
tr1_filt = medfilt(tr1, 3)
ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD, label='stim')
ax.plot(t_plot, tr0_filt, color=C_POST_EARLY, linewidth=1.5, label='Early')
ax.plot(t_plot, tr1_filt, color=C_POST_LATE, linewidth=1.5, label='Late')
ax.set_xlim(xlims)
vis = (t_plot >= xlims[0]) & (t_plot <= xlims[1])
yvals = np.concatenate([tr0_filt[vis], tr1_filt[vis]])
ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals))
ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)
ax.set_xlabel('Time from stim offset (s)', fontsize=FS_LABEL)
ax.set_ylabel('dF/F', fontsize=FS_LABEL)
ax.set_title(f'Postsynaptic neuron {ni}\n$\\Delta W$ = {best_dw:.4f}', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (1,2): presynaptic photostim ---
ax = fig.add_subplot(gs[0, 1])
tr0 = np.nanmean(favg_list[0][:, cl, gi], axis=1)
tr1 = np.nanmean(favg_list[1][:, cl, gi], axis=1)
tr0 -= np.nanmean(tr0[pw0[0]:pw0[1]])
tr1 -= np.nanmean(tr1[pw1[0]:pw1[1]])
ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD, label='stim')
ax.plot(t_plot, tr0, color=C_PRE_EARLY, linewidth=1.5, label='Early')
ax.plot(t_plot, tr1, color=C_PRE_LATE, linewidth=1.5, label='Late')
ax.set_xlim(xlims)
ax.set_xlabel('Time from stim offset (s)', fontsize=FS_LABEL)
ax.set_ylabel('dF/F', fontsize=FS_LABEL)
targ_dw = np.mean(AMP[1][cl, gi] - AMP[0][cl, gi])
ax.set_title(f'Presynaptic (n={len(cl)})\n$\\Delta W$ = {targ_dw:.4f}', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (2,1): split into top (activity) and bottom (RPE) ---
gs_inner = gs[1, 0].subgridspec(2, 1, hspace=0.4)

r_pre_trials = np.mean(epoch_act[cl, :], axis=0)
r_post_dev = epoch_act[ni, :] - bl_mean[ni]
pre_binned = np.array([np.mean(r_pre_trials[ws:ws+WIN_SIZE]) for ws in win_starts])
post_binned = np.array([np.mean(r_post_dev[ws:ws+WIN_SIZE]) for ws in win_starts])
pre_z = (pre_binned - np.mean(pre_binned)) / (np.std(pre_binned) + 1e-10)
post_z = (post_binned - np.mean(post_binned)) / (np.std(post_binned) + 1e-10)

# Top: binned activity
ax = fig.add_subplot(gs_inner[0])
ax.plot(win_centers, pre_z, 'o-', color=C_PRE_LATE, linewidth=1.5,
        markersize=4, label='Presynaptic')
ax.plot(win_centers, post_z, 's-', color=C_POST_LATE, linewidth=1.5,
        markersize=4, label='Postsynaptic')
ax.set_ylabel('Activity (z)', fontsize=FS_LABEL)
dev2_corr_val = ranked[RANK]['dev2_corr']
ax.set_title(f'dev2 corr = {dev2_corr_val:.3f}', fontsize=FS_TITLE, fontweight='bold')
ax.legend(frameon=False, fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom: binned RPE
ax = fig.add_subplot(gs_inner[1])
ax.plot(win_centers, rpe_per_win, 'o-', color=C_3RD, linewidth=1.5,
        markersize=4)
ax.set_xlabel('Trial', fontsize=FS_LABEL)
ax.set_ylabel('RPE', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- (2,2): scatter of r_pre * delta_r_post vs RPE ---
ax = fig.add_subplot(gs[1, 1])
cc_per_win = np.array([
    np.sum(r_pre_trials[ws:ws+WIN_SIZE] * r_post_dev[ws:ws+WIN_SIZE])
    for ws in win_starts
])
rho_rpe_val = ranked[RANK]['rho_rpe']
ax.scatter(rpe_per_win, cc_per_win, s=30, color=C_POST_LATE, alpha=0.6, edgecolor='none')
if np.std(rpe_per_win) > 0:
    coeffs = np.polyfit(rpe_per_win, cc_per_win, 1)
    xr = np.array([np.min(rpe_per_win), np.max(rpe_per_win)])
    ax.plot(xr, coeffs[0] * xr + coeffs[1], 'k--', linewidth=1.5)
ax.set_xlabel('RPE', fontsize=FS_LABEL)
ax.set_ylabel('$r_{pre} \\times \\Delta r_{post}$', fontsize=FS_LABEL)
ax.set_title(f'CC vs RPE\n$\\rho$ = {rho_rpe_val:.3f}', fontsize=FS_TITLE, fontweight='bold')
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle(f'{mouse} {session} — group {gi}, rank {RANK}', fontsize=FS_TITLE, fontweight='bold')

# Save with unique name
PAIR_DIR = os.path.join(RESULTS_DIR, 'example_pairs')
os.makedirs(PAIR_DIR, exist_ok=True)
fname = f'{mouse}_{session}_g{gi}_n{ni}'
fig.savefig(os.path.join(PAIR_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PAIR_DIR, f'{fname}.svg'))
plt.show()

print(f"Saved to {PAIR_DIR}/{fname}.png/.svg")
print(f"Postsynaptic {ni}: AMP pre={AMP[0][ni,gi]:.4f}, post={AMP[1][ni,gi]:.4f}, "
      f"dW={best_dw:.4f}, dist={stimDist[ni,gi]:.0f} um")
print(f"dev2_corr={dev2_corr_val:.3f}, rho_RPE={rho_rpe_val:.3f}")
