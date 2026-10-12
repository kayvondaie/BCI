#%% ============================================================================
# CELL 1: Setup
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
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *
import plotting_functions as pf

list_of_dirs = session_counting.counter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
PANEL_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written\3-factor learning paper\claude code 032226\meta_analysis_results\panels'
os.makedirs(PANEL_DIR, exist_ok=True)

#%% ============================================================================
# CELL 2: Load session and compute pairs / CC per window
# ============================================================================
mouse = "BCI102"
session_inds = np.where(
    (list_of_dirs['Mouse'] == mouse) &
    (list_of_dirs['Has data_main.npy'] == True)
)[0]
si = session_inds[6]
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
n_frames = F.shape[0]
tsta = np.arange(0, 12, dt_si)
tsta = tsta - tsta[int(2 / dt_si)]

data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)

# Behavioral variables
rt = np.array([x[0] if len(x) > 0 else np.nan
               for x in data['reward_time']], dtype=float)
hit = np.isfinite(rt)
rt_filled = rt.copy()
rt_filled[~np.isfinite(rt_filled)] = 30.0
tau_elig = 10
rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)

# Pre-trial epoch activity
F_nan = F.copy()
F_nan[np.isnan(F_nan)] = 0
ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
epoch_pre = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1]+1, :, :], axis=0)  # (N, trl)
epoch_post = epoch_pre.copy()  # same epoch, no lag

N_BASELINE = 20
bl_post_mean = np.nanmean(epoch_post[:, :min(N_BASELINE, trl)], axis=1)

# ---- Build pair selection ----
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

# ---- Sliding windows: compute CC (dev2) per window ----
WIN_SIZE = 5
WIN_STEP = 5
win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
n_wins = len(win_starts)

cc_per_win = np.full((n_wins, n_pairs), np.nan)
hi_slope = np.full(n_wins, np.nan)
hi_intercept = np.full(n_wins, np.nan)
rpe_per_win = np.full(n_wins, np.nan)

for wi, ws in enumerate(win_starts):
    trial_idx = np.arange(ws, ws + WIN_SIZE)
    rpe_per_win[wi] = np.nanmean(rt_rpe[trial_idx])

    pre_act = cl_weights @ epoch_pre[:, trial_idx]      # (n_pairs, win_size)
    post_dev = epoch_post[all_nt, :][:, trial_idx] - bl_post_mean[all_nt, np.newaxis]
    cc_pair = np.sum(pre_act * post_dev, axis=1)         # (n_pairs,)
    cc_per_win[wi, :] = cc_pair

    if np.std(cc_pair) > 0:
        A = np.column_stack([np.ones(n_pairs), cc_pair])
        coeffs = np.linalg.lstsq(A, Y_T, rcond=None)[0]
        hi_intercept[wi] = coeffs[0]
        hi_slope[wi] = coeffs[1]

win_centers = win_starts + WIN_SIZE // 2
print(f"{n_pairs} pairs, {n_wins} windows")
print(f"HI range: {np.nanmin(hi_slope):.4f} to {np.nanmax(hi_slope):.4f}")

#%% ============================================================================
# CELL 3: Didactic figure — HI time series + CC vs dW for max/min windows
# ============================================================================
from matplotlib.gridspec import GridSpec

# Figure settings — 8pt Arial, exact dimensions for Inkscape
FIG_W_MM = 140
FIG_H_MM = 100
FIG_W = FIG_W_MM / 25.4
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
    'svg.fonttype': 'none',
})

# Color scheme
C_PRE_LATE  = '#1d4ed8'
C_POST_LATE = '#dc2626'
C_3RD       = '#ea580c'

# Find windows with max and min Hebbian index
wi_max = np.nanargmax(hi_slope)
wi_min = np.nanargmin(hi_slope)
# wi_max = np.argsort(hi_slope)[-3]
# wi_min = np.argsort(hi_slope)[3]

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2],
              left=0.10, right=0.95, bottom=0.12, top=0.88,
              wspace=0.35, hspace=0.50)

# --- Top: HI time series spanning both columns ---
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.plot(win_centers, hi_slope, 'o-', color='k', linewidth=1, markersize=3)
ax_ts.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)

# Highlight max and min windows
ax_ts.plot(win_centers[wi_max], hi_slope[wi_max], 'o', color=C_POST_LATE,
           markersize=8, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
ax_ts.plot(win_centers[wi_min], hi_slope[wi_min], 'o', color=C_PRE_LATE,
           markersize=8, zorder=5, markeredgecolor='k', markeredgewidth=0.5)

ax_ts.set_xlabel('Trial')
ax_ts.set_ylabel('Hebbian index')
ax_ts.set_title(f'{mouse} {session}')
ax_ts.spines['top'].set_visible(False)
ax_ts.spines['right'].set_visible(False)

# --- Bottom row: CC vs dW scatter for max (left) and min (right) ---
bin_X_all, bin_Y_all = [], []
for col, (wi, label, color) in enumerate([
    (wi_max, 'Max HI', C_POST_LATE),
    (wi_min, 'Min HI', C_PRE_LATE),
]):
    ax = fig.add_subplot(gs[1, col])
    cc = cc_per_win[wi, :]
    dw = Y_T
    ok = np.isfinite(cc) & np.isfinite(dw)
    cc_ok, dw_ok = cc[ok], dw[ok]

    plt.sca(ax)
    X_bin, Y_bin, _ = pf.mean_bin_plot(cc_ok, dw_ok, col=4, color='k')
    bin_X_all.append(X_bin)
    bin_Y_all.append(Y_bin)

    # Regression line (Hebbian index) in orange — clipped to binned x range
    if np.std(cc_ok) > 0:
        A_mat = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A_mat, dw_ok, rcond=None)[0]
        xr = np.array([X_bin[0], X_bin[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, color=C_3RD, linewidth=1.5)

    ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
    ax.axvline(0, color='k', ls='--', alpha=0.2, linewidth=0.5)

    slope_val = hi_slope[wi]
    trial_range = f'trials {win_starts[wi]}-{win_starts[wi]+WIN_SIZE}'
    ax.set_title(f'{label} (HI={slope_val:+.4f})\n{trial_range}',
                 color=color)
    ax.set_xlabel('CC (dev2, pre epoch)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Match x and y ranges across bottom panels — tight to binned means
all_bx = np.concatenate(bin_X_all)
all_by = np.concatenate(bin_Y_all)
xpad = 0.15 * (np.max(all_bx) - np.min(all_bx))
ypad = 0.15 * (np.max(all_by) - np.min(all_by))
xl = [np.min(all_bx) - xpad, np.max(all_bx) + xpad]
yl = [np.min(all_by) - ypad, np.max(all_by) + ypad]
for ax in fig.axes[1:]:
    ax.set_xlim(xl)
    ax.set_ylim(yl)

fig.axes[1].set_ylabel(r'$\Delta W$')

fname = f'hebbian_index_didactic_{mouse}_{session}'
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname}.png/.svg")

#%% ============================================================================
# CELL 4: CC dev2 matrices (for exploration, separate figure)
# ============================================================================
CC_mats = {}
for wi in [wi_max, wi_min]:
    trial_idx = np.arange(win_starts[wi], win_starts[wi] + WIN_SIZE)
    pre = epoch_pre[:, trial_idx]
    post_dev = epoch_post[:, trial_idx] - bl_post_mean[:, None]
    CC_mats[wi] = pre @ post_dev.T

b = np.argsort(np.sum(CC_mats[wi_max] - CC_mats[wi_min], axis=0))
vmax = max(np.nanmax(np.abs(CC_mats[wi_max])), np.nanmax(np.abs(CC_mats[wi_min]))) * .1

fig_mat, axes_mat = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.5),
                                  gridspec_kw={'left': 0.08, 'right': 0.92,
                                               'bottom': 0.15, 'top': 0.82,
                                               'wspace': 0.35})

for col, (wi, label, color) in enumerate([
    (wi_max, 'Max HI', C_POST_LATE), (wi_min, 'Min HI', C_PRE_LATE),
]):
    ax = axes_mat[col]
    CC_sorted = CC_mats[wi][b, :][:, b]
    im = ax.imshow(CC_sorted, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    slope_val = hi_slope[wi]
    trial_range = f'trials {win_starts[wi]}-{win_starts[wi]+WIN_SIZE}'
    ax.set_title(f'{label} (HI={slope_val:+.4f})\n{trial_range}', color=color)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')
    plt.colorbar(im, ax=ax, shrink=0.8, label='CC dev2')

fname_mat = f'hebbian_index_matrices_{mouse}_{session}'
fig_mat.savefig(os.path.join(PANEL_DIR, f'{fname_mat}.png'), dpi=300)
fig_mat.savefig(os.path.join(PANEL_DIR, f'{fname_mat}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname_mat}.png/.svg")

#%% ============================================================================
# CELL 5: Pooled version — average CC across all HI>0 and HI<0 windows
# ============================================================================

hi_pos = np.where(np.isfinite(hi_slope) & (hi_slope > 0))[0]
hi_neg = np.where(np.isfinite(hi_slope) & (hi_slope < 0))[0]

# Average CC across windows in each group
cc_pos = np.nanmean(cc_per_win[hi_pos, :], axis=0)
cc_neg = np.nanmean(cc_per_win[hi_neg, :], axis=0)

fig5 = plt.figure(figsize=(FIG_W, FIG_H))
gs5 = GridSpec(2, 2, figure=fig5, height_ratios=[1, 1.2],
               left=0.10, right=0.95, bottom=0.12, top=0.88,
               wspace=0.35, hspace=0.50)

# --- Top: HI time series, highlight positive (red) and negative (blue) ---
ax_ts = fig5.add_subplot(gs5[0, :])
ax_ts.plot(win_centers, hi_slope, 'o-', color='k', linewidth=1, markersize=3)
ax_ts.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
ax_ts.plot(win_centers[hi_pos], hi_slope[hi_pos], 'o', color=C_POST_LATE,
           markersize=5, zorder=5, markeredgecolor='none')
ax_ts.plot(win_centers[hi_neg], hi_slope[hi_neg], 'o', color=C_PRE_LATE,
           markersize=5, zorder=5, markeredgecolor='none')
ax_ts.set_xlabel('Trial')
ax_ts.set_ylabel('Hebbian index')
ax_ts.set_title(f'{mouse} {session}')
ax_ts.spines['top'].set_visible(False)
ax_ts.spines['right'].set_visible(False)

# --- Bottom row: pooled CC vs dW ---
bin_X_all, bin_Y_all = [], []
for col, (cc_pool, label, color, n_win) in enumerate([
    (cc_pos, 'HI > 0', C_POST_LATE, len(hi_pos)),
    (cc_neg, 'HI < 0', C_PRE_LATE, len(hi_neg)),
]):
    ax = fig5.add_subplot(gs5[1, col])
    ok = np.isfinite(cc_pool) & np.isfinite(Y_T)
    cc_ok, dw_ok = cc_pool[ok], Y_T[ok]

    plt.sca(ax)
    X_bin, Y_bin, _ = pf.mean_bin_plot(cc_ok, dw_ok, col=4, color='k')
    bin_X_all.append(X_bin)
    bin_Y_all.append(Y_bin)

    # Regression line in orange
    if np.std(cc_ok) > 0:
        A_mat = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A_mat, dw_ok, rcond=None)[0]
        xr = np.array([X_bin[0], X_bin[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, color=C_3RD, linewidth=1.5)

    ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
    ax.axvline(0, color='k', ls='--', alpha=0.2, linewidth=0.5)
    ax.set_title(f'{label} ({n_win} windows)', color=color)
    ax.set_xlabel('CC (dev2, pre epoch)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Match ranges — tight to binned means
all_bx = np.concatenate(bin_X_all)
all_by = np.concatenate(bin_Y_all)
xpad = 0.15 * (np.max(all_bx) - np.min(all_bx))
ypad = 0.15 * (np.max(all_by) - np.min(all_by))
xl = [np.min(all_bx) - xpad, np.max(all_bx) + xpad]
yl = [np.min(all_by) - ypad, np.max(all_by) + ypad]
for ax in fig5.axes[1:]:
    ax.set_xlim(xl)
    ax.set_ylim(yl)
fig5.axes[1].set_ylabel(r'$\Delta W$')

fname5 = f'hebbian_index_pooled_{mouse}_{session}'
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.png'), dpi=300)
fig5.savefig(os.path.join(PANEL_DIR, f'{fname5}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname5}.png/.svg")

#%% ============================================================================
# CELL 6: Combined figure — rastermap + CC matrices + HI + CC vs dW
# ============================================================================
# Layout (3 x 4.6 inches):
#   Row 0: Rastermap heatmap (full width)
#   Row 1: CC dev2 matrix max HI (left), CC dev2 matrix min HI (right)
#   Row 2: HI time series (full width)
#   Row 3: CC vs dW scatter max HI (left), CC vs dW scatter min HI (right)

from rastermap import Rastermap

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',
})

# ---- Rastermap sorting ----
df_cl = data['df_closedloop']
n_neurons_cl = df_cl.shape[0]
n_clust = min(20, n_neurons_cl // 10)
n_pcs = min(200, n_neurons_cl - 1)

import rastermap.rastermap as _rm
_orig_fit = _rm.Rastermap.fit
def _safe_fit(self, X, **kwargs):
    try:
        return _orig_fit(self, X, **kwargs)
    except IndexError:
        return self
_rm.Rastermap.fit = _safe_fit

model = Rastermap(n_PCs=n_pcs, n_clusters=n_clust, locality=0.75,
                  time_lag_window=5).fit(df_cl.astype(np.float32))
_rm.Rastermap.fit = _orig_fit

isort = model.isort
df_sorted = df_cl[isort, :]

# Z-score each neuron
row_mean = np.nanmean(df_sorted, axis=1, keepdims=True)
row_std = np.nanstd(df_sorted, axis=1, keepdims=True)
row_std[row_std == 0] = 1.0
df_sorted = (df_sorted - row_mean) / row_std

n_total_frames = df_sorted.shape[1]
n_n = df_sorted.shape[0]

# Bin in time only
BIN_T = 50
n_t_bins = n_total_frames // BIN_T
df_binned = df_sorted[:, :n_t_bins * BIN_T].reshape(n_n, n_t_bins, BIN_T).mean(axis=2)
vmax_rm = np.percentile(np.abs(df_binned[np.isfinite(df_binned)]), 95)

# Trial boundaries in continuous frame space
ops_rm = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
frames_per_file_rm = ops_rm['frames_per_file']
trial_starts_frame = np.cumsum([0] + list(frames_per_file_rm[:-1]))

# Window center frames
win_center_frames = []
for wi, ws in enumerate(win_starts):
    we = ws + WIN_SIZE - 1
    if we < len(trial_starts_frame):
        f_start = trial_starts_frame[ws]
        f_end = trial_starts_frame[we] + frames_per_file_rm[we]
        win_center_frames.append((f_start + f_end) / 2)
    else:
        win_center_frames.append(trial_starts_frame[ws])
win_center_frames = np.array(win_center_frames)

# ---- CC dev2 matrices for max/min HI windows ----
CC_mats = {}
for wi in [wi_max, wi_min]:
    tidx = np.arange(win_starts[wi], win_starts[wi] + WIN_SIZE)
    pre = epoch_pre[:, tidx]
    post_dev = epoch_post[:, tidx] - bl_post_mean[:, None]
    CC_mats[wi] = pre @ post_dev.T

b = np.argsort(np.sum(CC_mats[wi_max] - CC_mats[wi_min], axis=0))
vmax_cc = max(np.nanmax(np.abs(CC_mats[wi_max])),
              np.nanmax(np.abs(CC_mats[wi_min]))) * 0.1

# ---- Figure: 3 x 4.6 inches ----
fig6 = plt.figure(figsize=(3.0, 4.6))
gs6 = GridSpec(4, 2, figure=fig6,
               height_ratios=[1.2, 1.0, 0.6, 1.0],
               left=0.14, right=0.92, bottom=0.07, top=0.95,
               wspace=0.40, hspace=0.45)

# === Row 0: Rastermap heatmap (spans both columns) ===
ax_rm = fig6.add_subplot(gs6[0, :])
ax_rm.imshow(df_binned, aspect='auto', cmap='coolwarm', vmin=-vmax_rm, vmax=vmax_rm,
             extent=[0, n_total_frames, n_n, 0],
             interpolation='nearest', rasterized=True)
for ti in range(0, trl, 20):
    if ti < len(trial_starts_frame):
        ax_rm.axvline(trial_starts_frame[ti], color='k', alpha=0.12, linewidth=0.3)

# Mark max/min HI windows
for wi, color in [(wi_max, C_POST_LATE), (wi_min, C_PRE_LATE)]:
    ws = win_starts[wi]
    we = ws + WIN_SIZE - 1
    if we < len(trial_starts_frame):
        x0 = trial_starts_frame[ws]
        x1 = trial_starts_frame[we] + frames_per_file_rm[we]
        ax_rm.axvspan(x0, x1, color=color, alpha=0.12, zorder=0)

ax_rm.set_ylabel('Neurons')
tick_step = max(1, trl // 6)
tick_trials = np.arange(0, trl, tick_step)
tick_frames = [trial_starts_frame[t] for t in tick_trials if t < len(trial_starts_frame)]
ax_rm.set_xticks(tick_frames[:len(tick_trials)])
ax_rm.set_xticklabels(tick_trials[:len(tick_frames)])
ax_rm.set_xlabel('Trial')

# === Row 1: CC dev2 matrices ===
for col, (wi, label, color) in enumerate([
    (wi_max, 'Max HI', C_POST_LATE), (wi_min, 'Min HI', C_PRE_LATE),
]):
    ax = fig6.add_subplot(gs6[1, col])
    CC_sorted = CC_mats[wi][b, :][:, b]
    im = ax.imshow(CC_sorted, cmap='coolwarm', vmin=-vmax_cc, vmax=vmax_cc,
                   aspect='equal', interpolation='nearest')
    ax.set_xlabel('Neuron')
    if col == 0:
        ax.set_ylabel('Neuron')
    # colorbar only on left panel to avoid overlap
    if col == 0:
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

# === Row 2: HI time series (spans both columns) ===
ax_hi = fig6.add_subplot(gs6[2, :])
ax_hi.plot(win_centers, hi_slope, 'o-', color='k', linewidth=0.8, markersize=2.5)
ax_hi.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)

ax_hi.plot(win_centers[wi_max], hi_slope[wi_max], 'o', color=C_POST_LATE,
           markersize=6, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
ax_hi.plot(win_centers[wi_min], hi_slope[wi_min], 'o', color=C_PRE_LATE,
           markersize=6, zorder=5, markeredgecolor='k', markeredgewidth=0.5)

ax_hi.set_ylabel('Hebbian index')
ax_hi.spines['top'].set_visible(False)
ax_hi.spines['right'].set_visible(False)

# === Row 3: CC vs dW scatter for max/min windows ===
bin_X_all, bin_Y_all = [], []
for col, (wi, label, color) in enumerate([
    (wi_max, 'Max HI', C_POST_LATE), (wi_min, 'Min HI', C_PRE_LATE),
]):
    ax = fig6.add_subplot(gs6[3, col])
    cc = cc_per_win[wi, :]
    dw = Y_T
    ok = np.isfinite(cc) & np.isfinite(dw)
    cc_ok, dw_ok = cc[ok], dw[ok]

    plt.sca(ax)
    X_bin, Y_bin, _ = pf.mean_bin_plot(cc_ok, dw_ok, col=4, color='k')
    bin_X_all.append(X_bin)
    bin_Y_all.append(Y_bin)

    if np.std(cc_ok) > 0:
        A_mat = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A_mat, dw_ok, rcond=None)[0]
        xr = np.array([X_bin[0], X_bin[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, color=C_3RD, linewidth=1.5)

    ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
    ax.axvline(0, color='k', ls='--', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('CC (dev2)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col == 0:
        ax.set_ylabel(r'$\Delta W$')

# Match x/y ranges across bottom scatter panels
all_bx = np.concatenate(bin_X_all)
all_by = np.concatenate(bin_Y_all)
xpad = 0.15 * (np.max(all_bx) - np.min(all_bx))
ypad = 0.15 * (np.max(all_by) - np.min(all_by))
xl = [np.min(all_bx) - xpad, np.max(all_bx) + xpad]
yl = [np.min(all_by) - ypad, np.max(all_by) + ypad]
for ax in [fig6.axes[-1], fig6.axes[-2]]:
    ax.set_xlim(xl)
    ax.set_ylim(yl)

fname6 = f'figure_correlations_{mouse}_{session}'
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.png'), dpi=300)
fig6.savefig(os.path.join(PANEL_DIR, f'{fname6}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname6}.png/.svg")
