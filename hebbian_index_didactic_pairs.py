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
pair_gi_list = []

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
    pair_gi_list.append(np.full(len(nontarg), gi, dtype=int))

Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
all_nt = np.concatenate(pair_nt_list)
all_gi = np.concatenate(pair_gi_list)
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
# CELL 3b: HI + RPE overlaid, first 40 trials (slide-friendly)
# ============================================================================
# Two identical-axis figures (with and without RPE) so the RPE trace can be
# faded in over the HI plot in PowerPoint. RPE is per-trial (matches the (1,1)
# panel of the 2x4 figure in CELL 12); HI is only defined per sliding window.
TRIAL_CUTOFF = 40
sel_win = win_centers < TRIAL_CUTOFF
n_trials_plot = min(TRIAL_CUTOFF, trl)

C_HI       = 'k'
C_HI_POS   = (1.0, 0.0, 0.0)   # post-strengthen (red)
C_HI_NEG   = (0.0, 0.0, 1.0)   # post-weaken (blue)
C_RPE      = '#F99D20'         # matches RPE_COLOR in the 2x4 figure

# Pre-compute fixed y-limits so the two plots have identical axes.
hi_x     = win_centers[sel_win].astype(float)
hi_vals  = hi_slope[sel_win]
rpe_vals = rt_rpe[:n_trials_plot]
hi_pad   = 0.12 * (np.nanmax(hi_vals)  - np.nanmin(hi_vals))
rpe_pad  = 0.12 * (np.nanmax(rpe_vals) - np.nanmin(rpe_vals))
ylim_hi  = (min(np.nanmin(hi_vals) - hi_pad, -hi_pad),
            max(np.nanmax(hi_vals) + hi_pad,  hi_pad))
ylim_rpe = (np.nanmin(rpe_vals) - rpe_pad, np.nanmax(rpe_vals) + rpe_pad)

# Fixed plot-box geometry so the two figures align perfectly for the fade.
FIG_SZ        = (10.0, 4.8)
MARGIN_LRTB   = dict(left=0.13, right=0.88, top=0.84, bottom=0.22)
TITLE_FONTSZ  = 22
LBL_FONTSZ    = 20
TICK_FONTSZ   = 16

def _make_overlay(show_rpe):
    fig, ax_hi = plt.subplots(1, 1, figsize=FIG_SZ)
    ax_rpe = ax_hi.twinx()

    # HI: thick line + white-edged markers
    ax_hi.plot(hi_x, hi_vals, '-', color=C_HI, linewidth=2.6, zorder=3)
    ax_hi.plot(hi_x, hi_vals, 'o', color=C_HI, markersize=11,
               markeredgecolor='white', markeredgewidth=1.5, zorder=4)

    if show_rpe:
        ax_rpe.plot(np.arange(n_trials_plot), rpe_vals,
                    color=C_RPE, linewidth=3.0, alpha=0.95, zorder=2)

    ax_hi.axhline(0, color='k', ls=(0, (5, 4)), alpha=0.45, linewidth=1.1)

    # Left axis (HI) — always shown
    ax_hi.set_xlabel('Trial', fontsize=LBL_FONTSZ, labelpad=6)
    ax_hi.set_ylabel('Hebbian index', color=C_HI,
                     fontsize=LBL_FONTSZ, labelpad=6)
    ax_hi.tick_params(axis='y', colors=C_HI, labelsize=TICK_FONTSZ)
    ax_hi.tick_params(axis='x', labelsize=TICK_FONTSZ)
    ax_hi.set_ylim(ylim_hi)
    ax_hi.set_xlim(0, TRIAL_CUTOFF)
    ax_hi.spines['top'].set_visible(False)
    ax_hi.spines['left'].set_color(C_HI)

    # Right axis (RPE) — same limits in both versions; visibility toggles
    ax_rpe.set_ylim(ylim_rpe)
    ax_rpe.spines['top'].set_visible(False)
    if show_rpe:
        ax_rpe.set_ylabel('RPE', color=C_RPE,
                          fontsize=LBL_FONTSZ, labelpad=6)
        ax_rpe.tick_params(axis='y', colors=C_RPE, labelsize=TICK_FONTSZ)
        ax_rpe.spines['right'].set_color(C_RPE)
    else:
        ax_rpe.set_ylabel('')
        ax_rpe.tick_params(axis='y', length=0, labelright=False)
        ax_rpe.spines['right'].set_visible(False)

    ax_hi.yaxis.grid(True, color='0.92', linewidth=0.8)
    ax_hi.set_axisbelow(True)

    ax_hi.set_title(f'{mouse} {session}  —  first {TRIAL_CUTOFF} trials',
                    fontsize=TITLE_FONTSZ, pad=14, loc='left')

    # Explicit layout so both versions have identical plot boxes
    fig.subplots_adjust(**MARGIN_LRTB)

    suffix = 'with_rpe' if show_rpe else 'no_rpe'
    fname = f'hebbian_index_rpe_overlay_first{TRIAL_CUTOFF}_{suffix}_{mouse}_{session}'
    fig.savefig(os.path.join(PANEL_DIR, f'{fname}.png'), dpi=300)
    fig.savefig(os.path.join(PANEL_DIR, f'{fname}.svg'))
    plt.show()
    print(f"Saved {fname}.png/.svg")

_make_overlay(show_rpe=False)
_make_overlay(show_rpe=True)

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

#%% ============================================================================
# CELL 7: Photostim responses for first- and last-bin pairs (max-HI window)
# ----------------------------------------------------------------------------
# For the max-HI window scatter (CC vs dW), the col=4 mean_bin_plot bins pairs
# by CC into 4 equal-count bins (sorted by CC). This cell averages photostim
# responses across the pairs in the first bin (lowest CC) and the last bin
# (highest CC), Early vs Late epoch, separately for postsynaptic neurons (the
# nontarget per pair) and presynaptic neurons (the cl group per pair). Mirrors
# the averaging in example_pair_simple.py Cell 3.
# ============================================================================
from scipy.signal import medfilt

# ---- Build photostim traces (favg) per epoch with artifact handling ----
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
art0 = artifact_list[0]
t_zero = art0[-1] * dt_si if art0.size > 0 else 0
t_plot = t_favg - t_zero
stim_start = art0[0] * dt_si - t_zero if art0.size > 0 else 0
stim_end = 0.0
xlims = (-0.2, 0.5)

# ---- Color scheme (match example_pair_simple) ----
C_PRE_EARLY  = '#93c5fd'
C_PRE_LATE_  = '#1d4ed8'
C_POST_EARLY = '#fca5a5'
C_POST_LATE_ = '#dc2626'
C_3RD_       = '#ea580c'

# ---- Identify first- and last-bin pairs for max-HI window ----
N_BINS = 4
cc_max = cc_per_win[wi_max, :]
ok_mask = np.isfinite(cc_max) & np.isfinite(Y_T)
ok_idx = np.where(ok_mask)[0]
order = ok_idx[np.argsort(cc_max[ok_idx])]
n_per_bin = len(order) // N_BINS
first_bin_pairs = order[:n_per_bin]
last_bin_pairs  = order[-n_per_bin:]

print(f"Max-HI window {wi_max}: {len(order)} valid pairs, "
      f"{n_per_bin} per bin")
print(f"  first bin CC range: [{cc_max[first_bin_pairs].min():+.3f}, "
      f"{cc_max[first_bin_pairs].max():+.3f}]  mean dW: "
      f"{Y_T[first_bin_pairs].mean():+.4f}")
print(f"  last  bin CC range: [{cc_max[last_bin_pairs].min():+.3f}, "
      f"{cc_max[last_bin_pairs].max():+.3f}]  mean dW: "
      f"{Y_T[last_bin_pairs].mean():+.4f}")

# ---- Aggregate photostim traces for a list of pair indices ----
# Same baseline subtraction as example_pair_simple.py Cell 4 (single-pair plot),
# no normalization — keeps units in raw dF/F to match the CC vs dW analysis.
def collect_traces(pair_inds):
    post_e, post_l, pre_e, pre_l = [], [], [], []
    for pi in pair_inds:
        gi_p = int(all_gi[pi])
        ni_p = int(all_nt[pi])
        cl_p = pair_cl_list_idx[pi]

        # Postsynaptic (single nontarg neuron)
        tr0 = favg_list[0][:, ni_p, gi_p].copy()
        tr1 = favg_list[1][:, ni_p, gi_p].copy()
        tr0 -= np.nanmean(tr0[pre_win_list[0][0]:pre_win_list[0][1]])
        tr1 -= np.nanmean(tr1[pre_win_list[1][0]:pre_win_list[1][1]])
        post_e.append(medfilt(tr0, 3))
        post_l.append(medfilt(tr1, 3))

        # Presynaptic (averaged across cl neurons)
        tr0 = np.nanmean(favg_list[0][:, cl_p, gi_p], axis=1)
        tr1 = np.nanmean(favg_list[1][:, cl_p, gi_p], axis=1)
        tr0 -= np.nanmean(tr0[pre_win_list[0][0]:pre_win_list[0][1]])
        tr1 -= np.nanmean(tr1[pre_win_list[1][0]:pre_win_list[1][1]])
        pre_e.append(tr0)
        pre_l.append(tr1)

    return (np.array(post_e), np.array(post_l),
            np.array(pre_e),  np.array(pre_l))

# Per-pair cl arrays (cl is identical for all pairs sharing a gi)
pair_cl_list_idx = []
offset_idx = 0
for gi_idx in range(len(dw_list)):
    n_nt = len(dw_list[gi_idx])
    cl_arr = pair_cl_list[gi_idx][0]  # cl is the same row repeated
    for _ in range(n_nt):
        pair_cl_list_idx.append(cl_arr)
        offset_idx += 1

post_e_first, post_l_first, pre_e_first, pre_l_first = collect_traces(first_bin_pairs)
post_e_last,  post_l_last,  pre_e_last,  pre_l_last  = collect_traces(last_bin_pairs)

def mean_sem(arr):
    arr = np.asarray(arr)
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
    return m, s

# ---- Plot: 2 rows (first bin, last bin) x 2 cols (post, pre) ----
fig7 = plt.figure(figsize=(FIG_W, FIG_H))
gs7 = GridSpec(2, 2, figure=fig7,
               left=0.10, right=0.95, bottom=0.12, top=0.88,
               wspace=0.35, hspace=0.55)

panels = [
    (0, 'First bin (lowest CC)', post_e_first, post_l_first, pre_e_first, pre_l_first,
     len(first_bin_pairs)),
    (1, 'Last bin (highest CC)', post_e_last, post_l_last, pre_e_last, pre_l_last,
     len(last_bin_pairs)),
]

vis = (t_plot >= xlims[0]) & (t_plot <= xlims[1])

def tight_ylim(ax, m0, m1):
    yvals = np.concatenate([m0[vis], m1[vis]])
    yvals = yvals[np.isfinite(yvals)]
    if yvals.size == 0:
        return
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)

for row, label, pe, pl, qe, ql, n_p in panels:
    # Post column
    ax = fig7.add_subplot(gs7[row, 0])
    m0, s0 = mean_sem(pe)
    m1, s1 = mean_sem(pl)
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_)
    ax.fill_between(t_plot, m0 - s0, m0 + s0, color=C_POST_EARLY, alpha=0.3)
    ax.fill_between(t_plot, m1 - s1, m1 + s1, color=C_POST_LATE_, alpha=0.3)
    ax.plot(t_plot, m0, color=C_POST_EARLY, linewidth=1, label='Early')
    ax.plot(t_plot, m1, color=C_POST_LATE_, linewidth=1, label='Late')
    ax.set_xlim(xlims)
    tight_ylim(ax, m0, m1)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'{label}\nPostsynaptic (n={n_p} pairs)')
    if row == 0:
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pre column
    ax = fig7.add_subplot(gs7[row, 1])
    m0, s0 = mean_sem(qe)
    m1, s1 = mean_sem(ql)
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_)
    ax.fill_between(t_plot, m0 - s0, m0 + s0, color=C_PRE_EARLY, alpha=0.3)
    ax.fill_between(t_plot, m1 - s1, m1 + s1, color=C_PRE_LATE_, alpha=0.3)
    ax.plot(t_plot, m0, color=C_PRE_EARLY, linewidth=1, label='Early')
    ax.plot(t_plot, m1, color=C_PRE_LATE_, linewidth=1, label='Late')
    ax.set_xlim(xlims)
    tight_ylim(ax, m0, m1)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Presynaptic (n={n_p} pairs)')
    if row == 0:
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig7.suptitle(f'{mouse} {session} — max-HI window (HI={hi_slope[wi_max]:+.4f})',
              fontsize=8, fontweight='bold', y=0.97)

fname7 = f'hebbian_bin_pairs_photostim_{mouse}_{session}'
fig7.savefig(os.path.join(PANEL_DIR, f'{fname7}.png'), dpi=300)
fig7.savefig(os.path.join(PANEL_DIR, f'{fname7}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname7}.png/.svg")

#%% ============================================================================
# CELL 8: Photostim difference (Late - Early) for first/last-bin pairs
# ----------------------------------------------------------------------------
# Same pairs as Cell 7, but plotting the per-pair Late minus Early difference.
# This is the trace-level analog of dW = AMP[1] - AMP[0].
# ============================================================================
fig8 = plt.figure(figsize=(FIG_W, FIG_H))
gs8 = GridSpec(2, 2, figure=fig8,
               left=0.10, right=0.95, bottom=0.12, top=0.88,
               wspace=0.35, hspace=0.55)

panels_diff = [
    (0, 'First bin (lowest CC)', post_l_first - post_e_first,
     pre_l_first - pre_e_first, len(first_bin_pairs)),
    (1, 'Last bin (highest CC)', post_l_last - post_e_last,
     pre_l_last - pre_e_last, len(last_bin_pairs)),
]

def tight_ylim_one(ax, m):
    yvals = m[vis]
    yvals = yvals[np.isfinite(yvals)]
    if yvals.size == 0:
        return
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)

for row, label, post_diff, pre_diff, n_p in panels_diff:
    # Post column
    ax = fig8.add_subplot(gs8[row, 0])
    m, s = mean_sem(post_diff)
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.fill_between(t_plot, m - s, m + s, color=C_POST_LATE_, alpha=0.3)
    ax.plot(t_plot, m, color=C_POST_LATE_, linewidth=1)
    ax.set_xlim(xlims)
    tight_ylim_one(ax, m)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel(r'$\Delta$ dF/F (Late - Early)')
    ax.set_title(f'{label}\nPostsynaptic (n={n_p} pairs)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pre column
    ax = fig8.add_subplot(gs8[row, 1])
    m, s = mean_sem(pre_diff)
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.fill_between(t_plot, m - s, m + s, color=C_PRE_LATE_, alpha=0.3)
    ax.plot(t_plot, m, color=C_PRE_LATE_, linewidth=1)
    ax.set_xlim(xlims)
    tight_ylim_one(ax, m)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel(r'$\Delta$ dF/F (Late - Early)')
    ax.set_title(f'Presynaptic (n={n_p} pairs)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig8.suptitle(f'{mouse} {session} — max-HI window (HI={hi_slope[wi_max]:+.4f}) — Late minus Early',
              fontsize=8, fontweight='bold', y=0.97)

fname8 = f'hebbian_bin_pairs_photostim_diff_{mouse}_{session}'
fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.png'), dpi=300)
fig8.savefig(os.path.join(PANEL_DIR, f'{fname8}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname8}.png/.svg")

#%% ============================================================================
# CELL 9: Single-pair example — high-CC last-bin pair with positive dW
# ----------------------------------------------------------------------------
# Filters last_bin_pairs (highest CC in max-HI window) to those with dW > 0,
# ranks by CC * dW (matches example_pair_simple.py ranking spirit), and plots
# Early vs Late photostim for the RANK-th pair. Single-pair format from
# example_pair_simple.py Cell 4.
# ============================================================================
RANK = 0  # change to flip through ranked candidates

# Filter last bin to positive-dW pairs with a detectable connection
# (AMP[0] > 0.15 or AMP[1] > 0.15, matching example_pair_simple.py)
AMP_THR = 0.15
def _has_connection(pi):
    gi_p = int(all_gi[pi])
    ni_p = int(all_nt[pi])
    return (AMP[0][ni_p, gi_p] > AMP_THR) or (AMP[1][ni_p, gi_p] > AMP_THR)

candidates = [pi for pi in last_bin_pairs if Y_T[pi] > 0 and _has_connection(pi)]
candidates.sort(key=lambda pi: cc_max[pi] * Y_T[pi], reverse=True)

if len(candidates) == 0:
    print("No last-bin pairs with positive dW.")
else:
    print(f"{len(candidates)} positive-dW pairs in last bin")
    for r, pi in enumerate(candidates[:10]):
        print(f"  rank {r}: pair {pi}  gi={int(all_gi[pi])}  ni={int(all_nt[pi])}  "
              f"CC={cc_max[pi]:+.3f}  dW={Y_T[pi]:+.4f}")

    pi_pick = candidates[RANK]
    gi_pick = int(all_gi[pi_pick])
    ni_pick = int(all_nt[pi_pick])
    cl_pick = pair_cl_list_idx[pi_pick]
    cc_pick = cc_max[pi_pick]
    dw_pick = Y_T[pi_pick]

    # Preserve for Cell 11 (heatmap markers)
    pi_pick_max = pi_pick
    gi_pick_max = gi_pick
    ni_pick_max = ni_pick
    cl_pick_max = cl_pick
    cc_pick_max = cc_pick
    dw_pick_max = dw_pick

    pw0, pw1 = pre_win_list[0], pre_win_list[1]

    # Postsynaptic traces (single neuron, medfilt)
    tr0_post = favg_list[0][:, ni_pick, gi_pick].copy()
    tr1_post = favg_list[1][:, ni_pick, gi_pick].copy()
    tr0_post -= np.nanmean(tr0_post[pw0[0]:pw0[1]])
    tr1_post -= np.nanmean(tr1_post[pw1[0]:pw1[1]])
    tr0_post = medfilt(tr0_post, 3)
    tr1_post = medfilt(tr1_post, 3)

    # Presynaptic traces (averaged across cl group, no filter)
    tr0_pre = np.nanmean(favg_list[0][:, cl_pick, gi_pick], axis=1)
    tr1_pre = np.nanmean(favg_list[1][:, cl_pick, gi_pick], axis=1)
    tr0_pre -= np.nanmean(tr0_pre[pw0[0]:pw0[1]])
    tr1_pre -= np.nanmean(tr1_pre[pw1[0]:pw1[1]])

    # ~1 in x 1 in per panel
    fig9 = plt.figure(figsize=(3.0, 1.6))
    gs9 = GridSpec(1, 2, figure=fig9,
                   left=0.14, right=0.95, bottom=0.22, top=0.78, wspace=0.40)

    vis_pick = (t_plot >= xlims[0]) & (t_plot <= xlims[1])

    # Post
    ax = fig9.add_subplot(gs9[0, 0])
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_, label='stim')
    ax.plot(t_plot, tr0_post, color=C_POST_EARLY, linewidth=1.5, label='Early')
    ax.plot(t_plot, tr1_post, color=C_POST_LATE_, linewidth=1.5, label='Late')
    ax.set_xlim(xlims)
    yvals = np.concatenate([tr0_post[vis_pick], tr1_post[vis_pick]])
    yvals = yvals[np.isfinite(yvals)]
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Postsynaptic neuron {ni_pick}\n$\\Delta W$ = {dw_pick:+.4f}')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pre
    ax = fig9.add_subplot(gs9[0, 1])
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_, label='stim')
    ax.plot(t_plot, tr0_pre, color=C_PRE_EARLY, linewidth=1.5, label='Early')
    ax.plot(t_plot, tr1_pre, color=C_PRE_LATE_, linewidth=1.5, label='Late')
    ax.set_xlim(xlims)
    yvals = np.concatenate([tr0_pre[vis_pick], tr1_pre[vis_pick]])
    yvals = yvals[np.isfinite(yvals)]
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Presynaptic (n={len(cl_pick)})\nCC = {cc_pick:+.3f}')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig9.suptitle(f'ex. 1 — {mouse} {session} — group {gi_pick}, rank {RANK} '
                  f'(last-bin, dW>0)', fontsize=8, fontweight='bold', y=0.97)

    fname9 = f'hebbian_lastbin_example_{mouse}_{session}_g{gi_pick}_n{ni_pick}'
    fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.png'), dpi=300)
    fig9.savefig(os.path.join(PANEL_DIR, f'{fname9}.svg'))
    plt.show()
    print(f"Saved to {PANEL_DIR}/{fname9}.png/.svg")

#%% ============================================================================
# CELL 10: Single-pair example — high-CC pair from min-HI plot with dW < 0
# ----------------------------------------------------------------------------
# Analog of Cell 9, but for the min-HI window (wi_min). HI < 0 means high CC
# is associated with negative dW. Picks last-bin (highest CC) pairs with
# dW < 0, ranked by |CC * dW|.
# ============================================================================
RANK_MIN = 4  # change to flip through ranked candidates

cc_min = cc_per_win[wi_min, :]
ok_mask_min = np.isfinite(cc_min) & np.isfinite(Y_T)
ok_idx_min = np.where(ok_mask_min)[0]
order_min = ok_idx_min[np.argsort(cc_min[ok_idx_min])]
n_per_bin_min = len(order_min) // N_BINS
last_bin_pairs_min = order_min[-n_per_bin_min:]

# Filter to negative-dW pairs with a detectable connection, rank by most
# negative product (largest |CC|*|dW|).
candidates_min = [pi for pi in last_bin_pairs_min
                  if Y_T[pi] < 0 and _has_connection(pi)]
candidates_min.sort(key=lambda pi: cc_min[pi] * Y_T[pi])  # ascending = most negative first

if len(candidates_min) == 0:
    print("No last-bin pairs with negative dW in min-HI window.")
else:
    print(f"{len(candidates_min)} negative-dW pairs in last bin of min-HI window")
    for r, pi in enumerate(candidates_min[:10]):
        print(f"  rank {r}: pair {pi}  gi={int(all_gi[pi])}  ni={int(all_nt[pi])}  "
              f"CC={cc_min[pi]:+.3f}  dW={Y_T[pi]:+.4f}")

    pi_pick = candidates_min[RANK_MIN]
    gi_pick = int(all_gi[pi_pick])
    ni_pick = int(all_nt[pi_pick])
    cl_pick = pair_cl_list_idx[pi_pick]
    cc_pick = cc_min[pi_pick]
    dw_pick = Y_T[pi_pick]

    # Preserve for Cell 11 (heatmap markers)
    pi_pick_min = pi_pick
    gi_pick_min = gi_pick
    ni_pick_min = ni_pick
    cl_pick_min = cl_pick
    cc_pick_min = cc_pick
    dw_pick_min = dw_pick

    pw0, pw1 = pre_win_list[0], pre_win_list[1]

    tr0_post = favg_list[0][:, ni_pick, gi_pick].copy()
    tr1_post = favg_list[1][:, ni_pick, gi_pick].copy()
    tr0_post -= np.nanmean(tr0_post[pw0[0]:pw0[1]])
    tr1_post -= np.nanmean(tr1_post[pw1[0]:pw1[1]])
    tr0_post = medfilt(tr0_post, 3)
    tr1_post = medfilt(tr1_post, 3)

    tr0_pre = np.nanmean(favg_list[0][:, cl_pick, gi_pick], axis=1)
    tr1_pre = np.nanmean(favg_list[1][:, cl_pick, gi_pick], axis=1)
    tr0_pre -= np.nanmean(tr0_pre[pw0[0]:pw0[1]])
    tr1_pre -= np.nanmean(tr1_pre[pw1[0]:pw1[1]])

    # ~1 in x 1 in per panel
    fig10 = plt.figure(figsize=(3.0, 1.6))
    gs10 = GridSpec(1, 2, figure=fig10,
                    left=0.14, right=0.95, bottom=0.22, top=0.78, wspace=0.40)

    vis_pick = (t_plot >= xlims[0]) & (t_plot <= xlims[1])

    # Post
    ax = fig10.add_subplot(gs10[0, 0])
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_, label='stim')
    ax.plot(t_plot, tr0_post, color=C_POST_EARLY, linewidth=1.5, label='Early')
    ax.plot(t_plot, tr1_post, color=C_POST_LATE_, linewidth=1.5, label='Late')
    ax.set_xlim(xlims)
    yvals = np.concatenate([tr0_post[vis_pick], tr1_post[vis_pick]])
    yvals = yvals[np.isfinite(yvals)]
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Postsynaptic neuron {ni_pick}\n$\\Delta W$ = {dw_pick:+.4f}')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pre
    ax = fig10.add_subplot(gs10[0, 1])
    ax.axvspan(stim_start, stim_end, alpha=0.12, color=C_3RD_, label='stim')
    ax.plot(t_plot, tr0_pre, color=C_PRE_EARLY, linewidth=1.5, label='Early')
    ax.plot(t_plot, tr1_pre, color=C_PRE_LATE_, linewidth=1.5, label='Late')
    ax.set_xlim(xlims)
    yvals = np.concatenate([tr0_pre[vis_pick], tr1_pre[vis_pick]])
    yvals = yvals[np.isfinite(yvals)]
    ypad = 0.1 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-12)
    ax.set_ylim(np.nanmin(yvals) - ypad, np.nanmax(yvals) + ypad)
    ax.set_xlabel('Time from stim offset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Presynaptic (n={len(cl_pick)})\nCC = {cc_pick:+.3f}')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig10.suptitle(f'ex. 2 — {mouse} {session} — group {gi_pick}, rank {RANK_MIN} '
                   f'(min-HI last-bin, dW<0)', fontsize=8, fontweight='bold', y=0.97)

    fname10 = f'hebbian_minHI_lastbin_example_{mouse}_{session}_g{gi_pick}_n{ni_pick}'
    fig10.savefig(os.path.join(PANEL_DIR, f'{fname10}.png'), dpi=300)
    fig10.savefig(os.path.join(PANEL_DIR, f'{fname10}.svg'))
    plt.show()
    print(f"Saved to {PANEL_DIR}/{fname10}.png/.svg")

#%% ============================================================================
# CELL 11: CC heatmaps with picked pairs marked
# ----------------------------------------------------------------------------
# Re-plots the max-HI and min-HI CC dev2 matrices (same as Cell 4) and overlays
# markers at the cells corresponding to the picks from Cell 9 (max-HI) and
# Cell 10 (min-HI). Each picked pair has multiple presynaptic cl neurons and
# one postsynaptic ni neuron — we mark each (cl, ni) cell. Convention:
# CC_mats[wi][i, j] = sum_t pre[i, t] * post_dev[j, t], so rows = pre, cols = post.
# ============================================================================
CC_mats_11 = {}
for wi in [wi_max, wi_min]:
    tidx = np.arange(win_starts[wi], win_starts[wi] + WIN_SIZE)
    pre = epoch_pre[:, tidx]
    post_dev = epoch_post[:, tidx] - bl_post_mean[:, None]
    CC_mats_11[wi] = pre @ post_dev.T

b_11 = np.argsort(np.sum(CC_mats_11[wi_max] - CC_mats_11[wi_min], axis=0))
inv_b_11 = np.argsort(b_11)  # original neuron c -> sorted position inv_b[c]
vmax_11 = max(np.nanmax(np.abs(CC_mats_11[wi_max])),
              np.nanmax(np.abs(CC_mats_11[wi_min]))) * 0.1

# Both picked pairs marked on both panels with text labels
picks_11 = [
    ('ex. 1', gi_pick_max, ni_pick_max, cl_pick_max,
     cc_pick_max, dw_pick_max),   # max-HI pick
    ('ex. 2', gi_pick_min, ni_pick_min, cl_pick_min,
     cc_pick_min, dw_pick_min),   # min-HI pick
]

MARKER_SIZE = 120  # bigger circles

# Match heatmap size from Cell 6 row 1 (~1 in x 1 in per panel)
fig11, axes11 = plt.subplots(1, 2, figsize=(3.0, 1.6),
                              gridspec_kw={'left': 0.14, 'right': 0.92,
                                           'bottom': 0.18, 'top': 0.78,
                                           'wspace': 0.40})

panels_11 = [
    (wi_max, 'Max HI', C_POST_LATE, axes11[0]),
    (wi_min, 'Min HI', C_PRE_LATE, axes11[1]),
]

for wi, label, title_color, ax in panels_11:
    CC_sorted = CC_mats_11[wi][b_11, :][:, b_11]
    im = ax.imshow(CC_sorted, cmap='coolwarm', vmin=-vmax_11, vmax=vmax_11,
                   aspect='equal', interpolation='nearest')

    # Overlay both picks; row = cl in sorted space, col = ni in sorted space
    n_neurons_sorted = CC_sorted.shape[0]
    for tag, _gi, ni_p, cl_p, _cc, _dw in picks_11:
        col_pos = inv_b_11[ni_p]
        rows_pos = [inv_b_11[c] for c in cl_p]
        ax.scatter([col_pos] * len(rows_pos), rows_pos,
                   s=MARKER_SIZE, marker='o',
                   facecolors='none', edgecolors='k', linewidths=1.5)
        # Label near the topmost dot, offset slightly to the right
        label_row = min(rows_pos)
        ax.text(col_pos + 0.04 * n_neurons_sorted, label_row, tag,
                ha='left', va='center', color='k', fontsize=8,
                fontweight='bold')

    slope_val = hi_slope[wi]
    trial_range = f'trials {win_starts[wi]}-{win_starts[wi]+WIN_SIZE}'
    ax.set_title(f'{label} (HI={slope_val:+.4f})\n{trial_range}',
                 color=title_color)
    ax.set_xlabel('Postsynaptic neuron')
    ax.set_ylabel('Presynaptic neuron')
    plt.colorbar(im, ax=ax, shrink=0.8, label='CC dev2')

fname11 = f'hebbian_index_matrices_marked_{mouse}_{session}'
fig11.savefig(os.path.join(PANEL_DIR, f'{fname11}.png'), dpi=300)
fig11.savefig(os.path.join(PANEL_DIR, f'{fname11}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname11}.png/.svg")

#%% ============================================================================
# CELL 12: Cell 6 figure + new bottom row with ex. 1 and ex. 2 post-only traces
# ----------------------------------------------------------------------------
# Same content as Cell 6 (rastermap + CC matrices with markers + HI + CC vs dW),
# plus an extra row showing only the postsynaptic photostim traces for the two
# picked example pairs (ex. 1 from Cell 9, ex. 2 from Cell 10), Early vs Late.
# Requires Cell 6 to have run (df_binned, trial_starts_frame, frames_per_file_rm,
# n_total_frames, n_n, vmax_rm) and Cells 7, 9, 10 (favg_list, picks).
# ============================================================================

# Optional trial-range clipping for rastermap, RPE, and HI rows.
# Set to None for the full session, or a (lo, hi) tuple (half-open).
TRIAL_RANGE = (0,40)  # e.g. (50, 200)

t_lo = 0 if TRIAL_RANGE is None else max(0, int(TRIAL_RANGE[0]))
t_hi = trl if TRIAL_RANGE is None else min(trl, int(TRIAL_RANGE[1]))

# CC matrices (same as Cell 4 / Cell 6 row 1)
CC_mats_12 = {}
for wi in [wi_max, wi_min]:
    tidx = np.arange(win_starts[wi], win_starts[wi] + WIN_SIZE)
    pre = epoch_pre[:, tidx]
    post_dev = epoch_post[:, tidx] - bl_post_mean[:, None]
    CC_mats_12[wi] = pre @ post_dev.T

b_12 = np.argsort(np.sum(CC_mats_12[wi_max] - CC_mats_12[wi_min], axis=0))
inv_b_12 = np.argsort(b_12)
vmax_cc_12 = max(np.nanmax(np.abs(CC_mats_12[wi_max])),
                 np.nanmax(np.abs(CC_mats_12[wi_min]))) * 0.1

picks_12 = [
    ('ex. 1', gi_pick_max, ni_pick_max, cl_pick_max),
    ('ex. 2', gi_pick_min, ni_pick_min, cl_pick_min),
]

# Postsynaptic photostim traces for each pick (medfilt as in Cell 9/10)
post_traces_12 = []  # list of (label, tr_early, tr_late, dw)
for tag, gi_p, ni_p, _cl_p, dw_p in [
    ('ex. 1', gi_pick_max, ni_pick_max, cl_pick_max, dw_pick_max),
    ('ex. 2', gi_pick_min, ni_pick_min, cl_pick_min, dw_pick_min),
]:
    tr0 = favg_list[0][:, ni_p, gi_p].copy()
    tr1 = favg_list[1][:, ni_p, gi_p].copy()
    tr0 -= np.nanmean(tr0[pre_win_list[0][0]:pre_win_list[0][1]])
    tr1 -= np.nanmean(tr1[pre_win_list[1][0]:pre_win_list[1][1]])
    tr0 = medfilt(tr0, 3)
    tr1 = medfilt(tr1, 3)
    post_traces_12.append((tag, tr0, tr1, dw_p))

# Horizontal 2x4 layout:
#   (1,1) RPE/gain + rastermap (sub-stacked)   (1,2) CC HI>0   (1,3) photostim ex.1   (1,4) dW vs CC HI>0
#   (2,1) HI vs trial                          (2,2) CC HI<0   (2,3) photostim ex.2   (2,4) dW vs CC HI<0
fig12 = plt.figure(figsize=(6.33, 2.84))
gs12 = GridSpec(2, 4, figure=fig12,
                width_ratios=[1.0, 1.0, 1.0, 1.0],
                height_ratios=[1.0, 1.0],
                left=0.07, right=0.96, bottom=0.16, top=0.82,
                wspace=0.55, hspace=0.75)
RPE_COLOR = '#F99D20'

# Reference-line style (lighter for breathing room)
REF_ALPHA = 0.15
REF_LW = 0.4

# Frame-range clip corresponding to trial range
n_trials_x = min(trl, len(trial_starts_frame))
t_hi_x = min(t_hi, n_trials_x)
f_lo = trial_starts_frame[t_lo] if t_lo < n_trials_x else 0
f_hi = (trial_starts_frame[t_hi_x - 1] + frames_per_file_rm[t_hi_x - 1]
        if t_hi_x > 0 else n_total_frames)

trial_x_centers = np.array([trial_starts_frame[t] + frames_per_file_rm[t] / 2.0
                            for t in range(n_trials_x)])

# Gain (g0/g) — drops when the upper threshold goes up; matches figure_threshold_learning.py
_lwr = BCI_thresholds[0, :].astype(float).copy()
for _i in range(1, _lwr.size):
    if np.isnan(_lwr[_i]):
        _lwr[_i] = _lwr[_i - 1]
if np.isnan(_lwr[0]) and np.any(np.isfinite(_lwr)):
    _lwr[0] = _lwr[np.isfinite(_lwr)][0]
_g = (BCI_thresholds[1, :n_trials_x] - _lwr[:n_trials_x]).astype(float)
_g0 = next((v for v in _g if np.isfinite(v) and v > 0), np.nan)
gain_per_trial = np.where((_g > 0) & np.isfinite(_g), _g0 / _g, np.nan)

# === (1,1): RPE/gain strip (top) + rastermap (bottom), flush ===
gs_topleft = gs12[0, 0].subgridspec(2, 1, height_ratios=[0.35, 0.85], hspace=0.0)
ax_rpe = fig12.add_subplot(gs_topleft[0])
ax_rpe.plot(trial_x_centers, rt_rpe[:n_trials_x],
            color=RPE_COLOR, linewidth=0.9)
ax_rpe.set_ylabel('RPE', color=RPE_COLOR, fontsize=7)
ax_rpe.tick_params(axis='y', labelcolor=RPE_COLOR, labelsize=6, length=2)
ax_rpe.set_xlim(f_lo, f_hi)
ax_rpe.set_xticks([])
for sp in ('top', 'bottom'):
    ax_rpe.spines[sp].set_visible(False)

ax_thr = ax_rpe.twinx()
ax_thr.plot(trial_x_centers, gain_per_trial, color='0.3', linewidth=0.9)
ax_thr.set_ylabel('Gain', color='0.3', fontsize=7)
ax_thr.tick_params(axis='y', labelsize=6, length=2, colors='0.3')
for sp in ('top', 'bottom'):
    ax_thr.spines[sp].set_visible(False)

ax_rm = fig12.add_subplot(gs_topleft[1])
rm_im = ax_rm.imshow(df_binned, aspect='auto', cmap='bwr',
                     vmin=-vmax_rm, vmax=vmax_rm,
                     extent=[0, n_total_frames, n_n, 0],
                     interpolation='nearest', rasterized=True)
for wi, color in [(wi_max, C_POST_LATE), (wi_min, C_PRE_LATE)]:
    ws = win_starts[wi]
    we = ws + WIN_SIZE - 1
    if we < len(trial_starts_frame):
        x0 = trial_starts_frame[ws]
        x1 = trial_starts_frame[we] + frames_per_file_rm[we]
        ax_rm.axvspan(x0, x1, color=color, alpha=0.18, zorder=0)
ax_rm.set_xlim(f_lo, f_hi)
ax_rm.set_ylabel('Neurons', fontsize=7)
TRIAL_LABEL_STEP = 20
_label_trials_1 = [1] + list(range(TRIAL_LABEL_STEP, n_trials_x + 1, TRIAL_LABEL_STEP))
_label_trials_1 = sorted(set(t for t in _label_trials_1
                             if t_lo + 1 <= t <= t_hi_x and (t - 1) < len(trial_starts_frame)))
_label_frames = [trial_starts_frame[t - 1] for t in _label_trials_1]
ax_rm.set_xticks(_label_frames)
ax_rm.set_xticklabels(_label_trials_1)
ax_rm.set_xlabel('Trial', fontsize=7)
ax_rm.tick_params(labelsize=6, length=2)

# === (2,1): HI vs trial (takeaway) ===
ax_hi = fig12.add_subplot(gs12[1, 0])
for wi, color in [(wi_max, C_POST_LATE), (wi_min, C_PRE_LATE)]:
    ws = win_starts[wi]
    we = ws + WIN_SIZE
    ax_hi.axvspan(ws, we, color=color, alpha=0.12, zorder=0)
ax_hi.axhline(0, color='k', alpha=REF_ALPHA, linewidth=REF_LW)
ax_hi.plot(win_centers, hi_slope, 'o-', color='k', linewidth=1.0, markersize=2.5)
ax_hi.plot(win_centers[wi_max], hi_slope[wi_max], 'o', color=C_POST_LATE,
           markersize=6, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
ax_hi.plot(win_centers[wi_min], hi_slope[wi_min], 'o', color=C_PRE_LATE,
           markersize=6, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
ax_hi.set_xlim(t_lo, t_hi_x)
ax_hi.set_ylabel('Hebbian index', fontsize=7)
ax_hi.set_xlabel('Trial', fontsize=7)
ax_hi.tick_params(labelsize=6, length=2)
ax_hi.spines['top'].set_visible(False)
ax_hi.spines['right'].set_visible(False)

# === (1,2)/(2,2): CC matrices ===
matrix_axes = []
matrix_im = None
for row, wi in enumerate([wi_max, wi_min]):
    ax = fig12.add_subplot(gs12[row, 1])
    matrix_axes.append(ax)
    CC_sorted = CC_mats_12[wi][b_12, :][:, b_12]
    matrix_im = ax.imshow(CC_sorted, cmap='bwr',
                          vmin=-vmax_cc_12, vmax=vmax_cc_12,
                          aspect='equal', interpolation='nearest')
    n_neurons_sorted = CC_sorted.shape[0]
    for tag, _gi, ni_p, cl_p in picks_12:
        col_pos = inv_b_12[ni_p]
        rows_pos = [inv_b_12[c] for c in cl_p]
        ax.scatter([col_pos] * len(rows_pos), rows_pos,
                   s=60, marker='o',
                   facecolors='none', edgecolors='k', linewidths=0.9)
        ax.text(col_pos + 0.04 * n_neurons_sorted, min(rows_pos), tag,
                ha='left', va='center', color='k', fontsize=6, fontweight='bold')
    ax.tick_params(labelsize=6, length=2)
    ax.set_ylabel('Neuron', fontsize=7)
    if row == 1:
        ax.set_xlabel('Neuron', fontsize=7)
    else:
        ax.set_xticklabels([])

# Small horizontal colorbars above col 1 (rastermap) and col 2 (matrices)
def _add_h_colorbar(im, anchor_bbox, label):
    """Place a small horizontal colorbar centered above anchor_bbox."""
    w = anchor_bbox.width * 0.75
    x = anchor_bbox.x0 + (anchor_bbox.width - w) / 2
    y = 0.86
    h = 0.020
    cax = fig12.add_axes([x, y, w, h])
    cb = fig12.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label(label, fontsize=6, labelpad=1)
    cb.outline.set_linewidth(0.4)
    return cb

_add_h_colorbar(rm_im, ax_rm.get_position(), 'z-score')
_add_h_colorbar(matrix_im, matrix_axes[0].get_position(), 'CC')

# === (1,3)/(2,3): Postsynaptic photostim traces — independent y-axes, scale bars ===
vis_12 = (t_plot >= xlims[0]) & (t_plot <= xlims[1])

def _nice_step(rng, frac=0.4):
    target = rng * frac
    if target <= 0:
        return 0.1
    exponent = np.floor(np.log10(target))
    base = target / (10 ** exponent)
    nice = 1.0 if base < 1.5 else (2.0 if base < 3.5 else (5.0 if base < 7.5 else 10.0))
    return nice * (10 ** exponent)

X_SCALE_S = 0.2
for row, (tag, tr0, tr1, dw_p) in enumerate(post_traces_12):
    ax = fig12.add_subplot(gs12[row, 2])
    ax.axvspan(stim_start, stim_end, alpha=0.10, color=C_3RD_)
    ax.plot(t_plot, tr0, color=C_POST_EARLY, linewidth=1.0, label='Early')
    ax.plot(t_plot, tr1, color=C_POST_LATE_, linewidth=1.4, label='Late')
    ax.set_xlim(xlims)
    yvals = np.concatenate([tr0[vis_12], tr1[vis_12]])
    yvals = yvals[np.isfinite(yvals)]
    y_lo_p, y_hi_p = np.nanmin(yvals), np.nanmax(yvals)
    ypad_p = 0.12 * (y_hi_p - y_lo_p + 1e-12)
    ax.set_ylim(y_lo_p - ypad_p * 2.2, y_hi_p + ypad_p)

    y_step = _nice_step(y_hi_p - y_lo_p, frac=0.4)
    x0 = xlims[0] + 0.04 * (xlims[1] - xlims[0])
    y0 = y_lo_p - ypad_p * 1.4
    ax.plot([x0, x0 + X_SCALE_S], [y0, y0],
            color='k', linewidth=1.2, solid_capstyle='butt', clip_on=False)
    ax.plot([x0, x0], [y0, y0 + y_step],
            color='k', linewidth=1.2, solid_capstyle='butt', clip_on=False)
    ax.text(x0 + X_SCALE_S / 2, y0 - ypad_p * 0.6, f'{X_SCALE_S:g} s',
            ha='center', va='top', fontsize=5.5)
    ax.text(x0 - 0.01 * (xlims[1] - xlims[0]), y0 + y_step / 2,
            f'{y_step:g} dF/F', ha='right', va='center', fontsize=5.5)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right', 'bottom', 'left'):
        ax.spines[sp].set_visible(False)
    if row == 0:
        ax.legend(frameon=False, fontsize=5.5, loc='upper right',
                  handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title(f'{tag}  $\\Delta W$={dw_p:+.3f}', fontsize=7)

# === (1,4)/(2,4): CC vs dW scatter ===
bin_X_all_12, bin_Y_all_12 = [], []
scatter_axes = []
for row, (wi, color) in enumerate([(wi_max, C_POST_LATE), (wi_min, C_PRE_LATE)]):
    ax = fig12.add_subplot(gs12[row, 3])
    scatter_axes.append(ax)
    cc = cc_per_win[wi, :]
    dw = Y_T
    ok = np.isfinite(cc) & np.isfinite(dw)
    cc_ok, dw_ok = cc[ok], dw[ok]
    plt.sca(ax)
    X_bin, Y_bin, _ = pf.mean_bin_plot(cc_ok, dw_ok, col=4, color='k')
    bin_X_all_12.append(X_bin)
    bin_Y_all_12.append(Y_bin)
    if np.std(cc_ok) > 0:
        A_mat = np.column_stack([np.ones(len(cc_ok)), cc_ok])
        coeffs = np.linalg.lstsq(A_mat, dw_ok, rcond=None)[0]
        xr = np.array([X_bin[0], X_bin[-1]])
        ax.plot(xr, coeffs[0] + coeffs[1] * xr, color=C_3RD, linewidth=1.2)
    ax.axhline(0, color='k', alpha=REF_ALPHA, linewidth=REF_LW)
    ax.axvline(0, color='k', ls='--', alpha=REF_ALPHA, linewidth=REF_LW)
    ax.tick_params(labelsize=6, length=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Delta W$', fontsize=7)
    if row == 1:
        ax.set_xlabel('CC (dev2)', fontsize=7)
    else:
        ax.set_xticklabels([])

all_bx = np.concatenate(bin_X_all_12)
all_by = np.concatenate(bin_Y_all_12)
xpad = 0.15 * (np.max(all_bx) - np.min(all_bx))
ypad = 0.15 * (np.max(all_by) - np.min(all_by))
xl = [np.min(all_bx) - xpad, np.max(all_bx) + xpad]
yl = [np.min(all_by) - ypad, np.max(all_by) + ypad]
for ax in scatter_axes:
    ax.set_xlim(xl); ax.set_ylim(yl)

fname12 = f'figure_correlations_with_examples_{mouse}_{session}'
fig12.savefig(os.path.join(PANEL_DIR, f'{fname12}.png'), dpi=300)
fig12.savefig(os.path.join(PANEL_DIR, f'{fname12}.svg'))
plt.show()
print(f"Saved to {PANEL_DIR}/{fname12}.png/.svg")

#%% ============================================================================
# CELL 13: Per-pair r_pre and (r_post - mean(r_post baseline)) traces
# ----------------------------------------------------------------------------
# One small figure per example pair, mirroring the binned-activity panel in
# example_pair_simple.py Cell 4. Each plot shows the windowed, z-scored
# presynaptic activity (cl-group mean) and the windowed, z-scored postsynaptic
# deviation (ni neuron minus its baseline mean), aligned to trial number.
# Axes sized ~1 in to match the main figure (Cell 12).
# ============================================================================

C_PRE_PLOT  = '#1d4ed8'  # presynaptic blue
C_POST_PLOT = '#dc2626'  # postsynaptic red

def _binned_z(x_trials, win_starts, WIN_SIZE):
    binned = np.array([np.mean(x_trials[ws:ws + WIN_SIZE]) for ws in win_starts])
    return (binned - np.mean(binned)) / (np.std(binned) + 1e-10)

for tag, ni_p, cl_p, dw_p in [
    ('ex. 1', ni_pick_max, cl_pick_max, dw_pick_max),
    ('ex. 2', ni_pick_min, cl_pick_min, dw_pick_min),
]:
    r_pre = np.mean(epoch_pre[cl_p, :], axis=0)
    r_post_dev = epoch_post[ni_p, :] - bl_post_mean[ni_p]
    pre_z = _binned_z(r_pre, win_starts, WIN_SIZE)
    post_z = _binned_z(r_post_dev, win_starts, WIN_SIZE)

    fig13 = plt.figure(figsize=(1.9, 1.55))
    gs13 = GridSpec(1, 1, figure=fig13,
                    left=0.22, right=0.96, bottom=0.24, top=0.84)
    ax = fig13.add_subplot(gs13[0])
    ax.plot(win_centers, pre_z, 'o-', color=C_PRE_PLOT, linewidth=1.0,
            markersize=2.5, label=r'$r_{pre}$')
    ax.plot(win_centers, post_z, 's-', color=C_POST_PLOT, linewidth=1.0,
            markersize=2.5, label=r'$r_{post} - \overline{r_{post}}$')
    ax.axhline(0, color='k', alpha=REF_ALPHA, linewidth=REF_LW)
    ax.set_xlabel('Trial', fontsize=7)
    ax.set_ylabel('z-score', fontsize=7)
    ax.set_title(f'{tag}  $\\Delta W$={dw_p:+.3f}', fontsize=7.5)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(frameon=False, fontsize=6, loc='best',
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fname13 = f'hebbian_pair_activity_{tag.replace(". ", "")}_{mouse}_{session}'
    fig13.savefig(os.path.join(PANEL_DIR, f'{fname13}.png'), dpi=300)
    fig13.savefig(os.path.join(PANEL_DIR, f'{fname13}.svg'))
    plt.show()
    print(f"Saved to {PANEL_DIR}/{fname13}.png/.svg")
