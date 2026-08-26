"""
Plot photostim AMP vs stimDist for a single session, using the artifact-free
amp computation (asymmetric dF/F, no artifact detection, stim window from
stim_params).

Loads the photostim2 epoch from BCI116/012826 the same way
inhibitory_artifact_removal.py does, then calls
compute_amp_from_photostim_artifact_free.
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import BCI_data_helpers
import importlib
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import compute_amp_from_photostim_artifact_free


# -----------------------------
# Session
# -----------------------------
mouse  = 'BCI116'
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI116/012726/pophys/'
ps_epoch = 'photostim2'   # 'photostim' or 'photostim2'

mouse = '855520'
folder = r'//allen/aind/scratch/BCI/2p-raw/855520/052126/'   # vglut3
ps_epoch = 'photostim'   # 'photostim' or 'photostim2'

# Baseline-floor mask for the asymmetric dF/F. compute_amp_from_photostim_artifact_free
# divides by the per-cell baseline (mean of first 3 frames, averaged across groups).
# For dim sessions that denominator is tiny or negative for many cells, which blows
# up AMP. Cells dimmer than this F0 percentile (or with a non-positive baseline) are
# NaN-masked out of AMP. Set to 0 to disable the percentile floor (non-positive
# baselines are always masked).
BASELINE_FLOOR_PCTILE = 10

# Target-responsiveness gate. A stim group is kept only if its target cell (the
# nearest cell to the stim site) has its own AMP above this threshold; weak /
# non-responding targets are excluded from every plot (scatter, binned means,
# heatmap, quartile traces). Set to -np.inf to disable.
target_amp_thresh = 0.05

SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# Find + load the photostim h5
# -----------------------------
def find_photostim_h5(folder, ps_epoch):
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        candidates = [
            p for p in all_h5
            if 'photostim' in os.path.basename(p).lower()
            and 'photostim2' not in os.path.basename(p).lower()
        ]
    elif ps_epoch == 'photostim2':
        candidates = [p for p in all_h5 if 'photostim2' in os.path.basename(p).lower()]
    else:
        raise ValueError("ps_epoch must be 'photostim' or 'photostim2'")
    if not candidates:
        raise FileNotFoundError(f'No h5 found for {ps_epoch} in {folder}')
    return candidates[0]


def load_h5_epoch(h5_path):
    """Load stimDist, favg_raw, and stim_params from a single photostim h5.

    Uses dataset[()] (works for both arrays and scalar datasets — [:] silently
    fails on scalars like total_duration).
    """
    d = {}
    with h5py.File(h5_path, 'r') as f:
        for key in ['stimDist', 'favg_raw']:
            if key in f:
                d[key] = f[key][()]
        if 'stim_params' in f:
            sp = {}
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][()]
                except Exception:
                    pass
            d['stim_params'] = sp
    return d


h5_path = find_photostim_h5(folder, ps_epoch)
print(f'Loading {h5_path}')
ep_data = load_h5_epoch(h5_path)


# -----------------------------
# Diagnostic dump of stim_params (so we can figure out the time layout)
# -----------------------------
_diag_path = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\stim_params_diagnostic.txt'

def _diag_lines(ep_data):
    lines = []
    lines.append(f'h5_path: {h5_path}')
    lines.append(f'ps_epoch: {ps_epoch}')
    lines.append('')
    lines.append(f'favg_raw shape: {np.asarray(ep_data["favg_raw"]).shape}')
    lines.append(f'stimDist shape: {np.asarray(ep_data["stimDist"]).shape}')
    lines.append('')
    sp = ep_data.get('stim_params', {})
    lines.append(f'stim_params keys: {list(sp.keys())}')
    lines.append('')
    for k, v in sp.items():
        v_arr = np.asarray(v)
        lines.append(f'--- {k} ---')
        lines.append(f'  shape: {v_arr.shape}')
        lines.append(f'  dtype: {v_arr.dtype}')
        flat = v_arr.ravel()
        lines.append(f'  ravel[:30]: {flat[:30].tolist()}')
        if flat.size > 30:
            lines.append(f'  ravel[-10:]: {flat[-10:].tolist()}')
        if np.issubdtype(v_arr.dtype, np.number):
            try:
                lines.append(f'  min/max: {np.nanmin(v_arr)} / {np.nanmax(v_arr)}')
            except Exception as e:
                lines.append(f'  min/max: ERROR {e}')
            try:
                zeros = np.where(np.isclose(flat, 0.0))[0]
                lines.append(f'  zero positions (first 20): {zeros[:20].tolist()}')
            except Exception as e:
                lines.append(f'  zero positions: ERROR {e}')
        lines.append('')
    return lines

try:
    _lines = _diag_lines(ep_data)
    _text = '\n'.join(_lines)
    print('\n========== STIM_PARAMS DIAGNOSTIC ==========')
    print(_text)
    print('=============================================\n')
    try:
        with open(_diag_path, 'w') as _f:
            _f.write(_text)
        print(f'Wrote diagnostic to {_diag_path}')
    except Exception as e:
        print(f'Could not write file: {e}')
except Exception as e:
    print(f'Diagnostic failed: {e}')
    import traceback
    traceback.print_exc()


# -----------------------------
# Build data dict in the shape compute_amp_from_photostim_artifact_free expects
# -----------------------------
# The artifact-free function only looks at one epoch key at a time (`photostim`
# or `photostim` + `photostim2`). We expose the loaded epoch under whichever
# key matches `ps_epoch`, and supply dt_si from the suite2p siHeader.
suite2p_folder = 'suite2p_photostim_single' if ps_epoch == 'photostim' else 'suite2p_photostim_single2'
si_path = os.path.join(folder, suite2p_folder, 'plane0', 'siHeader.npy')
if not os.path.exists(si_path):
    si_path = os.path.join(folder, 'suite2p_photostim_single', 'plane0', 'siHeader.npy')
siHeader = np.load(si_path, allow_pickle=True).tolist()
dt_si = 1.0 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])

data = {'dt_si': dt_si}
if ps_epoch == 'photostim':
    data['photostim'] = ep_data
else:
    # function loops over photostim then photostim2 — populate both so the
    # photostim2 epoch is the one returned. For a session with only photostim2,
    # we duplicate the data into 'photostim' as a placeholder.
    data['photostim']  = ep_data
    data['photostim2'] = ep_data


AMP, stimDist = compute_amp_from_photostim_artifact_free(mouse, data, folder)

# AMP is a list (one entry per epoch); pick the one that corresponds to ps_epoch
amp = AMP[-1] if ps_epoch == 'photostim2' else AMP[0]

print(f'AMP shape: {amp.shape}, stimDist shape: {stimDist.shape}')


# -----------------------------
# Baseline-floor mask: drop dim / non-positive-baseline cells
# -----------------------------
# Recompute the per-cell baseline the same way the helper does (mean of first 3
# frames per group, averaged across groups), then NaN out cells whose baseline is
# non-positive (sign-flips dF/F) or below the percentile floor (1/F0 blow-up).
favg_raw_all = np.asarray(ep_data['favg_raw'], dtype=float)
bl_per_group = np.nanmean(favg_raw_all[0:3, :, :], axis=0)   # (n_cells, n_groups)
bl_per_cell  = np.nanmean(bl_per_group, axis=-1)             # (n_cells,)

pos_bl = bl_per_cell[np.isfinite(bl_per_cell) & (bl_per_cell > 0)]
floor_F = np.nanpercentile(pos_bl, BASELINE_FLOOR_PCTILE) if pos_bl.size else 0.0

bad_cells = (~np.isfinite(bl_per_cell)) | (bl_per_cell <= 0) | (bl_per_cell < floor_F)
print(
    f'baseline floor: pctile {BASELINE_FLOOR_PCTILE} -> F0 >= {floor_F:.3f}; '
    f'masking {int(bad_cells.sum())}/{bl_per_cell.size} cells '
    f'(non-positive or dim baseline)'
)

amp = amp.copy()
amp[bad_cells, :] = np.nan


# -----------------------------
# Target-responsiveness gate (applied to all plots)
# -----------------------------
# The target of each group is the nearest cell to the stim site. Groups whose
# target's own AMP is at/below target_amp_thresh are weak / non-responders; NaN
# their whole column so they drop out of every downstream plot.
n_cells, n_groups = amp.shape
target_cell_idx = np.argmin(stimDist, axis=0)                # (n_groups,)
target_amp = amp[target_cell_idx, np.arange(n_groups)]       # (n_groups,)
target_dist = stimDist[target_cell_idx, np.arange(n_groups)] # (n_groups,)

responsive = np.isfinite(target_amp) & (target_amp > target_amp_thresh)
keep_groups = np.where(responsive)[0]
print(f'{responsive.sum()}/{n_groups} targets responsive (target amp > {target_amp_thresh})')
if responsive.any():
    print(f'  median target distance: {np.nanmedian(target_dist[responsive]):.1f} µm')

amp[:, ~responsive] = np.nan


#%%
# -----------------------------
# Scatter: AMP vs stimDist
# -----------------------------
amp_flat  = amp.ravel()
dist_flat = stimDist.ravel()

mask = np.isfinite(amp_flat) & np.isfinite(dist_flat)
amp_flat  = amp_flat[mask]
dist_flat = dist_flat[mask]

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(dist_flat, amp_flat, s=6, alpha=0.25, color='black', edgecolor='none')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Distance from stim site (µm)', fontsize=12)
ax.set_ylabel('AMP (ΔF/F)', fontsize=12)
ax.set_title(
    f'{mouse} {ps_epoch}: AMP vs stim distance '
    f'(responsive targets, n={len(keep_groups)})',
    fontsize=12,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'amp_vs_stimDist_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Binned mean ± SEM
# -----------------------------
edges   = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500])
centers = 0.5 * (edges[:-1] + edges[1:])
mean_b  = np.full(len(centers), np.nan)
sem_b   = np.full(len(centers), np.nan)
n_b     = np.zeros(len(centers), dtype=int)

for k in range(len(centers)):
    sel = (dist_flat >= edges[k]) & (dist_flat < edges[k + 1])
    if sel.sum() > 0:
        v = amp_flat[sel]
        mean_b[k] = np.nanmean(v)
        sem_b[k]  = np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v)))
        n_b[k]    = int(np.sum(np.isfinite(v)))

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.errorbar(centers, mean_b, yerr=sem_b, fmt='o-', color='#0096FF',
            linewidth=1.5, markersize=5, capsize=3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(30, color='red', linestyle='--', linewidth=1.0,
           label='targeted / non-targeted boundary')
ax.set_xlabel('Distance from stim site (µm)', fontsize=12)
ax.set_ylabel('Mean AMP (ΔF/F)', fontsize=12)
ax.set_title(
    f'{mouse} {ps_epoch}: binned AMP vs distance '
    f'(responsive targets, n={len(keep_groups)})',
    fontsize=12,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Inset: zoom in on the > 30 µm (non-targeted) portion
nontarg = centers > 30
ax_in = ax.inset_axes([0.45, 0.45, 0.5, 0.45])
ax_in.errorbar(centers[nontarg], mean_b[nontarg], yerr=sem_b[nontarg],
               fmt='o-', color='#0096FF', linewidth=1.2, markersize=4, capsize=2)
ax_in.axhline(0, color='gray', linewidth=0.5)
ax_in.set_title('non-targeted (>30 µm)', fontsize=9)
ax_in.tick_params(axis='both', labelsize=8)
ax_in.spines['top'].set_visible(False)
ax_in.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'amp_vs_stimDist_binned.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Per-target connectivity heatmap (non-targeted only, > 30 µm)
#
# Each row = one stim group whose target cell is itself responsive
# (target amp > target_amp_thresh). Columns = distance bins. Cell color
# = mean AMP across cells in that (target, distance) bin. Rows are
# sorted by net effect on the network (mean AMP over all cells > 30 µm).
# Heterogeneity in connectivity should show as variation across rows.
# -----------------------------

# target_cell_idx, target_amp, target_dist, responsive, keep_groups are all
# computed once above (target-responsiveness gate) and reused here.

# Distance bins for the heatmap (only non-targeted cells, > 30 µm)
hm_edges   = np.arange(30, 501, 50)
hm_centers = 0.5 * (hm_edges[:-1] + hm_edges[1:])
n_hm_bins  = len(hm_centers)

heat       = np.full((len(keep_groups), n_hm_bins), np.nan)
net_effect = np.full(len(keep_groups), np.nan)

for r, g in enumerate(keep_groups):
    d = stimDist[:, g]
    a = amp[:, g]
    valid = np.isfinite(a) & np.isfinite(d)
    for b in range(n_hm_bins):
        sel = valid & (d >= hm_edges[b]) & (d < hm_edges[b + 1])
        if sel.any():
            heat[r, b] = np.nanmean(a[sel])
    nontarg_sel = valid & (d > 30)
    if nontarg_sel.any():
        net_effect[r] = np.nanmean(a[nontarg_sel])

# Sort rows by net effect (most suppressive at top, most facilitating at bottom)
order = np.argsort(net_effect)
heat_sorted   = heat[order, :]
net_sorted    = net_effect[order]
target_sorted = target_amp[keep_groups][order]
groups_sorted = keep_groups[order]

# Symmetric colormap centered on 0 — captures both inhibition and facilitation
vmax = np.nanpercentile(np.abs(heat_sorted), 95)
if not np.isfinite(vmax) or vmax == 0:
    vmax = np.nanmax(np.abs(heat_sorted)) if np.isfinite(np.nanmax(np.abs(heat_sorted))) else 0.05

# Layout: [direct cbar | direct | heatmap | heatmap cbar | net effect]
# Explicit gridspec keeps colorbars narrow and ensures labels don't overlap.
fig = plt.figure(figsize=(10, 5.5))
gs = fig.add_gridspec(
    1, 5,
    width_ratios=[0.08, 0.35, 3.2, 0.08, 1.0],
    wspace=0.55,
)
cax_d = fig.add_subplot(gs[0, 0])
ax_d  = fig.add_subplot(gs[0, 1])
ax_h  = fig.add_subplot(gs[0, 2], sharey=ax_d)
cax_h = fig.add_subplot(gs[0, 3])
ax_n  = fig.add_subplot(gs[0, 4], sharey=ax_d)

# Left: direct response (target's own AMP)
direct_vmax = np.nanpercentile(target_sorted, 99)
im_d = ax_d.imshow(
    target_sorted[:, None], aspect='auto', cmap='Reds',
    vmin=0, vmax=direct_vmax, interpolation='nearest',
    extent=[0, 1, len(target_sorted), 0],
)
ax_d.set_xticks([0.5])
ax_d.set_xticklabels(['target'], fontsize=9)
ax_d.set_ylabel('Target (sorted by net effect)', fontsize=11)
ax_d.set_title('Direct', fontsize=10)
cbar_d = fig.colorbar(im_d, cax=cax_d)
cbar_d.set_label('Target AMP', fontsize=8)
cbar_d.ax.tick_params(labelsize=7)
cax_d.yaxis.set_label_position('left')
cax_d.yaxis.tick_left()

# Middle: per-distance-bin AMP heatmap
im = ax_h.imshow(
    heat_sorted, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
    extent=[hm_edges[0], hm_edges[-1], heat_sorted.shape[0], 0],
    interpolation='nearest',
)
ax_h.set_xlabel('Distance from target (µm)', fontsize=11)
ax_h.set_title(
    f'{mouse} {ps_epoch}: per-target connectivity '
    f'(target amp > {target_amp_thresh}, n={len(keep_groups)} targets)',
    fontsize=10,
)
ax_h.tick_params(axis='y', labelleft=False)
cbar = fig.colorbar(im, cax=cax_h)
cbar.set_label('Mean AMP (ΔF/F)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Right: net effect per target
ax_n.plot(net_sorted, np.arange(len(net_sorted)), color='black', linewidth=1)
ax_n.axvline(0, color='gray', linewidth=0.5)
ax_n.set_xlabel('Net effect\n(mean AMP, >30 µm)', fontsize=9)
ax_n.invert_yaxis()
ax_n.tick_params(axis='y', labelleft=False)
ax_n.spines['top'].set_visible(False)
ax_n.spines['right'].set_visible(False)
ax_n.tick_params(axis='both', labelsize=8)

plt.savefig(os.path.join(SAVE_DIR, 'amp_vs_stimDist_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Quartile dF/F traces
# Split the heatmap rows (sorted by net effect) into 4 quartiles, then plot
# distance-binned dF/F traces (same six bins as inhibitory_artifact_removal.py)
# overlaid by quartile in each bin.
# -----------------------------
import math as _math

# Stim window for shading + baseline indexing
_time_arr = np.asarray(ep_data['stim_params']['time']).ravel()
_total_dur = float(np.asarray(ep_data['stim_params']['total_duration']).ravel()[0])
_start_f = int(np.where(np.isclose(_time_arr, 0.0))[0][0])
_end_hits = np.where(_time_arr >= _total_dur)[0]
_end_f = int(_end_hits[0]) if len(_end_hits) > 0 else _start_f + 16
_pre_end = _start_f - 1

favg_raw_full = np.asarray(ep_data['favg_raw'], dtype=float)  # (n_frames, n_cells, n_groups)
n_frames = favg_raw_full.shape[0]
t_arr_full = np.arange(n_frames) * dt_si

# Restrict plotting to t <= 2.0 s
_t_max = 2.0
_crop = int(np.searchsorted(t_arr_full, _t_max, side='right'))
t_arr = t_arr_full[:_crop]

# Per-cell baseline averaged across groups (asymmetric dF/F denominator).
# Same baseline floor as the AMP mask above: percentile over positive baselines,
# and the bl_global <= _floor check below also drops non-positive baselines.
_F0_per_group = np.nanmean(favg_raw_full[0:_pre_end, :, :], axis=0)
_F0_per_cell  = np.nanmean(_F0_per_group, axis=-1)
_pos_F0 = _F0_per_cell[np.isfinite(_F0_per_cell) & (_F0_per_cell > 0)]
_floor = np.nanpercentile(_pos_F0, BASELINE_FLOOR_PCTILE) if _pos_F0.size else 0.0

distance_bins = [
    ("target (< 30 µm)",  lambda d: d < 30),
    ("30–100 µm",          lambda d: (d >= 30) & (d < 100)),
    ("> 100 µm",           lambda d: d >= 200),
]

# Quartile splits — groups_sorted is in ascending net-effect order
n_targets = len(groups_sorted)
q_edges = np.linspace(0, n_targets, 5, dtype=int)
quartile_groups = [groups_sorted[q_edges[q]:q_edges[q + 1]] for q in range(4)]
quartile_labels = ['Q1 (most suppressive)', 'Q2', 'Q3', 'Q4 (most facilitating)']
quartile_colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728']

n_rows = len(quartile_groups)
n_cols = len(distance_bins)

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(10, 8),
    sharex=True, sharey='col',
)

for r, (qg, qlabel, qcolor) in enumerate(zip(quartile_groups, quartile_labels, quartile_colors)):
    for c, (label, cond) in enumerate(distance_bins):
        ax = axes[r, c]
        traces = []
        for g in qg:
            d = stimDist[:, g]
            ind = np.where(cond(d))[0]
            for j in ind:
                F = favg_raw_full[:, j, g]
                bl = np.nanmean(F[0:_pre_end])
                bl_global = _F0_per_cell[j]
                if (not np.isfinite(bl_global)) or (bl_global <= _floor):
                    continue
                traces.append((F - bl) / bl_global)
        if traces:
            traces = np.vstack(traces)[:, :_crop]
            avg = np.nanmean(traces, axis=0)
            sem = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(traces.shape[0])
            ax.plot(t_arr, avg, color=qcolor, linewidth=1.5)
            ax.fill_between(t_arr, avg - sem, avg + sem, alpha=0.2,
                            color=qcolor, edgecolor='none')
            n_text = f'n={traces.shape[0]}'
        else:
            n_text = 'n=0'
        ax.axvspan(_start_f * dt_si, _end_f * dt_si, color='#FFBF00', alpha=0.2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.text(0.97, 0.95, n_text, transform=ax.transAxes,
                fontsize=8, ha='right', va='top', color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=9)

        if r == 0:
            ax.set_title(label, fontsize=11)
        if c == 0:
            ax.set_ylabel(qlabel, fontsize=10, color=qcolor)

fig.supxlabel('Time (s)', fontsize=12)
fig.supylabel('ΔF/F', fontsize=12, x=0.02)
fig.suptitle(
    f'{mouse} {ps_epoch}: dF/F by distance × quartile '
    f'(target amp > {target_amp_thresh}, n={n_targets} targets)',
    fontsize=11, y=1.00,
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'amp_vs_stimDist_quartile_traces.png'),
            dpi=150, bbox_inches='tight')
plt.show()
