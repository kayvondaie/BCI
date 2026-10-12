"""
Significance tests on per-trial photostim responses for BCI116/012726/photostim2.

Pulls per-trial Fstim from ddc.stimDist_single_cell, runs paired pre-vs-post
t-tests across trials per (cell, group), then surfaces example single
non-targets (> 30 µm from the targeted cell) that are significantly
excited or inhibited.
"""

import sys
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

sys.path.append(r'C:\Users\christina.wang\Downloads\BCI_code_local')
import data_dict_create_module_iscell as ddc

# -----------------------------
# Session
# -----------------------------
folder    = r'//allen/aind/scratch/BCI/2p-raw/BCI116/012726/pophys/'
ps_epoch  = 'photostim2'
suite2p_folder = 'suite2p_photostim_single' if ps_epoch == 'photostim' else 'suite2p_photostim_single2'
ps_dir    = os.path.join(folder, suite2p_folder, 'plane0')

SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# Load suite2p + per-trial Fstim
# -----------------------------
iscell   = np.load(os.path.join(ps_dir, 'iscell.npy'),   allow_pickle=True)
stat     = np.load(os.path.join(ps_dir, 'stat.npy'),     allow_pickle=True)
F        = np.load(os.path.join(ps_dir, 'F.npy'),        allow_pickle=True)
ops      = np.load(os.path.join(ps_dir, 'ops.npy'),      allow_pickle=True).tolist()
siHeader = np.load(os.path.join(ps_dir, 'siHeader.npy'), allow_pickle=True).tolist()

(Fstim, seq, favg, stimDist_pix, stimPosition, centroidX, centroidY,
 slmDist, stimID, Fstim_raw, favg_raw_full, stim_params) = ddc.stimDist_single_cell(
    ops, F, siHeader, stat, 0
)
# Use the explicit raw, per-trial array; Fstim (the first return) may be
# normalized or shaped differently in this loader.
Fstim = Fstim_raw

print(f'Fstim_raw shape:    {Fstim_raw.shape}')
print(f'favg_raw_full shape:{favg_raw_full.shape}')
print(f'seq shape:          {np.asarray(seq).shape}, '
      f'first 10 = {np.asarray(seq).ravel()[:10].tolist()}')

umPerPix = (
    1000
    / float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])
    / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
)
dt_si = 1.0 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
stimDist = stimDist_pix * umPerPix      # (n_cells, n_groups), µm

# -----------------------------
# Stim window from h5 stim_params (the ddc loader's stim_params uses a
# different time convention; the h5's `time` is stim-relative with the
# zero-crossing landing at the actual stim-onset frame).
#   pre  = (start - before, start)          # 200 ms immediately before stim
#   post = (start, end)                     # exactly the stim window
# -----------------------------
def _find_h5(folder, ps_epoch):
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5 if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        raise FileNotFoundError(f'No h5 found for {ps_epoch} in {folder}')
    return cand[0]

h5_path = _find_h5(folder, ps_epoch)
with h5py.File(h5_path, 'r') as _f:
    h5_time           = np.asarray(_f['stim_params']['time'][()]).ravel()
    h5_total_duration = float(np.asarray(_f['stim_params']['total_duration'][()]).ravel()[0])

start = int(np.where(np.isclose(h5_time, 0.0))[0][0])
end_hits = np.where(h5_time >= h5_total_duration)[0]
end = int(end_hits[0]) if len(end_hits) > 0 else start + 16
before = int(np.floor(0.2 / dt_si))
pre_start = start - before

print(f'h5 time first 5: {h5_time[:5].tolist()}, zero at {start}')
print(f'h5 total_duration: {h5_total_duration:.4f} s  → end frame {end}')
print(f'pre window frames:  [{pre_start}, {start})  (200 ms)')
print(f'post window frames: [{start}, {end})')

if pre_start < 0 or end > Fstim.shape[0]:
    raise ValueError(
        f'window indices out of range for Fstim with {Fstim.shape[0]} frames'
    )

# -----------------------------
# Per-cell baseline averaged across all pre-stim windows of all trials
# (asymmetric dF/F denominator — same convention as the artifact-free amp).
# Defined here so the t-test loop can express amp in ΔF/F units.
# -----------------------------
F0_per_cell = np.nanmean(np.nanmean(Fstim[pre_start:start, :, :], axis=0), axis=-1)
F0_per_cell_safe = np.where(np.isfinite(F0_per_cell) & (F0_per_cell > 0),
                            F0_per_cell, np.nan)


# -----------------------------
# Paired t-test: post vs pre, across trials, per (cell, group)
# Stores amp in ΔF/F units (raw amp / F0_per_cell), so it matches the
# trace plots that divide by the same F0.
# -----------------------------
seq_arr = np.asarray(seq).ravel() - 1
n_frames, n_cells, n_trials = Fstim.shape
n_groups = stimDist.shape[1]

pv  = np.full((n_cells, n_groups), np.nan)
amp = np.full((n_cells, n_groups), np.nan)

print(f'seq_arr: shape={seq_arr.shape}, unique={np.unique(seq_arr)[:20]}, '
      f'min={seq_arr.min()}, max={seq_arr.max()}')

trials_per_group = []
for gi in range(n_groups):
    trial_idx = np.where(seq_arr == gi)[0]
    trials_per_group.append(len(trial_idx))
    if len(trial_idx) < 3:
        continue
    pre  = np.nanmean(Fstim[pre_start:start, :, trial_idx], axis=0)   # (n_cells, n_trials_gi)
    post = np.nanmean(Fstim[start:end,        :, trial_idx], axis=0)
    # paired t-test using nan_policy='omit' so NaN trials are dropped per cell
    t_stat, p_value = ttest_rel(post, pre, axis=1, nan_policy='omit')
    pv[:, gi]  = p_value
    amp[:, gi] = np.nanmean(post - pre, axis=1) / F0_per_cell_safe

print(f'trials per group: min={min(trials_per_group)}, max={max(trials_per_group)}, '
      f'median={int(np.median(trials_per_group))}')
print(f'pv finite: {int(np.sum(np.isfinite(pv)))} / {pv.size}')
print(f'amp finite: {int(np.sum(np.isfinite(amp)))} / {amp.size}')


# -----------------------------
# Identify significant non-targets
# -----------------------------
NONTARG_DIST = 30
ALPHA        = 0.05
CELLS_ONLY   = True

cell_mask = (iscell[:, 0] == 1) if CELLS_ONLY else np.ones(n_cells, dtype=bool)

nontarg = stimDist > NONTARG_DIST
sig     = pv < ALPHA
sig_exc = sig & (amp > 0) & nontarg & cell_mask[:, None]
sig_inh = sig & (amp < 0) & nontarg & cell_mask[:, None]

print(f'Non-target significant excited:   {int(sig_exc.sum())}')
print(f'Non-target significant inhibited: {int(sig_inh.sum())}')


# -----------------------------
# Plot example single-(cell, group) traces (sorted by p-value)
# -----------------------------
t_axis = np.arange(n_frames) * dt_si - start * dt_si


T_MAX = 2.0
_crop = int(np.searchsorted(t_axis, T_MAX, side='right'))
t_plot = t_axis[:_crop]


def plot_examples(mask, n_show=12, color='#0096FF', suptitle='',
                  savename=None, sort='amp_desc'):
    cells, groups = np.where(mask)
    if len(cells) == 0:
        print(f'No examples for {suptitle}')
        return
    p_vals = pv[cells, groups]
    a_vals = amp[cells, groups]
    if sort == 'amp_desc':
        order = np.argsort(-a_vals)        # largest excitation first
    elif sort == 'amp_asc':
        order = np.argsort(a_vals)         # largest inhibition first
    else:
        order = np.argsort(p_vals)         # most significant first
    cells, groups = cells[order], groups[order]
    p_vals, a_vals = p_vals[order], a_vals[order]
    n_show = min(n_show, len(cells))

    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2), sharex=True,
    )
    axes = np.atleast_2d(axes).ravel()

    for k in range(n_show):
        ax = axes[k]
        ci, gi = cells[k], groups[k]
        trial_idx = np.where(seq_arr == gi)[0]
        traces = Fstim[:, ci, trial_idx].astype(float)
        bl = np.nanmean(traces[pre_start:start, :], axis=0, keepdims=True)
        # Asymmetric dF/F: per-trial baseline subtract, per-cell across-trial divide
        f0 = F0_per_cell[ci]
        if not np.isfinite(f0) or f0 <= 0:
            continue
        traces = ((traces - bl) / f0)[:_crop, :]
        m = np.nanmean(traces, axis=1)
        s = np.nanstd(traces, axis=1, ddof=1) / np.sqrt(traces.shape[1])
        ax.plot(t_plot, m, color=color, linewidth=1.2)
        ax.fill_between(t_plot, m - s, m + s, color=color, alpha=0.25,
                        edgecolor='none')
        ax.axvspan(0, (end - start) * dt_si, color='#FFBF00', alpha=0.2)
        ax.axhline(0, color='gray', linewidth=0.4)
        ax.set_xlim(t_plot[0], t_plot[-1])
        d = stimDist[ci, gi]
        ax.set_title(
            f'cell {ci}, grp {gi}\n'
            f'd={d:.0f} µm, amp={a_vals[k]:.2f}, p={p_vals[k]:.1e}',
            fontsize=8,
        )
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for k in range(n_show, len(axes)):
        fig.delaxes(axes[k])

    fig.supxlabel('Time from stim onset (s)', fontsize=10)
    fig.supylabel('ΔF/F', fontsize=10)
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()
    if savename:
        plt.savefig(os.path.join(SAVE_DIR, savename), dpi=150, bbox_inches='tight')
    plt.show()


plot_examples(
    sig_exc, n_show=12, color='#d62728', sort='p',
    suptitle=f'Top non-target excited (p<{ALPHA}, d>{NONTARG_DIST} µm), sorted by p',
    savename='photostim_significance_examples_excited.png',
)

plot_examples(
    sig_inh, n_show=12, color='#1f77b4', sort='p',
    suptitle=f'Top non-target inhibited (p<{ALPHA}, d>{NONTARG_DIST} µm), sorted by p',
    savename='photostim_significance_examples_inhibited.png',
)


#%%
# -----------------------------
# Quick summary: # significant per distance bin
# -----------------------------
edges = np.array([30, 50, 75, 100, 150, 200, 300, 500])
centers = 0.5 * (edges[:-1] + edges[1:])
frac_e = np.zeros(len(centers))
frac_i = np.zeros(len(centers))
for k in range(len(centers)):
    in_bin = (stimDist >= edges[k]) & (stimDist < edges[k + 1]) & cell_mask[:, None]
    n_total = int(in_bin.sum())
    if n_total == 0:
        continue
    frac_e[k] = (in_bin & sig & (amp > 0)).sum() / n_total
    frac_i[k] = (in_bin & sig & (amp < 0)).sum() / n_total

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(centers, frac_e, width=np.diff(edges), color='#d62728', alpha=0.7,
       edgecolor='white', label='Excited (p<0.05, amp>0)')
ax.bar(centers, -frac_i, width=np.diff(edges), color='#1f77b4', alpha=0.7,
       edgecolor='white', label='Inhibited (p<0.05, amp<0)')
ax.axhline(0, color='black', linewidth=0.6)
ax.axvline(NONTARG_DIST, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Distance from stim target (µm)', fontsize=11)
ax.set_ylabel('Fraction significant', fontsize=11)
ax.set_title(f'BCI116 {ps_epoch}: significance vs distance', fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_significance_fraction_vs_dist.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# I→I connectivity matrix
# Rows = receivers (target cells with target_amp > 0.05)
# Cols = targets   (groups whose target cells passed the same threshold)
# Cell M[i, j] = amp of target_i when group_j is stimulated.
# Direct activation entries (target_i within 30 µm of group_j's stim) are
# masked (NaN), as is the diagonal — these would just measure direct
# stimulation, not network connectivity.
# -----------------------------
TARGET_AMP_THRESH = 0.05
DIRECT_DIST       = 30   # µm; mask out responses within this radius of stim

target_cell_idx = np.argmin(stimDist, axis=0)                          # (n_groups,)
target_amp      = amp[target_cell_idx, np.arange(n_groups)]            # (n_groups,)

resp_mask = np.isfinite(target_amp) & (target_amp > TARGET_AMP_THRESH)
if CELLS_ONLY:
    resp_mask &= cell_mask[target_cell_idx]
resp_groups  = np.where(resp_mask)[0]
resp_targets = target_cell_idx[resp_groups]
n_resp = len(resp_groups)
print(f'I→I matrix: {n_resp} responsive targets '
      f'(target amp > {TARGET_AMP_THRESH})')

M = np.full((n_resp, n_resp), np.nan)
for i, ti in enumerate(resp_targets):
    for j, gj in enumerate(resp_groups):
        if i == j:
            continue
        if stimDist[ti, gj] < DIRECT_DIST:
            continue
        M[i, j] = amp[ti, gj]

# Sort rows and columns by direct-response amplitude (largest at top/left)
order          = np.argsort(-target_amp[resp_groups])
M_sorted       = M[order][:, order]
direct_sorted  = target_amp[resp_groups][order]

vmax = np.nanpercentile(np.abs(M_sorted), 95)
if not np.isfinite(vmax) or vmax == 0:
    vmax = 0.05

fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(
    1, 4, width_ratios=[0.06, 0.4, 5.0, 0.06], wspace=0.45,
)
cax_d = fig.add_subplot(gs[0, 0])
ax_d  = fig.add_subplot(gs[0, 1])
ax_m  = fig.add_subplot(gs[0, 2], sharey=ax_d)
cax_m = fig.add_subplot(gs[0, 3])

# Direct response strip on the left
direct_vmax = np.nanpercentile(direct_sorted, 99)
im_d = ax_d.imshow(
    direct_sorted[:, None], aspect='auto', cmap='Reds',
    vmin=0, vmax=direct_vmax, interpolation='nearest',
    extent=[0, 1, n_resp, 0],
)
ax_d.set_xticks([0.5])
ax_d.set_xticklabels(['target'], fontsize=8)
ax_d.set_ylabel('Non-target (sorted by direct response)', fontsize=10)
ax_d.set_title('Direct', fontsize=9)
cbar_d = fig.colorbar(im_d, cax=cax_d)
cbar_d.set_label('Target AMP', fontsize=8)
cbar_d.ax.tick_params(labelsize=7)
cax_d.yaxis.set_label_position('left')
cax_d.yaxis.tick_left()

# Connectivity matrix
im = ax_m.imshow(
    M_sorted, aspect='auto', cmap='RdBu_r',
    vmin=-vmax, vmax=vmax, interpolation='nearest',
    extent=[0, n_resp, n_resp, 0],
)
ax_m.set_xlabel('Stimulated target (target)', fontsize=10)
ax_m.set_title(
    f'BCI116 {ps_epoch}: I→I connectivity, n={n_resp} targets '
    f'(rows = non-target response, cols = stimulated target; '
    f'diag and direct overlaps masked)', fontsize=9,
)
ax_m.tick_params(axis='y', labelleft=False)
cbar = fig.colorbar(im, cax=cax_m)
cbar.set_label('AMP (ΔF/F)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_connectivity_matrix.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Same matrix, sorted by net output (mean column value)
# Each cell's "outgoing strength" = average effect it has on the rest of the
# population when stimulated. Most-suppressive targets go to the left;
# most-facilitating go to the right. Same order applied to rows so the
# diagonal still corresponds to (cell i, group of cell i).
# -----------------------------
net_output = np.nanmean(M, axis=0)        # column mean: how each target affects others
net_input  = np.nanmean(M, axis=1)        # row mean:    how each receiver is affected
sort_order = np.argsort(net_input)        # ascending: most-inhibited receivers first

M_sorted2      = M[sort_order][:, sort_order]
direct_sorted2 = target_amp[resp_groups][sort_order]
net_out_s      = net_output[sort_order]
net_in_s       = net_input[sort_order]

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(
    2, 6,
    width_ratios=[0.06, 0.4, 5.0, 0.6, 0.06, 0.05],
    height_ratios=[0.5, 5.0],
    wspace=0.32, hspace=0.08,
)
ax_top = fig.add_subplot(gs[0, 2])              # net output bar (top)
cax_d  = fig.add_subplot(gs[1, 0])              # cbar for direct
ax_d   = fig.add_subplot(gs[1, 1])              # direct strip
ax_m   = fig.add_subplot(gs[1, 2], sharey=ax_d, sharex=ax_top)   # main matrix
ax_in  = fig.add_subplot(gs[1, 3], sharey=ax_d) # net input bar (right)
cax_m  = fig.add_subplot(gs[1, 4])              # cbar for matrix

# Top panel: net OUTPUT per target
ax_top.bar(np.arange(n_resp) + 0.5, net_out_s, width=1.0,
           color=['#1f77b4' if v < 0 else '#d62728' for v in net_out_s],
           edgecolor='none')
ax_top.axhline(0, color='gray', linewidth=0.5)
ax_top.set_ylabel('Net out\n(mean ΔF/F)', fontsize=8)
ax_top.tick_params(axis='both', labelsize=7)
ax_top.set_xticks([])
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)

# Direct response strip
direct_vmax = np.nanpercentile(direct_sorted2, 99)
im_d = ax_d.imshow(
    direct_sorted2[:, None], aspect='auto', cmap='Reds',
    vmin=0, vmax=direct_vmax, interpolation='nearest',
    extent=[0, 1, n_resp, 0],
)
ax_d.set_xticks([0.5])
ax_d.set_xticklabels(['target'], fontsize=8)
ax_d.set_ylabel('Non-target (sorted by net input)', fontsize=10)
ax_d.set_title('Direct', fontsize=9)
cbar_d = fig.colorbar(im_d, cax=cax_d)
cbar_d.set_label('Target AMP', fontsize=8)
cbar_d.ax.tick_params(labelsize=7)
cax_d.yaxis.set_label_position('left')
cax_d.yaxis.tick_left()

# Connectivity matrix (sorted)
im = ax_m.imshow(
    M_sorted2, aspect='auto', cmap='RdBu_r',
    vmin=-vmax, vmax=vmax, interpolation='nearest',
    extent=[0, n_resp, n_resp, 0],
)
ax_m.set_xlim(0, n_resp)
ax_top.set_xlim(0, n_resp)
ax_m.set_xlabel('Stimulated target (target, sorted by net input)', fontsize=10)
ax_m.set_title(
    f'BCI116 {ps_epoch}: I→I connectivity '
    f'(n={n_resp} targets, sort = net input)', fontsize=10,
)
ax_m.tick_params(axis='y', labelleft=False)
cbar = fig.colorbar(im, cax=cax_m)
cbar.set_label('AMP (ΔF/F)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Right panel: net INPUT per receiver (horizontal bar)
y_centers = np.arange(n_resp) + 0.5
ax_in.barh(y_centers, net_in_s, height=1.0,
           color=['#1f77b4' if v < 0 else '#d62728' for v in net_in_s],
           edgecolor='none')
ax_in.axvline(0, color='gray', linewidth=0.5)
ax_in.invert_yaxis()
ax_in.set_xlabel('Net in\n(mean ΔF/F)', fontsize=8)
ax_in.tick_params(axis='both', labelsize=7)
ax_in.tick_params(axis='y', labelleft=False)
ax_in.spines['top'].set_visible(False)
ax_in.spines['right'].set_visible(False)

plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_connectivity_matrix_clustered.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Overlay: all per-target dF/F traces for the non-target with the largest |net input|
# Each line = trial-averaged response of this one cell when one of the
# responsive targets is being stimulated. Colored by amp (this cell's
# response to that target) so the strongest inhibitory / excitatory
# inputs visually pop out.
# -----------------------------
worst_idx = int(np.nanargmax(net_input))           # most positively-driven receiver
worst_cell = int(resp_targets[worst_idx])
worst_input = float(net_input[worst_idx])
print(f'Most-excited non-target: cell {worst_cell}  (net input = {worst_input:+.3f})')

f0 = F0_per_cell[worst_cell]
amps_for_cell = M[worst_idx, :]                    # response to each target (NaN where masked)

# Color each trace by its amp value (matrix-style RdBu_r)
amp_vmax = max(np.nanpercentile(np.abs(amps_for_cell), 95), 1e-6)
cmap = plt.get_cmap('RdBu_r')

fig, ax = plt.subplots(figsize=(7, 4.5))

for j, gj in enumerate(resp_groups):
    if j == worst_idx:
        continue                                   # skip self
    a = amps_for_cell[j]
    if not np.isfinite(a):
        continue                                   # direct overlap (masked)
    trial_idx = np.where(seq_arr == gj)[0]
    if len(trial_idx) == 0:
        continue
    traces = Fstim[:, worst_cell, trial_idx].astype(float)
    bl = np.nanmean(traces[pre_start:start, :], axis=0, keepdims=True)
    dff = ((traces - bl) / f0)[:_crop, :]
    m = np.nanmean(dff, axis=1)
    color = cmap(0.5 + 0.5 * np.clip(a / amp_vmax, -1, 1))
    ax.plot(t_plot, m, color=color, linewidth=1.0, alpha=0.85)

ax.axvspan(0, (end - start) * dt_si, color='#FFBF00', alpha=0.2)
ax.axhline(0, color='gray', linewidth=0.4)
ax.set_xlabel('Time from stim onset (s)', fontsize=11)
ax.set_ylabel('ΔF/F', fontsize=11)
ax.set_title(
    f'Cell {worst_cell}: response to each of {n_resp - 1} other targets\n'
    f'(largest positive net input in the matrix; net input = {worst_input:+.3f})',
    fontsize=10,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar matching the line colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-amp_vmax, vmax=amp_vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('AMP from this target (ΔF/F)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_traces_largest_input.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Single largest and smallest I→I connections
# Pick the (non-target, target) pair with the most positive amp and the
# most negative amp in M. Plot per-trial traces (thin) with mean ± SEM
# (thick) for each.
# -----------------------------
# Build a pv-matrix in the same shape as M, with the same direct-overlap
# and diagonal masking, so we can rank connections by significance.
P = np.full_like(M, np.nan)
for i, ti in enumerate(resp_targets):
    for j, gj in enumerate(resp_groups):
        if i == j or stimDist[ti, gj] < DIRECT_DIST:
            continue
        P[i, j] = pv[ti, gj]

# Most significant excitation: smallest p where amp > 0
# Most significant inhibition: smallest p where amp < 0
P_exc = np.where(M > 0, P, np.nan)
P_inh = np.where(M < 0, P, np.nan)

i_max, j_max = np.unravel_index(np.nanargmin(P_exc), P_exc.shape)
i_min, j_min = np.unravel_index(np.nanargmin(P_inh), P_inh.shape)

print(f'Most significant excitation: non-target idx {i_max}, target idx {j_max}, '
      f'amp = {M[i_max, j_max]:+.3f}, p = {P[i_max, j_max]:.2e}')
print(f'Most significant inhibition: non-target idx {i_min}, target idx {j_min}, '
      f'amp = {M[i_min, j_min]:+.3f}, p = {P[i_min, j_min]:.2e}')


def _trial_traces(non_target_idx, target_idx):
    """Return (mean, sem, all_trials) trial-traces in ΔF/F up to T_MAX."""
    cell_id  = int(resp_targets[non_target_idx])
    group_id = int(resp_groups[target_idx])
    f0 = F0_per_cell[cell_id]
    trial_idx = np.where(seq_arr == group_id)[0]
    raw = Fstim[:, cell_id, trial_idx].astype(float)
    bl = np.nanmean(raw[pre_start:start, :], axis=0, keepdims=True)
    dff = ((raw - bl) / f0)[:_crop, :]
    m = np.nanmean(dff, axis=1)
    s = np.nanstd(dff, axis=1, ddof=1) / np.sqrt(dff.shape[1])
    d = stimDist[cell_id, group_id]
    return m, s, dff, cell_id, group_id, d


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, (ii, jj, color, title_prefix) in zip(
    axes,
    [(i_max, j_max, '#d62728', 'Most significant excitation'),
     (i_min, j_min, '#1f77b4', 'Most significant inhibition')],
):
    m, s, dff, cell_id, group_id, d = _trial_traces(ii, jj)
    # Per-trial thin grey lines
    for k in range(dff.shape[1]):
        ax.plot(t_plot, dff[:, k], color='gray', linewidth=0.5, alpha=0.4)
    # Mean ± SEM
    ax.plot(t_plot, m, color=color, linewidth=2.0)
    ax.fill_between(t_plot, m - s, m + s, color=color, alpha=0.3, edgecolor='none')
    ax.axvspan(0, (end - start) * dt_si, color='#FFBF00', alpha=0.2)
    ax.axhline(0, color='gray', linewidth=0.4)
    ax.set_xlabel('Time from stim onset (s)', fontsize=11)
    ax.set_title(
        f'{title_prefix}\n'
        f'non-target cell {cell_id}, target {group_id}\n'
        f'amp = {M[ii, jj]:+.3f}, p = {P[ii, jj]:.1e}, d = {d:.0f} µm, n = {dff.shape[1]} trials',
        fontsize=10,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('ΔF/F', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_extreme_connections.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Null check for the I→I matrix
# 1) Histogram real M vs shuffled-seq M (does the real distribution have
#    fatter tails than chance?)
# 2) Fraction significant (p < 0.05) in real vs shuffled, within the same
#    I→I masked subset (~5 % under null).
# -----------------------------
rng = np.random.default_rng(42)
seq_shuf = rng.permutation(seq_arr)

amp_shuf = np.full((n_cells, n_groups), np.nan)
pv_shuf  = np.full((n_cells, n_groups), np.nan)

for gi in range(n_groups):
    trial_idx_s = np.where(seq_shuf == gi)[0]
    if len(trial_idx_s) < 3:
        continue
    pre_s  = np.nanmean(Fstim[pre_start:start, :, trial_idx_s], axis=0)
    post_s = np.nanmean(Fstim[start:end,        :, trial_idx_s], axis=0)
    _, p_s = ttest_rel(post_s, pre_s, axis=1, nan_policy='omit')
    pv_shuf[:, gi]  = p_s
    amp_shuf[:, gi] = np.nanmean(post_s - pre_s, axis=1) / F0_per_cell_safe

# Apply the same I→I masking as M
M_shuf = np.full((n_resp, n_resp), np.nan)
P_shuf = np.full((n_resp, n_resp), np.nan)
for i, ti in enumerate(resp_targets):
    for j, gj in enumerate(resp_groups):
        if i == j or stimDist[ti, gj] < DIRECT_DIST:
            continue
        M_shuf[i, j] = amp_shuf[ti, gj]
        P_shuf[i, j] = pv_shuf[ti, gj]

real_vals = M[np.isfinite(M)]
shuf_vals = M_shuf[np.isfinite(M_shuf)]

real_sig = np.sum(P[np.isfinite(P)] < 0.05) / max(np.isfinite(P).sum(), 1)
shuf_sig = np.sum(P_shuf[np.isfinite(P_shuf)] < 0.05) / max(np.isfinite(P_shuf).sum(), 1)
print(f'Real fraction significant in I→I matrix: {real_sig*100:.1f}%')
print(f'Shuf fraction significant in I→I matrix: {shuf_sig*100:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel 1: histogram overlay
ax = axes[0]
v_lim = max(np.nanpercentile(np.abs(real_vals), 99),
            np.nanpercentile(np.abs(shuf_vals), 99))
bins = np.linspace(-v_lim, v_lim, 41)
ax.hist(shuf_vals, bins=bins, alpha=0.5, color='gray',
        density=True, label=f'Shuffled (n={len(shuf_vals)})')
ax.hist(real_vals, bins=bins, alpha=0.5, color='#0096FF',
        density=True, label=f'Real (n={len(real_vals)})')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('AMP (ΔF/F)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('I→I amp distribution: real vs shuffled-seq null', fontsize=10)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: fraction significant bar
ax = axes[1]
ax.bar(['Real', 'Shuffled'], [real_sig, shuf_sig],
       color=['#0096FF', 'gray'], width=0.5)
ax.axhline(0.05, color='red', linestyle='--', linewidth=1,
           label='Chance (α=0.05)')
ax.set_ylabel('Fraction p < 0.05', fontsize=11)
ax.set_title('Fraction significant in the I→I matrix', fontsize=10)
ax.set_ylim(0, max(real_sig, shuf_sig, 0.05) * 1.4)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_null_check.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Extended connectivity matrix: target + non-target receivers
# Rows = responsive targets (top block) + clean non-target cells (bottom block)
# Cols = same responsive targets (the only cells we can stim)
# Cell M_ext[i, j] = amp of receiver_i when group_j is stimulated, with
# direct activations (< DIRECT_DIST) masked.
# Histogram compares target-target weights vs target→non-target weights.
# -----------------------------
min_dist_per_cell = np.nanmin(stimDist, axis=1)
is_clean_nontarget = min_dist_per_cell > DIRECT_DIST
if CELLS_ONLY:
    is_clean_nontarget &= cell_mask
nontarg_idx = np.where(is_clean_nontarget)[0]
n_nontarg = len(nontarg_idx)
print(f'Extended I→I: {n_resp} target receivers + {n_nontarg} non-target receivers')

M_ext = np.full((n_resp + n_nontarg, n_resp), np.nan)

# Top block: target receivers (same masking as M)
for i, ti in enumerate(resp_targets):
    for j, gj in enumerate(resp_groups):
        if i == j or stimDist[ti, gj] < DIRECT_DIST:
            continue
        M_ext[i, j] = amp[ti, gj]

# Bottom block: clean non-target receivers
for ki, ci in enumerate(nontarg_idx):
    for j, gj in enumerate(resp_groups):
        if stimDist[ci, gj] < DIRECT_DIST:
            continue           # safety; non-targets are already >30µm from every stim
        M_ext[n_resp + ki, j] = amp[ci, gj]

# Sort columns by target direct response (descending = strongest stim on left)
col_order = np.argsort(-target_amp[resp_groups])
# Sort target rows by their net incoming I→I (so sub-blocks are visually grouped)
target_block_unsorted = M_ext[:n_resp, :]
target_row_order = np.argsort(np.nanmean(target_block_unsorted, axis=1))
# Sort non-target rows by net input from target population
nt_block_unsorted = M_ext[n_resp:, :]
nt_row_order = np.argsort(np.nanmean(nt_block_unsorted, axis=1))

target_block_sorted = target_block_unsorted[target_row_order][:, col_order]
nt_block_sorted     = nt_block_unsorted[nt_row_order][:, col_order]

# Visual gap between targeted-target rows and never-targeted non-target rows
GAP_ROWS = 3
gap_block = np.full((GAP_ROWS, n_resp), np.nan)
M_ext_sorted = np.vstack([target_block_sorted, gap_block, nt_block_sorted])
total_rows = n_resp + GAP_ROWS + n_nontarg

vmax_ext = np.nanpercentile(np.abs(M_ext), 95)
if not np.isfinite(vmax_ext) or vmax_ext == 0:
    vmax_ext = 0.05

# Pull weight distributions for the histogram panel
tt_weights = target_block_sorted[np.isfinite(target_block_sorted)]
tn_weights = nt_block_sorted[np.isfinite(nt_block_sorted)]
print(f'  target→target weights: n={len(tt_weights)}, mean={np.mean(tt_weights):+.4f}')
print(f'  target→non-target    : n={len(tn_weights)}, mean={np.mean(tn_weights):+.4f}')


fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(
    1, 4, width_ratios=[5.0, 0.06, 0.5, 3.0], wspace=0.4,
)
ax_m = fig.add_subplot(gs[0, 0])
cax  = fig.add_subplot(gs[0, 1])
ax_h = fig.add_subplot(gs[0, 3])

# Stacked heatmap; NaN gap between blocks renders as transparent rows
_cmap_gap = plt.cm.RdBu_r.copy()
_cmap_gap.set_bad(color='white')
im = ax_m.imshow(
    M_ext_sorted, aspect='auto', cmap=_cmap_gap,
    vmin=-vmax_ext, vmax=vmax_ext, interpolation='nearest',
    extent=[0, n_resp, total_rows, 0],
)
ax_m.set_xlabel('Stimulated target (sender, sorted by direct response ↓)',
                fontsize=10)
ax_m.set_ylabel('Receiver (target above gap, non-target below)', fontsize=10)
ax_m.set_title(
    f'BCI116 {ps_epoch}: extended I→I connectivity\n'
    f'{n_resp} targets + {n_nontarg} non-targets   (direct overlaps masked)',
    fontsize=11,
)
# Annotate the two row blocks
ax_m.text(-0.6, n_resp / 2,                            'targets',
          ha='right', va='center', rotation=90, fontsize=9, color='black')
ax_m.text(-0.6, n_resp + GAP_ROWS + n_nontarg / 2,     'non-targets',
          ha='right', va='center', rotation=90, fontsize=9, color='black')

cbar = fig.colorbar(im, cax=cax)
cbar.set_label('AMP (ΔF/F)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Histogram comparison
hi = max(np.nanpercentile(np.abs(tt_weights), 99) if tt_weights.size else 0.05,
         np.nanpercentile(np.abs(tn_weights), 99) if tn_weights.size else 0.05)
bins = np.linspace(-hi, hi, 41)
ax_h.hist(tn_weights, bins=bins, color='gray', alpha=0.5, density=True,
          label=f'target → non-target (n={len(tn_weights)})')
ax_h.hist(tt_weights, bins=bins, color='#0096FF', alpha=0.5, density=True,
          label=f'target → target (n={len(tt_weights)})')
ax_h.axvline(0, color='black', linewidth=0.5)
ax_h.set_xlabel('AMP (ΔF/F)', fontsize=10)
ax_h.set_ylabel('Density', fontsize=10)
ax_h.set_title('Weight distributions', fontsize=11)
ax_h.legend(fontsize=8, frameon=False, loc='upper left')
ax_h.spines['top'].set_visible(False)
ax_h.spines['right'].set_visible(False)

plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_connectivity_matrix_extended.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Multi-session: aggregate target→target vs target→non-target weights
# across every session that passes the QC CSV filter (Type=Inhibitory,
# Pre OR Post = Good/Ok, no non-zero offset notes).
# -----------------------------
import csv as _csv
import re  as _re
from scipy.stats import wilcoxon, mannwhitneyu

DATA_ROOT = r'//allen/aind/scratch/BCI/2p-raw'
QC_CSV    = r'C:\Users\kayvon.daie\Documents\claude_code\BCI_030926\BCI\Inhibitory & Pan-neuronal BCI summary - Sheet1.csv'
_OK_QUALITY    = {'good', 'ok'}
_BAD_OFFSET_RE = _re.compile(r'offset\s*=\s*[1-9]', _re.IGNORECASE)

# Standardize the non-target pool to have similar baseline brightness as the
# target pool (drops dim/garbage ROIs that inflate near-zero noise in tn).
# F0 of each non-target must fall between MATCH_F0_PCTILE_LO and ..._HI of
# the target pool's F0 distribution. Set both to None to disable matching.
MATCH_F0_PCTILE_LO = 10
MATCH_F0_PCTILE_HI = 90


def load_qc_sessions():
    """List of (mouse, session) tuples passing QC."""
    keep = set()
    cur_m = cur_t = None
    with open(QC_CSV, encoding='utf-8') as f:
        rows = list(_csv.reader(f))
    for row in rows[2:]:
        if not row:
            continue
        subj  = row[0].strip() if len(row) > 0 else ''
        type_ = row[1].strip() if len(row) > 1 else ''
        if subj:  cur_m = subj
        if type_: cur_t = type_
        if (cur_t or '').lower() != 'inhibitory':
            continue
        date  = row[2].strip()        if len(row) > 2 else ''
        pre   = row[4].strip().lower() if len(row) > 4 else ''
        post  = row[5].strip().lower() if len(row) > 5 else ''
        notes = row[6]                 if len(row) > 6 else ''
        if (not date or cur_m is None or
            (pre not in _OK_QUALITY and post not in _OK_QUALITY) or
            _BAD_OFFSET_RE.search(notes or '')):
            continue
        try:
            m, d, y = date.split('/')
            sess = f'{int(m):02d}{int(d):02d}{int(y):02d}'
        except Exception:
            continue
        keep.add((cur_m, sess))
    return sorted(keep)


def session_weight_pools(mouse, session, ps_epoch=ps_epoch,
                          target_amp_thresh=TARGET_AMP_THRESH,
                          direct_dist=DIRECT_DIST):
    """For one session, build the extended connectivity matrix and return:
       (tt_weights, tn_weights, n_targets, n_nontargets).
    Re-runs the full per-(cell, group) t-test pipeline on this session's
    suite2p data so the threshold conventions match the headline analysis.
    """
    folder_i = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
    suite2p = ('suite2p_photostim_single' if ps_epoch == 'photostim'
               else 'suite2p_photostim_single2')
    ps_dir_i = os.path.join(folder_i, suite2p, 'plane0')

    iscell_i = np.load(os.path.join(ps_dir_i, 'iscell.npy'),   allow_pickle=True)
    stat_i   = np.load(os.path.join(ps_dir_i, 'stat.npy'),     allow_pickle=True)
    F_i      = np.load(os.path.join(ps_dir_i, 'F.npy'),        allow_pickle=True)
    ops_i    = np.load(os.path.join(ps_dir_i, 'ops.npy'),      allow_pickle=True).tolist()
    siHeader_i = np.load(os.path.join(ps_dir_i, 'siHeader.npy'), allow_pickle=True).tolist()

    out = ddc.stimDist_single_cell(ops_i, F_i, siHeader_i, stat_i, 0)
    Fstim_raw_i = out[9]
    seq_i       = out[1]
    stimDist_pix_i = out[3]

    umPerPix_i = (
        1000
        / float(siHeader_i['metadata']['hRoiManager']['scanZoomFactor'])
        / int(siHeader_i['metadata']['hRoiManager']['pixelsPerLine'])
    )
    dt_si_i = 1.0 / float(siHeader_i['metadata']['hRoiManager']['scanVolumeRate'])
    stimDist_i = stimDist_pix_i * umPerPix_i

    # h5 stim_params for the proper stim window
    all_h5 = sorted(glob.glob(os.path.join(folder_i, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5 if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        raise FileNotFoundError(f'no h5 for {mouse} {session} {ps_epoch}')
    with h5py.File(cand[0], 'r') as hf:
        h5_time_i = np.asarray(hf['stim_params']['time'][()]).ravel()
        h5_total_i = float(np.asarray(hf['stim_params']['total_duration'][()]).ravel()[0])
    start_i = int(np.where(np.isclose(h5_time_i, 0.0))[0][0])
    end_hits_i = np.where(h5_time_i >= h5_total_i)[0]
    end_i = int(end_hits_i[0]) if len(end_hits_i) > 0 else start_i + 16
    before_i = int(np.floor(0.2 / dt_si_i))
    pre_start_i = start_i - before_i

    Fstim_i = Fstim_raw_i
    if pre_start_i < 0 or end_i > Fstim_i.shape[0]:
        raise ValueError(f'window indices out of range for {mouse} {session}')

    # Per-cell baseline averaged across all pre-stim windows of all trials
    F0_per_cell_i = np.nanmean(np.nanmean(
        Fstim_i[pre_start_i:start_i, :, :], axis=0), axis=-1)
    F0_safe = np.where(np.isfinite(F0_per_cell_i) & (F0_per_cell_i > 0),
                       F0_per_cell_i, np.nan)

    seq_arr_i = np.asarray(seq_i).ravel() - 1
    n_cells_i = Fstim_i.shape[1]
    n_groups_i = stimDist_i.shape[1]
    amp_i = np.full((n_cells_i, n_groups_i), np.nan)
    for gi in range(n_groups_i):
        trial_idx = np.where(seq_arr_i == gi)[0]
        if len(trial_idx) < 3:
            continue
        pre_w  = np.nanmean(Fstim_i[pre_start_i:start_i, :, trial_idx], axis=0)
        post_w = np.nanmean(Fstim_i[start_i:end_i,        :, trial_idx], axis=0)
        amp_i[:, gi] = np.nanmean(post_w - pre_w, axis=1) / F0_safe

    target_cell_idx_i = np.argmin(stimDist_i, axis=0)
    target_amp_i = amp_i[target_cell_idx_i, np.arange(n_groups_i)]
    cell_mask_i = (iscell_i[:, 0] == 1) if CELLS_ONLY else np.ones(n_cells_i, dtype=bool)
    resp_mask_i = np.isfinite(target_amp_i) & (target_amp_i > target_amp_thresh)
    if CELLS_ONLY:
        resp_mask_i &= cell_mask_i[target_cell_idx_i]
    resp_groups_i  = np.where(resp_mask_i)[0]
    resp_targets_i = target_cell_idx_i[resp_groups_i]
    n_resp_i = len(resp_groups_i)

    nontarg_pre_mask_i = np.nanmin(stimDist_i, axis=1) > direct_dist
    if CELLS_ONLY:
        nontarg_pre_mask_i &= cell_mask_i

    # F0-match the non-target pool to the target pool so dim/garbage ROIs
    # don't inflate the near-zero noise distribution.
    is_clean_nontarget_i = nontarg_pre_mask_i.copy()
    F0_band = (np.nan, np.nan)
    if (MATCH_F0_PCTILE_LO is not None and MATCH_F0_PCTILE_HI is not None
        and resp_targets_i.size > 0):
        target_F0_arr = F0_per_cell_i[resp_targets_i]
        target_F0_arr = target_F0_arr[np.isfinite(target_F0_arr)]
        if target_F0_arr.size > 0:
            f0_lo = float(np.percentile(target_F0_arr, MATCH_F0_PCTILE_LO))
            f0_hi = float(np.percentile(target_F0_arr, MATCH_F0_PCTILE_HI))
            F0_band = (f0_lo, f0_hi)
            f0_in_band = (np.isfinite(F0_per_cell_i) &
                          (F0_per_cell_i >= f0_lo) &
                          (F0_per_cell_i <= f0_hi))
            is_clean_nontarget_i = is_clean_nontarget_i & f0_in_band
    nontarg_idx_i = np.where(is_clean_nontarget_i)[0]
    n_nontarg_i = len(nontarg_idx_i)

    # F0 category arrays for the matching-diagnostic plot
    F0_target_arr      = F0_per_cell_i[resp_targets_i]
    F0_nontarg_kept    = F0_per_cell_i[is_clean_nontarget_i]
    F0_nontarg_disc    = F0_per_cell_i[nontarg_pre_mask_i & ~is_clean_nontarget_i]

    # target→target weights (non-diagonal, non-direct)
    tt = []
    for i_loc, ti in enumerate(resp_targets_i):
        for j_loc, gj in enumerate(resp_groups_i):
            if i_loc == j_loc or stimDist_i[ti, gj] < direct_dist:
                continue
            v = amp_i[ti, gj]
            if np.isfinite(v):
                tt.append(v)
    # target→non-target weights
    tn = []
    for ci in nontarg_idx_i:
        for gj in resp_groups_i:
            v = amp_i[ci, gj]
            if np.isfinite(v):
                tn.append(v)
    return dict(
        tt=np.array(tt), tn=np.array(tn),
        n_targets=n_resp_i, n_nontargets=n_nontarg_i,
        F0_target=F0_target_arr,
        F0_nontarg_kept=F0_nontarg_kept,
        F0_nontarg_discarded=F0_nontarg_disc,
        F0_band=F0_band,
    )


qc_sessions = load_qc_sessions()
print(f'Aggregating across {len(qc_sessions)} QC sessions ({ps_epoch})...')

per_session = []   # list of dicts
all_tt = []        # pooled target→target weights
all_tn = []        # pooled target→non-target weights

for mouse_i, session_i in qc_sessions:
    try:
        out = session_weight_pools(mouse_i, session_i)
    except Exception as e:
        print(f'  Skipping {mouse_i} {session_i}: {type(e).__name__}: {e}')
        continue
    tt_i, tn_i = out['tt'], out['tn']
    n_t_i, n_n_i = out['n_targets'], out['n_nontargets']
    if tt_i.size == 0 or tn_i.size == 0:
        print(f'  Skipping {mouse_i} {session_i}: empty weight pool')
        continue
    per_session.append(dict(
        mouse=mouse_i, session=session_i,
        mean_tt=float(np.mean(tt_i)),     mean_tn=float(np.mean(tn_i)),
        absmean_tt=float(np.mean(np.abs(tt_i))),
        absmean_tn=float(np.mean(np.abs(tn_i))),
        std_tt=float(np.std(tt_i, ddof=1)) if tt_i.size > 1 else np.nan,
        std_tn=float(np.std(tn_i, ddof=1)) if tn_i.size > 1 else np.nan,
        n_tt=int(tt_i.size), n_tn=int(tn_i.size),
        n_targets=n_t_i, n_nontargets=n_n_i,
        F0_target=out['F0_target'],
        F0_nontarg_kept=out['F0_nontarg_kept'],
        F0_nontarg_discarded=out['F0_nontarg_discarded'],
        F0_band=out['F0_band'],
    ))
    all_tt.extend(tt_i.tolist())
    all_tn.extend(tn_i.tolist())
    print(f'  {mouse_i} {session_i}: targets={n_t_i}, non-targets={n_n_i} '
          f'(F0-band {MATCH_F0_PCTILE_LO}–{MATCH_F0_PCTILE_HI} pct of targets), '
          f'mean tt={np.mean(tt_i):+.4f}, mean tn={np.mean(tn_i):+.4f}')

all_tt = np.array(all_tt)
all_tn = np.array(all_tn)
n_sessions = len(per_session)

# Per-session arrays for paired tests
mean_tt_per    = np.array([s['mean_tt']    for s in per_session])
mean_tn_per    = np.array([s['mean_tn']    for s in per_session])
absmean_tt_per = np.array([s['absmean_tt'] for s in per_session])
absmean_tn_per = np.array([s['absmean_tn'] for s in per_session])
std_tt_per     = np.array([s['std_tt']     for s in per_session])
std_tn_per     = np.array([s['std_tn']     for s in per_session])


def _paired_wilcoxon(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return (np.nan, np.nan)
    return wilcoxon(a[valid], b[valid])

w_mean_stat, w_mean_p     = _paired_wilcoxon(mean_tt_per,    mean_tn_per)
w_abs_stat,  w_abs_p      = _paired_wilcoxon(absmean_tt_per, absmean_tn_per)
w_std_stat,  w_std_p      = _paired_wilcoxon(std_tt_per,     std_tn_per)

# Mann–Whitney on the pooled weights (treats each weight as independent)
if all_tt.size and all_tn.size:
    mw_stat, mw_p = mannwhitneyu(all_tt, all_tn, alternative='two-sided')
    mw_abs_stat, mw_abs_p = mannwhitneyu(np.abs(all_tt), np.abs(all_tn),
                                         alternative='two-sided')
else:
    mw_stat = mw_p = mw_abs_stat = mw_abs_p = np.nan

print(f'\nPooled: tt n={all_tt.size} mean={np.mean(all_tt):+.4f} | '
      f'tn n={all_tn.size} mean={np.mean(all_tn):+.4f}')
print(f'Per-session paired Wilcoxon, mean   (n={n_sessions}): stat={w_mean_stat:.2f}, p={w_mean_p:.2e}')
print(f'Per-session paired Wilcoxon, |mean| (n={n_sessions}): stat={w_abs_stat:.2f},  p={w_abs_p:.2e}')
print(f'Per-session paired Wilcoxon, std    (n={n_sessions}): stat={w_std_stat:.2f},  p={w_std_p:.2e}')
print(f'Pooled MWU on raw weights : stat={mw_stat:.2f},   p={mw_p:.2e}')
print(f'Pooled MWU on |weights|   : stat={mw_abs_stat:.2f}, p={mw_abs_p:.2e}')


# -----------------------------
# Plot: pooled histogram + three per-session paired scatters
#   - mean    (central tendency, signed)
#   - |mean|  (magnitude regardless of sign — catches "bigger + AND bigger −")
#   - std     (spread — catches symmetric heavy tails)
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# (0, 0) Pooled histogram
ax = axes[0, 0]
hi = max(np.nanpercentile(np.abs(all_tt), 99) if all_tt.size else 0.05,
         np.nanpercentile(np.abs(all_tn), 99) if all_tn.size else 0.05)
bins = np.linspace(-hi, hi, 41)
ax.hist(all_tn, bins=bins, color='gray',    alpha=0.5, density=True,
        label=f'target → non-target (n={all_tn.size})')
ax.hist(all_tt, bins=bins, color='#0096FF', alpha=0.5, density=True,
        label=f'target → target (n={all_tt.size})')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('AMP (ΔF/F)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(
    f'Pooled across {n_sessions} sessions\n'
    f'MWU on raw p={mw_p:.2e}, MWU on |amp| p={mw_abs_p:.2e}',
    fontsize=10,
)
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


def _paired_scatter(ax, a, b, ylabel, title):
    for ai, bi in zip(a, b):
        if not (np.isfinite(ai) and np.isfinite(bi)):
            continue
        ax.plot([0, 1], [ai, bi], 'o-', color='gray', alpha=0.5, markersize=5)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.any():
        a_v, b_v = a[valid], b[valid]
        ax.errorbar(
            [0, 1], [np.mean(a_v), np.mean(b_v)],
            yerr=[np.std(a_v, ddof=1) / np.sqrt(valid.sum()),
                  np.std(b_v, ddof=1) / np.sqrt(valid.sum())],
            fmt='s', color='#d62728', markersize=10, capsize=5, linewidth=2,
            zorder=5,
        )
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['target → target', 'target → non-target'], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


_paired_scatter(
    axes[0, 1], mean_tt_per, mean_tn_per,
    'Per-session mean AMP (ΔF/F)',
    f'Per-session mean (n={n_sessions})  Wilcoxon p={w_mean_p:.2e}',
)
_paired_scatter(
    axes[1, 0], absmean_tt_per, absmean_tn_per,
    'Per-session mean |AMP| (ΔF/F)',
    f'Per-session |mean| (n={n_sessions})  Wilcoxon p={w_abs_p:.2e}',
)
_paired_scatter(
    axes[1, 1], std_tt_per, std_tn_per,
    'Per-session std of AMP (ΔF/F)',
    f'Per-session std (n={n_sessions})  Wilcoxon p={w_std_p:.2e}',
)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_extended_multisession.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# F0-matching diagnostic (standalone — reads from per_session, no rerun)
# Pools each cell's F0 across all QC sessions into 3 categories:
#   target              (responsive I cells, the comparator)
#   non-target kept     (passed the F0-band match)
#   non-target discarded (clean non-target but F0 fell outside the band)
# Shows the raw distributions plus a per-session normalized version
# (each session's F0 axis divided by that session's median target F0,
# so different imaging gains don't pile up as separate modes).
# -----------------------------
def _safe_concat(arrs):
    arrs = [np.asarray(a)[np.isfinite(a)] for a in arrs if len(a) > 0]
    if not arrs:
        return np.array([])
    return np.concatenate(arrs)


F0_t_pool   = _safe_concat([s['F0_target']             for s in per_session])
F0_kept_pool= _safe_concat([s['F0_nontarg_kept']       for s in per_session])
F0_disc_pool= _safe_concat([s['F0_nontarg_discarded']  for s in per_session])

# Per-session normalization: divide each cell's F0 by that session's median
# target F0. Lets the pool reflect "relative brightness vs typical target."
def _normalized(field):
    out = []
    for s in per_session:
        target_med = np.nanmedian(s['F0_target'])
        if not np.isfinite(target_med) or target_med <= 0:
            continue
        x = np.asarray(s[field])
        x = x[np.isfinite(x)] / target_med
        out.append(x)
    return _safe_concat(out)


F0_t_norm    = _normalized('F0_target')
F0_kept_norm = _normalized('F0_nontarg_kept')
F0_disc_norm = _normalized('F0_nontarg_discarded')

print(f'F0 pools: target={F0_t_pool.size}, kept={F0_kept_pool.size}, '
      f'discarded={F0_disc_pool.size}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# -- Left: raw F0 (per-session imaging-gain differences will appear) --
ax = axes[0]
hi_raw = np.nanpercentile(np.concatenate([F0_t_pool, F0_kept_pool, F0_disc_pool]),
                          99.5)
bins_raw = np.linspace(0, hi_raw, 51)
ax.hist(F0_disc_pool, bins=bins_raw, color='lightgray', alpha=0.6,
        density=True, label=f'Non-target discarded (n={F0_disc_pool.size})')
ax.hist(F0_kept_pool, bins=bins_raw, color='#0096FF', alpha=0.5,
        density=True, label=f'Non-target kept (n={F0_kept_pool.size})')
ax.hist(F0_t_pool,    bins=bins_raw, color='#d62728', alpha=0.5,
        density=True, label=f'Target (n={F0_t_pool.size})')
ax.set_xlabel('Baseline F (raw)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('F0 distributions — pooled raw', fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -- Right: F0 normalized by per-session median target F0 --
ax = axes[1]
hi_norm = np.nanpercentile(np.concatenate([F0_t_norm, F0_kept_norm, F0_disc_norm]),
                           99.5)
bins_norm = np.linspace(0, hi_norm, 51)
ax.hist(F0_disc_norm, bins=bins_norm, color='lightgray', alpha=0.6,
        density=True, label=f'Non-target discarded (n={F0_disc_norm.size})')
ax.hist(F0_kept_norm, bins=bins_norm, color='#0096FF', alpha=0.5,
        density=True, label=f'Non-target kept (n={F0_kept_norm.size})')
ax.hist(F0_t_norm,    bins=bins_norm, color='#d62728', alpha=0.5,
        density=True, label=f'Target (n={F0_t_norm.size})')
ax.axvline(1.0, color='black', linewidth=0.8, linestyle='--',
           label='Median target F0')
ax.set_xlabel('F0  /  median(target F0) per session', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(
    f'F0 normalized per session\n'
    f'F0-match band = pct {MATCH_F0_PCTILE_LO}–{MATCH_F0_PCTILE_HI} of targets',
    fontsize=11,
)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(
    f'F0-matching diagnostic across {len(per_session)} QC sessions',
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_F0_matching_diagnostic.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# -----------------------------
# Multi-session: changes in connection strength
#   dW = AMP[photostim2] - AMP[photostim]
# Same target → target / target → non-target split as the AMP analysis above,
# but on the difference across the two photostim epochs.
#
# The per-epoch suite2p folders use different cell IDs, so we cannot simply
# subtract the amp arrays from session_weight_pools(). Instead we use the
# ddct loader, which maps both photostim epochs into a unified cell space,
# and compute_amp_from_photostim_artifact_free, which returns AMP[0] and
# AMP[1] in that shared space.
# -----------------------------
import data_dict_create_module_test as ddct
from BCI_data_helpers import compute_amp_from_photostim_artifact_free

DW_TARGET_DIST     = DIRECT_DIST          # < this µm: receiver IS a target of group g
DW_NONTARG_LO      = DIRECT_DIST          # > this µm from EVERY group: clean non-target
DW_NONTARG_HI      = 1000                 # upper bound (µm) on non-target distance
DW_TARGET_AMP_THR  = TARGET_AMP_THRESH    # group is "responsive" if target > thr in BOTH epochs


def _load_stim_params_h5(folder, ps_epoch):
    """ddct.load_hdf5 doesn't read sub-groups; pull stim_params via h5py."""
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))
    if ps_epoch == 'photostim':
        cand = [p for p in all_h5
                if 'photostim' in os.path.basename(p).lower()
                and 'photostim2' not in os.path.basename(p).lower()]
    else:
        cand = [p for p in all_h5
                if 'photostim2' in os.path.basename(p).lower()]
    if not cand:
        return None
    sp = {}
    with h5py.File(cand[0], 'r') as hf:
        if 'stim_params' in hf:
            for k in hf['stim_params'].keys():
                try:
                    sp[k] = hf['stim_params'][k][()]
                except Exception:
                    pass
    return sp


def session_dW_pools(mouse, session,
                      target_dist=DW_TARGET_DIST,
                      nontarg_lo=DW_NONTARG_LO,
                      nontarg_hi=DW_NONTARG_HI,
                      amp_thr=DW_TARGET_AMP_THR):
    """For one session, build dW pools for tt and tn pairs.
    Returns (tt_dw, tn_dw, n_targets, n_nontargets).
    """
    folder_i = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
    photostim_keys = ['stimDist', 'favg_raw']
    bci_keys = ['df_closedloop', 'F', 'mouse', 'session',
                'conditioned_neuron', 'dt_si']
    data = ddct.load_hdf5(folder_i, bci_keys, photostim_keys)
    # ddct.load_hdf5 can't read the stim_params sub-group; load it via h5py
    # so compute_amp_from_photostim_artifact_free finds the stim window.
    for ep in ('photostim', 'photostim2'):
        if ep in data:
            sp = _load_stim_params_h5(folder_i, ep)
            if sp is not None:
                data[ep]['stim_params'] = sp
    AMP, stimDist_i = compute_amp_from_photostim_artifact_free(
        mouse, data, folder_i)
    if len(AMP) < 2:
        raise ValueError('need both photostim and photostim2 to compute dW')

    n_cells_i, n_groups_i = stimDist_i.shape
    target_cell_idx_i = np.argmin(stimDist_i, axis=0)

    target_amp_0 = AMP[0][target_cell_idx_i, np.arange(n_groups_i)]
    target_amp_1 = AMP[1][target_cell_idx_i, np.arange(n_groups_i)]
    resp_mask_i = (np.isfinite(target_amp_0) & (target_amp_0 > amp_thr) &
                   np.isfinite(target_amp_1) & (target_amp_1 > amp_thr))
    resp_groups_i  = np.where(resp_mask_i)[0]
    resp_targets_i = target_cell_idx_i[resp_groups_i]
    n_resp_i = len(resp_groups_i)
    if n_resp_i == 0:
        return dict(tt=np.array([]), tn=np.array([]),
                    n_targets=0, n_nontargets=0)

    dW = AMP[1] - AMP[0]

    min_dist_per_cell = np.nanmin(stimDist_i, axis=1)
    is_clean_nontarget_i = ((min_dist_per_cell > nontarg_lo) &
                            (min_dist_per_cell < nontarg_hi))
    nontarg_idx_i = np.where(is_clean_nontarget_i)[0]

    # tt: target receiver ti when stimulating group gj, excluding diagonal
    # (group's own target) and any direct overlap (< target_dist).
    tt_dw = []
    for i_loc, ti in enumerate(resp_targets_i):
        for j_loc, gj in enumerate(resp_groups_i):
            if i_loc == j_loc or stimDist_i[ti, gj] < target_dist:
                continue
            v = dW[ti, gj]
            if np.isfinite(v):
                tt_dw.append(v)
    # tn: clean non-target receivers
    tn_dw = []
    for ci in nontarg_idx_i:
        for gj in resp_groups_i:
            v = dW[ci, gj]
            if np.isfinite(v):
                tn_dw.append(v)

    return dict(tt=np.array(tt_dw), tn=np.array(tn_dw),
                n_targets=n_resp_i, n_nontargets=len(nontarg_idx_i))


print(f'\nAggregating dW across {len(qc_sessions)} QC sessions...')

per_session_dw = []
all_tt_dw = []
all_tn_dw = []

for mouse_i, session_i in qc_sessions:
    try:
        out_dw = session_dW_pools(mouse_i, session_i)
    except Exception as e:
        print(f'  Skipping {mouse_i} {session_i}: {type(e).__name__}: {e}')
        continue
    tt_i, tn_i = out_dw['tt'], out_dw['tn']
    if tt_i.size == 0 or tn_i.size == 0:
        print(f'  Skipping {mouse_i} {session_i}: empty dW pool')
        continue
    per_session_dw.append(dict(
        mouse=mouse_i, session=session_i,
        mean_tt=float(np.mean(tt_i)),    mean_tn=float(np.mean(tn_i)),
        absmean_tt=float(np.mean(np.abs(tt_i))),
        absmean_tn=float(np.mean(np.abs(tn_i))),
        std_tt=float(np.std(tt_i, ddof=1)) if tt_i.size > 1 else np.nan,
        std_tn=float(np.std(tn_i, ddof=1)) if tn_i.size > 1 else np.nan,
        n_tt=int(tt_i.size), n_tn=int(tn_i.size),
        n_targets=out_dw['n_targets'], n_nontargets=out_dw['n_nontargets'],
    ))
    all_tt_dw.extend(tt_i.tolist())
    all_tn_dw.extend(tn_i.tolist())
    print(f'  {mouse_i} {session_i}: tt n={tt_i.size} mean={np.mean(tt_i):+.4f} | '
          f'tn n={tn_i.size} mean={np.mean(tn_i):+.4f}')

all_tt_dw = np.array(all_tt_dw)
all_tn_dw = np.array(all_tn_dw)
n_sessions_dw = len(per_session_dw)

mean_tt_dw_per    = np.array([s['mean_tt']    for s in per_session_dw])
mean_tn_dw_per    = np.array([s['mean_tn']    for s in per_session_dw])
absmean_tt_dw_per = np.array([s['absmean_tt'] for s in per_session_dw])
absmean_tn_dw_per = np.array([s['absmean_tn'] for s in per_session_dw])
std_tt_dw_per     = np.array([s['std_tt']     for s in per_session_dw])
std_tn_dw_per     = np.array([s['std_tn']     for s in per_session_dw])

w_mean_dw_stat, w_mean_dw_p = _paired_wilcoxon(mean_tt_dw_per,    mean_tn_dw_per)
w_abs_dw_stat,  w_abs_dw_p  = _paired_wilcoxon(absmean_tt_dw_per, absmean_tn_dw_per)
w_std_dw_stat,  w_std_dw_p  = _paired_wilcoxon(std_tt_dw_per,     std_tn_dw_per)

if all_tt_dw.size and all_tn_dw.size:
    mw_dw_stat,     mw_dw_p     = mannwhitneyu(all_tt_dw, all_tn_dw,
                                               alternative='two-sided')
    mw_abs_dw_stat, mw_abs_dw_p = mannwhitneyu(np.abs(all_tt_dw), np.abs(all_tn_dw),
                                               alternative='two-sided')
else:
    mw_dw_stat = mw_dw_p = mw_abs_dw_stat = mw_abs_dw_p = np.nan

print(f'\nPooled dW: tt n={all_tt_dw.size} mean={np.mean(all_tt_dw):+.4f} | '
      f'tn n={all_tn_dw.size} mean={np.mean(all_tn_dw):+.4f}')
print(f'Per-session paired Wilcoxon, mean   (n={n_sessions_dw}): stat={w_mean_dw_stat:.2f}, p={w_mean_dw_p:.2e}')
print(f'Per-session paired Wilcoxon, |mean| (n={n_sessions_dw}): stat={w_abs_dw_stat:.2f},  p={w_abs_dw_p:.2e}')
print(f'Per-session paired Wilcoxon, std    (n={n_sessions_dw}): stat={w_std_dw_stat:.2f},  p={w_std_dw_p:.2e}')
print(f'Pooled MWU on raw dW : stat={mw_dw_stat:.2f},   p={mw_dw_p:.2e}')
print(f'Pooled MWU on |dW|   : stat={mw_abs_dw_stat:.2f}, p={mw_abs_dw_p:.2e}')


fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
hi_dw = max(np.nanpercentile(np.abs(all_tt_dw), 99) if all_tt_dw.size else 0.05,
            np.nanpercentile(np.abs(all_tn_dw), 99) if all_tn_dw.size else 0.05)
bins_dw = np.linspace(-hi_dw, hi_dw, 41)
ax.hist(all_tn_dw, bins=bins_dw, color='gray',    alpha=0.5, density=True,
        label=f'target → non-target (n={all_tn_dw.size})')
ax.hist(all_tt_dw, bins=bins_dw, color='#0096FF', alpha=0.5, density=True,
        label=f'target → target (n={all_tt_dw.size})')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('dW = AMP[photostim2] - AMP[photostim] (ΔF/F)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(
    f'Δ connection strength, pooled across {n_sessions_dw} sessions\n'
    f'MWU on raw p={mw_dw_p:.2e}, MWU on |dW| p={mw_abs_dw_p:.2e}',
    fontsize=10,
)
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

_paired_scatter(
    axes[0, 1], mean_tt_dw_per, mean_tn_dw_per,
    'Per-session mean dW (ΔF/F)',
    f'Per-session mean (n={n_sessions_dw})  Wilcoxon p={w_mean_dw_p:.2e}',
)
_paired_scatter(
    axes[1, 0], absmean_tt_dw_per, absmean_tn_dw_per,
    'Per-session mean |dW| (ΔF/F)',
    f'Per-session |mean| (n={n_sessions_dw})  Wilcoxon p={w_abs_dw_p:.2e}',
)
_paired_scatter(
    axes[1, 1], std_tt_dw_per, std_tn_dw_per,
    'Per-session std of dW (ΔF/F)',
    f'Per-session std (n={n_sessions_dw})  Wilcoxon p={w_std_dw_p:.2e}',
)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'photostim_II_dW_extended_multisession.png'),
            dpi=150, bbox_inches='tight')
plt.show()
