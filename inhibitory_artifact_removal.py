# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:44:41 2026

@author: christina.wang
"""

import sys
import os
import glob
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter

sys.path.append(r'C:\Users\christina.wang\Downloads\BCI_code_local')
import data_dict_create_module_iscell as  ddc

# -----------------------------
# Fast single-session h5 loading
# -----------------------------

folder = r'//allen/aind/scratch/BCI/2p-raw/BCI116/012826/pophys/'
ps_epoch = 'photostim2'   # 'photostim' or 'photostim2'

SAVE_DIR = r'C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\claude_code\inhibitory_photostim_BCI_meeting_050126'
os.makedirs(SAVE_DIR, exist_ok=True)


def find_photostim_h5(folder, ps_epoch):
    all_h5 = sorted(glob.glob(os.path.join(folder, '*.h5')))

    if ps_epoch == 'photostim':
        candidates = [
            p for p in all_h5
            if 'photostim' in os.path.basename(p).lower()
            and 'photostim2' not in os.path.basename(p).lower()
        ]
    elif ps_epoch == 'photostim2':
        candidates = [
            p for p in all_h5
            if 'photostim2' in os.path.basename(p).lower()
        ]
    else:
        raise ValueError("ps_epoch must be 'photostim' or 'photostim2'")

    if len(candidates) == 0:
        raise FileNotFoundError(f'No h5 found for {ps_epoch} in {folder}')

    return candidates[0]


def load_h5_epoch(h5_path):
    d = {}

    with h5py.File(h5_path, 'r') as f:
        for key in ['stimDist', 'favg_raw', 'centroidX', 'centroidY']:
            if key in f:
                d[key] = f[key][:]

        if 'stim_params' in f:
            sp = {}
            for k in f['stim_params'].keys():
                try:
                    sp[k] = f['stim_params'][k][:]
                except Exception:
                    pass
            d['stim_params'] = sp

    return d


def load_si_and_ops(folder, ps_epoch):
    if ps_epoch == 'photostim':
        suite2p_folder = 'suite2p_photostim_single'
    else:
        suite2p_folder = 'suite2p_photostim_single2'

    si_path = os.path.join(folder, suite2p_folder, 'plane0', 'siHeader.npy')
    ops_path = os.path.join(folder, suite2p_folder, 'plane0', 'ops.npy')

    if not os.path.exists(si_path):
        si_path = os.path.join(folder, 'suite2p_photostim_single', 'plane0', 'siHeader.npy')

    si = np.load(si_path, allow_pickle=True).tolist()

    trial_dur = None
    if os.path.exists(ops_path):
        ops = np.load(ops_path, allow_pickle=True).tolist()
        values, counts = np.unique(ops['frames_per_file'], return_counts=True)
        trial_dur = int(values[np.argmax(counts)])

    return si, trial_dur


def stim_timing(ep_data, dt):
    sp = ep_data.get('stim_params', {})
    time_arr = sp.get('time', None)

    if time_arr is not None:
        time_arr = np.asarray(time_arr).ravel()
        hits = np.where(np.isclose(time_arr, 0.0))[0]
        start = int(hits[0]) if len(hits) > 0 else 10
    else:
        start = 10

    pre_end = start - 1

    total_duration = sp.get('total_duration', None)
    if time_arr is not None and total_duration is not None:
        total_duration = float(np.asarray(total_duration).ravel()[0])
        hits = np.where(time_arr >= total_duration)[0]
        end = int(hits[0]) if len(hits) > 0 else start + 16
    else:
        end = start + 16

    return start, pre_end, end


def estimate_session_artifact_value_from_suite2p(
    folder,
    ps_epoch,
    start,
    end,
    pre_frames=8,
    far_thresh_um=100,
    dim_pctile=10,
    use_median=False,
):
    """
    Estimate one scalar artifact value (raw F units) using the full suite2p ROI set:
      far ROI + dim ROI + iscell == 0.

    Light leakage is a constant additive offset across the frame, so the artifact
    is estimated and returned in raw F units. Subtraction should be applied to F
    before computing dF/F.
    """

    if ps_epoch == 'photostim':
        suite2p_folder = 'suite2p_photostim_single'
    else:
        suite2p_folder = 'suite2p_photostim_single2'

    ps_dir = os.path.join(folder, suite2p_folder, 'plane0')

    if not os.path.exists(os.path.join(ps_dir, 'iscell.npy')):
        raise FileNotFoundError(f'Could not find suite2p files in {ps_dir}')

    iscell = np.load(os.path.join(ps_dir, 'iscell.npy'), allow_pickle=True)
    stat = np.load(os.path.join(ps_dir, 'stat.npy'), allow_pickle=True)
    Ftrace = np.load(os.path.join(ps_dir, 'F.npy'), allow_pickle=True)
    ops = np.load(os.path.join(ps_dir, 'ops.npy'), allow_pickle=True).tolist()
    siHeader = np.load(os.path.join(ps_dir, 'siHeader.npy'), allow_pickle=True).tolist()

    (
        Fstim,
        seq,
        favg,
        stimDist_full,
        stimPosition,
        centroidX,
        centroidY,
        slmDist,
        stimID,
        Fstim_raw,
        favg_raw_full,
        stim_params,
    ) = ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat, 0)

    F0 = np.nanmean(np.nanmean(Fstim_raw[:pre_frames, :, :], axis=0), axis=-1)

    favg_raw_bs = favg_raw_full - np.nanmean(
        favg_raw_full[:pre_frames, :, :],
        axis=0,
        keepdims=True
    )

    dim_thresh = np.nanpercentile(F0, dim_pctile)
    dim_mask = F0 <= dim_thresh
    noncell_mask = iscell[:, 0] == 0

    artifact_traces = []
    n_roi_group_pairs = 0

    for gi in range(favg_raw_bs.shape[2]):
        far_rois = np.where(
            (stimDist_full[:, gi] > far_thresh_um)
            & dim_mask
            & noncell_mask
        )[0]

        if len(far_rois) == 0:
            continue

        for ri in far_rois:
            artifact_traces.append(favg_raw_bs[:, ri, gi])
        n_roi_group_pairs += len(far_rois)

    if len(artifact_traces) == 0:
        raise ValueError('No valid far + dim + iscell=0 samples found for artifact estimation.')

    artifact_traces = np.vstack(artifact_traces)
    artifact_trace_mean = np.nanmean(artifact_traces, axis=0)

    window_vals = artifact_traces[:, start:end].ravel()
    window_vals = window_vals[np.isfinite(window_vals)]

    if window_vals.size == 0:
        raise ValueError('No finite samples in stim window for artifact estimation.')

    if use_median:
        artifact_value = np.nanmedian(window_vals)
    else:
        artifact_value = np.nanmean(window_vals)

    return float(artifact_value), artifact_trace_mean, window_vals.size, n_roi_group_pairs


h5_path = find_photostim_h5(folder, ps_epoch)
print(f'Loading h5: {h5_path}')

ep_data = load_h5_epoch(h5_path)
si, trial_dur = load_si_and_ops(folder, ps_epoch)

umPerPix = (
    1000
    / float(si['metadata']['hRoiManager']['scanZoomFactor'])
    / int(si['metadata']['hRoiManager']['pixelsPerLine'])
)

fraw = np.array(ep_data['favg_raw'], dtype=float)
stimDist = np.array(ep_data['stimDist'], dtype=float) * umPerPix

dt = 1.0 / float(si['metadata']['hRoiManager']['scanVolumeRate'])
t = np.arange(fraw.shape[0]) * dt

start, pre_end, end = stim_timing(ep_data, dt)
crop = trial_dur if trial_dur is not None else fraw.shape[0]

F0_all = np.nanmean(fraw[0:pre_end, :, :], axis=0)            # per-(cell, group)
F0_per_cell = np.nanmean(F0_all, axis=-1)                     # per-cell, avg across groups (more stable dF/F denominator)
floor_pctile = 1   # percentile of F0_per_cell below which cells are excluded from dF/F averages
floor = np.nanpercentile(F0_per_cell, floor_pctile)

artifact_raw, artifact_trace, n_artifact_samples, n_roi_group_pairs = estimate_session_artifact_value_from_suite2p(
    folder=folder,
    ps_epoch=ps_epoch,
    start=start,
    end=end,
    pre_frames=8,
    far_thresh_um=100,
    dim_pctile=10,
    use_median=False,
)

print(f'fraw shape: {fraw.shape}')
print(f'start={start}, pre_end={pre_end}, end={end}, crop={crop}')
print(f'baseline floor = {floor}')
print(f'estimated artifact_raw = {artifact_raw:.6f}')
print(f'artifact samples = {n_artifact_samples}, ROI-group pairs = {n_roi_group_pairs}')


distance_bins = [
    ("< 10 µm",    lambda d: d < 10),
    ("15–30 µm",   lambda d: (d > 15) & (d < 30)),
    ("30–50 µm",   lambda d: (d > 30) & (d < 50)),
    ("50–100 µm",  lambda d: (d > 50) & (d < 100)),
    ("100–200 µm", lambda d: (d > 100) & (d < 200)),
    ("> 200 µm",   lambda d: d > 200),
]

artifact_win = slice(start, end)


def plot_distance_bin_traces(artifact_value, suptitle=None, mode='dff', floor=None, savepath=None):
    """
    Plot per-cell traces in six distance bins.

    artifact_value : scalar in raw F units to subtract from each cell's F
                     during the stim window. Pass 0.0 for the uncorrected version.
    mode    : 'dff'    → plot (F - F0) / F0
              'raw_bs' → plot F - F0 in raw fluorescence units (no normalization)
    floor   : minimum baseline F0 to include a cell in the average. Cells with
              bl <= floor are dropped. Defaults to the module-level `floor`.
              Raise this to exclude dim cells that amplify noise in dF/F space.
    savepath: if not None, save figure to this path (PNG) before show.
    """

    if mode not in ('dff', 'raw_bs'):
        raise ValueError("mode must be 'dff' or 'raw_bs'")

    if floor is None:
        floor = globals()['floor']

    n_bins, n_cols = len(distance_bins), 3
    n_rows = math.ceil(n_bins / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5), sharex=True)
    axes = axes.ravel()

    avg_legend = None
    smooth_legend = None
    se_patch = None
    photostim_patch = None

    for ax, (label, cond) in zip(axes, distance_bins):
        A = []
        A_s = []

        for i in range(fraw.shape[2]):
            ind = np.where(cond(stimDist[:, i]))[0]
            if len(ind) == 0:
                continue

            dffs = []
            for j in ind:
                F = fraw[:, j, i].copy()
                F[artifact_win] = F[artifact_win] - artifact_value

                bl = np.nanmean(F[0:pre_end])           # per-(cell, group): for subtraction
                bl_global = F0_per_cell[j]              # per-cell, across groups: for division

                if (not np.isfinite(bl_global)) or (bl_global <= floor):
                    continue

                if mode == 'dff':
                    trace = (F - bl) / bl_global
                else:
                    trace = F - bl

                dffs.append(trace)

            if len(dffs) == 0:
                continue

            a = np.nanmean(dffs, axis=0)
            A.append(a)
            A_s.append(median_filter(a, size=5, mode='reflect'))

        if len(A) == 0:
            ax.set_title(label + " (no cells)", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        A = np.vstack(A)
        A_s = np.vstack(A_s)

        avg = np.nanmean(A, axis=0)
        sem = np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0])
        smooth_avg = np.nanmean(A_s, axis=0)

        avg_line, = ax.plot(
            t[:crop],
            avg[:crop],
            linewidth=2,
            color='#0096FF',
            label='Avg'
        )

        smooth_line, = ax.plot(
            t[:crop],
            smooth_avg[:crop],
            linewidth=2.4,
            color='#FFA500',
            label='Smoothed avg'
        )

        ax.fill_between(
            t[:crop],
            avg[:crop] - sem[:crop],
            avg[:crop] + sem[:crop],
            alpha=0.2,
            edgecolor='none',
            facecolor=avg_line.get_color()
        )

        ax.axvspan(start * dt, end * dt, color='#FFBF00', alpha=0.3)

        if avg_legend is None:
            avg_legend = avg_line
            smooth_legend = smooth_line
            se_patch = mpatches.Patch(
                facecolor=avg_line.get_color(),
                alpha=0.2,
                label='SE'
            )
            photostim_patch = mpatches.Patch(
                facecolor='#FFBF00',
                alpha=0.3,
                label='Photostim event'
            )

        ax.set_title(label, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for k in range(len(distance_bins), len(axes)):
        fig.delaxes(axes[k])

    ylabel = "ΔF/F" if mode == 'dff' else "F − F₀ (raw)"
    fig.supxlabel("Time (s)", fontsize=18, fontname="Arial", x=0.55, y=0.07)
    fig.supylabel(ylabel, fontsize=18, fontname="Arial", y=0.55)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=1.02)

    if avg_legend is not None:
        fig.legend(
            handles=[avg_legend, smooth_legend, se_patch, photostim_patch],
            loc="upper right",
            fontsize=14,
            bbox_to_anchor=(1.3, 0.98),
            frameon=False
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


#%%
# --------------------------------------------
# Six distance-bin traces with artifact removal
# --------------------------------------------
plot_distance_bin_traces(
    artifact_raw, suptitle='Artifact subtracted',
    savepath=os.path.join(SAVE_DIR, 'inhibitory_dff_corrected.png'),
)


#%%
# --------------------------------------------
# Estimated artifact trace (raw F units)
# --------------------------------------------
fig_a, ax_a = plt.subplots(figsize=(5.5, 3.5))

ax_a.plot(
    t[:crop],
    artifact_trace[:crop],
    linewidth=2,
    color='black',
    label='Mean far+dim+non-cell ROIs'
)

ax_a.axhline(
    artifact_raw,
    color='#0096FF',
    linestyle='--',
    linewidth=1.5,
    label=f'Window mean = {artifact_raw:.3f}'
)

ax_a.axhline(0, color='gray', linewidth=0.5)
ax_a.axvspan(start * dt, end * dt, color='#FFBF00', alpha=0.3, label='Photostim event')

ax_a.set_xlabel('Time (s)', fontsize=12)
ax_a.set_ylabel('Artifact (raw F, baseline-subtracted)', fontsize=12)
ax_a.set_title(f'Estimated photostim artifact (n={n_roi_group_pairs} ROI-group pairs)', fontsize=12)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.legend(fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'inhibitory_artifact_trace.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# --------------------------------------------
# Same six distance-bin traces, uncorrected
# --------------------------------------------
plot_distance_bin_traces(
    0.0, suptitle='Uncorrected (no artifact subtraction)',
    savepath=os.path.join(SAVE_DIR, 'inhibitory_dff_uncorrected.png'),
)


#%%
# --------------------------------------------
# Raw F − F0 version, artifact subtracted
# --------------------------------------------
plot_distance_bin_traces(
    artifact_raw,
    suptitle='Artifact subtracted (raw F − F₀)',
    mode='raw_bs',
    savepath=os.path.join(SAVE_DIR, 'inhibitory_rawF_corrected.png'),
)


#%%
# --------------------------------------------
# Raw F − F0 version, uncorrected
# --------------------------------------------
plot_distance_bin_traces(
    0.0,
    suptitle='Uncorrected (raw F − F₀)',
    mode='raw_bs',
    savepath=os.path.join(SAVE_DIR, 'inhibitory_rawF_uncorrected.png'),
)


#%%
# --------------------------------------------
# Floor sweep: dF/F, no artifact subtraction
# --------------------------------------------
# Diagnostic: the apparent "artifact" in dF/F is dim cells amplifying small
# offsets via 1/F0. Raising the floor excludes them. As floor goes up, the
# stim-window bump in the bin averages should shrink and eventually vanish.

for pct in [1, 10, 25, 50]:
    f = np.nanpercentile(F0_per_cell, pct)
    plot_distance_bin_traces(
        0.0,
        suptitle=f'Uncorrected, floor = F0 pctile {pct} ({f:.2f})',
        mode='dff',
        floor=f,
        savepath=os.path.join(SAVE_DIR, f'inhibitory_floor_sweep_pct{pct:02d}.png'),
    )
