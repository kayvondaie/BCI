import bci_time_series as bts
from BCI_data_helpers import *
import data_dict_create_module_iscell as ddc
folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/092525/pophys/';
#folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/101125/pophys/';
#folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/101625/pophys/';
data = ddc.main(folder)
#%%
dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
hit = np.isnan(rt)==0;
rt[np.isnan(rt)] = 30;
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
    folder, data, rt, dt_si)
# Ftrace = np.load(folder + '/suite2p_BCI/plane0/F.npy', allow_pickle=True)
# df = 0*Ftrace
df = data['df_closedloop']

def get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 2)):
     
    nC, nFrames = df.shape
    reward_frames = np.where(reward_vector > 0)[0]

    pre_frames  = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_reward = np.arange(-pre_frames, post_frames) * dt_si

    F_reward = np.full((win_len, nC, len(reward_frames)), np.nan)

    for ri, r in enumerate(reward_frames):
        start = r - pre_frames
        stop  = r + post_frames
        if start < 0 or stop > nFrames:
            continue
        F_reward[:, :, ri] = df[:, start:stop].T  # (time, cells)

    return F_reward, t_reward


rta, t_reward = get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 6))
sta, t_reward = get_reward_aligned_df(df, trial_start_vector, dt_si, window=(-2, 10))

ts = np.arange(F.shape[0]) * dt_si - 2
tr = np.arange(rta.shape[0]) * dt_si - 2

rta = np.nanmean(rta,2)
sta = np.nanmean(sta,2)
#%%

import numpy as np
import matplotlib.pyplot as plt

# Epoch colors
epoch_colors = {
    "Pre": "#33b983",    # teal
    "Early": "#1077f3",  # blue
    "Late": "#0050ae",   # dark blue
    "Rew": "#bf8cfc"     # lavender
}

# --- Compute mean vectors ---
ind = np.where((tr > 2) & (tr < 3))[0]
aft = np.nanmean(rta[ind, :], 0)

ind = np.where((ts < -1))[0]
pre = np.nanmean(sta[ind, :], 0)
ind = np.where((ts > 0) & (ts < .2))[0]
early = np.nanmean(sta[ind, :], 0)
ind = np.where((tr < 0) & (tr > -2))[0]
late = np.nanmean(rta[ind, :], 0)
ind = np.where((tr > 1) & (tr < 2))[0]
rew = np.nanmean(rta[ind, :], 0)

# Subtract baselines
rew2 = rew - late
early2 = early - pre
late2 = late - pre
pre2 = pre - rew

pre = pre2; late = late2; early = early2; rew = rew2

# Restrict to valid cells
ind = np.where((pre < 11))[0]

# --- Plot ---
plt.figure(figsize=(6, 8))
epochs = [("Pre", pre), ("Early", early), ("Late", late), ("Rew", rew)]

for i, (label, vec) in enumerate(epochs, 1):
    ts = np.arange(sta.shape[0]) * dt_si - 2
    tr = np.arange(rta.shape[0]) * dt_si - 2

    y1 = sta[:, ind] @ vec[ind]
    y2 = rta[:, ind] @ vec[ind]
    color = epoch_colors[label]

    ax1 = plt.subplot(4, 2, 2 * i - 1)
    ax2 = plt.subplot(4, 2, 2 * i)

    # Plot traces
    ax1.plot(ts, y1, color=color, linewidth=2)
    ax2.plot(tr, y2, color=color, linewidth=2)

    # Shared y-limits
    ymin = min(y1.min(), y2.min())
    ymax = max(y1.max(), y2.max())
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # Remove axis frames & ticks
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Titles
    ax1.set_title(f"{label} (trial-start aligned)", fontsize=10, color=color)
    ax2.set_title(f"{label} (reward-aligned)", fontsize=10, color=color)

    # --- Add scale bars (0.5 s horizontally, 0.2 arbitrary units vertically) ---
    xscale = 0.5
    yscale = 0.2 * (ymax - ymin)
    x0 = ts.min() + 0.2
    y0 = ymin + 0.05 * (ymax - ymin)
    for ax in [ax1, ax2]:
        ax.plot([x0, x0 + xscale], [y0, y0], color='k', lw=2)
        ax.plot([x0, x0], [y0, y0 + yscale], color='k', lw=2)
        ax.plot([0,0],ax.get_ylim(),':',color = (.5,.5,.5))
        ax.text(x0 + xscale / 2, y0 - 0.03 * (ymax - ymin), f"{xscale:.1f}s",
                ha='center', va='top', fontsize=8)
        ax.text(x0 - 0.05, y0 + yscale / 2, f"{yscale:.2f}",
                ha='right', va='center', fontsize=8)

plt.tight_layout()
plt.show()

#%%
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import re

path = folder + r'stack_00001.tif'

# --- Load TIFF ---
with tiff.TiffFile(path) as tif:
    datas = tif.asarray()
    try:
        desc = tif.pages[0].tags['ImageDescription'].value
    except Exception:
        desc = ''
    print(f"Raw TIFF data shape: {datas.shape}")
    print(desc[:300], '...')

# --- Parse metadata to infer channels/planes if needed ---
def get_value(text, key):
    match = re.search(rf'{key}\s*=\s*(\[.*?\]|[^\r\n]*)', text)
    if not match:
        return None
    val = match.group(1).strip()
    if val.startswith('['):
        try:
            arr = np.fromstring(val.strip('[]'), sep=' ')
            return arr
        except Exception:
            return val
    else:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

# --- Handle already-shaped data ---
if datas.ndim == 5:
    # (planes, frames, channels, y, x)
    n_planes, n_frames, n_channels, height, width = datas.shape
elif datas.ndim == 3:
    # (frames, y, x)
    n_channels = get_value(desc, 'state.acq.numberOfChannelsSave') or 1
    zs = get_value(desc, 'state.acq.zs')
    n_planes = len(zs) if isinstance(zs, np.ndarray) and zs.size > 0 else 1
    n_total_frames = datas.shape[0]
    frames_per_vol = n_planes * n_channels
    n_volumes = n_total_frames // frames_per_vol
    height, width = datas.shape[1:]
    datas = datas.reshape(n_volumes, n_planes, n_channels, height, width)
    n_planes, n_frames, n_channels = datas.shape[1:4]
else:
    raise ValueError(f"Unexpected data shape: {data.shape}")

print(f"Final interpreted shape: (planes={n_planes}, frames={n_frames}, channels={n_channels}, y={height}, x={width})")

# --- Select central plane, channel 2 (index 1) ---
plane_idx = n_planes // 2
if n_channels < 2:
    raise ValueError("This TIFF only contains one channel — no channel 2 available.")
#%%
plt.figure(figsize=(12, 6))
names = ['Green channel','Red channel']
for i in range(2):
    avg_frame = np.mean(datas[plane_idx, :, i, :, :], axis=0)
    plt.subplot(1,2,i+1)
    
    plt.imshow(avg_frame, cmap='gray',
               vmin=np.percentile(avg_frame, 1),
               vmax=np.percentile(avg_frame, 99.7))
    plt.title(names[i])
    plt.axis('off')
plt.show()
#%%
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Load Suite2p ROI definitions ---
suite2p_folder = folder + r'suite2p_BCI/plane0'
stat = np.load(os.path.join(suite2p_folder, 'stat.npy'), allow_pickle=True)
ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).tolist()
iscell = np.load(os.path.join(suite2p_folder, 'iscell.npy'), allow_pickle=True)

# Filter to real cells only
cell_inds = np.where(iscell[:, 0] == 1)[0]
stat = [stat[i] for i in cell_inds]

# --- Compute mean brightness for each cell in the red channel image ---
brightness = np.zeros(len(stat))
for i, roi in enumerate(stat):
    ypix = roi['ypix']
    xpix = roi['xpix']
    # Optional: use ROI weights (area weighting)
    if 'lam' in roi:
        lam = roi['lam']
        brightness[i] = np.sum(avg_frame[ypix, xpix] * lam) / np.sum(lam)
    else:
        brightness[i] = np.mean(avg_frame[ypix, xpix])

# --- Normalize and visualize ---
plt.figure(figsize=(4, 3))
plt.hist(brightness, bins=50, color='r', alpha=0.6)
plt.xlabel('Mean brightness (red channel)')
plt.ylabel('Cell count')
plt.title('Brightness across ROIs')
plt.show()

# --- Example: overlay top bright cells ---
topN = 10
idx_sorted = np.argsort(-brightness)
overlay = np.zeros((*avg_frame.shape, 3))
overlay[..., 0] = avg_frame / np.percentile(avg_frame, 99)  # grayscale red background

for k in idx_sorted[:topN]:
    roi = stat[k]
    ypix, xpix = roi['ypix'], roi['xpix']
    overlay[ypix, xpix, :] = [1, 1, 0]  # yellow overlay for bright cells

plt.figure(figsize=(6, 6))
plt.imshow(overlay, origin='upper')
plt.axis('off')
plt.show()
#%%

ind = np.where((pre < 11) & (brightness > np.percentile(brightness,80)))[0]
plt.figure(figsize=(6,8))

for i,(vec,label) in enumerate([(pre,"Pre"),(early,"Early"),(late,"Late"),(rew,"Rew")],1):
    ts = np.arange(sta.shape[0]) * dt_si - 2
    tr = np.arange(rta.shape[0]) * dt_si - 2

    y1 = sta[:,ind] @ vec[ind]
    y2 = rta[:,ind] @ vec[ind]

    ax1 = plt.subplot(4,2,2*i-1); ax1.plot(ts, y1)
    ax2 = plt.subplot(4,2,2*i);   ax2.plot(tr, y2)

    ymin = min(y1.min(),y2.min())
    ymax = max(y1.max(),y2.max())
    ax1.set_ylim(ymin,ymax)
    ax2.set_ylim(ymin,ymax)

    ax1.set_title(f"{label} (trial-start aligned)")
    ax2.set_title(f"{label} (reward-aligned)")
    ax1.set_xlabel("Time (s)")
    ax2.set_xlabel("Time (s)")
    ax1.set_ylabel("Projection")
    ax2.set_ylabel("Projection")

plt.tight_layout()
#%%
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
titles = ['Pre', 'Early', 'Late', 'Reward']

for ax, v, title in zip(axes, [pre, early, late, rew], titles):
    ax.scatter(brightness, v, color='k', s=10)
    plt.sca(ax)
    #pf.mean_bin_plot(brightness, v)
    #ax.set_title(title)
    ax.set_xlabel('Red fluorescence')
    ax.set_ylabel(title + r' (DF/F)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))

for i in range(2):
    if i == 0:
        Y = rta;
        t = tr
    else:
        Y = sta;
        t = ts
    plt.subplot(1,2,i+1)
    avg_red = np.nanmean(Y[:,brightness > np.percentile(brightness,80)],1);
    avg_not = np.nanmean(Y[:,brightness < np.percentile(brightness,20)],1);
    avg_red = avg_red - np.nanmean(avg_red[0:20])
    avg_not  = avg_not - np.nanmean(avg_not [0:20])
    plt.plot(t,avg_red,'r');
    plt.plot(t,avg_not,'k')    
    plt.legend(('Red cells','Not red'))
    if i == 0:
        plt.xlabel('Time from reward (s)')
    else:
        plt.xlabel('Time from trial start (s)')
#%%
from scipy.signal import medfilt

def medfilt_real_data(favg, artifact, kernel_size=5):
    """Apply median filter only to real (non-artifact) frames."""
    favg_filt = np.copy(favg)
    T = favg.shape[0]
    artifact_mask = np.zeros(T, dtype=bool)
    artifact_mask[artifact] = True

    # Indices of real frames
    real_idx = np.where(~artifact_mask)[0]

    # Apply median filter only to real frames, one ROI at a time
    for c in range(favg.shape[1]):  # cell
        for g in range(favg.shape[2]):  # stim group
            trace = favg[:, c, g]
            real_trace = trace[real_idx]

            if np.sum(~np.isnan(real_trace)) > kernel_size:
                # Apply medfilt to the valid subset
                smoothed = medfilt(real_trace, kernel_size=kernel_size)
                favg_filt[real_idx, c, g] = smoothed

            # keep artifact frames as NaN
            favg_filt[artifact_mask, c, g] = np.nan

    return favg_filt

def compute_amp_from_photostim(mouse, data, folder, return_favg=False):
    import numpy as np
    from scipy.signal import medfilt

    AMP = []
    favg_all = []
    favg_filt_all = []
    # --- Load ScanImage header to compute µm per pixel ---
    siHeader_path = folder + r'/suite2p_BCI/plane0/siHeader.npy'
    siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
    umPerPix = 1000 / float(siHeader['metadata']['hRoiManager']['scanZoomFactor']) / int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])

    # --- Determine number of epochs ---
    n_epochs = 2 if 'photostim2' in data else 1
    for epoch_i in range(n_epochs):
        if epoch_i == 0:
            stimDist = data['photostim']['stimDist'] * umPerPix
            favg_raw = data['photostim']['favg_raw']
        else:
            stimDist = data['photostim2']['stimDist'] * umPerPix
            favg_raw = data['photostim2']['favg_raw']

        # --- Normalize ΔF/F traces by baseline (first 8 frames) ---
        favg = np.zeros_like(favg_raw)
        baseline = np.nanmean(favg_raw[0:7, :, :], axis=0)
        favg = (favg_raw - baseline) / baseline

        dt_si = data['dt_si']
        after = int(np.floor(0.2 / dt_si))
        before = int(np.floor(0.2 / dt_si))
        if mouse == "BCI103":
            after = int(np.floor(0.5 / dt_si))

        # --- Detect stimulation artifact ---
        artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > 0.5)[0]
        artifact = artifact[artifact < 25]
        #artifact = np.concatenate(([artifact[0] - 1], artifact,[artifact[-1]+1]))
        #artifact = np.concatenate(([artifact[0] - 2,artifact[0] - 1], artifact,[artifact[-1]+1,artifact[-1]+2,artifact[-1]+3]))

        if artifact.size == 0:
            AMP.append(np.full(favg_raw.shape[1:], np.nan))
            favg_all.append(favg)
            continue

        # --- Define pre- and post-stim windows ---
        pre = (int(artifact[0] - before), int(artifact[0] - 2))
        post = (int(artifact[-1] + 2), int(artifact[-1] + after))

        # --- Mask artifact region ---
        favg[artifact, :, :] = np.nan

        # --- Interpolate early NaNs (if any) ---
        favg[0:30, :, :] = np.apply_along_axis(
            lambda m: np.interp(
                np.arange(len(m)),
                np.where(~np.isnan(m))[0] if np.any(~np.isnan(m)) else [0],
                m[~np.isnan(m)] if np.any(~np.isnan(m)) else [0]
            ),
            axis=0,
            arr=favg[0:30, :, :]
        )

        # --- Apply median filter AFTER artifact removal ---
        favg_filt = np.apply_along_axis(medfilt, 0, favg, kernel_size=5)
        favg = np.apply_along_axis(medfilt, 0, favg, kernel_size=5)
        
        #favg = medfilt_real_data(favg, artifact, kernel_size=5)


        # --- Compute AMP (Δmean_post − Δmean_pre) ---
        amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
        AMP.append(amp)
        favg_all.append(favg)
        favg_filt_all.append(favg_filt)

    if return_favg:
        return AMP, stimDist, favg_all, artifact, favg_filt_all
    else:
        return AMP, stimDist


#%%
AMP, stimDist,FAVG,artifact,favg_filt = compute_amp_from_photostim('BCI116', data, folder,return_favg= True)
plt.scatter(stimDist.flatten(),AMP[0].flatten())
#%%
epoch = 0;
favg = FAVG[epoch]
favg2 = favg_filt[epoch]
amp = AMP[epoch];
targs = np.argmin(stimDist,0)
amp_targ = amp[targs, np.arange(amp.shape[1])]
plt.plot(brightness[targs],amp_targ,'ko')
plt.xlabel('Red brightness')
plt.ylabel('Target response')
plt.show()
#%%
t = np.arange(0,favg.shape[0]*dt_si,dt_si)
plt.figure(figsize = (10,3))
b = np.argsort(-amp_targ)
gi = b[6]
plt.subplot(141)
plt.plot(t[0:30],favg[0:30,targs[gi],gi])
plt.title('Target')
plt.xlabel('Time from photostim (s)')
plt.subplot(142)
plt.scatter(stimDist[:,gi],amp[:,gi])
plt.title('Distance from target' + str(gi))
plt.subplot(143)
plt.scatter(brightness,amp[:,gi])
plt.xlabel('Brightness')
plt.ylabel('Response to target ' + str(gi))
plt.subplot(144)
b = np.argsort(-amp[:,gi])
b = b[b<100]
ci = 30
ci = b[ci]
ci = 8
plt.plot(t[0:30],favg[0:30,ci,gi])
plt.title('non-target ' + str(ci))
plt.xlabel('Time from photostim (s)')
plt.tight_layout()
#%%
plt.figure(figsize=(12, 6))
num_good = 400

# --- Select group and compute SVD ---
b = np.argsort(-amp_targ)
gi = b[6]           # target group index
vi = 1             # which singular vector to show

x = favg[0:25, 0:num_good, gi].copy()
x[np.isnan(x)] = 0
if targs[gi] < x.shape[0]:
    x[:, targs[gi]] = 0

u, s, v = np.linalg.svd(x)
v = v.T

vp = v[:, vi] * (v[:, vi] > 0)
vn = -v[:, vi] * (v[:, vi] < 0)

x = favg[0:25, 0:num_good, gi].copy()
x[np.isnan(x)] = 0
if targs[gi] < x.shape[0]:
    x[:, targs[gi]] = 0

# --- Time vector and helper for photostim mark ---
t = np.arange(x.shape[0]) * dt_si - 2  # adjust baseline as needed

def ps_time():
    plt.axvspan(8*dt_si - 2, 13*dt_si - 2, color='magenta', alpha=0.2)

# ========== TOP ROW ==========
plt.subplot(2, 4, 1)
plt.plot(t, x @ v[:, vi], '.-')
ps_time()
plt.title('Mode time course')
plt.xlabel('Time (s)')
plt.ylabel('Projection (a.u.)')

plt.subplot(2, 4, 2)
plt.plot(t, x @ vp, '.-', color='r')
ps_time()
plt.title('Positive mode (co-active cells)')
plt.xlabel('Time (s)')
plt.ylabel('Projection (a.u.)')

plt.subplot(2, 4, 3)
plt.plot(t, x @ vn, '.-', color='b')
ps_time()
plt.title('Negative mode (suppressed cells)')
plt.xlabel('Time (s)')
plt.ylabel('Projection (a.u.)')

plt.subplot(2, 4, 4)
#plt.plot(brightness[0:num_good], v[:, vi], 'k.')
pf.mean_bin_plot(brightness[0:num_good], v[:, vi])
plt.xlabel('Red-channel brightness')
plt.ylabel('SVD weight')
plt.title('Spatial weights vs. opsin expression')

# ========== BOTTOM ROW (target neuron traces) ==========
# Photostim-aligned
plt.subplot(2, 4, 5)
plt.plot(t, favg[0:25, targs[gi], gi], 'k.-')
ps_time()
plt.title(f'Target cell {targs[gi]} photostim response')
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F')

# Trial start aligned
plt.subplot(2, 4, 6)
plt.plot(ts, sta[:, targs[gi]], 'k')
plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
plt.title('Target cell — trial start aligned')
plt.xlabel('Time from trial start (s)')
plt.ylabel('ΔF/F')

# Reward aligned
plt.subplot(2, 4, 7)
plt.plot(tr, rta[:, targs[gi]], 'k')
plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
plt.title('Target cell — reward aligned')
plt.xlabel('Time from reward (s)')
plt.ylabel('ΔF/F')

# Info / placeholder panel
plt.subplot(2,4,8);
#plt.plot(stimDist[0:num_good,gi],v[:,vi],'k.')
pf.mean_bin_plot(stimDist[0:num_good,gi],v[:,vi])
plt.xlabel('Distance from target (um)')
plt.ylabel('SVD weight')

plt.tight_layout()
plt.show()
amp2 = np.nanmean(favg[7:11, :, :], axis=0)    # shape: (n_cells, n_groups)
rebound = np.nanmean(favg[15:22, :, :], axis=0) - np.nanmean(favg[7:11, :, :], axis=0)
#%%
import matplotlib.pyplot as plt
import numpy as np

# --- configurable parameters ---
X = 8          # number of example top/bottom cells to show (beyond the first two columns)
n_frames = 25

# --- sort neurons by SVD weight ---
b = np.argsort(v[:, vi])
t = np.arange(n_frames) * dt_si - 2

plt.figure(figsize=(3 * (X + 2), 6))

# ---------------- TOP ROW ----------------
# (1) target neuron
plt.subplot(2, X + 2, 1)
plt.plot(t, favg[0:n_frames, targs[gi], gi], 'k')
plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.2)
plt.title('Target neuron')
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F')

# (2) SVD projections (same 3 traces from before)
plt.subplot(2, X + 2, 2)

plt.plot(t, x @ vp, 'r', label='Positive')

plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.2)
plt.title('SVD projections')
plt.xlabel('Time (s)')
plt.legend(frameon=False)

# (3-...) top-weighted example neurons
for i in range(X):
    ci = b[-(i + 1)]   # top cells
    plt.subplot(2, X + 2, 3 + i)
    plt.plot(t, favg[0:n_frames, ci, gi], 'r')
    plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.2)
    plt.title(f'Top {i+1}')
    plt.xlabel('Time (s)')

# ---------------- BOTTOM ROW ----------------
# (1) heatmap of all neurons sorted by weight
plt.subplot(2, X + 2, X + 3)
plt.imshow(favg[0:n_frames, b, gi].T, aspect='auto', cmap='bwr',
           vmin=-0.8, vmax=0.8, origin='lower')
plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.15)
plt.colorbar(label='ΔF/F', fraction=0.046, pad=0.04)
plt.title('All cells (sorted)')
plt.xlabel('Time (s)')
plt.ylabel('Neuron index (sorted)')

# (2) again show SVD projections for reference
plt.subplot(2, X + 2, X + 4)

plt.plot(t, x @ vn, 'b')
plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.2)
plt.title('SVD projections')
plt.xlabel('Time (s)')

# (3-...) bottom-weighted example neurons
for i in range(X):
    ci = b[i]   # bottom cells
    plt.subplot(2, X + 2, X + 5 + i)
    plt.plot(t, favg[0:n_frames, ci, gi], 'b')
    plt.axvspan(8 * dt_si - 2, 13 * dt_si - 2, color='m', alpha=0.2)
    plt.title(f'Bottom {i+1}')
    plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()
#%%
import numpy as np
from scipy.stats import pearsonr

# average photostim response over the key frames
amp2 = np.nanmean(favg[7:11, :, :], axis=0)    # shape: (n_cells, n_groups)
rebound = np.nanmean(favg[15:22, :, :], axis=0) - np.nanmean(favg[7:11, :, :], axis=0)

n_groups = amp2.shape[1]
corrs = np.full(n_groups, np.nan)

for gi in range(n_groups):
    # mask out the target neuron
    mask = np.ones(amp2.shape[0], dtype=bool)
    mask[targs[gi]] = False
    mask[200:] = False;

    # compute correlation between brightness and response (excluding target)
    valid = np.isfinite(brightness) & np.isfinite(amp2[:, gi]) & mask
    if np.sum(valid) > 1:
        corrs[gi], _ = pearsonr(brightness[valid], amp2[valid, gi])
        pf.mean_bin_plot(brightness[valid], rebound[valid, gi])
        plt.show()

# 'corrs' now holds correlation coefficient for each target group
ind = np.where(amp_targ > .2)[0]
plt.figure(figsize=(4,3))
plt.hist(corrs[ind], bins=20, color='k', alpha=0.7)
plt.xlabel('r (brightness vs. ΔF/F amplitude)')
plt.ylabel('Count')
plt.title('Brightness-response correlation across targets')
plt.tight_layout()
plt.show()



#%%
a, t_reward = get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 6))
g = np.nanmean(a[90:120,targs[gi],:],0)
rtr = rt[rt!=30]
pf.mean_bin_plot(rtr,g,4)
#%%
a = 0*df
for i in range(df.shape[0]):
    a[i,:] = medfilt(df[i,:],21)
cc = np.corrcoef(a);
import numpy as np
import matplotlib.pyplot as plt

# Sort neuron indices by brightness
order = np.argsort(brightness)

# Reorder correlation matrix
cc_sorted = cc[order, :][:, order]

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(cc_sorted, cmap='bwr', vmin=-.5, vmax=.5, origin='upper')
plt.colorbar(label='Correlation coefficient')
plt.title('Pairwise correlations (sorted by brightness)')
plt.xlabel('Neuron index (sorted)')
plt.ylabel('Neuron index (sorted)')

# Optional: mark the boundary between bright and dim neurons (e.g., 80th percentile)
thr = np.percentile(brightness, 80)
cut_idx = np.sum(brightness[order] < thr)
plt.axhline(cut_idx, color='k', linestyle='--', alpha=0.4)
plt.axvline(cut_idx, color='k', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
#%%
