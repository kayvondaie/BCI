import bci_time_series as bts
from BCI_data_helpers import *
folder = '//allen/aind/scratch/BCI/2p-raw/BCI116/092525/pophys/';
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


ind = np.where((tr>2) & (tr<3))[0]
aft = np.nanmean(rta[ind,:],0)

ind = np.where((ts<-1))[0]
pre = np.nanmean(sta[ind,:],0)
ind = np.where((ts>0) & (ts<.2))[0]
early = np.nanmean(sta[ind,:],0)
ind = np.where((tr<-1) & (tr>-2))[0]
late = np.nanmean(rta[ind,:],0)
ind = np.where((tr>1) & (tr<2))[0]
rew = np.nanmean(rta[ind,:],0)

rew2 = rew - late
early2 = early - pre
late2 = late - pre
pre2 = pre - rew

pre = pre2;late = late2;early = early2;rew = rew2;

# pre = pre - aft
# rew = rew - aft
# early = early - aft
# late = late - aft

ind = np.where((pre < 11))[0]
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
avg_frame = np.mean(datas[plane_idx, :, 1, :, :], axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(avg_frame, cmap='gray',
           vmin=np.percentile(avg_frame, 1),
           vmax=np.percentile(avg_frame, 99.7))
plt.title(f"Average frame — plane {plane_idx+1}/{n_planes}, channel 2")
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
    ax.set_title(title)
    ax.set_xlabel('Brightness')
    ax.set_ylabel('Value')

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
    avg_red = np.nanmean(Y[:,brightness > np.percentile(brightness,70)],1);
    avg_not = np.nanmean(Y[:,brightness < np.percentile(brightness,50)],1);
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
def compute_amp_from_photostim(mouse, data, folder, return_favg=False):
    import numpy as np
    from scipy.signal import medfilt

    AMP = []
    favg_all = []

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
        baseline = np.nanmean(favg_raw[0:8, :, :], axis=0)
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
        artifact = artifact[artifact < 40]

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
        favg = np.apply_along_axis(medfilt, 0, favg, kernel_size=5)

        # --- Compute AMP (Δmean_post − Δmean_pre) ---
        amp = np.nanmean(favg[post[0]:post[1], :, :], axis=0) - np.nanmean(favg[pre[0]:pre[1], :, :], axis=0)
        AMP.append(amp)
        favg_all.append(favg)

    if return_favg:
        return AMP, stimDist, favg_all, artifact
    else:
        return AMP, stimDist


#%%
AMP, stimDist,FAVG,artifact = compute_amp_from_photostim('BCI116', data, folder,return_favg= True)
#%%
plt.scatter(stimDist.flatten(),AMP[0].flatten())
#%%
epoch = 0;
favg = FAVG[epoch]
amp = AMP[epoch];
targs = np.argmin(stimDist,0)
amp_targ = amp[targs, np.arange(amp.shape[1])]
plt.plot(brightness[targs],amp_targ,'ko')
plt.xlabel('Red brightness')
plt.ylabel('Target response')
plt.show()
#%%
t = np.arange(0,favg.shape[0]*dt_si,dt_si)
t = t - t[artifact[np.where(np.diff(artifact) > 2)[0][0]]]
plt.figure(figsize = (10,3))
b = np.argsort(-amp_targ)
gi = b[3]
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
ci = 11

plt.plot(t[0:30],favg[0:30,b[ci],gi])
plt.title('non-target')
plt.xlabel('Time from photostim (s)')
plt.tight_layout()
#%%
t = np.arange(0,favg.shape[0]*dt_si,dt_si)
t = t - t[artifact[np.where(np.diff(artifact) > 2)[0][0]]]
ind = np.where((stimDist[:,gi]>20)&(amp[:,gi] > .4))[0]
ind = np.where((stimDist[:,gi]>30)&(stimDist[:,gi]<50))[0]
ind = np.where((amp[:,gi] > .5))[0]

plt.plot(t[0:30],np.nanmedian(favg[0:30,ind,gi],1))
ind = artifact[np.arange(0,np.where(np.diff(artifact) > 2)[0]+1)]

plt.xlim((-.2,.5))
plt.show()

