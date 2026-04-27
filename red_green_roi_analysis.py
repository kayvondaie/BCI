import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from BCI_data_helpers import compute_amp_from_photostim

# Paths
folder = 'Y:/2p-raw/BCI93/020425/'
tif_path = folder + 'pophys/stack_00001.tif'
stat_path = folder + 'pophys/suite2p_BCI/plane0/stat.npy'

# Read TIF — shape is (21, 200, 2, 256, 512): (planes, frames, channels, H, W)
img = tf.imread(tif_path)
mid_plane = img.shape[0] // 2  # central plane (index 10)
green_avg = np.mean(img[mid_plane, :, 0, :, :], axis=0)
red_avg = np.mean(img[mid_plane, :, 1, :, :], axis=0)
#%%
# Load suite2p ROIs
stat = np.load(stat_path, allow_pickle=True)
iscell = np.load(folder + 'pophys/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
cells = np.where(iscell[:, 0] == 1)[0]

# Load photostim data directly from h5
h5_folder = folder + 'pophys/'
data = {'photostim': {}, 'photostim2': {}}
for key in ['stimDist', 'favg_raw']:
    with h5py.File(h5_folder + 'data_photostim.h5', 'r') as f:
        data['photostim'][key] = f[key][:]
    with h5py.File(h5_folder + 'data_photostim2.h5', 'r') as f:
        data['photostim2'][key] = f[key][:]
with h5py.File(h5_folder + 'data_main.h5', 'r') as f:
    data['dt_si'] = f['dt_si'][()]

#%% Compute per-ROI metrics and classify E/I (uses aligned images after alignment cell runs)
N = len(stat)
G = np.zeros(N)
R = np.zeros(N)
cc = np.zeros(N)

donut = np.zeros(N)

for i in range(N):
    ypix = stat[i]['ypix']
    xpix = stat[i]['xpix']
    g = green_avg_aligned[ypix, xpix]
    r = red_avg_aligned[ypix, xpix]
    G[i] = np.nanmean(g)
    R[i] = np.nanmean(r)
    cc[i] = np.corrcoef(g, r)[0, 1]
    # Donutness: correlation of red intensity with distance from ROI centroid
    cy, cx = np.mean(ypix), np.mean(xpix)
    dist = np.sqrt((ypix - cy)**2 + (xpix - cx)**2)
    if len(dist) > 2:
        donut[i] = np.corrcoef(dist, r)[0, 1]

ratio = G / R

i_cells = np.where((ratio[cells] > 0.4) | (cc[cells] < 0))[0]
e_cells = np.where((ratio[cells] <= 0.4) & (cc[cells] >= 0))[0]

# Plot E/I classification
plt.figure()
plt.plot(ratio[cells[e_cells]], cc[cells[e_cells]], 'k.', markersize=4, label='E')
plt.plot(ratio[cells[i_cells]], cc[cells[i_cells]], 'r.', markersize=4, label='I')
plt.xlabel('Green/Red ratio')
plt.ylabel('Pixelwise RG correlation')
plt.legend()
plt.title('BCI93 020425 - E/I classification')
plt.show()

#%% Align stack green_avg to suite2p meanImg via cross-correlation
from scipy.signal import fftconvolve

ops = np.load(folder + 'pophys/suite2p_BCI/plane0/ops.npy', allow_pickle=True).item()
meanImg = ops['meanImg']

# Crop to matching size if needed
h = min(green_avg.shape[0], meanImg.shape[0])
w = min(green_avg.shape[1], meanImg.shape[1])
a = green_avg[:h, :w] - np.mean(green_avg[:h, :w])
b = meanImg[:h, :w] - np.mean(meanImg[:h, :w])

xcorr = fftconvolve(b, a[::-1, ::-1], mode='full')
peak = np.unravel_index(np.argmax(xcorr), xcorr.shape)
dy = peak[0] - (h - 1)
dx = peak[1] - (w - 1)
print(f'Alignment shift: dy={dy}, dx={dx}')

# Apply shift to green_avg and red_avg
from scipy.ndimage import shift as ndi_shift
green_avg_aligned = ndi_shift(green_avg, [dy, dx])
red_avg_aligned = ndi_shift(red_avg, [dy, dx])

#%% ROI overlay on aligned green image
g_norm = green_avg_aligned / np.percentile(green_avg_aligned, 99.5)
g_norm = np.clip(g_norm, 0, 1)
overlay = np.stack([g_norm, g_norm, g_norm], axis=2)

alpha = 0.4
for i in cells:
    ypix, xpix = stat[i]['ypix'], stat[i]['xpix']
    overlay[ypix, xpix, 0] = alpha * 1.0 + (1 - alpha) * overlay[ypix, xpix, 0]
    overlay[ypix, xpix, 1] = (1 - alpha) * overlay[ypix, xpix, 1]
    overlay[ypix, xpix, 2] = (1 - alpha) * overlay[ypix, xpix, 2]

plt.figure(figsize=(10, 5))
plt.imshow(np.clip(overlay, 0, 1))
plt.title('Green image + ROIs (red)')
plt.axis('off')
plt.show()

#%% Amp vs stimDist
AMP, stimDist = compute_amp_from_photostim('BCI93', data, h5_folder)
amp = AMP[0]
print(f'amp shape: {amp.shape}, stimDist shape: {stimDist.shape}, max(cells): {cells.max()}, len(stat): {len(stat)}')

# Limit to cells that fit within amp dimensions
valid = cells[cells < amp.shape[0]]
e_valid = valid[np.isin(valid, cells[e_cells])]
i_valid = valid[np.isin(valid, cells[i_cells])]

plt.figure()
plt.subplot(121)
plt.plot(stimDist[e_valid, :], amp[e_valid, :], 'k.', markersize=2)
plt.ylim(-1, 3)
plt.xlabel('Distance (px)')
plt.ylabel('Amplitude')
plt.title('Excitatory')
plt.subplot(122)
plt.plot(stimDist[i_valid, :], amp[i_valid, :], 'r.', markersize=2)
plt.ylim(-1, 3)
plt.xlabel('Distance (px)')
plt.title('Inhibitory')
plt.tight_layout()
plt.show()

#%% Mean non-target amp per neuron (excluding stimDist < 30 um)
amp_masked = amp.copy()
amp_masked[stimDist < 30] = np.nan
mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

plt.figure()
plt.hist(mean_amp_nontarg[cells], bins=50, color='k', edgecolor='w')
plt.xlabel('Mean amplitude (excl. <30 um)')
plt.ylabel('Count')
plt.title('Non-target mean amp per neuron')
plt.show()

#%% G/R vs correlation colored by mean non-target amp
plt.figure()
sc = plt.scatter(ratio[cells], cc[cells], c=mean_amp_nontarg[cells],
                 s=8, cmap='hot', vmin=0, vmax=np.percentile(mean_amp_nontarg[cells], 99))
plt.colorbar(sc, label='Mean non-target amp')
plt.xlabel('Green/Red ratio')
plt.ylabel('Pixelwise RG correlation')
plt.title('G/R vs correlation, colored by non-target amp')
plt.show()

#%% Donutness vs other metrics
plt.figure()
sc = plt.scatter(ratio[cells], donut[cells], c=mean_amp_nontarg[cells],
                 s=8, cmap='hot', vmin=0, vmax=np.percentile(mean_amp_nontarg[cells], 99))
plt.colorbar(sc, label='Mean non-target amp')
plt.xlabel('Green/Red ratio')
plt.ylabel('Red donutness (dist-intensity corr)')
plt.title('G/R vs donutness, colored by non-target amp')
plt.show()

#%% Save red and green channel images as PNGs
from matplotlib.colors import Normalize

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(green_avg_aligned, cmap='gray', vmin=0, vmax=np.percentile(green_avg_aligned, 99.5))
ax.set_title('Green channel')
ax.axis('off')
fig.savefig(folder + 'green_channel.png', dpi=200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(red_avg_aligned, cmap='gray', vmin=0, vmax=np.percentile(red_avg_aligned, 99.5))
ax.set_title('Red channel')
ax.axis('off')
fig.savefig(folder + 'red_channel.png', dpi=200, bbox_inches='tight')
plt.show()

print(f'Saved to {folder}green_channel.png and {folder}red_channel.png')
