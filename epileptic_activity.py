# -*- coding: utf-8 -*-
"""
Steinmetz-style epileptiform event detection for 2-photon FOV
Data-driven clustering adapted to modality
Author: Kayvon Daie
"""

import numpy as np, matplotlib.pyplot as plt, tifffile as tiff, extract_scanimage_metadata, os

#fname = r'//allen/aind/scratch/BCI/2p-raw/BCI98/110724/pophys/spont_00003.tif'
# fname = r'//allen/aind/scratch/BCI/2p-raw/BCI102/021125/pophys/spontPost_00001.tif'
# fname = r'//allen/aind/scratch/BCI/2p-raw/BCI103/020325/pophys/spont_00003.tif'
#fname = '//allen/aind/scratch/BCI/2p-raw/BCI105/012325/pophys/spont_00007.tif'
# fname = '//allen/aind/scratch/BCI/2p-raw/BCI112/051325/pophys/spont_00014.tif'
fname = '//allen/aind/scratch/BCI/2p-raw/BCI98/030625_spont/pophys/spont_00002.tif'
#fname = '//allen/aind/scratch/BCI/2p-raw/BCI103/020625/pophys/spont_00001.tif'
#fname = '//allen/aind/scratch/BCI/2p-raw/BCI31/052422/pophys/spont_00011.tif'
folder = os.path.dirname(fname)
base = os.path.basename(fname)
mouse = folder.split('/')[-3]
session = folder.split('/')[-2]
# prefix = 'spont_'
prefix = base.split('_')[0] + '_'

# match only: prefix + 5 digits + '.tif'
files = sorted([
    os.path.join(folder,f)
    for f in os.listdir(folder)
    if f.startswith(prefix) and f[len(prefix):len(prefix)+5].isdigit() and f.endswith('.tif')
])

print('n files:', len(files))

dff_list = []
for f in files:
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(f)
    dt_si = 1 / float(siHeader['metadata']['hRoiManager']['scanVolumeRate'])
    data = tiff.imread(f)
    T = data.shape[0]
    frame_means = data.reshape(T,-1).mean(axis=1)
    f0 = np.percentile(frame_means,20)
    dff = (frame_means - f0)/f0
    dff_list.append(dff)

#%%
# ----------------------------------------------------------
# 3. Peak detection (positive + negative)
# ----------------------------------------------------------
pk_pos, prop_pos = find_peaks(dff, prominence=0, width=0)
pk_neg, prop_neg = find_peaks(-dff, prominence=0, width=0)

peaks = np.concatenate([pk_pos])
prom = np.concatenate([prop_pos['prominences'], prop_neg['prominences']])
width = np.concatenate([prop_pos['widths'], prop_neg['widths']]) * dt_si

# sort by peak time
ix = np.argsort(peaks)
peaks = peaks[ix]
prom = prom[ix]
width = width[ix]

print(f"Found {len(peaks)} total peaks")

# ----------------------------------------------------------
# 4. Data-driven Steinmetz cluster detection
#    (your data's epileptiform events = high prominence, narrow width)
# ----------------------------------------------------------
prom_mean = np.mean(prom)
prom_std  = np.std(prom)

# Epileptiform if prominence is unusually large and width is narrow
big_idx = (prom > .3) & (width < .4)

print(f"Detected {big_idx.sum()} large epileptiform events")
plt.figure(figsize=(8,2))
plt.subplot(121)
df = [x[7:] for x in dff_list]
dff  = np.concatenate(df,0)
t = np.arange(len(dff)) * dt_si

plt.plot(t, dff, lw=1)
plt.plot(t, dff, 'k',lw=1);
plt.xlim((0,23));plt.ylim((-.1,.5))
plt.xlabel('time (s)')
plt.ylabel('ΔF/F')
plt.title(mouse + ' ' + session)


# ----------------------------------------------------------
# 6. PLOT: Prominence vs Width (Steinmetz-style)
# ----------------------------------------------------------
plt.subplot(143)
plt.scatter(width, prom, s=10, color='gray', alpha=0.3, label='Background')
plt.scatter(width[big_idx], prom[big_idx], s=25, color='red', label='Epileptiform')
plt.xlabel('Width (s)')
plt.ylabel('Prominence (ΔF/F)')
plt.tight_layout()
plt.xlim((0,.6));
plt.ylim((0,1))

# ----------------------------------------------------------
# 7. Event-triggered average (only true epileptiform events)
# ----------------------------------------------------------
win = int(0.5 / dt_si)   # ±0.5 s window
event_peaks = peaks[big_idx]

aligned = []
for p in event_peaks:
    if p - win >= 0 and p + win < len(dff):
        aligned.append(dff[p-win:p+win])

aligned = np.array(aligned)

if len(aligned) > 0:
    tt = np.arange(-win, win) * dt_si    
    plt.subplot(144)
    plt.plot(tt, aligned.T, color='gray', alpha=0.25)
    plt.plot(tt, aligned.mean(axis=0), color='red', lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('ΔF/F')
    plt.ylim((-.1,.5))

    plt.tight_layout()
    plt.show()
else:
    print("No large events to align.")
