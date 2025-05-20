import numpy as np
import matplotlib.pyplot as plt

sii = 2
mouse = list_of_dirs['Mouse'][session_inds[sii]]
session = list_of_dirs['Session'][session_inds[sii]]
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
photostim_keys = ['stimDist', 'favg_raw', 'Fstim', 'seq']
bci_keys = ['dt_si']
data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
#%%
# Extract data
Fstim = data['photostim']['Fstim']  # shape: [time, cell, stim]
seq = data['photostim']['seq'] - 1
stimDist = data['photostim']['stimDist'] * umPerPix

# Stim group to examine
gi = 10
ind = np.where(seq == gi)[0]

# Cell closest to stim site
cl = np.argmin(stimDist[:, gi])

# Compute amplitude: post - pre average
post = np.arange(18, 22)
pre = np.arange(0, 8)
amp_trl = np.nanmean(Fstim[post, :, :], axis=0) - np.nanmean(Fstim[pre, :, :], axis=0)  # shape: [cell, stim]
f = amp_trl[:, ind]                        # [cell, trials for group gi]
amp_avg = np.nanmean(f, axis=1)           # [cell]

# Find strong responders >30 µm from stim
non = np.where((stimDist[:, gi] > 30) & (stimDist[:, gi] < 100))[0]
b = np.argsort(-amp_avg[non])             # descending sort
ci = non[b[0]]                             # 5th strongest responder

# Plot
plt.figure(figsize=(6, 2))

# Closest cell
plt.subplot(131)
plt.plot(Fstim[:, cl, ind], 'k', linewidth=0.4)
plt.plot(np.nanmean(Fstim[:, cl, ind], axis=1), 'r', linewidth=1)
plt.title(str(round(stimDist[cl,gi])) + 'um')


# Strong responder
plt.subplot(132)
plt.plot(np.nanmean(Fstim[:, ci, ind], axis=1), 'r', linewidth=1)
plt.title(str(round(stimDist[ci,gi])) + 'um')
# Amplitude correlation
plt.subplot(133)
plt.plot(f[cl, :], f[ci, :], 'k.')
plt.xlabel('direct amp')
plt.ylabel('indirect ')

plt.tight_layout()
plt.show()
#%%
SLOPE, R2 = [], []
for gi in range(stimDist.shape[1]):
    ind = np.where(seq == gi)[0]
    
    # Cell closest to stim site
    cl = np.argmin(stimDist[:, gi])
    
    # Compute amplitude: post - pre average
    post = np.arange(18, 22)
    pre = np.arange(0, 8)
    amp_trl = np.nanmean(Fstim[post, :, :], axis=0) - np.nanmean(Fstim[pre, :, :], axis=0)  # shape: [cell, stim]
    f = amp_trl[:, ind]  # shape: [n_cells, n_trials for group gi]
    
    # Filter to non-local cells
    non = np.where((stimDist[:, gi] > -30) & (stimDist[:, gi] < 10000))[0]
    
    # Trial amplitudes of closest cell
    x = f[cl, :]  # shape: [n_trials]
    
    # Initialize array for slopes
    from scipy.stats import linregress
    slopes = []
    r2s = []
    
    for ci in non:
        y = f[ci, :]
        x = f[cl, :]
        valid = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(valid) > 1:
            xv = x[valid]
            yv = y[valid]
            beta = np.dot(xv, yv) / np.dot(xv, xv); r2 = 1 - np.sum((yv - beta * xv)**2) / np.sum(yv**2)
            slopes.append(beta)
            r2s.append(r2)
        else:
            slopes.append(np.nan)
            r2s.append(np.nan)
    
    slopes = np.array(slopes)
    r2 = np.array(r2s)
    R2.append(r2s)
    SLOPE.append(slopes)
