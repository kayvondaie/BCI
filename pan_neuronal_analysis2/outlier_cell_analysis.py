import numpy as np
import matplotlib.pyplot as plt
import h5py
from BCI_data_helpers import compute_amp_from_photostim

# Paths
folder = '/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal/BCI93/020425/'
h5_folder = folder + 'pophys/'

# Load suite2p ROIs
stat = np.load(folder + 'pophys/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
iscell = np.load(folder + 'pophys/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
cells = np.where(iscell[:, 0] == 1)[0]

# Load photostim data from h5
data = {'photostim': {}, 'photostim2': {}}
for key in ['stimDist', 'favg_raw']:
    with h5py.File(h5_folder + 'data_photostim.h5', 'r') as f:
        data['photostim'][key] = f[key][:]
    with h5py.File(h5_folder + 'data_photostim2.h5', 'r') as f:
        data['photostim2'][key] = f[key][:]
with h5py.File(h5_folder + 'data_main.h5', 'r') as f:
    data['dt_si'] = f['dt_si'][()]

# Compute amp and stimDist
AMP, stimDist = compute_amp_from_photostim('BCI93', data, h5_folder)
amp = AMP[0]

# Mean non-target amp per neuron (excluding stimDist < 30 um)
amp_masked = amp.copy()
amp_masked[stimDist < 30] = np.nan
mean_amp_nontarg = np.nanmean(amp_masked, axis=1)

#%% Classify outliers and plot
cells = cells[cells < amp.shape[0]]

mu = np.nanmean(mean_amp_nontarg[cells])
sd = np.nanstd(mean_amp_nontarg[cells])
n_sd = 2
threshold = mu + n_sd * sd
outlier_idx = cells[mean_amp_nontarg[cells] > threshold]
rest_idx = cells[mean_amp_nontarg[cells] <= threshold]
print(f'Threshold: {mu:.4f} + {n_sd}*{sd:.4f} = {threshold:.4f}, n_outliers: {len(outlier_idx)}, n_rest: {len(rest_idx)}')

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(stimDist[rest_idx, :], amp[rest_idx, :], 'k.', markersize=2)
plt.ylim(-1, 3)
plt.xlabel('Distance (um)')
plt.ylabel('Amplitude')
plt.title(f'Rest (n={len(rest_idx)})')

plt.subplot(122)
plt.plot(stimDist[outlier_idx, :], amp[outlier_idx, :], 'r.', markersize=2)
plt.ylim(-1, 3)
plt.xlabel('Distance (um)')
plt.title(f'Outliers (n={len(outlier_idx)})')

plt.suptitle('Amp vs stimDist')
plt.tight_layout()
plt.show()
