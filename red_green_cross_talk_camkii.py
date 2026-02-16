# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:58:56 2025

@author: kayvon.daie
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths to your Suite2p outputs
# ------------------------------------------------------------
f_green_path = '//allen/aind/scratch/BCI/2p-raw/CK003/112625/pophys/suite2p_BCI/plane0/F.npy'
f_red_path   = '//allen/aind/scratch/BCI/2p-raw/CK003/112625/pophys/suite2p_BCI/plane0/F_chan2.npy'

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
F  = np.load(f_green_path)      # green channel (ROIs × frames)
F2 = np.load(f_red_path)        # red channel

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
# F, F2: (n_cells, n_frames)
g = F.mean(axis=1)   # mean green per ROI
r = F2.mean(axis=1)  # mean red per ROI

# stp = 60*60*15
# g = medfilt(np.nanmean(F[:,0:stp],0),111);
# r = np.nanmean(F2[:,0:stp],0)

# --- unweighted linear regression with intercept ---
# build design matrix [r_i, 1]
X = np.vstack([r, np.ones_like(r)]).T       # shape (n_cells, 2)
beta, *_ = np.linalg.lstsq(X, g, rcond=None)
a, b = beta  # slope and intercept

print(f"slope a = {a:.3f}, intercept b = {b:.3f}")

# --- plot with non-zero intercept line ---
plt.figure(figsize=(6,6))
plt.scatter(r, g, s=10)
xline = np.linspace(r.min(), r.max(), 200)
yline = a * xline + b
plt.plot(xline, yline, 'r', label=f'y = {a:.3f} x + {b:.3f}')
plt.xlabel("mean red (per ROI)")
plt.ylabel("mean green (per ROI)")
plt.title("ROI-wise mean brightness relationship")
plt.legend()
plt.tight_layout()
plt.show()
#%%
b = np.argsort(-r)
b = b[8]
#b = 44
plt.figure(figsize = (8,3))
# plt.subplot(422)
# plt.plot(F2[b,:],'m',lw = .2)
# plt.plot(F[b,:],'g',lw = .2)
# plt.subplot(424);
# plt.plot(F[b,:] - F2[b,:]*.25,'g',lw = .2)

plt.subplot(131)
plt.scatter(r, g, s=10)
xline = np.linspace(r.min(), r.max(), 200)
a, bb = beta  # slope and intercept
yline = a * xline + bb
plt.plot(xline, yline, 'r', label=f'y = {a:.3f} x + {bb:.3f}')
plt.xlabel("mean red (per ROI)")
plt.ylabel("mean green (per ROI)")
plt.legend()

plt.subplot(132);
plt.imshow(F,aspect = 'auto',vmin = np.percentile(F.flatten(),5),vmax = np.percentile(F.flatten(),95))
plt.title('Green raw')
plt.xlabel('Frame #')
plt.ylabel('ROI')

plt.subplot(133);
fc = F - F2*.6
plt.imshow(fc,aspect = 'auto',vmin =np.percentile(fc.flatten(),5),vmax = np.percentile(fc.flatten(),95))
plt.title('Green corrected')
plt.xlabel('Frame #')
plt.tight_layout()

