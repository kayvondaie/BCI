# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:37:20 2025

@author: kayvon.daie
"""


import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]
mice = ["BCI102"]
si = 5
for mi in range(len(mice)):
    
    HI = []
    RT = []
    HIT = []
    HIa= []
    HIb = []
    HIc = []
    DOT = []
    TRL = []
    THR = []
    RPE = []
    FIT = []
    GRP = []
    RPE_FIT = []
    DW = []
    XALL,YALL = [],[]
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'lasso'     #ridge, pinv
    alpha         =  100        #only used for ridge
    num_bins      =  10        # number of bins to calculate correlations
    tau_elig      =  10
    shuffle       =  0
    plotting      =  1
    mouse = mice[mi]
    session_inds = np.where((list_of_dirs['Mouse'] == mouse) & (list_of_dirs['Has data_main.npy']==True))[0]
    #for sii in range(len(session_inds)):
    for sii in range(si,si+1):
        print(sii)
        mouse = list_of_dirs['Mouse'][session_inds[sii]]
        session = list_of_dirs['Session'][session_inds[sii]]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        photostim_keys = ['stimDist', 'favg_raw']
        bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
        try:
            data = ddct.load_hdf5(folder, bci_keys, photostim_keys)
        except FileNotFoundError:
            print(f"Skipping session {mouse} {session} — file not found.")
            continue  # <--- Skip to next session
        BCI_thresholds = data['BCI_thresholds']
        AMP = []
        siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
        umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
        AMP, stimDist, FAVG = compute_amp_from_photostim(mouse, data, folder,return_favg = True)
        ops = np.load(os.path.join(folder, 'suite2p_BCI/plane0/ops.npy'), allow_pickle=True).tolist()

        stat = np.load(os.path.join(folder, 'suite2p_BCI/plane0/stat.npy'), allow_pickle=True).tolist()
        iscell = np.load(os.path.join(folder, 'suite2p_BCI/plane0/iscell.npy'), allow_pickle=True)  # keep as ndarray
        
        cell_inds = np.where(iscell[:, 0] == 1)[0]
        stat = [stat[i] for i in cell_inds]   # filtered stat list
#%%


#%%
scores = [np.mean(np.sort(AMP[epoch][:, gi])[-5:]) for gi in range(AMP[epoch].shape[1])]
best_gi = np.argmax(scores)
epoch = 1
gi = best_gi
x = AMP[epoch][:, gi].copy()

# exclude near cells
near = np.where((stimDist[:, gi] < 30))[0]
far = np.where((stimDist[:, gi] > 100))[0]
x[near] = 0;x[far] = 0

# sort and pick top 5
b = np.argsort(-x)
top5 = b[:5]

# target cell
targ = np.argmin(stimDist[:, gi])

plt.figure(figsize=(8, 2))


# zoom: compute centroid of target ROI
yc = np.mean(stat[targ]['ypix'])
xc = np.mean(stat[targ]['xpix'])
zoom = 100  # pixels half-width of zoom window
plt.xlim(xc - zoom, xc + zoom)
plt.ylim(yc + zoom, yc - zoom)  # flip y to match image coords

# --- Bottom row: traces ---
for j, cell in enumerate(top5):
    plt.subplot(1, 5, j + 1)
    plt.plot(FAVG[epoch][0:50, cell, gi], 'k')
    plt.title(f"cell {cell}", fontsize=8)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

#%%
# --- Background image ---
rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
normalized_img = img / img.max()
rgba_img[..., 0] = normalized_img
rgba_img[..., 1] = normalized_img
rgba_img[..., 2] = normalized_img
rgba_img[..., 3] = 1.0

# --- Overlay ---
overlay = np.zeros_like(rgba_img)

# Target cell (magenta: R + B)
roi = stat[targ]
ypix, xpix = roi['ypix'], roi['xpix']
overlay[ypix, xpix, 0] = 1.0   # Red
overlay[ypix, xpix, 2] = 1.0   # Blue
overlay[ypix, xpix, 3] = 0.5   # Alpha

# Top5 non-targets (blue)
for cell in top5:
    if cell == targ:
        continue
    roi = stat[cell]
    ypix, xpix = roi['ypix'], roi['xpix']
    overlay[ypix, xpix, 0] = 1.0   # Blue
    overlay[ypix, xpix, 3] = 0.5   # Alpha

# --- Zoom around target cell ---
yc = np.mean(stat[targ]['ypix'])
xc = np.mean(stat[targ]['xpix'])
zoom = 80  # half-width of zoom window in pixels

fig, ax = plt.subplots()
ax.imshow(img/5, cmap='gray', vmin=0, vmax=50)
ax.imshow(overlay, alpha=0.5)

ax.set_xlim(xc - zoom, xc + zoom)
ax.set_ylim(yc + zoom, yc - zoom)  # flip y-axis
ax.axis('off')

# --- Scalebar (100 µm) ---
um_per_pix = 1000 / img.shape[1]
scalebar_pix = 100 / um_per_pix

# Place scalebar in bottom-left of zoomed area
x_start = xc - zoom + 10
x_end   = x_start + scalebar_pix
y_pos   = yc + zoom - 20

ax.plot([x_start, x_end], [y_pos, y_pos], color='w', linewidth=3)
ax.text((x_start + x_end)/2, y_pos - 10, "100 µm",
        color='w', ha='center', va='bottom', fontsize=8)

plt.show()
