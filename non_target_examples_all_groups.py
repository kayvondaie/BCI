import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import math
import session_counting
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import ttest_1samp
import re

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
        photostim_keys = ['stimDist', 'favg_raw','Fstim','seq']
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
        AMP, stimDist, FAVG,_ = compute_amp_from_photostim(mouse, data, folder,return_favg = True)
        ops = np.load(os.path.join(folder, 'suite2p_BCI/plane0/ops.npy'), allow_pickle=True).tolist()

        stat = np.load(os.path.join(folder, 'suite2p_BCI/plane0/stat.npy'), allow_pickle=True).tolist()
        iscell = np.load(os.path.join(folder, 'suite2p_BCI/plane0/iscell.npy'), allow_pickle=True)  # keep as ndarray
        
        cell_inds = np.where(iscell[:, 0] == 1)[0]
        stat = [stat[i] for i in cell_inds]   # filtered stat list


#%%

epoch = 'photostim'
stimDist = data[epoch]['stimDist']
Fstim = data[epoch]['Fstim']  # T x cells x trials
favg = data[epoch]['favg_raw']  # T x cells x groups
seq = data[epoch]['seq'] - 1
G = stimDist.shape[1]

# --- Compute amplitudes and p-values
amp = np.nanmean(favg[26:35, :, :], axis=0) - np.nanmean(favg[10:19, :, :], axis=0)
pv = np.full((Fstim.shape[1], G), np.nan)

for gi in range(G):    
    ind = np.where(seq == gi)[0]
    post = np.nanmean(Fstim[27:35, :, ind], axis=0)
    pre  = np.nanmean(Fstim[10:16, :, ind], axis=0)
    valid = np.where(np.sum(np.isnan(pre[0:10, :]), axis=0) == 0)[0]
    if len(valid) > 0:
        _, p_value = ttest_ind(post[:, valid], pre[:, valid], axis=1)
        pv[:, gi] = p_value
#%%
import math
from scipy.signal import medfilt

# --- Select n_best smallest p-values with distance > threshold
dist_thresh = 25
n_best = 25
mask = stimDist > dist_thresh
pv_masked = np.where(mask, pv, np.nan)
flat_idx = np.argsort(pv_masked.flatten())[:n_best]
cell_gi_pairs = [np.unravel_index(idx, pv.shape) for idx in flat_idx]

# --- Tunable parameters
artifact_start = 19
artifact_end   = 24
trim_len = 50
medfilt_kernel = 3   # must be odd (e.g. 3, 5, 7)

# Grid layout
n_cols = math.ceil(np.sqrt(n_best))
n_rows = math.ceil(n_best / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5*n_cols, 1.5*n_rows))
axes = axes.flatten()

for ax, (cell_idx, gi_idx) in zip(axes, cell_gi_pairs):
    ind = np.where(seq == gi_idx)[0]
    trials = Fstim[:, cell_idx, ind]

    # Interpolate stim artifact
    trials_interp = trials.copy()
    for k in range(trials.shape[1]):
        y = trials[:, k].copy()
        lx, rx = artifact_start-1, artifact_end+1
        if not np.isnan(y[lx]) and not np.isnan(y[rx]):
            ramp = np.linspace(y[lx], y[rx], artifact_end - artifact_start + 1)
            y[artifact_start:artifact_end+1] = ramp
        trials_interp[:, k] = y

    # Mean ± SEM
    mean_trace = np.nanmean(trials_interp, axis=1)
    sem_trace  = np.nanstd(trials_interp, axis=1) / np.sqrt(trials_interp.shape[1])

    # Apply median filter to smooth traces
    mean_trace = medfilt(mean_trace, kernel_size=medfilt_kernel)
    sem_trace  = medfilt(sem_trace, kernel_size=medfilt_kernel)

    # Trim
    x = np.arange(10,trim_len)
    mean_trim = mean_trace[10:trim_len]
    sem_trim  = sem_trace[10:trim_len]
    if np.nanmean(mean_trim)<0:
        c = 'b'
    else:
        c = 'r'
    # Plot
    ax.plot(x, mean_trim, color='k', lw=1)
    ax.fill_between(x, mean_trim - sem_trim, mean_trim + sem_trim, color=c, alpha=0.3)
    ax.axvspan(artifact_start, artifact_end, color='red', alpha=0.1)

    ax.axis('off')

# --- Scale bars on bottom-left panel
scalex = 6   # frames
scaley = 0.1  # ΔF/F units
sb_ax = axes[-1]
ylim = sb_ax.get_ylim()
xlim = sb_ax.get_xlim()

sb_ax.plot([xlim[0]+5, xlim[0]+5+scalex], [ylim[0]+0.05, ylim[0]+0.05], 'k', lw=2)
sb_ax.plot([xlim[0]+5, xlim[0]+5], [ylim[0]+0.05, ylim[0]+0.05+scaley], 'k', lw=2)

fig.tight_layout(pad=1)
plt.show()
