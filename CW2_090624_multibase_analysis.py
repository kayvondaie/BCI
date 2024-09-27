# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:00:38 2024

@author: kayvon.daie
"""
import folder_props_fun    
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']
print(bases)
ind = input('pick indices of bases for BCI1, BCI2')
ind = np.fromstring(ind[1:-1], sep=',')
#%%
import tifffile as tiff
import time
# Extract base names from the TIFF files
bases = [x[:x.rfind('_')] for x in folder_props['siFiles']]
base = bases[int(ind[0])]

siFiles = [x for x, b in zip(folder_props['siFiles'], bases) if b == base]
num_trials = len(siFiles)
siFiles = siFiles[:num_trials]
len_files = np.zeros(len(siFiles))

# Iterate through each TIFF file and count the frames
for i, si_file in enumerate(siFiles):
    start_time = time.time()
    tiff_file_path = os.path.join(folder, si_file)
    with tiff.TiffFile(tiff_file_path) as tiff_obj:
        frame_count = len(tiff_obj.pages)
    len_files[i] = frame_count
    print(f"Time for {si_file}: {time.time() - start_time} seconds")
#%%
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)        
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()

#%%
import glob
import pandas as pd
base = bases[int(ind[0])]
csv_files = glob.glob(os.path.join(folder, base+'_IntegrationRois' + '_*.csv'))
csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
csv_data = []
for i in range(len(csv_files)):
    csv_file = csv_files[i]
    csv_data.append(pd.read_csv(csv_file))        
    data['roi_csv'] = np.concatenate(csv_data)
#%%   
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])

behav_file = folder + r'/behavior/' + folder[-7:-1]+r'-bpod_zaber.npy'
_, _, _, _,rt1 = ddc.create_zaber_info(behav_file,bases[int(ind[0])],ops,dt_si)
rt = np.array([x[0] if len(x) > 0 else np.nan for x in rt1])
rew = ~np.isnan(rt)
roi = np.copy(data['roi_csv'])
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)
from scipy.interpolate import interp1d
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)
#%%
strt = 0  
dt_si = np.median(np.diff(roi[:, 0]))
fcn = np.empty((350, len(len_files) - 1))
FCN = np.empty((350, len(len_files) - 1))
t_si = np.empty((350, len(len_files) - 1))
strts = np.empty(len(rt1) - 1, dtype=int)  # Initialize with the correct length
avg = np.empty((len(rt1) - 1,))
BCI_threshold = data['BCI_thresholds'][:,55]
cn_ind = data['cn_csv_index'][0]

fun = lambda x: np.minimum((x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)[0]) * 3.3, 3.3)
for i in range(len(rt1) - 1):
    strts[i] = strt  # Literal translation of strts(i) = strt
    ind = np.arange(strt, strt + len_files[i], dtype=int)  # Ensure ind is an array of integers
    ind = np.clip(ind, 0, len(roi_interp) - 1)
    # Extract and process roi_interp data for fcn and t_si
    a = roi_interp[ind.astype(int), cn_ind + 2]

    # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
    a_padded = np.concatenate([a, np.full(400, np.nan)])
    fcn[:, i] = a_padded[:350]
    FCN[:, i] = a_padded[:350]

    # Repeat for t_si (first column of roi_interp)
    a = roi_interp[ind.astype(int), 0]
    a = a - a[0]  # Shift time values

    # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
    a_padded = np.concatenate([a, np.full(400, np.nan)])
    t_si[:, i] = a_padded[:350]

    strt = strt + len_files[i]  # Update strt for the next trial

    # Determine the stopping point
    if rew[i]:
        stp = np.max(np.where(t_si[:, i] < rt[i])[0])
    else:
        stp = t_si.shape[0]

    # Calculate average for this trial
    avg[i] = np.nanmean(fun(fcn[:stp, i]))
  
