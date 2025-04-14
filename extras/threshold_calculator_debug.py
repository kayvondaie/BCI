folder = r'//allen/aind/scratch/BCI/2p-raw/BCI109/030525/pophys/'
data = np.load(folder + 'data_main.npy',allow_pickle=True)
#%%
import scipy.io
import os
import re
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
stem = siHeader['siBase'][0]

pattern = re.compile(rf'^{re.escape(stem)}_threshold_(\d+)\.mat$')

max_file = None
max_num = -1

for fname in os.listdir(folder):
    match = pattern.match(fname)
    if match:
        num = int(match.group(1))
        if num > max_num:
            max_num = num
            max_file = fname

#%%
filepath = os.path.join(folder, max_file)
bpod = scipy.io.loadmat(filepath)
abc = bpod['abcdef'].squeeze()

# Access fields using field names, like a dict or structured array
bpod_info = abc['bpod_info'].item()[0]
cn_trace = abc['cn_trace'].item()[0]

plt.plot(data['roi_csv'][0:47000,data['cn_csv_index'][0]+2])
plt.plot(cn_trace[0:47000],linewidth=.5)

plt.title(folder)

#%% from 4/8
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:17:13 2025

@author: kayvon.daie
"""

folder = r'//allen/aind/scratch/BCI/2p-raw/BCI109/040225/pophys/'
data = np.load(folder + 'data_main.npy',allow_pickle=True)
#%%
import scipy.io
import os
import re
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
stem = siHeader['siBase'][0]

pattern = re.compile(rf'^{re.escape(stem)}_params_(\d+)\.mat$')

max_file = None
max_num = -1

for fname in os.listdir(folder):
    match = pattern.match(fname)
    if match:
        num = int(match.group(1))
        if num > max_num:
            max_num = num
            max_file = fname
print(max_file)
#%%
filepath = os.path.join(folder, max_file)
bpod = scipy.io.loadmat(filepath)
abc = bpod['temp_params'].squeeze()

# Access fields using field names, like a dict or structured array
bpod_info = abc['bpod_info'].item()[0]
cn_trace = abc['cn_trace'].item()[0]

plt.plot(data['roi_csv'][0:47000,data['cn_csv_index'][0]+2])
plt.plot(cn_trace[0:47000],linewidth=.5)

plt.title(folder)

param_file = filepath
#%%

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load .mat file
aa = loadmat(param_file, simplify_cells=True)
temp_params = aa['temp_params']

output = np.array(temp_params['cn_output'])
cn_trace = np.array(temp_params['cn_trace'])
ROI = data['roi_csv']

time = np.zeros(len(cn_trace))
time[:len(ROI)] = ROI[:, 0]

# Trial structure and reward detection
bpod_info_diff = np.diff(temp_params['bpod_info'])
thresholds = np.array(temp_params['current_thresholds']).T
dt = 0.03

ts = np.concatenate([[1], np.where(bpod_info_diff > 0.1)[0] + 1])  # +1 to convert from diff indices
rew = np.where(bpod_info_diff < -0.1)[0] + 1

rt2 = []
avg = []
tot = []
REW = []
lwr = []
upr = []

for i in range(len(ts) - 1):
    ind = np.arange(ts[i], ts[i + 1])
    times = time[ind]
    times -= times[0]
    a = np.intersect1d(rew, ind)
    
    if a.size > 0:
        REW.append(1)
        a_idx = a[0]
        rt2.append(times[a_idx - ind[0]])
        avg.append(np.nanmean(output[ind[0]:a_idx]))
        tot.append(np.nansum(output[ind[0]:a_idx]))
        lwr.append(thresholds[ind[0], 0])
        upr.append(thresholds[ind[0], 1])
    else:
        rt2.append(np.nan)
        REW.append(0)

# Plot avg vs. rt2
plt.figure()
plt.scatter(avg, rt2)
plt.title('avg vs. rt2')
plt.show()
plt.figure(figsize = (6,3))
thr_cross = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
st = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
rt = rt - st;
thr_cross = thr_cross - st;
plt.subplot(121)
plt.plot(thr_cross[0:100],rt2[0:100],'k.')
plt.xlabel('Thr. cross (bpod)')
plt.ylabel('Thr. cross (SI)')
plt.subplot(122)
plt.plot(rt[0:100],rt2[0:100],'k.')
plt.xlabel('Rew. time (bpod)')
plt.ylabel('Thr. cross (SI)')
plt.tight_layout()
#%%
# Plot upper thresholds
upr = np.array(upr)
lwr = np.array(lwr)
avg = np.array(avg)
rt2 = np.array(rt2)

plt.figure()
plt.plot(upr)
switches = np.where(np.diff(lwr) != 0 | np.diff(upr) != 0)[0]
if switches.size > 0:
    ind = np.arange(switches[-1], len(avg))
    plt.figure()
    plt.scatter(avg[ind], rt2[ind])
    plt.title('avg vs. rt2 (after threshold switch)')


