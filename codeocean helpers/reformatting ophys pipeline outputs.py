# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:16:33 2025

@author: kayvon.daie
"""

import h5py
import matplotlib.pyplot as plt
import os
import json

# Path to your HDF5 file
file_dff             = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_dff.h5'
file_extraction      = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_extraction.h5'
file_trial_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/trial_locations.json'
file_epoch_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/epoch_locations.json'
# Open the HDF5 file and load all neuron traces
with h5py.File(file_dff, 'r') as f:
    Ftrace = f['data'][:]  # Access all rows and all timepoints

# Optional: check shape
print(f"Ftrace shape: {Ftrace.shape}")  # Should be (neurons, timepoints)
#%%
# Open and load the JSON file
with open(file_epoch_locations, "r") as f:
    epoch_locations = json.load(f)

# Print the content to inspect
print(epoch_locations)
start, end = epoch_locations["spont_slm"]
print("Start:", start)
print("End:", end)

#%%
import json

# File paths


#tif_file = "/data/single-plane-ophys_740369_2025-02-04_17-33-11_processed_2025-02-05_08-01-42/740369_2025-02-04_17-33-11/motion_correction/trial_locations.json"
tif_file = '//allen/aind/scratch/BCI/2p-raw/BCI93/020425/pophys/neuron15_00002.tif'
# Load JSON files
with open(file_epoch_locations, 'r') as f:
    epoch_data = json.load(f)

with open(file_trial_locations, 'r') as f:
    tif_data = json.load(f)

# Get first epoch and its index range
first_epoch = list(epoch_data.keys())[3]  # e.g., "spont"
start_idx, end_idx = epoch_data[first_epoch]  # e.g., [0, 13526]

# Extract TIFFs that overlap with the first epoch's range
matching_tifs = [
    tif for tif, (tif_start, tif_end) in tif_data.items() 
    if not (tif_end < start_idx or tif_start > end_idx)  # Overlapping condition
]

#%%
import numpy as np

# Create a list to store extracted neuron activity for each matching TIFF
Fstim = []
ops = dict()
ops2 = dict()
ops['frames_per_file'] = []
ops2['frames_per_file'] = []
for i in range(len(matching_tifs)):
    tif_start, tif_end = tif_data[matching_tifs[i]]  # Get start and end frames
    ops['frames_per_file'].append(tif_end - tif_start)
    ind = list(range(tif_start,tif_end));
    ops2['frames_per_file'].append(len(ind))
    # Extract the corresponding frames from Ftrace
    extracted_frames = Ftrace[:, ind[0]-5:ind[-1]+ 1]  # All neurons, frames in range
    
    # Append extracted data (keeping it as a NumPy array)
    Fstim.append(extracted_frames)

# Print summary
print(f"Fstim contains {len(Fstim)} entries, one for each matching TIFF.")
print(f"Example entry shape: {Fstim[0].shape} (neurons x frames)")
#%%

import numpy as np

# Define constants
max_frames = 50  # Fixed frame length
num_neurons = Ftrace.shape[0]  # Number of neurons
num_tifs = len(Fstim) - 1  # Exclude the last trial

# Initialize a (50, neurons, TIFFs) array filled with NaN
Fstim_padded = np.full((max_frames, num_neurons, num_tifs), np.nan)

# Copy each TIFF's data into the new array, handling cases where frames > 50
for i, extracted_frames in enumerate(Fstim[:-1]):  # Exclude last trial
    frame_count = extracted_frames.shape[1]  # Number of frames in this TIFF

    if frame_count > max_frames:
        # Trim to 50 frames if too long
        extracted_frames = extracted_frames[:, :max_frames]
    elif frame_count < max_frames:
        # Pad with NaNs if too short
        pad_width = max_frames - frame_count
        extracted_frames = np.pad(extracted_frames, ((0, 0), (0, pad_width)), constant_values=np.nan)

    # Store in padded array
    Fstim_padded[:, :, i] = extracted_frames.T  # Transpose to fit (time, neuron)

# Print final shape
print(f"Fstim_padded shape: {Fstim_padded.shape} (time x neurons x TIFFs, excluding last trial)")
Fstim = Fstim_padded.copy()

#%%
import extract_scanimage_metadata
file = "/data/single-plane-ophys_740369_2025-02-04_17-33-11/pophys/spont_slm_00001.tif"
file = '//allen/aind/scratch/BCI/2p-raw/BCI93/020425/pophys/spontpost_slm_00001.tif'
siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)

photostim_groups = siHeader['metadata']['json']['RoiGroups']['photostimRoiGroups']
seq = siHeader['metadata']['hPhotostim']['sequenceSelectedStimuli']
seq_clean = seq.strip('[]')
if ';' in seq_clean:
    list_nums = seq_clean.split(';')
else:
    list_nums = seq_clean.split()
seq = [int(num) for num in list_nums if num]
seq = seq*40
seqPos = int(siHeader['metadata']['hPhotostim']['sequencePosition'])-1;
seq = seq[seqPos:]
seq = np.asarray(seq)

#%%
import h5py

# Your specific file path:
# Recursive function to inspect contents clearly:
def inspect_h5(group, prefix=''):
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            print(f"\nGroup: {path}")
            # print attributes if available
            for attr in item.attrs:
                print(f"  Attribute - {attr}: {item.attrs[attr]}")
            # recursively inspect subgroup
            inspect_h5(item, path)
        elif isinstance(item, h5py.Dataset):
            print(f"Dataset: {path}, shape: {item.shape}, dtype: {item.dtype}")
            # print attributes of dataset
            for attr in item.attrs:
                print(f"  Attribute - {attr}: {item.attrs[attr]}")

# Inspect the file
with h5py.File(file_extraction, 'r') as f:
    print("Root-level datasets and groups:")
    inspect_h5(f)
#%%
import sparse
import h5py

with h5py.File(file_extraction) as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]
pixelmasks = sparse.COO(coords, data, shape).todense()

import numpy as np
import matplotlib.pyplot as plt
plt.imshow(np.nanmean(pixelmasks,axis=0))

#%%
stat = dict()
for ci in range(pixelmasks.shape[0]):
    ypix, xpix = np.nonzero(pixelmasks[ci, :, :])
    stat[ci] = {'ypix': ypix, 'xpix': xpix}

#%%
#trip detection doesn't work, expects raw fluorescence which isn't available in the CO pipeline (yet?)
# Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw = ddc.stimDist_single_cell(ops,Ftrace.copy()*20,siHeader,stat)

#%%
deg = siHeader['metadata']['hRoiManager']['imagingFovDeg']
g = [i for i in range(len(deg)) if deg.startswith(" ",i)]
gg = [i for i in range(len(deg)) if deg.startswith(";",i)]
for i in gg:
    g.append(i)
g = np.sort(g)
num = [];
for i in range(len(g)-1):
    num.append(float(deg[g[i]+1:g[i+1]]))
dim = int(siHeader['metadata']['hRoiManager']['linesPerFrame']),int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
degRange = (num[4] - num[0],num[1] - num[5])
#degRange = np.max(num) - np.min(num)
#pixPerDeg = dim/degRange
pixPerDeg = np.array(dim) / np.array(degRange)

centroidX = []
centroidY = []
for i in range(len(stat)):
    centroidX.append(np.mean(stat[i]['xpix']))
    centroidY.append(np.mean(stat[i]['ypix']))

favg = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
favg_raw = np.zeros((Fstim.shape[0],Fstim.shape[1],len(photostim_groups)))
stimDist = np.zeros([Fstim.shape[1],len(photostim_groups)])
slmDist = np.zeros([Fstim.shape[1],len(photostim_groups)])

coordinates = photostim_groups[0]['rois'][1]['scanfields']['slmPattern']
coordinates = np.array([[0, 0, 0, 0]])
coordinates = np.asarray(coordinates)
if np.ndim(coordinates) == 1:
    coordinates = coordinates.reshape(1,-1)
xy = coordinates[:,:2] + photostim_groups[0]['rois'][1]['scanfields']['centerXY']
stimPos = np.zeros(np.shape(xy))
stimPosition = np.zeros([stimPos.shape[0],stimPos.shape[1],len(photostim_groups)])

#%%
seq = seq[0:Fstim.shape[2]]
for gi in range(len(photostim_groups)):        
    coordinates = photostim_groups[gi]['rois'][1]['scanfields']['slmPattern']
    coordinates = np.array([[0, 0, 0, 0]])
    coordinates = np.asarray(coordinates)
    if np.ndim(coordinates) == 1:
        coordinates = np.asarray(coordinates)
        coordinates = coordinates.reshape(1,-1)
    galvo = photostim_groups[gi]['rois'][1]['scanfields']['centerXY']
    
    coordinates = np.asarray(coordinates)

    xy = coordinates[:,:2] + galvo
    xygalvo = coordinates[:,:2]*0 + galvo
    stimPos = np.zeros(np.shape(xy))
    galvoPos = np.zeros(np.shape(xy))
    for i in range(np.shape(xy)[0]):
        stimPos[i,:] = np.array(xy[i,:]-[num[-1], num[0]])*pixPerDeg
        galvoPos[i,:] = np.array(xygalvo[i,:]-[num[-1], num[0]])*pixPerDeg
    sd = np.zeros([np.shape(xy)[0],favg.shape[1]])        
    for i in range(np.shape(xy)[0]):
        for j in range(favg.shape[1]):
            sd[i,j] = np.sqrt(sum((stimPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))
            slmDist[j,gi] = np.sqrt(sum((galvoPos[i,:] - np.asarray([centroidX[j], centroidY[j]]))**2))                
    stimDist[:,gi] = np.min(sd,axis=0)
    ind = np.where(seq == gi+1)[0]
    favg[:,:,gi] = np.nanmean(Fstim[:,:,ind],axis = 2)
    stimPosition[:,:,gi] = stimPos
#%%
plt.plot(np.nanmean(np.nanmean(favg,axis=2),axis=1))
#%%
amp = np.nanmean(favg[11:18,:,:],axis=0) - np.nanmean(favg[0:4,:,:],axis=0)
plt.plot(stimDist.flatten(),amp.flatten(),'k.')
plt.xlabel('Distance from photostim')
plt.ylabel('Response amp.')
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
t = np.arange(0,dt_si*favg.shape[0],dt_si)
t = t - t[4]
for gi in range(12):
    plt.subplot(3,4,gi+1)
    cl = np.argmin(stimDist[:,gi]);
    plt.plot(t,favg[:,cl,gi])
    plt.tight_layout()

#%%
strt = 0;
numTrl = len(ops)
AA = np.zeros(numTrl)
 for ti in range(numTrl):
     pre_pad = np.arange(strt-pre,strt)
     ind = list(range(strt,strt+ops['frames_per_file'][ti]))
     strt = ind[-1]+1
     post_pad = np.arange(ind[-1]+1,ind[-1]+post)
     ind = np.concatenate((pre_pad,np.asarray(ind)),axis=0)
     ind = np.concatenate((ind,post_pad),axis = 0)
     ind[ind > F.shape[1]-1] = F.shape[1]-1;
     ind[ind < 0] = 0
     stimID[ind[pre+1]] = seq[ti]
     a = F[:,ind].T
     g = F[:,ind].T
     bl = np.tile(np.mean(a[0:pre,:],axis = 0),(a.shape[0],1))
     a = (a-bl) / bl
     if a.shape[0]>Fstim.shape[0]:
         a = a[0:Fstim.shape[0],:]
     Fstim[0:a.shape[0],:,ti] = a
     try:
         Fstim_raw[0:a.shape[0],:,ti] = g
     except ValueError as e:
         print(f"Skipping trial {ti} due to shape mismatch: {e}")