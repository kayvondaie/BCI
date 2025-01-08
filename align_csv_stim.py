# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:35:39 2023

@author: scanimage
"""

import numpy as np

# Read data from file
# filename = folder + ops['tiff_list'][0][0:-4] + r'.stim'
def read_stim(filename):
    hFile = open(filename, 'rb')  # Use 'rb' for reading binary file
    phtstimdata = np.fromfile(hFile, dtype=np.float32)
    hFile.close()
    
    # Sanity check for file size
    datarecordsize = 3
    lgth = len(phtstimdata)
    if lgth % datarecordsize != 0:
        print('Unexpected size of photostim log file')
        lgth = (lgth // datarecordsize) * datarecordsize
        phtstimdata = phtstimdata[:lgth]
    
    # Reshape the data
    phtstimdata = np.reshape(phtstimdata, (lgth // datarecordsize, datarecordsize))
    
    # Extract x, y, and beam power
    out = {}
    out['X'] = phtstimdata[:, 0]
    out['Y'] = phtstimdata[:, 1]
    out['Beam'] = phtstimdata[:, 2]
    return out


def find_blocks(data, zero_threshold):
    blocks = []
    block_start = None
    zero_count = 0

    for i, value in enumerate(data):
        if value == 1:
            if block_start is None:  # Start of a new block
                block_start = i
            zero_count = 0  # Reset zero count
        elif value == 0 and block_start is not None:
            zero_count += 1
            if zero_count > zero_threshold:  # End the current block
                blocks.append((block_start, i - zero_count))
                block_start = None  # Reset block start
                zero_count = 0  # Reset zero count

    # Handle case where the last block doesn't have enough trailing zeros to trigger an end
    if block_start is not None:
        blocks.append((block_start, len(data)-1))

    return blocks

def laser_on_off(stim_data,blocks,zero_threshold,dt):
    laser_signal = [1 if item != 0 else 0 for item in stim_data['Beam']]
    time_points = np.arange(len(stim_data['Beam'])) * dt
    on_off_times = [(time_points[start], time_points[end]) for start, end in blocks]
    return on_off_times

def process_rois(rois):
    rois.set_index('frameNumber', inplace=True)
    full_index = range(rois.index.min(), rois.index.max() + 1)
    rois = rois.reindex(full_index).interpolate(method='linear').reset_index()
    return rois

def laser_status(rois, on_off_times):
    timestamps = rois['timestamp']
    laser_status = []
    for i in range(len(timestamps) - 1):
        frame_times = np.arange(timestamps[i], timestamps[i + 1], step=0.001)
        overlap = int(any(start <= t < end for t in frame_times for start, end in on_off_times))
        laser_status.append(overlap)
    laser_status.append(0)  # no laser last frame
    rois.insert(2, 'laser_status', laser_status)
    return rois


