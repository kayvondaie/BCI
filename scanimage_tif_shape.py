# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:48:26 2025

@author: kayvon.daie
"""

from ScanImageTiffReader import ScanImageTiffReader
import numpy as np

tif_path = r'//allen/aind/scratch/BCI/2p-raw/BCI93/011025/pophys/stack_00001.tif'

# Open using ScanImage's own reader
reader = ScanImageTiffReader(tif_path)

# Read the image data as a NumPy array
data = reader.data()

# Parse metadata
meta = reader.metadata()

# Determine basic shape
print("Raw data shape:", data.shape)

# Try to extract ScanImage dimension info from metadata
try:
    lines = [l for l in meta.split('\n') if 'frame' in l.lower() or 'channel' in l.lower() or 'plane' in l.lower()]
    for l in lines:
        print(l)
except Exception as e:
    print("Could not parse metadata:", e)

# Estimate dimension order
# ScanImageTiffReader loads data as (frames, y, x)
frames, ypix, xpix = data.shape
print(f"x pixels: {xpix}")
print(f"y pixels: {ypix}")
print(f"frames:   {frames}")

# You can parse channels/planes from metadata
import re
channels = re.search(r'SI.hChannels.channelSave = (\[.*?\])', meta)
planes = re.search(r'SI.hStackManager.numSlices = (\d+)', meta)

if channels:
    n_channels = len(eval(channels.group(1)))
else:
    n_channels = 1

if planes:
    n_planes = int(planes.group(1))
else:
    n_planes = 1

print(f"channels: {n_channels}")
print(f"planes:   {n_planes}")

reader.close()
