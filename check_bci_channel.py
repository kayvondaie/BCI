# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:41:59 2026

@author: kayvon.daie
"""


mouse = "BCI116";
session = '020126';
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
siHeader = np.load(folder +r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
siHeader['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois'][0]['scanfields']['channel']
#%%
import os
import numpy as np
from pathlib import Path

mouse = "BCI120"
base_folder = Path(r'//allen/aind/scratch/BCI/2p-raw/') / mouse

# Dictionary to store results
channel_data = {}

# Find all session folders (assuming they're direct subdirectories of the mouse folder)
if base_folder.exists():
    for session_folder in base_folder.iterdir():
        if session_folder.is_dir():
            session = session_folder.name
            
            # Construct the path to siHeader.npy
            siHeader_path = session_folder / 'pophys' / 'suite2p_BCI' / 'plane0' / 'siHeader.npy'
            
            # Check if the file exists
            if siHeader_path.exists():
                try:
                    # Load the header
                    siHeader = np.load(siHeader_path, allow_pickle=True).tolist()
                    
                    # Extract the channel information
                    channel = siHeader['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois'][0]['scanfields']['channel']
                    
                    # Store in dictionary
                    channel_data[session] = channel
                    print(f"Session {session}: channel = {channel}")
                    
                except (KeyError, IndexError, Exception) as e:
                    print(f"Session {session}: Error extracting channel - {e}")
            else:
                print(f"Session {session}: siHeader.npy not found")
else:
    print(f"Base folder not found: {base_folder}")

# Display all collected data
print("\n=== Summary ===")
for session, channel in channel_data.items():
    print(f"{session}: {channel}")