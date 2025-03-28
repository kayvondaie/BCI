# run_photostim_analysis.py
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:16:33 2025
Main script for photostim analysis

@author: kayvon.daie
"""

import sys
import os

# Add 'codeocean_helpers' folder inside the current directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'codeocean helpers'))

import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import data_dict_create_module_test as ddc  # Your existing module
import ophys_pipeline_helper as oph  # New modular script you're about to create
import glob
import pandas as pd


data = dict()

folder = r'//allen/aind/scratch/BCI/2p-raw/BCI93/020425/pophys/'

file_dff             = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_dff.h5'
file_extraction      = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_extraction.h5'
file_trial_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/trial_locations.json'
file_epoch_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/epoch_locations.json'
behav_file           = '//allen/aind/scratch/BCI/2p-raw/BCI93/020425/behavior/020425-bpod_zaber.npy'


with open(file_epoch_locations, 'r') as f:
    epoch_data = json.load(f)
bases = list(epoch_data.items())
tif_file             = folder + bases[2][0] + '_00001.tif'

epoch_inds = (3,4);
stim_tif = []
stim_tif.append(folder + bases[epoch_inds[0]][0] + '_00001.tif')
stim_tif.append(folder + bases[epoch_inds[1]][0] + '_00001.tif')
             


for index in range(1,3):    
    Ftrace, stat, ops, siHeader = oph.run_all_preprocessing(file_dff, file_extraction, file_trial_locations, epoch_data, epoch_inds[index-1], stim_tif[index-1])
    key_name = f'photostim{index if index > 1 else ""}'
    data[key_name] = dict()
    data[key_name]['Ftrace'] = Ftrace.copy()
    data[key_name]['Fstim'], data[key_name]['seq'], data[key_name]['favg'], data[key_name]['stimDist'], \
    data[key_name]['stimPosition'], data[key_name]['centroidX'], data[key_name]['centroidY'], \
    data[key_name]['slmDist'], data[key_name]['stimID'], data[key_name]['Fstim_raw'], \
    data[key_name]['favg_raw'] = ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat, 0)

#%%
epoch_inds = (2);
Ftrace, stat, ops, siHeader = oph.run_all_preprocessing(file_dff, file_extraction, file_trial_locations, epoch_data, epoch_inds, tif_file)
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
if dt_si < 0.05:
    post = round(10/0.05 * 0.05/dt_si)
    pre = round(2/0.05 * 0.05/dt_si)
else:
    post = round(10/0.05)
    pre = round(2/0.05)
ops['frames_per_file'] = ops['frames_per_file'][0:-1]
data['F'], data['Fraw'],data['df_closedloop'],data['centroidX'],data['centroidY'] = ddc.create_BCI_F(Ftrace,ops,stat,pre,post);   
data['dist'], data['conditioned_neuron_coordinates'], data['conditioned_neuron'], data['cn_csv_index'] = ddc.find_conditioned_neurons(siHeader,stat)
data['dt_si'] = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])
base,_ = list(epoch_data.items())[2]
data['reward_time'], data['step_time'], data['trial_start'], data['SI_start_times'],data['threshold_crossing_time'] = ddc.create_zaber_info(behav_file,base,ops,dt_si)
#%%
numtrl = data['F'].shape[2]
BCI_thresholds = np.full((2, numtrl), np.nan)
base,_ = list(epoch_data.items())[2]

# Iterate over trials and attempt to load the corresponding threshold files
for i in range(numtrl):
    try:
        st = folder + base + r'_threshold_' + str(i+1) + r'.mat'
        
        # Check if the file exists before trying to load it
        if os.path.exists(st):
            threshold_data = scipy.io.loadmat(st)
            BCI_thresholds[:, i] = threshold_data['BCI_threshold'].flatten()
            
    except:
        pass  # Ignore any exceptions and continue with the next iteration
data['BCI_thresholds'] = BCI_thresholds
pophys_subfolder = os.path.join(folder, 'pophys')
if os.path.isdir(pophys_subfolder):
    csv_folder = pophys_subfolder
else:
    csv_folder = folder
csv_files = glob.glob(os.path.join(csv_folder, base+'_IntegrationRois' + '_*.csv'))
csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
csv_data = []
for i in range(len(csv_files)):
    csv_file = csv_files[i]
    csv_data.append(pd.read_csv(csv_file))        
data['roi_csv'] = np.concatenate(csv_data)