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

data = dict()

file_dff             = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_dff.h5'
file_extraction      = 'C:/Users/kayvon.daie/Documents/data/BCI93/740369_2025-02-04_17-33-11_extraction.h5'
file_trial_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/trial_locations.json'
file_epoch_locations = 'C:/Users/kayvon.daie/Documents/data/BCI93/epoch_locations.json'
tif_file             = '//allen/aind/scratch/BCI/2p-raw/BCI93/020425/pophys/neuron15_00002.tif'


with open(file_epoch_locations, 'r') as f:
    epoch_data = json.load(f)


# data['photostim'] = []
# Fstim, seq, favg, stimDist, stimPosition, centroidX, centroidY, slmDist, stimID, Fstim_raw, favg_raw = \
#     ddc.stimDist_single_cell(ops, F, siHeader, stat)

for index in range(1,3):
    epoch_inds = (3,4);
    Ftrace, stat, ops, siHeader = oph.run_all_preprocessing(file_dff, file_extraction, file_trial_locations, epoch_data, epoch_inds[index-1], tif_file)
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
data['F'], data['Fraw'],data['df_closedloop'],data['centroidX'],data['centroidY'] = ddc.create_BCI_F(Ftrace,ops,stat,pre,post);   