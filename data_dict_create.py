# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:51:06 2023

@author: scanimage
"""
# =============================================================================
#List of Keys
# dist: distance from cn (in pixels)
# F: time x neuron x trial deltaF/F during behavior
# Fraw: time x neuron x trial raw fluorescence intensity during behavior
# conditioned_neuron: index of conditioned neuron
# conditioned_coordinates: location of CN (in pixels)
# dat_filefile containing all data from that session
# mouse: name of mouse
# session: date of session
# dt_si: time step for F (seconds)
# trace_corr: NxN pairwise correlation matrix for full fluorescence (df/f) traces during closed loop behavior.
# seq: trial x 1 vector indicating which stim. group was stimulated on trial t

# =============================================================================
import data_dict_create_module as ddc
import numpy as np
import re
folder = r'D:/KD/BCI_data/BCI_2022/BCI48/032923/'
data = dict()

# BCI data
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
Ftrace = np.load(folder +r'/suite2p_BCI/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
iscell = np.load(folder + r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()

data['trace_corr'] = np.corrcoef(Ftrace.T, rowvar=False)

# metadata
data['dat_file'] = folder + r'/suite2p_BCI/plane0/'
slash_indices = [match.start() for match in re.finditer('/', folder)]
data['session'] = folder[slash_indices[-2]+1:slash_indices[-1]]
data['mouse'] = folder[slash_indices[-3]+1:slash_indices[-2]]

# create F and Fraw
data['F'], data['Fraw'] = ddc.create_BCI_F(Ftrace,ops);

# create dist, conditioned_neuron, conditioned_coordinates
data['dist'], data['conditioned_neuron_coordinates'], data['conditioned_neuron'] = ddc.find_conditioned_neurons(siHeader,stat)
data['dt_si'] = 1/float(siHeader['metadata']['hRoiManager']['scanFrameRate'])

# photostim data
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)#note that this is only defined in the BCI folder
Ftrace = np.load(folder +r'/suite2p_photostim/plane0/F.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_photostim/plane0/ops.npy', allow_pickle=True).tolist()
siHeader = np.load(folder + r'/suite2p_photostim/plane0/siHeader.npy', allow_pickle=True).tolist()

data['Fstim'], data['seq'], data['favg'], data['stimDist'], data['stimPosition'] = ddc.create_photostim_Fstim(ops, Ftrace,siHeader,stat)

# spont data
data['spont'] = np.load(folder +r'/suite2p_spont/plane0/F.npy', allow_pickle=True)

np.save(folder + r'data_'+data['session']+'.npy',data)

