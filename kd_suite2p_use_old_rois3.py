# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:13:14 2023

@author: scanimage
"""
import suite2p
import os;
#os.chdir('H:/My Drive/Python Scripts/BCI_analysis/')
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir)
os.chdir(relative_path)
import re
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
import extract_scanimage_metadata
#import registration_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
import copy
import shutil
from collections import Counter

folder = [r'//allen/aind/scratch/BCI/2p-raw/BCI109/032525/pophys/']
#old_folder = r'//allen/aind/scratch/BCI/2p-raw/BCI102/021125/pophys/'
#folder = [r'\\allen\aind\scratch\david.feng\BCI_43_032423/']
#old_folder = r'C:/Users/Kayvon Daie/Documents/BCI_data/BCI58/082923/'
#folder = [r'D:/KD/BCI_data/BCI_2022/BCI54/072423/']
#old_folder = r'D:/KD/BCI_data/BCI_2022/BCI45/050123/suite2p_BCI/'
#old_folder = r'D:/KD/BCI_data/BCI_2022/BCI48/042623/suite2p_BCI/' 
if 'old_folder' in locals():
    try:
        stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
        ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
        iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')
    except:
        stat_old = np.load(old_folder + r'suite2p_spont/' + r'/plane0/stat.npy',allow_pickle = 'True')
        ops_old = np.load(old_folder + r'suite2p_spont/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
        iscell_old = np.load(old_folder +r'suite2p_spont/' + r'/plane0/iscell.npy',allow_pickle = 'True')    

savefolders = dict()
savefolders[0] = 'BCI';
savefolders[1] = 'photostim_single';
savefolders[2] = 'photostim_single2';
savefolders[3] = 'spont_pre';
savefolders[4] = 'spont_post';
#savefolders[2] = 'spont';
#savefolders[2] = 'photostim2';

ops = s2p_default_ops()#run_s2p.default_ops()
ops['data_path'] = folder
folder = ops['data_path'][0]
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']

tif_bases = [fname.rsplit('_', 1)[0] for fname in folder_props['siFiles']]

# Count occurrences of each base in the TIFF files
base_counts = Counter(tif_bases)

# Print the bases with their corresponding counts
for index, base in enumerate(bases):
    count = base_counts.get(base, 0)  # Get the count for the base, default to 0 if not found
    print(f"{index}: {base} ({count} TIFFs)")


# Assign save folders based on keywords
ind = input('pick indices of bases for BCI, photostim, spont, photostim2 in that order')
ind = [int(x) for x in ind[1:-1].split(',')]

# # Initialize the savefolders dictionary
# savefolders = {}

# for i, index in enumerate(ind):
#     base_name = bases[index]
    
#     # Determine save folder based on keywords
#     if 'neuron' in base_name.lower():
#         savefolders[i] = "BCI"
#     elif 'spont' in base_name.lower():
#         savefolders[i] = "spont"
#     elif 'photostim' in base_name.lower():
#         savefolders[i] = "photostim_single"
#     else:
#         savefolders[i] = "unknown"  # Default if no keyword matches

# # Reorganize savefolders and ind to put "BCI" first
# if "BCI" in savefolders.values():
#     bci_index = next(key for key, value in savefolders.items() if value == "BCI")
#     # Move the "BCI" index to the front
#     ind = [ind[bci_index]] + [v for i, v in enumerate(ind) if i != bci_index]
#     #savefolders = {0: savefolders[bci_index]} | {i + 1: savefolders[k] for i, k in enumerate(savefolders) if k != bci_index}

# Print the save folder assignments
print("Reorganized savefolders:")
print(savefolders)
print("Reorganized ind:")
print(ind)

#%%
for ei in range(0,len(ind)):
    if ei == 1:
        old_folder = folder
    if 'old_folder' in locals():
        try:
            stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
            ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
            iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')
        except:
            stat_old = np.load(old_folder + r'suite2p_spont/' + r'/plane0/stat.npy',allow_pickle = 'True')
            ops_old = np.load(old_folder + r'suite2p_spont/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
            iscell_old = np.load(old_folder +r'suite2p_spont/' + r'/plane0/iscell.npy',allow_pickle = 'True')    

    #base = {base for i, base in enumerate(bases) if str(i) in ind}
    base = bases[int(ind[ei])]
    siFiles = folder_props['siFiles']
    files = os.listdir(folder)
    good = np.zeros([1,np.shape(files)[0]])
    for fi in range(0,np.shape(files)[0]):
        str = files[fi]
        a = str.find('.tif')
        if a > -1:
            #b = str.find('_')
            #b2 = str.find('_',b+1)
            #b = max([b,b2]);
            b = max([i for i, char in enumerate(str) if char == '_'])
            b = str[0:b]
            if b == base:
                good[0][fi] = 1
    #        if b == base2:
    #            good[0][fi] = 1
    
    good = np.where(good == 1)
    good = good[1]
    
    ops['tiff_list'] = [files[i] for i in good]
    #ops['tiff_list'] = [files[i] for i in range(2,7)]
    
    ops['do_registration']=True
    ops['save_mat'] = True
    
    ops['do_bidiphase'] = True
    ops['reg_tif'] = False # save registered movie as tif files
    ops['delete_bin'] = 0
    ops['keep_movie_raw'] = 0
    ops['fs'] = 20
    ops['nchannels'] = 1
    ops['tau'] = 1
    ops['nimg_init'] = 500
    ops['nonrigid'] = False
    ops['smooth_sigma'] = .5
    ops['threshold_scaling'] = .7
    ops['batch_size'] = 250
    ops['do_registration'] = True
    if 'old_folder' in locals():
        ops['roidetect'] = False
        ops['refImg'] = ops_old['refImg'];
        ops['force_refImg'] = True
    else:
        ops['roidetect'] = True
    ops['do_regmetrics'] = False
    ops['allow_overlap'] = False
    ops['save_folder'] = 'suite2p_' + savefolders[ei]    
    ops = suite2p.run_s2p(ops)    
    if 'old_folder' in locals():
        #stat_new, model_robust = registration_functions.shift_rois_affine(ops_old['meanImg'],ops['meanImg'],stat_old)
        stat_new = copy.deepcopy(stat_old)
        
        from suite2p.extraction.masks import create_masks
        from suite2p.extraction.extract import extract_traces_from_masks
        
        cell_masks, neuropil_masks = create_masks(stat_new,ops['Ly'],ops['Lx'],ops)
        F, Fneu, F_chan2, Fneu_chan2 = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
        np.save(folder + ops['save_folder'] + r'/plane0/F.npy',F)
        np.save(folder + ops['save_folder'] + r'/plane0/Fneu.npy',Fneu)
        np.save(folder + ops['save_folder'] + r'/plane0/stat.npy',stat_new)
        np.save(folder + ops['save_folder'] + r'/plane0/iscell.npy',iscell_old)
        
        folder = ops['data_path'][0]
        file = folder + ops['tiff_list'][0]                
        shutil.copy(ops['save_path0'] +r'/suite2p/plane0/data.bin',ops['save_path0']+ops['save_folder'] + r'/plane0/data.bin')
    file = folder + ops['tiff_list'][0]                
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(file)
    siBase = dict()
    for i in range(3):
        try:
            siBase[i] = bases[int(ind[i])]
        except:
            siBase[i] = ''
    siHeader['siBase']=siBase
    siHeader['savefolders'] = savefolders
    np.save(folder + 'suite2p_' + savefolders[ei] + r'/plane0/siHeader.npy',siHeader)

import data_dict_create_module_test as ddc
data = ddc.main(folder)

del str
