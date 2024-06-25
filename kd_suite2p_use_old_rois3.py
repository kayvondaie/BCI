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

folder = [r'//allen/aind/scratch/BCI/2p-raw/BCI85/062424/']
old_folder = r'//allen/aind/scratch/BCI/2p-raw/BCI85/061824/'
#folder = [r'\\allen\aind\scratch\david.feng\BCI_43_032423/']
#old_folder = r'C:/Users/Kayvon Daie/Documents/BCI_data/BCI58/082923/'
#folder = [r'D:/KD/BCI_data/BCI_2022/BCI54/072423/']
#old_folder = r'D:/KD/BCI_data/BCI_2022/BCI45/050123/suite2p_BCI/'
#old_folder = r'D:/KD/BCI_data/BCI_2022/BCI48/042623/suite2p_BCI/' 
if 'old_folder' in locals():
    stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
    ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
    iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')

savefolders = dict()
savefolders[0] = 'BCI';
savefolders[1] = 'spont';
#savefolders[2] = 'spont';
#savefolders[2] = 'photostim2';

ops = s2p_default_ops()#run_s2p.default_ops()
ops['data_path'] = folder
folder = ops['data_path'][0]
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']

print(bases)
ind = input('pick indices of bases for BCI, photostim, spont, photostim2 in that order')
ind = np.fromstring(ind[1:-1], sep=',')
#%%
for ei in range(0,len(ind)):
    if ei == 1:
        old_folder = folder
    if 'old_folder' in locals():
        stat_old = np.load(old_folder + r'suite2p_BCI/' + r'/plane0/stat.npy',allow_pickle = 'True')
        ops_old = np.load(old_folder + r'suite2p_BCI/' +r'/plane0/ops.npy',allow_pickle = 'True').tolist()
        iscell_old = np.load(old_folder +r'suite2p_BCI/' + r'/plane0/iscell.npy',allow_pickle = 'True')

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

import data_dict_create_module as ddc
data = ddc.main(folder)
if data['mouse'] == 'BCI69':
    bci69_folder = 'H:/My Drive/Learning rules/BCI_data/BCI_69/'
    np.save(bci69_folder  + r'data_'+data['mouse']+r'_'+data['session']+r'.npy',data)

#%%
os.chdir(r'C:\Users\scanimage\Documents\Python Scripts\BCI_analysis')
import extract_scanimage_metadata
s2pfolder = folder + '/suite2p/plane0/'
F = np.load(s2pfolder + 'F.npy');
iscell = np.load(s2pfolder + 'iscell.npy');
ops = np.load(s2pfolder + 'ops.npy', allow_pickle = True)
ops = ops.tolist()
stat = np.load(s2pfolder + 'stat.npy', allow_pickle = True)

file = folder + ops['tiff_list'][0]
header=extract_scanimage_metadata.extract_scanimage_metadata(file)

#%%


integration_roi_data = {}
integration_roi_data['outputChannelsEnabled'] = np.asarray(metadata['metadata']['hIntegrationRoiManager']['outputChannelsEnabled'].strip('[]').split(' '))=='true'
integration_roi_data['outputChannelsNames'] = metadata['metadata']['hIntegrationRoiManager']['outputChannelsNames'].strip("{}").replace("'","")
integration_roi_data['outputChannelsRoiNames'] = eval(metadata['metadata']['hIntegrationRoiManager']['outputChannelsRoiNames'].replace('{','[').replace('}',']'))
integration_roi_data['outputChannelsFunctions'] = eval(metadata['metadata']['hIntegrationRoiManager']['outputChannelsFunctions'].replace('{','[').replace('}',']').replace(' ',','))

metadata['metadata']['hIntegrationRoiManager']['outputChannelsRoiNames']


try:
    conditioned_neuron_name = (integration_roi_data['outputChannelsRoiNames'])[0]
except:
    conditioned_neuron_name = (integration_roi_data['outputChannelsRoiNames'])




if len(conditioned_neuron_name) == 0:
     conditioned_neuron_name = ''
rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']   
if type(rois) is not list:
     rois = [rois]
roinames_list = list() 
for roi in rois:
     try:
         roinames_list.append(roi['name'])
     except:
         roinames_list.append(None)
try:
     roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
except:
     try:
         print('ROI names in scanimage header does not match up: {}'.format(conditioned_neuron_name))
         conditioned_neuron_name = ' '.join(conditioned_neuron_name.split(","))
         roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
     except:
         print('no usable ROI idx, skipping')
         roi_idx = None
         





