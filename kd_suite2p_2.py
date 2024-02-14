# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:13:14 2023

@author: scanimage
"""

import suite2p
import os
import re
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import folder_props_fun
ops = s2p_default_ops()#run_s2p.default_ops()
ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/BCI48/032823/']
folder = ops['data_path'][0]
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']

print(bases)
ind = input('which ones?')
#base = {base for i, base in enumerate(bases) if str(i) in ind}
base = bases[int(ind)]
siFiles = folder_props['siFiles']
files = os.listdir(folder)
good = np.zeros([1,np.shape(files)[0]])
for fi in range(0,np.shape(files)[0]):
    str = files[fi]
    a = str.find('.tif')
    if a > -1:
        b = str.find('_')
        b2 = str.find('_',b+1)
        b = max([b,b2]);
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

ops['batch_size'] = 250
ops['do_registration'] = True
ops['roidetect'] = True
ops['do_regmetrics'] = False
ops['allow_overlap'] = False
ops['save_folder'] = 'suite2p_' + base
#ops['refImg'] = ref;
#ops['force_refImg'] = True

#ops = suite2p.run_s2p(ops)

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
         





