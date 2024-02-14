# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:13:14 2023

@author: scanimage
"""

import suite2p
import os
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import matplotlib.pyplot as plt
ops = s2p_default_ops()#run_s2p.default_ops()
ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/ChRmine transgenic test/BCI48/files/']
ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/BCI_42/020323/']
folder = ops['data_path'][0]
files = os.listdir(folder)
amps = []
ref = np.zeros([800,800])
for fi in range(2,86):
    tif = ScanImageTiffReader(folder+files[fi]).data();
    sz = np.shape(tif)
    amp = np.empty(shape=(np.shape(tif)[0], 1), dtype=int)
    for frm in range(0,np.shape(tif)[0]):        
        amp[frm] = np.mean(np.mean(tif[frm,:,:]))
    non = np.where(amp<np.percentile(amp,75))[0]
    img = [tif[i,:,:] for i in non];
    img = np.mean(img,0)
    ref = ref + img
    amps.append(np.transpose(amp))
    
plt.imshow(ref)
d = np.concatenate(amps,axis = 1);
bad = np.where(d>40)
bad_frames = bad[1]
badfile = folder + 'badfile.npy'
np.save(badfile, bad_frames)
#%%
import suite2p
import os
from suite2p import default_ops as s2p_default_ops
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np

files = os.listdir(folder)
base = 'file25'
good = np.zeros([1,np.shape(files)[0]-1])
for fi in range(0,np.shape(files)[0]):
    str = files[fi]
    a = str.find('.tif')
    if a > -1:
        b = str.find('_')
        b = str[0:b]
        if b == base:
            good[0][fi] = 1

good = np.where(good == 1)
good = good[1]




ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/BCI_42/020323/']
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
ops['refImg'] = ref;
ops['force_refImg'] = True