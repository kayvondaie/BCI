# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:07:06 2021

@author: scanimage
"""

#%% segment day 1
import suite2p
from suite2p import default_ops as s2p_default_ops
ops = s2p_default_ops()#run_s2p.default_ops()
#ops['look_one_level_down']=True

ops['data_path'] = [r'D:/KD/BCI_data/BCI21_111121_closed_loop_registered']
#ops['data_path'] = r'D:\KD\BCI_data\BCI21_111121_closed_loop_registered\'
ops['do_registration']=False
ops['save_mat'] = 1
ops = run_s2p(ops)
#%% binary file day 2
ops = s2p_default_ops()#run_s2p.default_ops()
#ops['look_one_level_down']=True
ops['roidetect']=False
ops['data_path'] = [r'D:/KD/BCI_data/BCI21_111521_closed_loop_registered']
ops['do_registration']=False
ops['save_mat'] = 1
ops = run_s2p(ops)
#%% extraction day 2 
#ops (dictionary) – ‘Ly’, ‘Lx’, ‘nframes’, ‘batch_size’
stat = np.load('D:/KD/BCI_data/BCI21_111121_closed_loop_registered/suite2p/plane0/stat.npy',allow_pickle='true')
ops = np.load('D:/KD/BCI_data/BCI21_111521_closed_loop_registered/suite2p/plane0/ops.npy',allow_pickle='true')
ops.tolist()['data_path']suite2p.extraction.extract.create_masks_and_extract(ops.tolist(), stat)
dat = suite2p.extraction.extract.create_masks_and_extract(ops.tolist(), stat)
ops_old = np.load('D:/KD/BCI_data/BCI21_111121_closed_loop_registered/suite2p/plane0/ops.npy',allow_pickle='true')
#%%
import suite2p
from suite2p import default_ops as s2p_default_ops
ops = s2p_default_ops()#run_s2p.default_ops()
#ops['look_one_level_down']=True

ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/BCI34/071822/registered/']
ops['data_path'] = [r'D:/KD/BCI_data/BCI_2022/KH_BCI3/090122/']

#ops['data_path'] = [r'I:/BCI32/fov2/061322/registered/nonstim/']
#ops['data_path'] = r'D:\KD\BCI_data\BCI21_111121_closed_loop_registered\'
ops['do_registration']=False
ops['save_mat'] = 1

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
ops['do_registration'] = 1
ops['roidetect'] = True
ops['do_regmetrics'] = False
ops['allow_overlap'] = False

#%%

