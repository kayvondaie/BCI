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
ops['roidetect'] = False
ops['do_regmetrics'] = False
ops['allow_overlap'] = False
ops['save_folder'] = 'suite2p_' + base
ops['force_refImg'] = True

ops_old = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
stat_old = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
ops['refImg'] = ops_old['refImg'];

ops = suite2p.run_s2p(ops)


#%%
ops = np.load(folder + ops['save_folder'] + r'/plane0/ops.npy', allow_pickle=True).tolist()
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces_from_masks
cell_masks, neuropil_masks = create_masks(stat_old,ops_old['Ly'],ops_old['Lx'],ops_old)
F, Fneu, F_chan2, Fneu_chan2 = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
np.save(folder + ops['save_folder'] + r'/plane0/F.npy',F)
np.save(folder + ops['save_folder'] + r'/plane0/Fneu.npy',Fneu)
