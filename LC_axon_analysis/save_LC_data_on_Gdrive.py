# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:23:40 2025

@author: kayvon.daie
"""

import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_027"],'010112',has_pophys=False)
AXON_REW, AXON_TS = [], []
SESSION = []
save_dir = r'G:\My Drive\Learning rules\BCI_data\Neuromodulator imaging'
import bci_time_series as bts
for i in range(len(sessions)):
    print(i)
    try:
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
        try:
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
            suffix = 'BCI'
            main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
            main_npy_path = os.path.join(folder, main_npy_filename)
            data = np.load(main_npy_path, allow_pickle=True)
        except:
            folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/'
            suffix = 'BCI'
            main_npy_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
            main_npy_path = os.path.join(folder, main_npy_filename)
            data = np.load(main_npy_path, allow_pickle=True)
        
        # Create a filename using mouse and session to match your convention
        save_filename = f"data_main_{mouse}_{session}_{suffix}.npy"
        save_path = os.path.join(save_dir, save_filename)
        
        # Save the data
        np.save(save_path, data)
    except:
        continue
