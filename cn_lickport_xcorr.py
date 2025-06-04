# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:45:17 2025

@author: kayvon.daie
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
# -*- coding: utf-8 -*-
"""
Cross-correlation of conditioned neuron and lickport trace during step period
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from bci_time_series import *
import data_dict_create_module_test as ddct
import bci_time_series as bts

# Load session list first in a separate cell:
# import session_counting
# list_of_dirs = session_counting.counter()

mice = ["BCI102"]
plot_results = True
window = 100
run_all = False  # Toggle this to run all sessions or just one
si = 11           # Index within session_inds to run if run_all is False

for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) &
                            (list_of_dirs['Has data_main.npy'] == True))[0]

    iter_range = session_inds if run_all else [session_inds[si]]

    for sii in iter_range:
        mouse = list_of_dirs['Mouse'][sii]
        session = list_of_dirs['Session'][sii]
        folder = rf'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'

        bci_keys = ['df_closedloop', 'F', 'mouse', 'session',
                    'conditioned_neuron', 'dt_si', 'step_time', 'reward_time','BCI_thresholds']
        try:
            data = ddct.load_hdf5(folder, bci_keys, [])
        except Exception as e:
            print(f"Skipping {mouse} {session}: {e}")
            continue

        try:
            cn = data['conditioned_neuron'][0][0]
            df = data['df_closedloop'][cn, :]
            dt_si = data['dt_si']
            rt = data['reward_time']
            
            
            def parse_hdf5_array_string(array_raw, trl):
                if isinstance(array_raw, str):
                    # Match both non-empty and empty arrays
                    pattern = r'array\(\[([^\]]*)\](?:, dtype=float64)?\)'
                    matches = re.findall(pattern, array_raw.replace('\n', ''))
        
                    parsed = []
                    for match in matches:
                        try:
                            if match.strip() == '':
                                parsed.append(np.array([]))
                            else:
                                arr = np.fromstring(match, sep=',')
                                parsed.append(arr)
                        except Exception as e:
                            print("Skipping array due to error:", e)
        
                    # Pad to match number of trials
                    pad_len = trl - len(parsed)
                    if pad_len > 0:
                        parsed += [np.array([])] * pad_len
        
                    return np.array(parsed, dtype=object)
        
                else:
                    # Already a list/array
                    if len(array_raw) < trl:
                        pad_len = trl - len(array_raw)
                        return np.array(list(array_raw) + [np.array([])] * pad_len, dtype=object)
                    return array_raw
        
            trl = data['F'].shape[2]
            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            hit = np.isnan(rt)==0;
            rt[np.isnan(rt)] = 20;
        

            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
                folder, data, rt, dt_si)

    

            xcorr = correlate(df, step_vector, mode='full')
            lags = np.arange(-len(df)+1, len(df)) * dt_si

            if plot_results:
                plt.figure(figsize=(5, 3))
                mid = len(lags) // 2
                plt.subplot(131)
                plt.plot(lags[mid - window:mid + window],
                         xcorr[mid - window:mid + window], color='teal')
                plt.axvline(0, linestyle='--', color='k')
                plt.xlabel('Lag (s)\n\n← CN leads        lickport leads →', fontsize=10)
                plt.ylabel('Cross-correlation')
                plt.title(f'CN vs Lick during Step: {mouse} {session}')
                plt.tight_layout()

                plt.subplot(132)
                plt.plot(hit)
                BCI_thresholds = data['BCI_thresholds']
                thr = BCI_thresholds[1,:]
                thr = thr / np.nanmax(thr)
                plt.plot(thr)
                
                fcn = data['F'][:,cn,:]
                fcn = np.nanmean(fcn,axis=0)
                window = 10;
                fcn = np.convolve(fcn,np.ones(window,)/window)[window:-window]
                plt.subplot(133)
                plt.plot(fcn,'k')
                plt.plot(thr)
                plt.tight_layout()
                plt.show()


        except KeyError as e:
            print(f"Missing key in {mouse} {session}: {e}")
        except Exception as e:
            print(f"Error in {mouse} {session}: {e}")
