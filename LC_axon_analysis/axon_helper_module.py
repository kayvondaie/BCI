# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:06:42 2025

@author: kayvon.daie
"""
import os
import numpy as np
def get_axon_data_dict(sessions,i):
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
    return data, folder

import numpy as np

def reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4)):
    # Parameters
    pre_s, post_s = window
    pre_pts  = int(np.round(pre_s / dt_si))
    post_pts = int(np.round(post_s / dt_si))
    total_pts = post_pts - pre_pts  # total points in window

    # Find reward times
    reward_times = np.where(reward_vector > 0)[0]

    # Only keep rewards with enough buffer on both sides
    valid_rewards = reward_times[(reward_times + post_pts < dfaxon.shape[1]) &
                                 (reward_times + pre_pts >= 0)]

    # Preallocate output: time x neuron x trial
    out = np.zeros((total_pts, dfaxon.shape[0], len(valid_rewards)))

    for i, rt in enumerate(valid_rewards):
        inds = np.arange(rt + pre_pts, rt + post_pts)
        out[:, :, i] = dfaxon[:, inds].T  # transpose to (time, neuron)

    return out  # shape: (time, neuron, trial)
