# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import medfilt, correlate
from axon_helper_module import *
from BCI_data_helpers import *
import bci_time_series as bts

# --- helper: detrend across trials ---
def detrend_across_trials(data, order=1, lam=1e-6):
    T, N = data.shape
    trial_means = np.nanmean(data, axis=0)  # (trials,)
    x = np.arange(N)

    # Design matrix
    X = np.column_stack([x**k for k in range(1, order+1)] + [np.ones_like(x)])
    I = np.eye(X.shape[1])

    mask = np.isfinite(trial_means)
    if mask.sum() < (order + 1):
        trend = np.full(N, np.nanmean(trial_means))
    else:
        beta = np.linalg.solve(X[mask].T @ X[mask] + lam * I, X[mask].T @ trial_means[mask])
        trend = X @ beta  # (trials,)

    return data - trend[None, :]

# --- storage for results ---
results = []

# --- loop over cre lines ---
for crei in range(3):
    if crei == 0:
        mice, creline = ["BCINM_034", "BCINM_031"], "5-HT"
    elif crei == 1:
        mice, creline = ["BCINM_027","BCINM_017"], "NE"
    elif crei == 2:
        mice, creline = ["BCINM_024","BCINM_021"], "Ach"

    sessions = session_counting.counter2(mice,'010112',has_pophys=False)

    processing_mode = 'all'
    si = 10
    inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)

    RTA_low, RTA_high = [], []
    time = None

    for i in inds:
        try:
            mouse = sessions['Mouse'][i]
            session = sessions['Session'][i]

            # Load data
            try:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
                main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
                data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
            except:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
                main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
                data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)

            # Behavior
            dt_si = data['dt_si']
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            rt[np.isnan(rt)] = 20
            dfaxon = data['ch1']['df_closedloop']

            step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

            # Compute RPE (not used directly here)
            rpe = compute_rpe(rt == 20, baseline=1, window=20, fill_value=50)

            # Reward-aligned responses
            rta,_ = trial_aligned_responses(dfaxon, trial_start_vector, reward_vector, rt, dt_si)

            # Average across neurons → (timepoints, trials)
            mean_rta = np.nanmean(rta, axis=1)  # (time, trials)
            mean_rta = detrend_across_trials(mean_rta, order=1)

            # Trial bins
            high_inds = np.where(rt < np.nanmedian(rt))[0]
            low_inds  = np.where(rt > np.nanmedian(rt))[0]

            # Time vector
            n_timepoints = mean_rta.shape[0]
            time = np.linspace(-2, 4, n_timepoints)

            # Accumulate
            RTA_low.append(mean_rta[:, low_inds])
            RTA_high.append(mean_rta[:, high_inds])

        except Exception as e:
            print(f"Skipped session {i}: {e}")
            continue

    # Save per-creline results
    results.append(dict(
        creline=creline,
        time=time,
        RTA_low=RTA_low,
        RTA_high=RTA_high
    ))
#%%
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 8,        # base font size
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8
})

# --- now do plotting outside loop ---
plt.figure(figsize=(7/3,7))
for crei, res in enumerate(results):
    plt.subplot(3,1,crei+1)
    time = res["time"]

    for typ, color, label in zip(["RTA_low","RTA_high"], ['b','r'], ['Slow','Fast']):
        a = np.concatenate(res[typ], axis=1)
        trace = np.nanmean(a, 1)
        sem   = np.nanstd(a, 1) / np.sqrt(a.shape[1])
        plt.plot(time, trace, color=color, label=label)
        plt.fill_between(time, trace - sem, trace + sem, color=color, alpha=0.3)
        plt.xlim((-2,1.5))

    plt.title(res["creline"])
    plt.axvline(0, color='k', ls='--')
    plt.xlabel("Time (s)")
    if crei == 0:
        plt.ylabel("Population avg dF/F")
    plt.legend(frameon=False)

plt.tight_layout()
plt.show()
