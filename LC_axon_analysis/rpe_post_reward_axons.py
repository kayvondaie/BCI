# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_027"],'010112',has_pophys=False)
#%%
from scipy.signal import medfilt, correlate
from axon_helper_module import *
import bci_time_series as bts

processing_mode = 'one'
si = 10
inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)
XCORR, LAGS, SESSION = [], [], []
num = 1000
plot = 1

for i in inds:
    print(i)
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

    # Timing and behavioral signals
    dt_si = data['dt_si']
    rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
    rt[np.isnan(rt)] = 20
    dfaxon = data['ch1']['df_closedloop']

    step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

    # Compute RPE
    rpe = compute_rpe(rt == 20, baseline=1, window=20, fill_value=50)

    # Reward-aligned responses
    rta = reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4))

    # Get rewarded frame indices
    rewarded_frames = np.where(reward_vector > 0)[0]
    pre_pts = int(2 / dt_si)
    post_pts = int(4 / dt_si)
    valid_reward_frames = rewarded_frames[
        (rewarded_frames - pre_pts >= 0) &
        (rewarded_frames + post_pts < dfaxon.shape[1])
    ]
    valid_reward_frames = valid_reward_frames[:rta.shape[2]]  # match rta

    # --- Map reward frames to trial indices using trial_start_vector ---
    trial_starts = np.where(trial_start_vector > 0)[0]  # one per trial
    assert len(trial_starts) == len(rpe), "Trial count mismatch"
    
    # Match number of reward-aligned frames to rta
    valid_reward_frames = valid_rewards[:rta.shape[2]]
    
    # Find the trial each reward frame belongs to
    reward_trial_inds = np.searchsorted(trial_starts, valid_reward_frames, side='right') - 1
    
    # Keep only valid trial indices
    reward_trial_inds = reward_trial_inds[(reward_trial_inds >= 0) & (reward_trial_inds < len(rpe))]
    
    # Get corresponding RPE values
    rpe_rewarded = rpe[reward_trial_inds]


    # Sanity check
    assert rta.shape[2] == len(rpe_rewarded), "Mismatch between rta trials and rpe_rewarded"

    # Average across neurons → (time, trials)
    mean_rta = np.nanmean(rta, axis=1)

    # Bin trials into low / medium / high RPE groups
    percentiles = np.percentile(rpe_rewarded, [33, 66])
    low_inds  = np.where(rpe_rewarded <= percentiles[0])[0]
    med_inds  = np.where((rpe_rewarded > percentiles[0]) & (rpe_rewarded <= percentiles[1]))[0]
    high_inds = np.where(rpe_rewarded > percentiles[1])[0]

    # Time vector
    n_timepoints = mean_rta.shape[0]
    time = np.linspace(-2, 4, n_timepoints)

    # Plot PSTHs
    plt.figure(figsize=(6, 4))
    for trial_inds, label, color in zip([low_inds, high_inds],
                                        ['Low RPE', 'High RPE'],
                                        ['blue', 'red']):
        trace = np.nanmean(mean_rta[:, trial_inds], axis=1)
        sem   = np.nanstd(mean_rta[:, trial_inds], axis=1) / np.sqrt(len(trial_inds))
        plt.plot(time, trace, label=label, color=color)
        plt.fill_between(time, trace - sem, trace + sem, color=color, alpha=0.3)

    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Time from reward (s)')
    plt.ylabel('Population avg dF/F')
    plt.title('Reward-aligned PSTH by RPE tercile')
    plt.legend()
    plt.tight_layout()
    plt.show()
