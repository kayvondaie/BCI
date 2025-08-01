# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_017"],'010112',has_pophys=False)
#%%
# for i in range(len(sessions)):
#     try:
#         mouse = sessions['Mouse'][i]
#         session = sessions['Session'][i]
#         folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
#         data = ddc.main(folder)
#     except:
#         mouse = sessions['Mouse'][i]
#         session = sessions['Session'][i]
#         folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/'
#         data = ddc.main(folder)
#%%
AXON_REW, AXON_TS = [], []
SESSION = []
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
            
        rt = data['reward_time'];dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        dfaxon = data['ch1']['df_closedloop']
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

        r_ind = np.where(reward_vector == 1)[0]
        pre = 40
        post = 40
        frew_axon = np.zeros((pre+post,len(r_ind)))
        time = np.arange(0,dt_si*(pre+post),dt_si)
        time = time - time[pre]
        for i in range(len(r_ind)):
            if r_ind[i] > pre and r_ind[i] + post < dfaxon.shape[1]:
                frew_axon[:,i] = np.nanmean(dfaxon[:,r_ind[i]-pre:r_ind[i]+post],axis = 0)
        plt.plot(time,np.nanmean(frew_axon,axis=1))
        AXON_REW.append(np.nanmean(frew_axon,axis=1))
        Faxon = data['ch1']['F']
        plt.show()
        AXON_TS.append(np.nanmean(np.nanmean(Faxon,axis=2),axis=1))
        plt.plot(np.nanmean(np.nanmean(Faxon,axis=2),axis=1))
        plt.show()
        SESSION.append(data['session'])
        
        plt.plot(np.nanmean(np.nanmean(data['ch1']['F'][40:,:,:],axis=1),axis=0))
        plt.plot(data['BCI_thresholds'][1,:]/200)
        plt.plot(~np.isnan(rt))
        plt.show()
    except:
        continue
#%%
axon_rew = np.asarray(AXON_REW)
depth = np.zeros(axon_rew.shape[0])
for i in range(axon_rew.shape[0]):
    axon_rew[i,:] = axon_rew[i,:] - np.nanmean(axon_rew[i,post+10:])
    depth[i] = np.nanmean(axon_rew[i,pre-10:pre]) - np.nanmean(axon_rew[i,pre+20:])
    axon_rew[i,:] = axon_rew[i,:] / np.max(axon_rew[i,:])
# Plot with time on x-axis
ind = np.where(depth > -20.1)[0]

early = 1;
late = 22
plt.figure(figsize = (6,3))

plt.subplot(221);
plt.plot(time,axon_rew[early,:],'c')
plt.plot((0,0),plt.ylim(),'k:')
plt.title(SESSION[early])

plt.subplot(223);
plt.plot(time,axon_rew[late,:],'m')
plt.title(SESSION[late])
plt.plot((0,0),plt.ylim(),'k:')
plt.xlabel('Time from reward (s)')

plt.subplot(122)
plt.imshow(axon_rew[ind,:], aspect='auto', vmin=0, vmax=1,
           extent=[time[0], time[-1], axon_rew.shape[0], 0])

plt.axvline(0, linestyle='--', color='k', linewidth=1)

# Add arrows on the correct rows (adjusted for ind[] and y-flip)
arrow_x = 1.5  # x-position of arrow
arrow_size = 50

# Map session index to row index in imshow plot
early_row = np.where(ind == early)[0][0]
late_row = np.where(ind == late)[0][0]

plt.scatter([arrow_x], [early_row], color='c', s=arrow_size, marker='^', edgecolors='k', linewidths=0.5)
plt.scatter([arrow_x], [late_row], color='m', s=arrow_size, marker='^', edgecolors='k', linewidths=0.5)

plt.xlabel('Time from reward (s)')
plt.ylabel('Session index')
plt.title(mouse)

plt.tight_layout()

peak_times = [time[np.argmax(trace)] for trace in axon_rew]


#%%
r_ind = np.where(step_vector == 1)[0]
pre = 20
post = 20
fstep_axon = np.zeros((pre+post,len(r_ind)))
time = np.arange(0,dt_si*(pre+post),dt_si)
time = time - time[pre]
for i in range(len(r_ind)):
    if r_ind[i] > pre and r_ind[i] + post < dfaxon.shape[1]:
        fstep_axon[:,i] = np.nanmean(dfaxon[:,r_ind[i]-pre:r_ind[i]+post],axis = 0)

plt.subplot(121)
plt.plot(time,np.nanmean(fstep_axon,axis=1))

plt.subplot(122);
plt.imshow(fstep_axon.T,aspect = 'auto')

#%%
strt = 1000
stp = 3000;
cn = data['conditioned_neuron'][0][0]
df = data['df_closedloop'][cn,:]
plt.plot(df[strt:stp]+4,linewidth = .3)
plt.plot(np.nanmean(dfaxon[:,strt:stp],axis=0)*2 - 4,linewidth=.3,color = 'k');
plt.plot(step_vector[strt:stp]*3,linewidth = .3)
plt.plot(-reward_vector[strt:stp])
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Time window
strt = 22000
stp = 22300

# Get conditioned neuron trace
cn = data['conditioned_neuron'][0][0]
df = data['df_closedloop'][cn, :]

# Get average LC axon trace
lc = np.nanmean(dfaxon[:, :], axis=0)

# Subtract mean (optional for cross-correlation)
df = df - np.nanmean(df)
lc = lc - np.nanmean(lc)

# Compute cross-correlation
xcorr = correlate(lc, df, mode='full')
lags = np.arange(-len(df)+1, len(df)) * dt_si  # convert lag to seconds

# Plot cross-correlation
plt.figure(figsize=(5, 3))
ind = round(len(lags)/2)
window = 100
plt.plot(lags[ind - window:ind + window], xcorr[ind-window:ind+window], color='purple')
plt.axvline(0, linestyle='--', color='k', linewidth=1)
plt.xlabel('Lag (s)')
plt.xlabel('Lag (s)\n\n← LC leads        CN leads →', fontsize=10)
plt.ylabel('Cross-correlation')
plt.title('Conditioned neuron vs LC axon')
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Time window
strt = 22000
stp = 22300

# Get conditioned neuron trace
cn = data['conditioned_neuron'][0][0]
df = data['df_closedloop'][cn, :]

# Get average LC axon trace
lc = np.nanmean(dfaxon[:, :], axis=0)

# Subtract mean (optional for cross-correlation)
df = df - np.nanmean(df)
lc = lc - np.nanmean(lc)

# Compute cross-correlation
xcorr = correlate(lc, step_vector, mode='full')
lags = np.arange(-len(df)+1, len(df)) * dt_si  # convert lag to seconds

# Plot cross-correlation
plt.figure(figsize=(5, 3))
ind = round(len(lags)/2)
window = 100
plt.plot(lags[ind - window:ind + window], xcorr[ind-window:ind+window], color='purple')
plt.axvline(0, linestyle='--', color='k', linewidth=1)
plt.xlabel('Lag (s)')
plt.xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=10)
plt.ylabel('Cross-correlation')
plt.title('lickport vs LC axon')
plt.tight_layout()
plt.show()

#%%
fts = np.array(AXON_TS).T
plt.plot(np.nanmean(fts,axis=1))
plt.imshow(fts.T,aspect = 'auto')
plt.show()
trial_start_modulation = np.nanmean(fts[39:42,:],axis=0) - np.nanmean(fts[30:35,:],axis=0)
plt.plot(trial_start_modulation,'k.-')


#%%
AXON_REW, AXON_TS, XCORR = [], [], []
SESSION = []
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
            
        rt = data['reward_time'];dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        dfaxon = data['ch1']['df_closedloop']
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

        r_ind = np.where(reward_vector == 1)[0]
        pre = 40
        post = 40
        frew_axon = np.zeros((pre+post,len(r_ind)))
        time = np.arange(0,dt_si*(pre+post),dt_si)
        time = time - time[pre]
        for i in range(len(r_ind)):
            if r_ind[i] > pre and r_ind[i] + post < dfaxon.shape[1]:
                frew_axon[:,i] = np.nanmean(dfaxon[:,r_ind[i]-pre:r_ind[i]+post],axis = 0)
        AXON_REW.append(np.nanmean(frew_axon,axis=1))
        Faxon = data['ch1']['F']
        AXON_TS.append(np.nanmean(np.nanmean(Faxon,axis=2),axis=1))
        SESSION.append(data['session'])
        
        
        rt = data['reward_time'];dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        dfaxon = data['ch1']['df_closedloop']
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
        
        # Get conditioned neuron trace
        cn = data['conditioned_neuron'][0][0]
        df = data['df_closedloop'][cn, :]

        # Get average LC axon trace
        lc = np.nanmean(dfaxon[:, :], axis=0)

        # Subtract mean (optional for cross-correlation)
        df = df - np.nanmean(df)
        lc = lc - np.nanmean(lc)

        # Compute cross-correlation
        xcorr = correlate(lc, step_vector, mode='full')
        lags = np.arange(-len(df)+1, len(df)) * dt_si  # convert lag to seconds

        # Plot cross-correlation
        plt.figure(figsize=(5, 3))
        ind = round(len(lags)/2)
        window = 100
        plt.plot(lags[ind - window:ind + window], xcorr[ind-window:ind+window], color='purple')
        XCORR.append(xcorr[ind-window:ind+window])
        plt.axvline(0, linestyle='--', color='k', linewidth=1)
        plt.xlabel('Lag (s)')
        plt.xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=10)
        plt.ylabel('Cross-correlation')
        plt.title('Conditioned neuron vs LC axon')
        plt.tight_layout()
        plt.show()
    except:
        continue

# Stack the cross-correlations
x = np.stack(XCORR, axis=1)  # shape: (window*2, n_sessions)

# Create lag vector in seconds
lag_range = np.arange(-window, window) * dt_si  # since window is in samples, multiply by dt

# Plot heatmap
plt.figure(figsize=(6, 4))
plt.imshow(x.T, aspect='auto', extent=[lag_range[0], lag_range[-1], 0, x.shape[1]],
           origin='lower', cmap='viridis')
plt.colorbar(label='Cross-correlation')
plt.xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=10)
plt.ylabel('Session')
plt.title('XCorr: Conditioned neuron vs LC axon' + ' ' + mouse)
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(5, 3))
ind = round(len(lags)/2)
window = 100
plt.plot(lags[ind - window:ind + window], np.nanmean(x,axis=1), color='purple')
XCORR.append(xcorr[ind-window:ind+window])
plt.axvline(0, linestyle='--', color='k', linewidth=1)
plt.xlabel('Lag (s)')
plt.xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=10)
plt.ylabel('Cross-correlation')
plt.title('Conditioned neuron vs LC axon')
plt.tight_layout()
plt.show()

#%%


#%%

i = 4
mouse = sessions['Mouse'][i]
session = sessions['Session'][i]
folder_base = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
folder = folder_base + 'pophys/' if os.path.exists(folder_base + 'pophys/') else folder_base
data = np.load(os.path.join(folder, f"data_main_{mouse}_{session}_BCI.npy"), allow_pickle=True)
import numpy as np
import matplotlib.pyplot as plt
#%%
rt = data['reward_time'];dt_si = data['dt_si']
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
dfaxon = data['ch1']['df_closedloop']
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

r_ind = np.where(reward_vector == 1)[0]
pre = 40
post = 40
frew_axon = np.zeros((pre+post,len(r_ind)))
time = np.arange(0,dt_si*(pre+post),dt_si)
time = time - time[pre]
for i in range(len(r_ind)):
    if r_ind[i] > pre and r_ind[i] + post < dfaxon.shape[1]:
        frew_axon[:,i] = np.nanmean(dfaxon[:,r_ind[i]-pre:r_ind[i]+post],axis = 0)
AXON_REW.append(np.nanmean(frew_axon,axis=1))
Faxon = data['ch1']['F']
AXON_TS.append(np.nanmean(np.nanmean(Faxon,axis=2),axis=1))
SESSION.append(data['session'])


rt = data['reward_time'];dt_si = data['dt_si']
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
dfaxon = data['ch1']['df_closedloop']
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)

# Get conditioned neuron trace
cn = data['conditioned_neuron'][0][0]
df = data['df_closedloop'][cn, :]

# Get average LC axon trace
lc = np.nanmean(dfaxon[:, :], axis=0)

# Subtract mean (optional for cross-correlation)
df = df - np.nanmean(df)
lc = lc - np.nanmean(lc)

# Compute cross-correlation
xcorr = correlate(lc, step_vector, mode='full')
lags = np.arange(-len(df)+1, len(df)) * dt_si  # convert lag to seconds

# Plot cross-correlation
plt.figure(figsize=(10, 5))
plt.subplot(2,5,1)
ind = round(len(lags)/2)
window = 100
plt.plot(lags[ind - window:ind + window], xcorr[ind-window:ind+window], color='purple')
XCORR.append(xcorr[ind-window:ind+window])
plt.axvline(0, linestyle='--', color='k', linewidth=1)
plt.xlabel('Lag (s)')
plt.xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=10)
plt.ylabel('Cross-correlation')
plt.title('Conditioned neuron vs LC axon')
plt.tight_layout()
# Set trial index
trial = 15
num = 2
for ti in range(trial,trial+9):
    plt.subplot(2,5,num)    
    # Extract and average LC axon activity for the trial
    lc_trial = data['ch1']['F'][:, :, ti]  # shape: (n_rois, n_timepoints)
    lc_mean = np.nanmean(lc_trial, axis=1)  # average across ROIs
    
    # Time vector
    t = data['t_bci']  # should match length of lc_mean
    
    step_times = np.array(data['step_time'][ti])  # list of timestamps in same reference frame as t_bci
    
    # Plot
    plt.plot(t[1:], lc_mean, label='LC axon avg dF/F', color='green')
    
    # Plot vertical lines for each step, but only label the first one
    for i, st in enumerate(step_times):    
        label = 'step' if i == 0 else ''
        plt.plot((st, st), (.2, 0.4), 'k', label=label)
    
    rt = data['reward_time'][ti]
    plt.plot((rt, rt), (.2, 0.4), 'b', label='Reward',linewidth = 2)
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F')
    plt.title(f'Trial {ti}')    
    if num == 2:
        plt.legend()
    num = num + 1
plt.tight_layout()
#%%
ti = 15
# Extract and average LC axon activity for the trial
lc_trial = data['ch1']['F'][:, :, ti]  # shape: (n_rois, n_timepoints)
lc_mean = np.nanmean(lc_trial, axis=1)  # average across ROIs

# Time vector
t = data['t_bci']  # should match length of lc_mean

step_times = np.array(data['step_time'][ti])  # list of timestamps in same reference frame as t_bci

# Plot
plt.plot(t[1:], lc_mean, label='LC axon avg dF/F', color='green')

# Plot vertical lines for each step, but only label the first one
for i, st in enumerate(step_times):    
    label = 'step' if i == 0 else ''
    plt.plot((st, st), (.2, 0.4), 'k', label=label)

rt = data['reward_time'][ti]
plt.plot((rt, rt), (.2, 0.4), 'b', label='Reward',linewidth = 2)
plt.xlabel('Time (s)')
plt.ylabel('dF/F')
plt.title(f'Trial {ti}')    