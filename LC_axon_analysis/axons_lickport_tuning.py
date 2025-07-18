# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_027"],'010112',has_pophys=False)
#%%
from scipy.signal import medfilt
from scipy.signal import correlate

processing_mode = 'one'
si = 11
if processing_mode == 'all':
    inds = np.arange(0,len(sessions))
else:
    inds = np.arange(si,si+1)
XCORR, LAGS = [], []
SESSION = []
num = 1000
plot = 1
import bci_time_series as bts
for i in inds:
    print(i)
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
    time = np.arange(0,dt_si*len(step_vector),dt_si)
    strt = 0 
    stp  = 2000
    
    cc = np.corrcoef(dfaxon)
    b = np.argsort(cc[0,:])
    if num > dfaxon.shape[0]:
        num = dfaxon.shape[0]
        plot = 0
    for cis in range(num):
        if cis == 0: 
            ci = 0;
        else:
            ci = b[cis-1]
    
        vel = np.convolve(step_vector, np.ones(10,)/3)
        f = medfilt(dfaxon[ci, strt:stp], kernel_size=11)
        vel_trimmed = vel[strt:stp]
        time_trimmed = time[strt:stp]
        
        # Cross-correlation setup
        lc = dfaxon[ci, :] - np.nanmean(dfaxon[ci, :])
        vel_zm = vel - np.nanmean(vel)
        xcorr = correlate(lc, vel_zm, mode='full')
        XCORR.append(xcorr[ind - window:ind + window])
        LAGS.append(lags[ind - window:ind + window])
        lags = np.arange(-len(vel) + 1, len(vel)) * dt_si
        ind = len(lags) // 2
        window = 100
        
        if plot == 1:
        # Create figure with custom layout
            fig = plt.figure(figsize=(10, 3))
            gs = fig.add_gridspec(1, 4)  # 4 columns: 1 for xcorr, 3 for timeseries
            
            # Left: Cross-correlation plot
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.plot(lags[ind - window:ind + window], xcorr[ind - window:ind + window], color='purple')
            ax0.axvline(0, linestyle='--', color='k', linewidth=1)
            ax0.set_xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=8)
            ax0.set_ylabel('Cross-corr')
            ax0.set_title('XCorr')
            ax0.tick_params(labelsize=8)
            
            # Right: Time series plot
            ax1 = fig.add_subplot(gs[0, 1:])
            ax1.plot(time_trimmed, f, 'g', linewidth=1, label='LC')
            ax1.plot(time_trimmed, vel_trimmed, 'k', linewidth=0.5, label='Lickport')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Signal')
            ax1.set_title('Filtered LC & Lickport' + ' Cell' + str(ci) + ' ' + mouse + '_' + session)
            ax1.legend(fontsize=8)
            ax1.tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.show()
# --- After the for cis in range(10): loop ---
if plot == 1:
    # Compute mean LC trace across all cells
    lc_avg = np.nanmean(dfaxon, axis=0)
    lc_avg_filt = medfilt(lc_avg[strt:stp], kernel_size=11)
    vel = np.convolve(step_vector, np.ones(10,) / 3)
    vel_trimmed = vel[strt:stp]
    time_trimmed = time[strt:stp]
    
    # Cross-correlation
    lc_avg_zm = lc_avg - np.nanmean(lc_avg)
    vel_zm = vel - np.nanmean(vel)
    xcorr = correlate(lc_avg_zm, vel_zm, mode='full')
    lags = np.arange(-len(vel) + 1, len(vel)) * dt_si
    ind = len(lags) // 2
    window = 100
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 3))
    gs = fig.add_gridspec(1, 4)
    
    # Left: Cross-correlation plot
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(lags[ind - window:ind + window], xcorr[ind - window:ind + window], color='purple')
    ax0.axvline(0, linestyle='--', color='k', linewidth=1)
    ax0.set_xlabel('Lag (s)\n\n← LC leads        lickport leads →', fontsize=8)
    ax0.set_ylabel('Cross-corr')
    ax0.set_title('Avg XCorr')
    ax0.tick_params(labelsize=8)
    
    # Right: Time series plot
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(time_trimmed, lc_avg_filt, 'g', linewidth=1, label='Avg LC')
    ax1.plot(time_trimmed, vel_trimmed, 'k', linewidth=0.5, label='Lickport')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal')
    ax1.set_title('Avg LC & Lickport: ' + mouse + '_' + session)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.show()
#%%
# Stack cross-correlations
xcorr = np.stack(XCORR)  # shape: (n_cells, n_lags)
lags = LAGS[1]           # shared lags vector



# Sort by lag of peak
mp = np.round(xcorr.shape[1]/2).astype(int)
leadlag = np.nanmean(xcorr[:,:mp],axis=1) - np.nanmean(xcorr[:,mp:],axis=1)
sort_order = np.argsort(-leadlag)
xcorr_sorted = xcorr[sort_order, :]

# Plot
plt.figure(figsize=(6, 5))
im = plt.imshow(xcorr_sorted, aspect='auto', extent=[lags[0], lags[-1], 0, len(XCORR)],
                cmap='RdBu_r', vmin=-np.max(np.abs(xcorr)), vmax=np.max(np.abs(xcorr)))
plt.colorbar(im, label='Cross-corr')
plt.xlabel('Lag (s)')
plt.ylabel('Sorted axons')
plt.title('Cross-correlations sorted by peak lag')
plt.xlabel('Lag (s)\n\n← Axon leads        lickport leads →', fontsize=8)
plt.tight_layout()
plt.show()
