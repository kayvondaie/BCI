import session_counting
import data_dict_create_module_test as ddc
import bci_time_series as bts
from scipy.signal import correlate
X_ts,X_rew,X_cre = [],[],[]
for crei in range(3):
    if crei == 0:
        mice = ["BCINM_034", "BCINM_031"];creline = ['5-HT']
    elif crei == 1:
        mice = ["BCINM_027","BCINM_017"];creline = ['NE']
    elif crei == 2:
        mice = ["BCINM_024","BCINM_021"];creline = ['Ach']
    AXON_REW, AXON_TS, XCORR = [], [], []
    SESSION = []    
    for mi in range(len(mice)):    
        sessions = session_counting.counter2([mice[mi]],'010112',has_pophys=False)
        for si in range(len(sessions)):
            print(si)
            try:
                mouse = sessions['Mouse'][si]
                session = sessions['Session'][si]
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
                ts_ind = np.where(trial_start_vector == 1)[0]
                pre = 40
                post = 40
                frew_axon = np.zeros((pre+post,len(r_ind)))
                time = np.arange(0,dt_si*(pre+post),dt_si)
                time = time - time[pre]
                frew_ax = np.zeros((900,len(r_ind)))
                fts_ax  = np.zeros((900,len(r_ind)))
                for i in range(len(r_ind)):
                    if r_ind[i] > pre and r_ind[i] + post < dfaxon.shape[1]:
                        frew_axon[:,i] = np.nanmean(dfaxon[:,r_ind[i]-pre:r_ind[i]+post],axis = 0)
                        a = np.arange(r_ind[i]-pre,ts_ind[np.where(ts_ind > r_ind[i])[0][0]])
                        if len(a) > 900:
                            a = a[0:900]
                        frew_ax[0:len(a),i] = np.nanmean(dfaxon[:,a],0)
                        
                        a = np.arange(ts_ind[i]-pre,r_ind[np.where(r_ind> ts_ind[i])[0][0]])
                        if len(a) > 900:
                            a = a[0:900]
                        fts_ax[0:len(a),i] = np.nanmean(dfaxon[:,a],0)
                AXON_REW.append(np.nanmean(frew_ax,axis=1))
                Faxon = data['ch1']['F']
                AXON_TS.append(np.nanmean(fts_ax,1))
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
                norm = np.sqrt(np.sum(lc**2) * np.sum(step_vector**2))
                xcorr = xcorr / norm
                lags = np.arange(-len(df)+1, len(df)) * dt_si  # convert lag to seconds
        
                # Plot cross-correlation
                ind = round(len(lags)/2)
                window = 100
                XCORR.append(xcorr[ind-window:ind+window])
           
            except Exception as e:
                print(f"Error: {type(e).__name__} - {e}")
                continue
    
    # Stack the cross-correlations
    x = np.stack(XCORR, axis=1)  # shape: (window*2, n_sessions)
    X_cre.append(np.nanmean(x,1))
    
    x = np.stack(AXON_TS, axis=1)  # shape: (window*2, n_sessions)
    X_ts.append(np.nanmean(x,1))
    
    x = np.stack(AXON_REW, axis=1)  # shape: (window*2, n_sessions)
    X_rew.append(np.nanmean(x,1))
    
    # Create lag vector in seconds
    lag_range = np.arange(-window, window) * dt_si  # since window is in samples, multiply by dt
    # Plot heatmap
    # plt.figure(figsize=(6, 4))
    # plt.plot(lag_range,np.nanmean(x,1));plt.plot((0,0),plt.ylim(),'k:')
    # plt.axvline(0, linestyle='--', color='k', linewidth=1)
    # plt.xlabel('Lag (s)')
    # plt.xlabel('Lag (s)\n\n← Axons lead        lickport leads →', fontsize=10)
    # plt.ylabel('Cross-correlation')
    # plt.tight_layout()
    # plt.show()
    # plt.tight_layout()
    # plt.show()
#%%
t = np.arange(0,dt_si*900,dt_si)
t = t-t[pre]
x = np.stack(AXON_REW, axis=1)  # shape: (window*2, n_sessions)
plt.figure(figsize=(6, 4))
plt.subplot(121)
a = np.arange(0,100)
plt.plot(t[a],np.nanmean(x[a,:],1),'k');plt.plot((0,0),plt.ylim(),'k:')
plt.xlabel('Time from reward (s)')
plt.subplot(122)
x = np.stack(AXON_TS, axis=1)  # shape: (window*2, n_sessions)
a = np.arange(0,100)
plt.plot(t[a],np.nanmean(x[a,:],1),'k');plt.plot((0,0),plt.ylim(),'k:')
plt.xlabel('Time from trial start (s)')
plt.tight_layout()
plt.show()
#%%
# Plot heatmap
plt.figure(figsize = (8,8/3))

creline = ['Serotonin','Norepinephrine','Acetylcholine']
for i in range(3):
    # Create lag vector in seconds
    plt.subplot(1,3,i+1)
    lag_range = np.arange(-window, window) * dt_si  # since window is in samples, multiply by dt    
    plt.plot(lag_range,X_cre[i],'k');
    plt.plot((0,0),plt.ylim(),'k:')
    plt.axvline(0, linestyle='--', color='k', linewidth=1)
    plt.xlabel('Lag (s)')
    if i == 1:
        plt.xlabel('Lag (s)\n\n← Axons lead        lickport leads →', fontsize=10)
    else:
        plt.xlabel('Lag (s)', fontsize=10)
    plt.title(creline[i])
    plt.ylabel('Cross-correlation')
    plt.ylim((-.04,.12))
plt.tight_layout()
plt.show()    
for j in range(2):
    if j == 0:
        A = X_ts.copy()
        string = 'trial start'
    else:
        A = X_rew.copy()
        string = 'reward'
    plt.figure(figsize = (8,8/3))
    t = np.arange(0,dt_si*900,dt_si)
    t = t-t[pre]
    creline = ['Serotonin','Norepinephrine','Acetylcholine']
    for i in range(3):
        # Create lag vector in seconds
        plt.subplot(1,3,i+1)
        lag_range = np.arange(-window, window) * dt_si  # since window is in samples, multiply by dt    
        plt.plot(t[0:100],A[i][0:100],'k');
        plt.plot((0,0),plt.ylim(),'k:')
        plt.axvline(0, linestyle='--', color='k', linewidth=1)
        plt.xlabel('Lag (s)')
        
        plt.xlabel('Time from ' + string + ' (s)', fontsize=10)
        plt.title(creline[i])
        plt.ylabel('$\Delta$F/F')
        
    plt.tight_layout()
    plt.show()    