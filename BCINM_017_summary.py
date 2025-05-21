# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
sessions = session_counting.counter2(["BCINM_017"],'010112',has_pophys=False)
#%%
for i in range(len(sessions)):
    try:
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/'
        data = ddc.main(folder)
    except:
        continue
#%%
AXON_REW = []
SESSION = []
import bci_time_series as bts
for i in range(len(sessions)):
    print(i)
    try:
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
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
        plt.show()
        SESSION.append(data['session'])
        
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
ind = np.where(depth > .1)[0]
plt.imshow(axon_rew[ind,:], aspect='auto', vmin=0, vmax=1,
           extent=[time[0], time[-1], axon_rew.shape[0], 0])

# Add vertical line at time = 0
plt.axvline(0, linestyle='--', color='k', linewidth=1)

# Label axes
plt.xlabel('Time (s)')
plt.ylabel('Session index')

plt.title(mouse)
plt.show()