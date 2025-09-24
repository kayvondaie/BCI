import bci_time_series as bts
from BCI_data_helpers import *
mice = ["BCI102","BCI105","BCI106","BCI109","BCI103","BCI104","BCI93","BCI107"]
mice = ["BCI102"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    #session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
    si = 7
    
    pairwise_mode = 'dot_prod_no_mean'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'ridge'     #ridge, pinv
    alpha         =  .1        #only used for ridge
    epoch         =  'reward'  # reward, step, trial_start
    
    #for sii in range(0,len(session_inds)):        
    for sii in range(si,si+1):
        num_bins      =  2000         # number of bins to calculate correlations
        print(sii)
        mouse = list_of_dirs['Mouse'][session_inds[sii]]
        session = list_of_dirs['Session'][session_inds[sii]]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        photostim_keys = ['stimDist', 'favg_raw']
        bci_keys = ['df_closedloop','F','mouse','session','conditioned_neuron','dt_si','step_time','reward_time','BCI_thresholds']
        try:
            data = ddct.load_hdf5(folder,bci_keys,photostim_keys )
        except:
            continue        
        BCI_thresholds = data['BCI_thresholds']        
        cn = data['conditioned_neuron'][0][0]
        dfcn = data['df_closedloop'][cn,:]
        
        AMP, stimDist, FAVG = compute_amp_from_photostim(mouse, data, folder,return_favg = True)
    
        dt_si = data['dt_si']
        F = data['F']
        
        def get_reward_aligned_F(data, window=(-2, 5)):
            F = data['F']  # (time, cells, trials)
            dt = data['dt_si']
            nT, nC, nTrials = F.shape
        
            # ensure reward_time is parsed
            reward_time = parse_hdf5_array_string(data['reward_time'], nTrials)
        
            pre_frames  = int(abs(window[0]) / dt)
            post_frames = int(abs(window[1]) / dt)
            win_frames  = pre_frames + post_frames
            F_reward = np.full((win_frames, nC, nTrials), np.nan)
            t_reward = np.arange(-pre_frames, post_frames) * dt
        
            for ti in range(nTrials):
                rewards = reward_time[ti]
                if len(rewards) == 0:
                    continue
                reward_sec = float(rewards[0])       # already numeric now
                reward_frame = int(np.round(reward_sec / dt))
        
                start = reward_frame - pre_frames
                stop  = reward_frame + post_frames
                if start < 0 or stop > nT:
                    continue
                F_reward[:, :, ti] = F[start:stop, :, ti]
        
            return F_reward, t_reward


        F_reward, t_reward = get_reward_aligned_F(data, window=(-2, 2))




#%%


trl = F.shape[2]
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
hit = np.isnan(rt)==0;
rt[np.isnan(rt)] = 30;
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
    folder, data, rt, dt_si)

def get_reward_aligned_df(data, reward_vector, dt_si, window=(-2, 2)):
    """
    Extract peri-reward responses from df_closedloop.

    Parameters
    ----------
    data : dict
        Contains 'df_closedloop'.
    reward_vector : np.ndarray
        Binary or indicator vector of rewards, shape (frames,).
    dt_si : float
        Sampling interval in seconds.
    window : tuple
        (pre, post) window in seconds around reward.

    Returns
    -------
    F_reward : np.ndarray
        (time, cells, n_rewards) peri-reward responses.
    t_reward : np.ndarray
        Time vector relative to reward (s).
    """

    df = data['df_closedloop']      # shape (cells, frames)
    nC, nFrames = df.shape
    reward_frames = np.where(reward_vector > 0)[0]

    pre_frames  = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_reward = np.arange(-pre_frames, post_frames) * dt_si

    F_reward = np.full((win_len, nC, len(reward_frames)), np.nan)

    for ri, r in enumerate(reward_frames):
        start = r - pre_frames
        stop  = r + post_frames
        if start < 0 or stop > nFrames:
            continue
        F_reward[:, :, ri] = df[:, start:stop].T  # (time, cells)

    return F_reward, t_reward


F_reward, t_reward = get_reward_aligned_df(data, reward_vector, dt_si, window=(-2, 4.1))

t_F = np.arange(F.shape[0]) * dt_si - 2
t_R = np.arange(F_reward.shape[0]) * dt_si - 2

responses = {}

# Pre: -2 to -1 (F)
pre_idx = np.where((t_F >= -2) & (t_F < -1))[0]
responses['pre'] = np.nanmean(F[pre_idx, :, :], axis=0)  # (cells, trials)

# Early: 0 to 2 (F)
early_idx = np.where((t_F >= 0) & (t_F < 2))[0]
responses['early'] = np.nanmean(F[early_idx, :, :], axis=0)

# Late: -1 to 0 (reward-aligned)
late_idx = np.where((t_R >= -1) & (t_R < 0))[0]
responses['late'] = np.nanmean(F_reward[late_idx, :, :], axis=0)  # (cells, rewards)

# Rew: 0 to 2 (reward-aligned)
rew_idx = np.where((t_R >= 0) & (t_R < 2))[0]
responses['rew'] = np.nanmean(F_reward[rew_idx, :, :], axis=0)

# Aft: 2 to 4 (reward-aligned)
aft_idx = np.where((t_R >= 2) & (t_R < 4))[0]
responses['aft'] = np.nanmean(F_reward[aft_idx, :, :], axis=0)

# --- Collapse over trials/rewards -> (cells,)
pre_resp   = np.nanmean(responses['pre'],  axis=1)
early_resp = np.nanmean(responses['early'],axis=1)
late_resp  = np.nanmean(responses['late'], axis=1)
rew_resp   = np.nanmean(responses['rew'],  axis=1)
aft_resp   = np.nanmean(responses['aft'],  axis=1)




#%%
far_dist = 50
# --- score each group: target pre response * far response ---
scores = []
for i in range(n_groups):
    target_idx = np.argmin(stimDist[:, i])

    far_idx = np.where((stimDist[:, i] > 30) & (stimDist[:, i] < far_dist))[0]
    if len(far_idx) > 0:
        far_resp = np.nanmean(AMP[1][far_idx, i])
    else:
        far_resp = np.nan

    scores.append(pre_resp[target_idx] - aft_resp[target_idx])  # or pre_resp * far_resp
scores = np.array(scores)
b = np.argsort(-scores)[:20]   # top 20 candidate groups
for i in range(3):
    plt.figure(figsize=(6, 2))
    
    # --- 1. Target cells ---
    plt.subplot(121)
    target_traces = []
    for i in b:
        target_idx = np.argmin(stimDist[:, i])
        targ_trace = favg[np.ix_(time_window, [target_idx], [i])].squeeze()
        target_traces.append(targ_trace)
    
    target_traces = np.array(target_traces)
    target_mean = np.nanmean(target_traces, axis=0)
    target_sem  = np.nanstd(target_traces, axis=0) / np.sqrt(target_traces.shape[0])
    
    plt.plot(t, target_mean, color='#33b983', linewidth=2)
    plt.fill_between(t,
                     target_mean - target_sem,
                     target_mean + target_sem,
                     color='#33b983', alpha=0.3)
    plt.plot(t_artifact, np.zeros_like(t_artifact), 'm')
    
    # remove box and ticks
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # manual y scale bar (0.5 → 1)
    x0 = t[0] - 0.1*(t[-1]-t[0])   # just outside left edge
    plt.plot([x0, x0], [0.5, 1.0], 'k', lw=2)
    plt.text(x0 - 0.02*(t[-1]-t[0]), 0.75, '0.5 ΔF/F',
             va='center', ha='right', rotation=90)
    
    
    
    # --- 2. Non-target cells ---
    plt.subplot(122)
    non_traces = []
    for i in b:
        target_idx = np.argmin(stimDist[:, i])
        non_idx = np.where((stimDist[:, i] > 20) & (stimDist[:, i] < far_dist))[0]
        non_trace = np.nanmean(favg[np.ix_(time_window, non_idx, [i])], axis=1).squeeze()
        non_traces.append(non_trace)
    
    non_traces = np.array(non_traces)
    non_mean = np.nanmean(non_traces, axis=0)
    non_sem  = np.nanstd(non_traces, axis=0) / np.sqrt(non_traces.shape[0])
    
    plt.plot(t, non_mean, color='k', linewidth=2)
    plt.fill_between(t,
                     non_mean - non_sem,
                     non_mean + non_sem,
                     color='k', alpha=0.3)
    plt.plot(t_artifact, np.zeros_like(t_artifact), 'm')
    
    # remove box and ticks
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # manual y scale bar (0.05 → 0.1)
    x1 = t[0]    # just outside right edge
    plt.plot([x1, x1], [0.05, 0.1], 'k', lw=2)
    plt.text(x1 + 0.02*(t[-1]-t[0]), 0.075, '0.05 ΔF/F',
              va='center', ha='left', rotation=90)
    
    
    plt.tight_layout()
    plt.show()

#%%

epoch_colors = {
    "pre":   "#33b983",   # Pretrial (greenish)
    "early": "#1077f3",   # Early trial (blue)
    "late":  "#0050ae",   # Late trial (dark blue)
    "rew":   "#bf8cfc",   # Reward (purple)
    "CN":    "#f98517"    # Conditioned neuron (orange)
}

# --- Non-target cells split by tuning into 4 subplots with epoch colors ---
fig, axes = plt.subplots(4, 1, figsize=(3, 6), sharey=True)

tuning = {
    'pre': pre_resp-aft_resp,
    'early': early_resp-aft_resp,
    'late': late_resp-aft_resp,
    'rew': rew_resp-aft_resp,
}

for ax, (key, resp) in zip(axes, tuning.items()):
    non_traces = []
    for i in b:
        target_idx = np.argmin(stimDist[:, i])
        non_idx = np.where((stimDist[:, i] > 20) & (stimDist[:, i] < far_dist))[0]

        # rank non-targets by their tuning value
        top_idx = np.argsort(-resp[non_idx])[:10]   # top 10 tuned cells
        sel_idx = non_idx[top_idx]

        # average their photostim traces
        non_trace = np.nanmean(favg[np.ix_(time_window, sel_idx, [i])], axis=1).squeeze()
        non_traces.append(non_trace)

    non_traces = np.array(non_traces)
    non_mean = np.nanmean(non_traces, axis=0)
    non_sem  = np.nanstd(non_traces, axis=0) / np.sqrt(non_traces.shape[0])

    # plot mean ± SEM using epoch color
    ax.plot(t, non_mean, color=epoch_colors[key], linewidth=2)
    ax.fill_between(t, non_mean - non_sem, non_mean + non_sem,
                    color=epoch_colors[key], alpha=0.2)

    # artifact marker
    ax.plot(t_artifact, np.zeros_like(t_artifact), 'm')

    # remove box and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # add title

# Add a single y scale bar on the last subplot
x1 = t[0]
axes[-1].plot([x1, x1], [0.05, 0.1], 'k', lw=2)
axes[-1].text(x1 + 0.02*(t[-1]-t[0]), 0.075, '0.05 ΔF/F',
              va='center', ha='left', rotation=90)

plt.tight_layout()
plt.show()
