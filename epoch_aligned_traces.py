import bci_time_series as bts
trl = F.shape[2]
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
hit = np.isnan(rt)==0;
rt[np.isnan(rt)] = 30;
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
    folder, data, rt, dt_si)
# Ftrace = np.load(folder + '/suite2p_BCI/plane0/F.npy', allow_pickle=True)
# df = 0*Ftrace
df = data['df_closedloop']

def get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 2)):
     
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


rta, t_reward = get_reward_aligned_df(df, reward_vector, dt_si, window=(-2, 6))
sta, t_reward = get_reward_aligned_df(df, trial_start_vector, dt_si, window=(-2, 10))

ts = np.arange(F.shape[0]) * dt_si - 2
tr = np.arange(F_reward.shape[0]) * dt_si - 2

rta = np.nanmean(rta,2)
sta = np.nanmean(sta,2)
#%%


ind = np.where((tr>2) & (tr<3))[0]
aft = np.nanmean(rta[ind,:],0)

ind = np.where((ts<-1))[0]
pre = np.nanmean(sta[ind,:],0)
ind = np.where((ts>0) & (ts<2))[0]
early = np.nanmean(sta[ind,:],0)
ind = np.where((tr<0) & (tr>-2))[0]
late = np.nanmean(rta[ind,:],0)
ind = np.where((tr>0) & (tr<2))[0]
rew = np.nanmean(rta[ind,:],0)
rew = rew - early
late = late - early
early = early - pre

# pre = pre - aft
# rew = rew - aft
# early = early - aft
# late = late - aft

#%%
ind = np.where(pre < .1)[0]
plt.figure(figsize=(6,8))

for i,(vec,label) in enumerate([(pre,"Pre"),(early,"Early"),(late,"Late"),(rew,"Rew")],1):
    ts = np.arange(sta.shape[0]) * dt_si - 2
    tr = np.arange(rta.shape[0]) * dt_si - 2

    y1 = sta[:,ind] @ vec[ind]
    y2 = rta[:,ind] @ vec[ind]

    ax1 = plt.subplot(4,2,2*i-1); ax1.plot(ts, y1)
    ax2 = plt.subplot(4,2,2*i);   ax2.plot(tr, y2)

    ymin = min(y1.min(),y2.min())
    ymax = max(y1.max(),y2.max())
    ax1.set_ylim(ymin,ymax)
    ax2.set_ylim(ymin,ymax)

    ax1.set_title(f"{label} (trial-start aligned)")
    ax2.set_title(f"{label} (reward-aligned)")
    ax1.set_xlabel("Time (s)")
    ax2.set_xlabel("Time (s)")
    ax1.set_ylabel("Projection")
    ax2.set_ylabel("Projection")

plt.tight_layout()

