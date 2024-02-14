# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 05:27:36 2024

@author: Kayvon Daie
"""

dt = data['dt_si']
t = np.arange(0, dt * (df.shape[1]), dt)
vel = 0*t;
trial_strt = 0*t;
rew = 0*t
steps = data['step_time']
strt = data['trial_start']
rewT = data['reward_time']
for i in range(len(steps)):
    ind = np.where(t>steps[i])[0][0]
    vel[ind] = 1
for i in range(len(strt)):
    ind = np.where(t>strt[i])[0][0]
    trial_strt[ind] = 1
    
for i in range(len(rewT)):
    ind = np.where(t>rewT[i])[0][0]
    rew[ind] = 1

pos = 0*t;
for i in range(len(pos)):
    pos[i] = pos[i-1]
    if vel[i] == 1:
        pos[i] = pos[i-1] + 1;
    if trial_strt[i]==1:
        pos[i] = 0
#%%
from scipy.signal import medfilt
k = np.where(np.nanmean(f[40:150,:],axis =0)>0)[0]
avg = np.nanmean(df[k,:],axis=0)
avg = avg+.93
avg = avg*100
ind = list(range(1400,1900))
lw = .6
plt.subplot(312)
plt.plot(t[ind],pos[ind],'k',linewidth = lw)
plt.ylabel('Port position')
plt.subplot(413)
plt.plot(t[ind],rew[ind],'k',linewidth = lw)
plt.xlabel('Time (s)')
plt.ylabel('Reward')
#plt.plot(trial_strt[ind],linewidth = lw)
plt.subplot(411)
plt.plot(t[ind],medfilt(avg[ind],11),'k',linewidth = lw)
plt.ylabel('LC Axons')
plt.show()