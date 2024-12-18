# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:09:36 2024

@author: kayvon.daie
"""

df = data['F']
dt_si = data['dt_si']
t = np.arange(0, dt * (df.shape[0]*42), dt)
trial_strt = 0*t;
rew = 0*t
steps = data['step_time']
strt = data['trial_start']
rewT = data['reward_time']
F = data['F']
vel = np.zeros((F.shape[0]*2,F.shape[2]))
offset = int(np.round(data['SI_start_times'][0]/dt_si)[0])
for i in range(len(steps)):
    if np.isnan(F[-1,0,i]):
        l = np.where(np.isnan(F[40:,0,i])==1)[0][0]+39;
    else:
        l = F.shape[0]*2
    v = np.zeros(l,)
    for si in range(len(steps[i])):
        ind = np.where(t>steps[i][si])[0][0]
        v[ind] = 1
    vel[0:l,i] = v
# for i in range(len(strt)):
#     ind = np.where(t>strt[i])[0][0]
#     trial_strt[ind] = 1

rew = np.zeros((F.shape[0],F.shape[2]))
for i in range(len(rewT)):
    try:
        ind = np.where(t>rewT[i])[0][0]
        rew[ind,i] = 1
    except:
        pass
        

pos = 0*vel;
for ti in range(pos.shape[1]):
    for i in range(pos.shape[0]):
        pos[i,ti] = pos[i-1,ti]
        if vel[i,ti] == 1:
            pos[i,ti] = pos[i-1,ti] + 1; 
#%%
plt.figure(figsize=(2,4))
t = np.arange(0,dt*517,dt)
cn = data['conditioned_neuron'][0][0]
trial = 65;
plt.subplot(312)
#plt.plot(np.nanmean(Faxons[39-offset:,0:,trial],axis=1)*1,'k')
plt.subplot(312)
plt.plot(t[0:500],vel[:500,trial],'k');
plt.xlim((0,8))
plt.ylabel('Lickport steps')

plt.subplot(313)
plt.plot(t[0:500],rew[:500,trial],'k');
plt.xlabel('Time (s)')
plt.xlim((0,8))
plt.ylabel('Reward')

plt.subplot(311)
plt.plot(t,F[119-offset:619,[7,51,107],trial]/6-.04,linewidth=.6)
plt.xlim((0,8))
plt.tight_layout()