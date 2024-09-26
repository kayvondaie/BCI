# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:32:10 2024

@author: kayvon.daie
"""

    
dt = data['dt_si']
t = np.arange(0, dt * (df.shape[1]), dt)
trial_strt = 0*t;
rew = 0*t
steps = data['step_time']
strt = data['trial_start']
rewT = data['reward_time']
vel = np.zeros((F.shape[0],F.shape[2]))
offset = int(np.round(data['SI_start_times'][0]/dt_si)[0])
for i in range(len(steps)):
    if np.isnan(F[-1,0,i]):
        l = np.where(np.isnan(F[40:,0,i])==1)[0][0]+39;
    else:
        l = F.shape[0]
    v = np.zeros(l,)
    for si in range(len(steps[i])):
        ind = np.where(t>steps[i][si])[0][0]
        v[ind] = 1
    vel[0:l,i] = v
for i in range(len(strt)):
    ind = np.where(t>strt[i])[0][0]
    trial_strt[ind] = 1

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
trial = 49;
plt.subplot(312)
plt.plot(np.nanmean(Faxons[39-offset:,0:,trial],axis=1)*1,'k')
plt.xlim((0,250))
plt.subplot(313)
plt.plot(vel[:,trial]);
plt.plot(pos[:,trial]/10);
plt.plot(rew[:,trial]);
plt.xlim((0,250))
plt.subplot(311)
plt.plot(F[39-offset:,cn,trial]/6-.04)
plt.plot(vel[:,trial],'k.');
plt.xlim((0,250))
#%%
plt.figure(figsize=(2,3))  # Width: 10 inches, Height: 6 inches
plt.rcParams.update({'font.size': 8})
trls = np.arange(45,50)
fcn = np.squeeze(F[39:,cn,trls])
fcn = np.reshape(fcn.T, (-1, 1))
ind = np.where(np.isnan(fcn)==0)[0];
fcn = fcn[ind]
t = np.arange(0,dt_si*len(fcn),dt_si)
t = t[0:len(fcn)]
t = t - t[0]
lc_ax = np.nanmean(Faxons[39:,0:,trls],axis=1)
lc_ax  = np.reshape(lc_ax .T, (-1, 1))
lc_ax = lc_ax[ind]
pos2 = np.reshape(pos[:-39,trls].T, (-1, 1))
pos2 = pos2[ind]/max(pos2)*7
rew2 = np.reshape(rew[:-39,trls].T, (-1, 1))
rew2 = 10*(rew2[ind]) - 5;
vel2 = np.reshape(vel[:-39,trls].T, (-1, 1))
vel2 = 10*(vel2[ind]) - 5;

plt.subplot(311)
plt.plot(t,fcn,'m',linewidth=.5)
plt.plot(t[:-offset],vel2[offset:]-6,'k.',markersize=1)
plt.ylim(-1.5,8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

plt.subplot(312)
plt.plot(t,lc_ax,'g',linewidth=.5)
plt.plot(t[:-offset],vel2[offset:]/15-.5,'k.',markersize=1)
plt.plot(t[:-offset],rew2[offset:]/5-1.25,'bo',markersize=2)
plt.ylim(-.3,.4)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

plt.subplot(313)
plt.plot(t[:-offset],pos2[offset:],'k',linewidth=.5)
plt.plot(t[:-offset],vel2[offset:]-6,'k.',markersize=1)
plt.plot(t[:-offset],rew2[offset:]-4,'bo',markersize=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(-3,8)
plt.tight_layout()

plt.savefig(r'C:\Users\kayvon.daie\OneDrive - Allen Institute\written\RPE grant 2024\figures\panels\lc_axon_traces.svg', format='svg', dpi=300)