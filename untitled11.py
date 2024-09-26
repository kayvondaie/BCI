# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:32:10 2024

@author: kayvon.daie
"""

trial = 48;
plt.subplot(312)
plt.plot(np.nanmean(Faxons[39:,0:,trial],axis=1)*1,'k')
plt.xlim((0,250))
plt.subplot(313)
plt.plot(vel[:,trial]);
plt.plot(pos[:,trial]/10);
plt.plot(rew[:,trial]);
plt.xlim((0,250))
plt.subplot(311)
plt.plot(F[39:,cn,trial]/6-.04)
plt.plot(vel[:,trial],'k.');
plt.xlim((0,250))
#%%
trls = np.arange(40,46)
fcn = np.squeeze(F[39:,cn,trls])
fcn = np.reshape(fcn.T, (-1, 1))
ind = np.where(np.isnan(fcn)==0)[0];
fcn = fcn[ind]
lc_ax = np.nanmean(Faxons[39:,0:,trls],axis=1)
lc_ax  = np.reshape(lc_ax .T, (-1, 1))
lc_ax = lc_ax[ind]
pos2 = np.reshape(pos[:-39,trls].T, (-1, 1))
pos2 = pos2[ind]
rew2 = np.reshape(rew[:-39,trls].T, (-1, 1))
rew2 = 10*(rew2[ind]) - 5;
vel2 = np.reshape(vel[:-39,trls].T, (-1, 1))
vel2 = 10*(vel2[ind]) - 5;

plt.subplot(311)
plt.plot(fcn,'m',linewidth=.5)
plt.plot(vel2[5:]-6,'k.',linewidth=.5)
plt.ylim(-1.5,8)


plt.subplot(313)
plt.plot(lc_ax,'g',linewidth=.5)

plt.subplot(312)
plt.plot(pos2,'k',linewidth=.5)
plt.plot(vel2-6,'k.',linewidth=.5)
plt.plot(rew2-4,'bo',linewidth=.5)
plt.ylim(-3,13)

