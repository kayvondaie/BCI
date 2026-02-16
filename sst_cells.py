# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:41:59 2026

@author: kayvon.daie
"""
import data_dict_create_module_test as ddc
from BCI_data_helpers import *
import bci_time_series as bts

mouse = "BCI120";
session = '010526';
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
data = ddc.load_data_dict(folder)
siHeader = np.load(folder +r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
dt_si = data['dt_si']
F = data['F']
trl = F.shape[2]
data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
hit = np.isnan(rt)==0;
rt[np.isnan(rt)] = 30;
step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(
    folder, data, rt, dt_si)
df = data['df_closedloop']
rta, t_reward = get_reward_aligned_df_truncated(df, reward_vector, trial_start_vector, dt_si, window=(-4, 10))
#sta, t_reward = get_reward_aligned_df(df, trial_start_vector, dt_si, window=(-2, 10))
sta, t_trial = get_trial_aligned_df_padded(df, trial_start_vector, reward_vector, dt_si, window=(-12, 10))

ts = np.arange(F.shape[0]) * dt_si - 2
tr = np.arange(rta.shape[0]) * dt_si - 4
#%%
from numpy import *;
from matplotlib.pyplot import *;
figure(figsize = (6,8))
favg_raw = data['photostim']['favg_raw']
favg = favg_raw*0
for i in range(favg.shape[1]):
    favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])


pre_end = 8;
F  = data['photostim']['favg_raw']                 
F0 = np.nanmean(F[0:pre_end, :, :], axis=0)
floor = np.nanpercentile(F0, 1) 
F0 = np.where(np.isfinite(F0) & (F0 > floor), F0, np.nan)
#favg = (F - F0[None, :, :]) / F0[None, :, :]         

ff = nanmean(favg,2)
a = nanmean(ff[15:20,:],0) - nanmean(ff[0:6,:],0);
b = argsort(-a)
ci = 2
umPerPix = 1000/float(siHeader['metadata']['hRoiManager']['scanZoomFactor'])/int(siHeader['metadata']['hRoiManager']['pixelsPerLine'])
tps = data['photostim']['stim_params']['time']
subplot(411)
plot(tps[0:80],ff[0:80,b[ci]])
title('Response to all photostims')

xlabel('Time from photostim (s)')

subplot(412)
amp = nanmean(favg[15:20,:,:],0) - nanmean(favg[0:6,:,:],0);
stimDist = data['photostim']['stimDist']
scatter(stimDist[b[ci],:],amp[b[ci],:])
xlabel('Distance from photostim (um)')
ylabel('Response (AU)')

subplot(413)
plot(tr[0:400],nanmean(rta[0:400,b[ci],:],1))
xlabel('Time from reward')

subplot(414)
plot(ts[0:200],nanmean(sta[0:200,b[ci],:],1))
xlabel('Time from trial start (s)')
tight_layout()
#%%
from scipy.signal import medfilt
N = rta.shape[1]
rr = nanmean(rta,2)
ss = nanmean(sta,2)
for i in range(N):
    rr[:,i] = rr[:,i] - nanmean(rr[0:60,i],0)
    ss[:,i] = ss[:,i] - nanmean(ss[0:60,i],0)
figure(figsize = (13,2))
subplot(121)
for i in range(10):
    offset = i*.01;
    l1, = plot(ts[0:140], medfilt(ss[0:140, b[i]], 11) + offset, lw=.5)
    col = l1.get_color()
    plot(tr[0:300] + 7, medfilt(rr[0:300, b[i]], 11) + offset, lw=.5, color=col)
ax = gca()

# vertical reference lines
ax.axvline(0, color='k', lw=0.8, alpha=0.5)   # ts = 0
ax.axvline(7, color='k', lw=0.8, alpha=0.5)   # tr = 0 (shifted by +7)

# ---- 1 second time bar ----
# place it in the bottom-right corner
x0 = ax.get_xlim()[1] - 1.2   # start 1.2s from right edge
y0 = ax.get_ylim()[0] + 0.02  # slightly above bottom

ax.plot([x0, x0+1], [y0, y0], 'k', lw=2)
ax.text(x0 + 0.5, y0 - 0.01, '1 s', ha='center', va='top')

# ---- hide axes ----
ax.set_axis_off()
ax.plot([x1, x1], [y1, y1+0.5], 'k', lw=2)


    #ylim((-.4,1))
# subplot(121)
# imshow(medfilt(ss[0:140,b[0:20]],5).T,aspect = 'auto',cmap = 'bwr',vmin = -.7,vmax = .7);colorbar()
# subplot(122)
# imshow(medfilt(rr[0:300,b[0:20]],5).T,aspect = 'auto',cmap = 'bwr',vmin = -.7,vmax = .7);colorbar()
# tight_layout()
#%%
figure(figsize=(3,6))
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

targets = np.argmin(stimDist,0)
sst, targs = [],[]
ta = zeros(50,)
for i in range(50):
    a = nanmean(favg[0:50,targets[i],i])
    c = argmin(stimDist[:,i])
    c = np.isin(c, b[:20])

    if (a > .2) & (c==False):
        ta[i] = 1    
ind = where(ta == 1)[0]
ind = delete(ind, 21)   # remove the 21st entry
targets = targets[ind]
for i in range(len(ind)):    
    targs.append(favg[0:50,targets,ind[i]])
    sst.append(favg[0:50,b[0:10],ind[i]])
targs = concatenate(targs,0)
sst = concatenate(sst,0)
#sst = delete(sst, idx, axis=1)

subplot(211)
imshow(targs.T,aspect = 'auto',interpolation = 'none',vmin = -2,vmax = 2,cmap = 'bwr')
# after: subplot(211); imshow(...)
step = 5   # label every 5th target (change to taste)
nsteps = len(ind)
block = favg.shape[2]
xt = arange(0, nsteps, step)*block + (block/2 - 0.5)
labels = [str(i+1) for i in range(0, nsteps, step)]

gca().set_xticks(xt)
gca().set_xticklabels(labels)

gca().tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
ax = gca()

ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

ax.set_xlabel('Target Neuron #')
ax.xaxis.set_label_position('top')
cb = colorbar()
cb.set_label('$\Delta F/F$')

# optional: draw separators between blocks
for k in range(1, nsteps):
    axvline(k*block - 0.5, color='k', linewidth=0.5, alpha=0.3)
ylabel('Neuron #')
subplot(212)
imshow(sst.T,aspect = 'auto',vmin = -2,vmax = 2,cmap = 'bwr')
ylabel('SST Sink neuron #')
tt = np.arange(0,dt_si*sst.shape[0],dt_si)
# bottom plot time ticks at 0, 10, 20 seconds
t_sec = array([0, 10, 20])
xt = t_sec / dt_si        # convert seconds -> frame index
cb = plt.colorbar()
cb.set_label('$\Delta F/F$')

gca().set_xticks(xt)
gca().set_xticklabels([str(t) for t in t_sec])

xlabel('Time (s)')
#%%
min_dist = np.nanmin(stimDist[b[:10], :], axis=0)
figure(figsize = (8,4))
aa = []
for i in range(favg.shape[2]):
    ind = where((stimDist[:,i] > 30) & (stimDist[:,i]<80))[0];
    #ind = setdiff1d(ind,b[0:10])
    #ind = intersect1d(ind, b[0:15])
    #a = medfilt(nanmean(favg[0:50,ind,i],1),11);
    a = convolve(nanmean(favg[0:50,ind,i],1),ones(10,))[0:]/10;
    aa.append(a)
a = stack(aa).T
subplot(121)
ind = where(min_dist < 10)[0]
plot(tps[0:50],nanmean(a[0:50,ind],1))
#ylim((-.05,.04))
xlabel('Time from photostim (s)')
title('Target Sink cells')
ylabel('DFF non-target')

subplot(122)
ind = where(min_dist > 60)[0]
plot(tps[0:50],nanmean(a[0:50,ind],1))
#ylim((-.05,.04))
title('Target Non-sink cells')
xlabel('Time from photostim (s)')
ylabel('DFF non-target')
tight_layout()