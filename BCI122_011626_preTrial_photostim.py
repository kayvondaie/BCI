# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:25:07 2026

@author: kayvon.daie
"""
mouse = "BCI122";session = '011626';
folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
data = ddc.load_data_dict(folder)
Ftrace = data['df_closedloop']
iscell = np.load(folder +r'/suite2p_BCI/plane0/iscell.npy', allow_pickle=True)
stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
siHeader = np.load(folder +r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()
ind = np.where(iscell[:,0] == 1)[0]
stat = stat[ind]
key_name = 'photostim'
data[key_name] = dict()
data[key_name]['Fstim'], data[key_name]['seq'], data[key_name]['favg'], data[key_name]['stimDist'], \
data[key_name]['stimPosition'], data[key_name]['centroidX'], data[key_name]['centroidY'], \
data[key_name]['slmDist'], data[key_name]['stimID'], data[key_name]['Fstim_raw'], \
data[key_name]['favg_raw'], data[key_name]['stim_params'] = ddc.stimDist_single_cell(ops, Ftrace, siHeader, stat, offset=0)
#%%

targets = [378, 38, 304, 44, 217, 379, 49, 236, 115, 309, 365]
targets = [x - 1 for x in targets]
#F = data['F']
Ftrace = data['df_closedloop']
B = Ftrace*0
for ci in range(Ftrace.shape[0]):
    a = Ftrace[ci,:];
    b = np.convolve(a,np.ones(lens),'same')/lens;    
    B[ci,:] = b

siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist() 
dt_si = 1/float(siHeader['metadata']['hRoiManager']['scanVolumeRate']);
if dt_si < 0.05:
    post = int(round(10/0.05 * 0.05/dt_si))
    pre = int(round(2/0.05 * 0.05/dt_si))
else:
    post = 200
    pre = 40
    
F2,_,_,_,_ = ddc.create_BCI_F(B,ops,stat,pre,post);
F = F2.copy()
ftarg = nanmean(F[:,targets,:],1)
fcn = F[:,cn,:]

a = nanmean(ftarg[100:150,:],0)
stim_trials = where(a > .8)[0]
ctrl_trials = where(a <= .8)[0]

figure(figsize = (10,5))

rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['threshold_crossing_time']])
st = np.array([x[0] if len(x) > 0 else np.nan for x in data['SI_start_times']])
rt = rt - st;

rew = ~np.isnan(rt)
# hit trials
hits = np.where(rew == True)[0]

# trials that FOLLOW a hit
aft_hits = hits + 1

# keep only ctrl trials that follow a hit
ctrl_trials = np.array(ctrl_trials)
ctrl_trials = ctrl_trials[np.isin(ctrl_trials, aft_hits)]


#%%
figure(figsize=(8,4))
ts = data['t_bci']
ln = 21
ffcn = fcn*0
fftarg = ftarg*0
for i in range(fcn.shape[1]):
    ffcn[:,i] = fcn[:,i] - nanmean(fcn[80:100,i],0)
    fftarg[:,i] = ftarg[:,i] - nanmean(ftarg[80:100,i],0)

def fun(x,width):    
    y = np.convolve(x,ones(width,))
    return y

subplot(121)
plot(ts,nanmean(ftarg[0:,stim_trials],1),'m')
plot(ts,nanmean(ftarg[0:,ctrl_trials],1),'k')
plot((ts[105],ts[105]),ylim(),'k:')
plot((0,0),ylim(),'k:')
xlabel('Time from trial start (s)')
title('Targets')

subplot(122)
plot(ts[ln+1:],fun(nanmedian(fcn[0:,stim_trials],1),ln)[ln:-ln],'m')
plot(ts[ln+1:],fun(nanmedian(fcn[0:,ctrl_trials],1),ln)[ln:-ln],'k')
plot((ts[105],ts[105]),ylim(),'k:')
plot((0,0),ylim(),'k:')
xlabel('Time from trial start (s)')
title('CN')
# subplot(133)
# plot(fun(nanmean(ffcn[0:200,stim_trials],1),ln),'m')
# plot(fun(nanmean(ffcn[0:200,ctrl_trials],1),ln),'k')
# plot((105,105),ylim(),'k:')
# plot((118,118),ylim(),'k:')
tight_layout()
#%%

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

ctrl = rt[ctrl_trials]
stim = rt[stim_trials]

# stats
m_ctrl = np.nanmean(ctrl)
m_stim = np.nanmean(stim)
sem_ctrl = np.nanstd(ctrl) / np.sqrt(np.sum(~np.isnan(ctrl)))
sem_stim = np.nanstd(stim) / np.sqrt(np.sum(~np.isnan(stim)))

tstat, pval = ttest_ind(ctrl, stim, nan_policy='omit', equal_var=False)

# plot
plt.figure(figsize=(4,4))

# scatter
x1 = np.ones(len(ctrl)) * 0
x2 = np.ones(len(stim)) * 1
plt.scatter(x1, ctrl, alpha=0.5)
plt.scatter(x2, stim, alpha=0.5)

# mean ± SEM
plt.errorbar(0, m_ctrl, yerr=sem_ctrl, fmt='o', capsize=5)
plt.errorbar(1, m_stim, yerr=sem_stim, fmt='o', capsize=5)

# formatting
plt.xticks([0,1], ['Control', 'Stim'])
plt.ylabel('Reaction time')
plt.title(f'p = {pval:.3e}')
plt.xlim(-0.5, 1.5)

plt.tight_layout()
plt.show()

