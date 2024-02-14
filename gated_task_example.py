# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 12:53:48 2023

@author: scanimage
"""
from scipy.signal import medfilt
import numpy as np
files_with_movies = []
for k in zaber['scanimage_file_names']:
    if str(k) == 'no movie for this trial':
        files_with_movies.append(False)
    else:
        files_with_movies.append(True)

trl_strt = zaber['trial_start_times'][files_with_movies]
trl_end = zaber['trial_end_times'][files_with_movies]
go_cue = zaber['go_cue_times'][files_with_movies]
trial_times = [(trl_end[i]-trl_strt[i]).total_seconds() for i in range(len(trl_strt))]
trial_hit = zaber['trial_hit'][files_with_movies]
lick_L = zaber['lick_L'][files_with_movies]
rewT = zaber['reward_L'];
threshold_crossing_times = zaber['threshold_crossing_times'][files_with_movies]

F_trial_strt = [];
t_trial_strt = [];
strt = 0;
dff = 0*F
t_si = np.arange(0,dt_si * F.shape[1],dt_si)
for i in range(np.shape(F)[0]):
    #bl = np.percentile(F[i,:],50)
    bl = np.std(F[i,:])
    dff[i,:] = (F[i,:] - bl)/bl
for i in range(len(ops['frames_per_file'])):
    ind = list(range(strt,strt+ops['frames_per_file'][i]))    
    f = F[:,ind]
    F_trial_strt.append(f)
    t_trial_strt.append(t_si[ind])
    strt = ind[-1]+1

for ti in np.arange(66,70):
    f = F_trial_strt[ti][19,:]
    bl = np.percentile(f,20)
    f = (f-bl) / bl
    t = t_trial_strt[ti]
    fbad = medfilt(np.mean(F_trial_strt[ti][500:,:],axis = 0),3)
    plt.subplot(3,1,2)
    st = np.where((t_si[stim_time]>t[0]) & (t_si[stim_time]<t[-1]))[0]
    plt.plot(t,medfilt(f,5),'m',linewidth = 1)
    plt.axis('tight')
    yl = plt.ylim()
    plt.plot((rewT[ti]+t[0],rewT[ti]+t[0]),yl,'c',linewidth = 1)
    for si in range(len(st)):
        plt.fill_between((t_si[stim_time[st[si]]]+0,t_si[stim_time[st[si]]]+3.1),(yl[0],yl[0]),(yl[1],yl[1]),alpha = .1,color = 'k',edgecolor = 'none')
    plt.ylabel('DF/F')
    xl = plt.xlim()
    
    plt.subplot(3,1,1)
    f = np.mean(F_trial_strt[ti][b[0:12],:],axis = 0)
    bl = np.percentile(f,20)
    f = (f-bl) / bl
    t = t_trial_strt[ti]
    fbad = medfilt(np.mean(F_trial_strt[ti][500:,:],axis = 0),1)
    st = np.where((t_si[stim_time]>t[0]) & (t_si[stim_time]<t[-1]))[0]    
    #for si in range(len(st)):
        #plt.plot((t_si[stim_time[st[si]]],t_si[stim_time[st[si]]]+3.1),(-1,-1),'r')
    plt.plot(t,medfilt(f,5),'r',linewidth = 1)
    yl = plt.ylim()
    plt.plot((rewT[ti]+t[0],rewT[ti]+t[0]),yl,'c',linewidth = 1)
    for si in range(len(st)):
        plt.fill_between((t_si[stim_time[st[si]]]+0,t_si[stim_time[st[si]]]+3.1),(yl[0],yl[0]),(yl[1],yl[1]),alpha = .1,color = 'k',edgecolor = 'none')
    plt.ylabel('DF/F')
    plt.xlim(xl)
    
    plt.subplot(3,1,3)
    steps = zaber['zaber_move_forward'][ti]
    pos = np.zeros((1,len(t)))[0]
    for si in range(len(steps)):
        ind = np.where(t>steps[si]+t[0])[0][0]
        pos[ind] = 1     
    #pos = np.cumsum(pos)
    #pos[-1] = 0
    plt.plot(t,pos,'b',linewidth = .5)
    for si in range(len(st)):
        plt.fill_between((t_si[stim_time[st[si]]]+0,t_si[stim_time[st[si]]]+3.1),(0,0),(1.3,1.3),alpha = .1,color = 'k',edgecolor = 'none')
    plt.xlabel('Time from sessions start (s)')
    plt.ylabel('Lickport speed')
    plt.xlim(xl)
plt.show()

