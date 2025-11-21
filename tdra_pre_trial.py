

import session_counting
import data_dict_create_module_test as ddct

list_of_dirs = session_counting.counter()
si = 29;
folder = str(list_of_dirs[si])
#%%
import bci_time_series as bts
from BCI_data_helpers import *
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import ttest_1samp
import re
HI, HIb, HIc = [], [], []
RT, RPE, HIT = [], [], []
HIT_RATE,D_HIT_RATE,DOT, TRL, THR = [], [], [], [], []
CC_RPE, CC_RT, CC_MIS, CORR_RPE, CORR_RT = [], [], [], [], []
RT_WINDOW, HIT_WINDOW, THR_WINDOW = [], [], []
PTRL, PVAL, RVAL = [], [], []
Ddirect, Dindirect, CCdirect = [], [], []
NUM_STEPS,TST, CN, MOUSE, SESSION = [], [], [], [],[]

mice = ["BCI102","BCI103","BCI104","BCI105","BCI106","BCI109"]
mice = ["BCI102"]
for mi in range(len(mice)):
    session_inds = np.where((list_of_dirs['Mouse'] == mice[mi]) & (list_of_dirs['Has data_main.npy']==True))[0]
    #session_inds = np.where((list_of_dirs['Mouse'] == 'BCI103') & (list_of_dirs['Session']=='012225'))[0]
    si = 5
    
    pairwise_mode = 'dot_prod'  #dot_prod, noise_corr,dot_prod_no_mean
    fit_type      = 'ridge'     #ridge, pinv
    alpha         =  .1        #only used for ridge
    epoch         =  'pre'  # reward, step, trial_start,pre
    
    #for sii in range(0,len(session_inds)):        
    for sii in range(si,si+1):
        num_bins      =  2000         # number of bins to calculate correlations
        print(sii)
        mouse = list_of_dirs['Mouse'][session_inds[sii]]
        session = list_of_dirs['Session'][session_inds[sii]]
        folder = r'//allen/aind/scratch/BCI/2p-raw/' + mouse + r'/' + session + '/pophys/'
        photostim_keys = ['stimDist', 'favg_raw']
        photostim_keys = ['stimDist', 'favg_raw']
        bci_keys = [
            'F',
            'df_closedloop',
            'dt_si',
            'step_time',
            'reward_time',
            'conditioned_neuron',
            'BCI_thresholds'
        ]
        data = ddct.load_hdf5(folder, bci_keys, photostim_keys)


#%%
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

def get_trial_aligned_df_padded(df, trial_start_vector, reward_vector, dt_si, window=(-20, 10)):
    """
    Aligns all trials so t=0 (trial start) occurs at the same frame index for every trial.
    Pads both pre-trial and post-trial regions with NaNs when data are missing
    (e.g., short ITIs or truncated recording).

    Parameters
    ----------
    df : ndarray (nCells, nFrames)
    trial_start_vector : binary ndarray, 1 at trial start frames
    reward_vector : binary ndarray, 1 at reward frames
    dt_si : float
        Frame duration (s)
    window : tuple
        Time window (sec) around trial start, e.g. (-20, 10)

    Returns
    -------
    F_trial : ndarray (time, cells, trials)
        Each trial aligned to t=0 at the same frame index.
    t_trial : ndarray
        Relative time (s) to trial start.
    """
    nC, nFrames = df.shape
    trial_starts = np.where(trial_start_vector > 0)[0]
    reward_frames = np.where(reward_vector > 0)[0]

    pre_frames = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_trial = np.arange(-pre_frames, post_frames) * dt_si

    F_trial = np.full((win_len, nC, len(trial_starts)), np.nan)

    for ti, tstart in enumerate(trial_starts):
        # Identify neighboring rewards (avoid contamination)
        prev_reward = reward_frames[reward_frames < tstart]
        next_reward = reward_frames[reward_frames > tstart]
        prev_reward_frame = prev_reward[-1] if len(prev_reward) > 0 else 0
        next_reward_frame = next_reward[0] if len(next_reward) > 0 else nFrames

        # Desired absolute frame indices for the full window
        desired_start = tstart - pre_frames
        desired_stop = tstart + post_frames

        # Actual available region within constraints
        actual_start = max(desired_start, prev_reward_frame)
        actual_stop = min(desired_stop, next_reward_frame, nFrames)

        # Compute how much to pad on left (pre) and right (post)
        left_pad = actual_start - desired_start
        right_pad = desired_stop - actual_stop

        # Extract valid data
        valid_data = df[:, actual_start:actual_stop]

        # Place valid data within padded window
        start_idx = int(left_pad)
        stop_idx = start_idx + valid_data.shape[1]
        F_trial[start_idx:stop_idx, :, ti] = valid_data.T

    return F_trial, t_trial




import numpy as np

def get_reward_aligned_df_truncated(df, reward_vector, trial_start_vector, dt_si, window=(-2, 2)):
    """
    Extracts reward-aligned ΔF/F windows, truncating post-reward response at the next trial start.

    Parameters
    ----------
    df : ndarray
        (nCells, nFrames)
    reward_vector : ndarray
        Binary vector marking reward times (1 at reward frame).
    trial_start_vector : ndarray
        Binary vector marking trial start times (1 at trial start frame).
    dt_si : float
        Frame duration (s)
    window : tuple
        Time window around reward (sec), e.g. (-2, 10)

    Returns
    -------
    F_reward : ndarray
        (time, cells, rewards)
    t_reward : ndarray
        Relative time (sec) to reward.
    """
    nC, nFrames = df.shape
    reward_frames = np.where(reward_vector > 0)[0]
    trial_start_frames = np.where(trial_start_vector > 0)[0]

    pre_frames  = int(abs(window[0]) / dt_si)
    post_frames = int(abs(window[1]) / dt_si)
    win_len = pre_frames + post_frames
    t_reward = np.arange(-pre_frames, post_frames) * dt_si

    F_reward = np.full((win_len, nC, len(reward_frames)), np.nan)

    for ri, r in enumerate(reward_frames):
        # find next trial start after this reward
        next_trial = trial_start_frames[trial_start_frames > r]
        next_trial_frame = next_trial[0] if len(next_trial) > 0 else nFrames

        start = max(r - pre_frames, 0)
        stop = min(r + post_frames, next_trial_frame)

        # compute actual length for truncation
        window_len = stop - start
        if window_len <= 0 or stop > nFrames:
            continue

        # place truncated trace into preallocated array
        F_reward[:window_len, :, ri] = df[:, start:stop].T

    return F_reward, t_reward


rta, t_reward = get_reward_aligned_df_truncated(df, reward_vector, trial_start_vector, dt_si, window=(-2, 10))
#sta, t_reward = get_reward_aligned_df(df, trial_start_vector, dt_si, window=(-2, 10))
sta, t_trial = get_trial_aligned_df_padded(df, trial_start_vector, reward_vector, dt_si, window=(-12, 10))

ts = np.arange(F.shape[0]) * dt_si - 2
tr = np.arange(rta.shape[0]) * dt_si - 2
#%% relate pre activity to rt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

alignment = 'trial'  # or 'reward'
cn = data['conditioned_neuron'][0][0]
pre_t = np.where(tr > 0)[0][0]

ind = np.where(hit)[0]
y_all = rt[ind]
y_all[y_all == 30] = 10

window_len = 0.4
window_pre = -1
window_post = window_pre + window_len

def window_to_frames(t_axis, window):
    s, e = window
    si = np.argmin(np.abs(t_axis - s))
    ei = np.argmin(np.abs(t_axis - e))
    return si, ei

if alignment == 'trial':
    s, e = window_to_frames(t_trial, (window_pre, window_post))
    slice_data = sta[s:e, :, :]
    offset = 0
else:
    s, e = window_to_frames(t_reward, (window_pre, window_post))
    slice_data = rta[s:e, :, :]
    offset = 1

xmat = np.nanmean(slice_data, axis=0)
valid_trial_mask = ~np.all(np.isnan(xmat), axis=0)

xmat_hit = xmat[:, ind]
valid_trial_mask_hit = valid_trial_mask[ind] if len(valid_trial_mask) >= len(ind) else np.ones_like(ind, bool)
xmat_hit = xmat_hit[:, valid_trial_mask_hit]
y = y_all[valid_trial_mask_hit]

if offset == 0:
    X = xmat_hit.T
    y_next = y
else:
    X = xmat_hit[:, :-offset].T
    y_next = y[offset:]

neuron_cov = np.mean(~np.isnan(X), axis=0)
X = X[:, neuron_cov >= 0.8]
trial_cov = np.mean(~np.isnan(X), axis=1)
mask = (trial_cov >= 0.8) & (~np.isnan(y_next))
X = X[mask]
y_next = y_next[mask]

if len(y_next) < 5 or X.shape[1] < 2:
    print(f"Skipping window {window_pre:.1f}–{window_post:.1f}s: insufficient data.")
    exit()

kf = KFold(n_splits=min(5, len(y_next)), shuffle=True)
y_pred = np.full_like(y_next, np.nan, float)

for tr_idx, te_idx in kf.split(X):
    Xtr, Xte = X[tr_idx].copy(), X[te_idx].copy()
    ytr = y_next[tr_idx]
    mu = np.nanmean(Xtr, axis=0)
    Xtr[np.isnan(Xtr)] = np.take(mu, np.where(np.isnan(Xtr))[1])
    Xte[np.isnan(Xte)] = np.take(mu, np.where(np.isnan(Xte))[1])
    beta = np.linalg.pinv(Xtr) @ ytr
    y_pred[te_idx] = Xte @ beta

mask = ~np.isnan(y_pred)
if np.sum(mask) < 2:
    print(f"Skipping window {window_pre:.1f}–{window_post:.1f}s: no valid predictions.")
else:
    r2 = r2_score(y_next[mask], y_pred[mask])
    cc,p = pearsonr(y_next[mask], y_pred[mask])
    print(f'p = {p:.8f}')
    plt.scatter(y_pred, y_next, s=10, c='k', alpha=0.7)
    plt.xlabel('Predicted RT (t+1)' if offset else 'Predicted RT (t)')
    plt.ylabel('Observed RT (t+1)' if offset else 'Observed RT (t)')
    plt.tight_layout()
    plt.show()


#%%
# --- Compute ITIs from trial_start_vector ---
trial_start_frames = np.where(trial_start_vector > 0)[0]

# Frame differences between consecutive trial starts
iti_frames = np.diff(trial_start_frames)

# Convert to seconds using ScanImage frame duration
iti_sec = iti_frames * dt_si

# --- Plot histogram ---
plt.figure(figsize=(5,3))
plt.hist(iti_sec, bins=np.arange(0, np.nanmax(iti_sec)+1, 1), color='k', alpha=0.7)
plt.xlabel('Inter-trial interval (s)')
plt.ylabel('Count')
plt.title(f'Mean ITI = {np.nanmean(iti_sec):.1f} s  (n={len(iti_sec)})')
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.signal import medfilt


alignment = 'trial'  # or 'reward'
cn = data['conditioned_neuron'][0][0]
pre_t = np.where(tr > 0)[0][0]

ind = np.where(hit)[0]
y_all = rt[ind]
y_all[y_all == 30] = 10
BETA, BETA1, BETA2, PROJ = [], [], [], []
window_len = 0.25
bins = np.arange(-6,-.5,window_len)
for bi in range(len(bins)):

    
    window_pre2 = bins[bi]
    window_post2 = window_pre2 + window_len
    
    window_pre = -window_len 
    window_post = window_pre + window_len
    
    def window_to_frames(t_axis, window):
        s, e = window
        si = np.argmin(np.abs(t_axis - s))
        ei = np.argmin(np.abs(t_axis - e))
        return si, ei
    
    if alignment == 'trial':
        s, e = window_to_frames(t_trial, (window_pre, window_post))
        slice_data = sta[s:e, :, :]
        
        s2, e2 = window_to_frames(t_trial, (window_pre2, window_post2))
        slice_data2 = sta[s2:e2, :, :]
        
        offset = 0
    else:
        s, e = window_to_frames(t_reward, (window_pre, window_post))
        slice_data = rta[s:e, :, :]
        offset = 1
    
    xmat = np.nanmean(slice_data, axis=0)
    xmat2 = np.nanmean(slice_data2, axis=0)
    valid_trial_mask = ~np.all(np.isnan(xmat+xmat2), axis=0)
    
    xmat_hit = xmat[:, ind]
    xmat_hit2 = xmat2[:, ind]
    valid_trial_mask_hit = valid_trial_mask[ind] if len(valid_trial_mask) >= len(ind) else np.ones_like(ind, bool)
    xmat_hit = xmat_hit[:, valid_trial_mask_hit]
    xmat_hit2 = xmat_hit2[:, valid_trial_mask_hit]
    y = y_all[valid_trial_mask_hit]
    
    if offset == 0:
        X = xmat_hit.T
        X2 = xmat_hit2.T
        y_next = y
    else:
        X = xmat_hit[:, :-offset].T
        X2 = xmat_hit2[:, :-offset].T
        y_next = y[offset:]
    
    neuron_cov = np.mean(~np.isnan(X), axis=0)
    X = X[:, neuron_cov >= 0.8]
    X2 = X2[:, neuron_cov >= 0.8]
    trial_cov = np.mean(~np.isnan(X), axis=1)
    mask = (trial_cov >= 0.8) & (~np.isnan(y_next))
    X = X[mask]
    X2 = X2[mask]
    y_next = y_next[mask]
    
    if len(y_next) < 5 or X.shape[1] < 2:
        print(f"Skipping window {window_pre:.1f}–{window_post:.1f}s: insufficient data.")
        exit()
    
    kf = KFold(n_splits=min(5, len(y_next)), shuffle=True)
    y_pred = np.full_like(y_next, np.nan, float)
    y_pred2 = np.full_like(y_next, np.nan, float)
    plt.figure(figsize = (4,6))
    B = []
    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx].copy(), X[te_idx].copy()
        Xtr2, Xte2 = X2[tr_idx].copy(), X2[te_idx].copy()
        ytr = y_next[tr_idx]
        mu = np.nanmean(Xtr, axis=0)
        Xtr[np.isnan(Xtr)] = np.take(mu, np.where(np.isnan(Xtr))[1])
        Xte[np.isnan(Xte)] = np.take(mu, np.where(np.isnan(Xte))[1])
        beta = np.linalg.pinv(Xtr) @ ytr
        y_pred[te_idx] = Xte @ beta
        a = Xtr @ beta
        beta2 = np.linalg.pinv(Xtr2) @ a
        beta2 = beta2/np.linalg.norm(beta2)
        y_pred2[te_idx] = Xte2 @ beta2
        B.append(beta2)    
    a = np.nanmean(rta[:,:,te_idx],2)[:,neuron_cov >= 0.8]
    for aaa in range(a.shape[1]):
        q = np.where(t_reward < 0)[0][-1]
        a[:,aaa] = a[:,aaa] - np.nanmean(a[q-10:q,aaa])
    a = a @ beta
    PROJ.append(medfilt(a,11))    
    BETA.append(np.nanmean(np.stack(B),0))
    BETA1.append(B[1])
    BETA2.append(B[2])
        
    
    mask = ~np.isnan(y_pred)
    if np.sum(mask) < 2:
        print(f"Skipping window {window_pre:.1f}–{window_post:.1f}s: no valid predictions.")
    else:
        r2 = r2_score(y_next[mask], y_pred[mask])
        cc,p = pearsonr(y_next[mask], y_pred[mask])
        print(f'p = {p:.8f}')
        plt.subplot(211)
        plt.scatter(y_pred, y_next, s=10, c='k', alpha=0.7)
        plt.xlabel('Predicted RT (t+1)' if offset else 'Predicted RT (t)')
        plt.ylabel('Observed RT (t+1)' if offset else 'Observed RT (t)')
        plt.subplot(212)
    
        cc,p = pearsonr(y_pred2[mask], y_pred[mask])
        print(f'p = {p:.8f}')
        plt.scatter(y_pred2, y_pred, s=10, c='k', alpha=0.7)
        plt.tight_layout()
        plt.show()
#%%        
cross_corr = np.corrcoef(np.vstack([beta1, beta2]))[:L, L:]

plt.figure(figsize=(5,5))
plt.imshow(cross_corr, vmin=0, vmax=0.5, origin='upper')

# Label only -5 and 0 on each axis
plt.xticks([0, L-1], ['-5', '0'])
plt.yticks([0, L-1], ['-5', '0'])

plt.xlabel('Time before trial start (s)')
plt.ylabel('Time before trial start (s)')
plt.colorbar()
plt.tight_layout()
plt.show()
#%%
a = np.stack(PROJ).T;
#plt.imshow(a[:,:].T,aspect = 'auto',interpolation = 'none')
plt.plot(a[q-30:350,(2,-6)])
