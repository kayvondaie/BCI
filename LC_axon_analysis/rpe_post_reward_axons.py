# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os
sessions = session_counting.counter2(["BCINM_027"],'010112',has_pophys=False)
#%%
from scipy.signal import medfilt, correlate
from scipy.stats import pearsonr
p = r"C:\Users\kayvon.daie\Documents\GitHub\BCI\LC_axon_analysis"
assert os.path.isdir(p), f"Not a directory: {p}"
if p not in sys.path:
    sys.path.insert(0, p)  # use insert so this path is searched first
from axon_helper_module import *  # or: from axon_helper_module import whatever_you_need
from BCI_data_helpers import *
import bci_time_series as bts

processing_mode = 'all'
si = 10
inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)
XCORR, LAGS, SESSION = [], [], []
RTA_low,RTA_high,CC_RPE,PP_RPE = [],[],[],[]
RPE,RT,RPE_all,RT_all= [],[],[],[]
RTA,STA = [],[]
num = 1000
plot = 1

for i in inds:
    try:
        print(i)
        mouse = sessions['Mouse'][i]
        session = sessions['Session'][i]
    
        # Load data
        try:
            folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
            main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
            data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
        except:
            folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
            main_npy_filename = f"data_main_{mouse}_{session}_BCI.npy"
            data = np.load(os.path.join(folder, main_npy_filename), allow_pickle=True)
    
        # Timing and behavioral signals
        dt_si = data['dt_si']
        rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
        rt[np.isnan(rt)] = 20
        dfaxon = data['ch1']['df_closedloop']
    
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
    
        # Compute RPE
        rpe = compute_rpe(rt == 20, baseline=1, window=20, fill_value=50)
    
        # Reward-aligned responses
        rta = reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4))
        sta = reward_aligned_responses(dfaxon, trial_start_vector, dt_si, window=(-2, 4))
    
        # Get rewarded frame indices
        rewarded_frames = np.where(reward_vector > 0)[0]
        pre_pts = int(2 / dt_si)
        post_pts = int(4 / dt_si)
        valid_reward_frames = rewarded_frames[
            (rewarded_frames - pre_pts >= 0) &
            (rewarded_frames + post_pts < dfaxon.shape[1])
        ]
        valid_reward_frames = valid_reward_frames[:rta.shape[2]]  # match rta
    
        # --- Map reward frames to trial indices using trial_start_vector ---
        trial_starts = np.where(trial_start_vector > 0)[0]  # one per trial
        assert len(trial_starts) == len(rpe), "Trial count mismatch"
        
        # Match number of reward-aligned frames to rta
        valid_reward_frames = valid_reward_frames[:rta.shape[2]]
        
        # Find the trial each reward frame belongs to
        reward_trial_inds = np.searchsorted(trial_starts, valid_reward_frames, side='right') - 1
        
        # Keep only valid trial indices
        reward_trial_inds = reward_trial_inds[(reward_trial_inds >= 0) & (reward_trial_inds < len(rpe))]
        
        # Get corresponding RPE values
        rpe_rewarded = rpe[reward_trial_inds]
    
    
        # Sanity check
        assert rta.shape[2] == len(rpe_rewarded), "Mismatch between rta trials and rpe_rewarded"
    
        # Average across neurons → (time, trials)
        mean_rta = np.nanmean(rta, axis=1)
    
        # Bin trials into low / medium / high RPE groups
        percentiles = np.percentile(rpe_rewarded, [33, 66])
        low_inds  = np.where(rpe_rewarded <= percentiles[0])[0]
        med_inds  = np.where((rpe_rewarded > percentiles[0]) & (rpe_rewarded <= percentiles[1]))[0]
        high_inds = np.where(rpe_rewarded > percentiles[1])[0]
    
        # Time vector
        n_timepoints = mean_rta.shape[0]
        time = np.linspace(-2, 4, n_timepoints)
        
        # for ii in range(mean_rta.shape[1]):
        #     bl = np.nanmean(mean_rta[35:45,ii])
        #     mean_rta[:,ii] = mean_rta[:,ii] - bl; 
        
        # Plot PSTHs
        # plt.figure(figsize=(6, 4))
        for trial_inds, label, color in zip([low_inds, high_inds],
                                            ['Low RPE', 'High RPE'],
                                            ['blue', 'red']):
            trace = np.nanmean(mean_rta[:, trial_inds], axis=1)
            sem   = np.nanstd(mean_rta[:, trial_inds], axis=1) / np.sqrt(len(trial_inds))
            # plt.plot(time, trace, label=label, color=color)
            # plt.fill_between(time, trace - sem, trace + sem, color=color, alpha=0.3)
    
        # plt.axvline(0, color='k', linestyle='--')
        # plt.xlabel('Time from reward (s)')
        # plt.ylabel('Population avg dF/F')
        # plt.title('Reward-aligned PSTH by RPE tercile')
        # plt.legend()
        # plt.tight_layout()
        RTA_low.append(np.nanmean(mean_rta[:,low_inds],1))
        RTA_high.append(np.nanmean(mean_rta[:,high_inds],1))
        # plt.show()
        pp,cc = [],[]
        for ti in range(mean_rta.shape[0]):
            r, p = pearsonr(mean_rta[ti, :], rpe_rewarded)
            cc.append(r)
            pp.append(p)
        CC_RPE.append(cc)
        PP_RPE.append(pp)
        RPE.append(rpe_rewarded)
        RPE_all.append(rpe)
        RT_all.append(rt)
        RTA.append(mean_rta)
        STA.append(np.nanmean(sta,axis=1))
        RT.append(rt[reward_trial_inds])
    except:
        continue
#%%
for i in range(2):
    if i == 1:
        color = 'b'
        a = np.stack(RTA_high)
    else:
        color = 'r'
        a = np.stack(RTA_low)
    trace = np.nanmean(a, 0)
    sem   = np.nanstd(a, 0) / np.sqrt(len(RTA_high))
    plt.plot(time, trace, label=label, color=color)
    plt.fill_between(time, trace - sem, trace + sem, color=color, alpha=0.3)


#%%

#%%
rt = np.concatenate(RT)
rpe = np.concatenate(RPE)
rpe_all = np.concatenate(RPE_all)
rta = np.concatenate(RTA,1)
sta = np.concatenate(STA,1)
y = np.nanmean(rta[0:190,:],0)
pre = np.nanmean(sta[0:20,:],0)
plt.subplot(211)
pf.mean_bin_plot(y,rpe)
plt.subplot(212)
pf.mean_bin_plot(y,rt)
plt.show()

#%%

# Pooled data (your style)
rt  = np.concatenate(RT).astype(float)
rpe = np.concatenate(RPE).astype(float)
rta = np.concatenate(RTA,1).astype(float)
y   = np.nanmean(rta[30:50,:],0).astype(float)     # window avg

# Design: [RPE, logRT, 1]
logrt = np.log(np.clip(rt, 1e-6, None))
X = np.c_[rpe, logrt, np.ones_like(rpe)]
m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
X = X[m]; y = y[m]

# OLS fit (raw coefficients)
b = np.linalg.lstsq(X, y, rcond=None)[0]
yhat = X @ b
R2 = 1 - np.sum((y - yhat)**2)/np.sum((y - np.mean(y))**2)

# Standardized coefficients (z-score X[:,0:2] and y, keep intercept)
Xz = X.copy(); Xz[:,:2] = (X[:,:2] - X[:,:2].mean(0)) / X[:,:2].std(0, ddof=0)
yz = (y - y.mean()) / y.std(ddof=0)
bz = np.linalg.lstsq(np.c_[Xz[:,:2], np.ones(len(y))], yz, rcond=None)[0]  # [b_RPE, b_logRT, b0]

# Plots
plt.figure(figsize=(9,4.2))
plt.subplot(1,2,1)
pf.mean_bin_plot(y, yhat)
mn, mx = np.nanmin([y,yhat]), np.nanmax([y,yhat])

plt.xlabel('True ΔF/F (window mean)'); plt.ylabel('Fitted')
plt.title(f'OLS fit with RPE + logRT   R²={R2:.2f}')

plt.subplot(1,2,2)
plt.bar(['RPE','logRT'], bz[:2], alpha=.85)
plt.axhline(0,color='k',lw=1)
plt.ylabel('Standardized coefficient'); plt.title('Effect per 1 SD predictor')
plt.tight_layout(); plt.show()

print(f'Raw coeffs: RPE={b[0]:+.4g}, logRT={b[1]:+.4g}, intercept={b[2]:+.4g}')
print(f'Standardized: RPE={bz[0]:+.3f}, logRT={bz[1]:+.3f}')


from numpy.linalg import lstsq, inv
from scipy.stats import t as tdist

# ---------- analytic significance for standardized coefs ----------
# Build standardized design (match your bar plot)
Z = X.copy()
Z[:,:2] = (X[:,:2] - X[:,:2].mean(0)) / X[:,:2].std(0, ddof=0)  # zRPE, zlogRT
Z = np.c_[Z[:,:2], np.ones(len(y))]  # [zRPE, zlogRT, 1]

# Fit on standardized (same 'bz' you computed)
bz = lstsq(Z, (y - y.mean())/y.std(ddof=0), rcond=None)[0]   # length 3
yhat_z = Z @ bz
resid  = (y - y.mean())/y.std(ddof=0) - yhat_z
n, p = Z.shape
dof = n - p
sigma2 = (resid @ resid) / dof
XtX_inv = inv(Z.T @ Z)
se = np.sqrt(np.diag(XtX_inv) * sigma2)        # SEs for [zRPE, zlogRT, intercept]
tvals = bz / se
pvals = 2 * tdist.sf(np.abs(tvals), dof)

# 95% CI for the first two (zRPE, zlogRT)
ci95_lo = bz[:2] - 1.96 * se[:2]
ci95_hi = bz[:2] + 1.96 * se[:2]

# ---------- plot with black bars + error bars + stars ----------
labels = ['RPE','logRT']
vals   = bz[:2]
yerr   = np.vstack([vals - ci95_lo, ci95_hi - vals])

plt.figure(figsize=(9,4.2))
plt.subplot(1,2,1)
pf.mean_bin_plot(y, yhat)  # your non-CV fit scatter (keep as-is)
plt.xlabel('Axon signal ($\Delta$F/F)'); 
plt.ylabel('Predicted Axon signal')


plt.subplot(1,2,2)
x = np.arange(2)
plt.bar(x, vals,
        facecolor='white', edgecolor='black', linewidth=1.5,
        yerr=yerr, capsize=5)
plt.axhline(0, color='k', lw=1)
for i,(v,pv) in enumerate(zip(vals, pvals[:2])):
    star = '***' if pv<1e-3 else '**' if pv<1e-2 else '*' if pv<0.05 else ''
    if star:
        plt.text(i, v + (0.05 if v>=0 else -0.05), star,
                 ha='center', va='bottom' if v>=0 else 'top', fontsize=14)
plt.xticks(x, labels)
plt.ylabel('Standardized coefficient')
plt.title('Effect per 1 SD predictor (95% CI, analytic)')
plt.tight_layout(); plt.show()

print(f"z-coef RPE = {bz[0]:+.3f} (SE={se[0]:.3f}, t={tvals[0]:.2f}, p={pvals[0]:.3g})")
print(f"z-coef logRT = {bz[1]:+.3f} (SE={se[1]:.3f}, t={tvals[1]:.2f}, p={pvals[1]:.3g})")

# Overall model F-test on raw design X (with intercept)
e   = y - yhat
n,p = X.shape
dof1, dof2 = p-1, n-p
SSR = np.sum((yhat - y.mean())**2)         # regression sum of squares
SSE = np.sum(e**2)                          # error sum of squares
F   = (SSR/dof1) / (SSE/dof2)
from scipy.stats import f as fdist
p_F = 1 - fdist.cdf(F, dof1, dof2)
print(f"Model F={F:.2f} (df={dof1},{dof2}), p={p_F:.3g}")


#%%
from scipy.stats import pearsonr
rng = np.random.default_rng(0)

# --- bootstrap standardized coefs (bz) ---
B = 2000
boot = np.zeros((B, 2))
for b in range(B):
    idx = rng.choice(len(y), len(y), replace=True)
    Xb, yb = X[idx,:2], y[idx]
    # z-score
    Xbz = (Xb - Xb.mean(0)) / Xb.std(0, ddof=0)
    ybz = (yb - yb.mean()) / yb.std(ddof=0)
    Ab = np.c_[Xbz, np.ones(len(yb))]
    bb = np.linalg.lstsq(Ab, ybz, rcond=None)[0]
    boot[b,:] = bb[:2]

ci_lo = np.percentile(boot, 2.5, axis=0)
ci_hi = np.percentile(boot, 97.5, axis=0)
means = bz[:2]

# significance stars
stars = []
for m, lo, hi in zip(means, ci_lo, ci_hi):
    if lo > 0 or hi < 0:
        stars.append('*')
    else:
        stars.append('')

# --- plot ---
plt.figure(figsize=(9,4.2))
plt.subplot(1,2,1)
pf.mean_bin_plot(y, yhat_cv)
plt.xlabel('True ΔF/F (window mean)'); plt.ylabel('CV-predicted')
plt.title(f'KFold CV (RPE + logRT)   R²={R2_cv:.2f}')

plt.subplot(1,2,2)
x = np.arange(len(means))
plt.bar(x, means, yerr=[means-ci_lo, ci_hi-means], color='k', alpha=0.8, capsize=5)
for xi, yi, s in zip(x, means, stars):
    if s:
        plt.text(xi, yi + (0.05 if yi>=0 else -0.05), s, ha='center', va='bottom' if yi>=0 else 'top', fontsize=14)
plt.axhline(0, color='k', lw=1)
plt.xticks(x, ['RPE','logRT'])
plt.ylabel('Standardized coefficient')
plt.title('Effect per 1 SD predictor\n(bootstrap 95% CI)')
plt.tight_layout(); plt.show()

# correlation of CV fit
r,p = pearsonr(y, yhat_cv)
print("Pearson r between y and yhat_cv =", r, "p =", p)
#%%
#%%
rt_all = np.concatenate(RT)
rpe = np.concatenate(RPE)
rta = np.concatenate(RTA,1)
pp,cc = [],[]
for ti in range(mean_rta.shape[0]):
    r, p = pearsonr(rta[ti, :], rpe)
    cc.append(r)
    pp.append(p)
plt.plot(time,-np.log(pp)*np.sign(cc))
#%%
# --- Bootstrap errorbars for correlation coefficients ---
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

B = 1000
rng = np.random.default_rng(0)

n_time = rta.shape[0]
lb = np.full(n_time, np.nan)   # lower 95% CI
ub = np.full(n_time, np.nan)   # upper 95% CI
boot_mean = np.full(n_time, np.nan)

for ti in range(n_time):
    valid = np.isfinite(rpe) & np.isfinite(rta[ti, :])
    n = valid.sum()
    if n < 3:
        continue

    idx = np.where(valid)[0]
    vals = np.empty(B)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        r_b, _ = pearsonr(rta[ti, samp], rpe[samp])
        vals[b] = r_b

    boot_mean[ti] = np.nanmean(vals)
    lb[ti], ub[ti] = np.nanpercentile(vals, [2.5, 97.5])


# Plot correlation trace with bootstrap CIs
plt.figure(figsize=(6, 4))
plt.plot(time, cc, lw=2, color="k", label="Correlation (point estimate)")
plt.fill_between(time, lb, ub, alpha=0.3, color="gray", label="95% bootstrap CI")
plt.plot(plt.xlim(),(0,0),'k:')
plt.xlabel("Time from reward (s)")
plt.ylabel("Pearson r")
plt.title("Correlation vs. time with bootstrap CIs")
plt.legend()
plt.tight_layout()

