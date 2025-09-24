# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os
#sessions = session_counting.counter2(["BCINM_027","BCINM_017"],'010112',has_pophys=False)
#sessions = session_counting.counter2(["BCINM_021","BCINM_024"],'010112',has_pophys=False)
sessions = session_counting.counter2(["BCINM_031","BCINM_034"],'010112',has_pophys=False)
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
BETA,RTA,STA = [],[],[]
num = 1000
plot = 1
# accumulators (population-avg per trial)
pre_list, early_list, late_list, rew_list = [], [], [], []
trl_list, rpe_list, rt_list, hit_list,hit_rpe_list,miss_rpe_list = [], [], [], [],[],[]
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
        dfaxon = data['ch1']['df_closedloop']
    
        step_vector, reward_vector, trial_start_vector = bts.bci_time_series_fun(folder, data, rt, dt_si)
        rt[np.isnan(rt)] = 20

        # Compute RPE
        rpe_window = 5
        hit_rpe = compute_rpe(rt == 20, baseline=1, window=rpe_window, fill_value=50)
        miss_rpe = compute_rpe(rt != 20, baseline=1, window=rpe_window, fill_value=50)
        rpe = compute_rpe(rt, baseline=5, window=rpe_window, fill_value=50)
    
        # Reward-aligned responses
        rta = reward_aligned_responses(dfaxon, reward_vector, dt_si, window=(-2, 4))
        F = data['ch1']['F']
    
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
        STA.append(np.nanmean(F,axis=1))
        RT.append(rt[reward_trial_inds])
        
        
        trl = np.arange(1,len(rpe)+1)
        pre = np.nanmean(F[1:20,:,:],0)
        rewr = np.nanmean(rta[40:80,:,:],0)
        hit = rt!=20
        miss = rt==20
        later  = np.nanmean(rta[20:40,:,:],0)
        early = np.nanmean(F[40:80,:,:],0)
        other = np.nanmean(F[-20:,:,:],0)
        late = pre*0;rew = pre*0
        late[:,hit] = later;
        late[:,miss] = other[:,miss];
        rew[:,hit] = rewr;
        rew[:,miss] = other[:,miss];
        
        import numpy as np
        from numpy.linalg import lstsq, inv
        from scipy.stats import t as tdist
        import matplotlib.pyplot as plt
        
        # --- detrend helper (linear by default) ---
        def detrend_trials(Y, trl, order=1, lam=1e-6):
            """
            Remove polynomial trend over trials from each ROI's timecourse.
            Y: (rois x trials); trl: (trials,)
            order=1 → linear; lam adds tiny ridge for stability.
            """
            Y = np.asarray(Y, float)
            t = np.asarray(trl, float)
            t = (t - np.nanmean(t)) / (np.nanstd(t) if np.nanstd(t) > 0 else 1.0)
            T = np.column_stack([t**k for k in range(1, order+1)] + [np.ones_like(t)])
            Yd = np.full_like(Y, np.nan, float)
            I = np.eye(T.shape[1])
            for r in range(Y.shape[0]):
                y = Y[r, :]
                valid = np.isfinite(y)
                if valid.sum() < (order + 1):
                    mu = np.nanmean(y)
                    Yd[r, :] = y - (mu if np.isfinite(mu) else 0.0)
                    continue
                M = T[valid, :]
                yr = y[valid]
                beta = np.linalg.solve(M.T @ M + lam * I, M.T @ yr)
                Yd[r, :] = y - (T @ beta)   # subtract trend for ALL trials
            return Yd
        
        # --- predictors (trials,) ---
        trl  = np.arange(1, len(rpe)+1).astype(float)
        hit  = (rt != 20).astype(float)                       # miss is baseline
        logrt = np.log(np.clip(rt.astype(float), 1e-6, None)) # or use raw rt
        
        # --- responses (ROIs x trials) already built upstream: pre, early, late, rew ---
        # Detrend each epoch across trials (per ROI), then population-average per trial
        pre_dt   = detrend_trials(pre,   trl, order=1)
        early_dt = detrend_trials(early, trl, order=1)
        late_dt  = detrend_trials(late,  trl, order=1)
        rew_dt   = detrend_trials(rew,   trl, order=1)
        
        Y = {
            "pre":   np.nanmean(pre_dt,   axis=0),
            "early": np.nanmean(early_dt, axis=0),
            "late":  np.nanmean(late_dt,  axis=0),
            "rew":   np.nanmean(rew_dt,   axis=0),
        }
        
        # shared design AFTER detrending: drop Trial (we removed its contribution)
        X = np.c_[hit, hit_rpe, 10-rt, rpe.astype(float), np.ones_like(hit, float)]  # [RPE, logRT, Hit, 1]
        preds = ["Hit", "$RPE_{hit}$", "Reward time", "$RPE_{RT}$"]
        
        def fit_one(X, y):
            m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X, y = X[m], y[m]
        
            # OLS fit (raw) for fitted values / R^2
            b  = lstsq(X, y, rcond=None)[0]
            yhat = X @ b
            R2 = 1 - np.sum((y-yhat)**2)/np.sum((y - np.mean(y))**2)
        
            # Standardized coefs (z-score predictors except intercept, and y)
            Xs = X.copy()
            Xs[:,:-1] = (Xs[:,:-1] - Xs[:,:-1].mean(0)) / Xs[:,:-1].std(0, ddof=0)
            ys = (y - y.mean()) / y.std(ddof=0)
            bz = lstsq(Xs, ys, rcond=None)[0]
        
            # Analytic SE/p for standardized coefs
            resid = ys - Xs @ bz
            n,p = Xs.shape
            dof = n - p
            sigma2 = (resid @ resid) / dof
            XtX_inv = inv(Xs.T @ Xs)
            se = np.sqrt(np.diag(XtX_inv) * sigma2)
            tvals = bz / se
            pvals = 2 * tdist.sf(np.abs(tvals), dof)
            return dict(b=b, bz=bz, se=se, p=pvals, R2=R2, y=y, yhat=yhat)
        
        def pstar(p): return '***' if p<1e-3 else '**' if p<1e-2 else '*' if p<0.05 else ''
        
        for key in ["pre","early","late","rew"]:
            res = fit_one(X, Y[key])
        
            # Left: your bin-plot of fitted vs true
            plt.figure(figsize=(9,4.2))
            plt.subplot(1,2,1)
            pf.mean_bin_plot(res["y"], res["yhat"])
            plt.xlabel(f"True ΔF/F ({key}, detrended)")
            plt.ylabel("Fitted")
            plt.title(f"{key}: OLS (Trial removed)  R²={res['R2']:.2f}")
        
            # Right: black-outline bars with 95% CI + stars (standardized coefs)
            vals = res["bz"][:len(names)]
            ci_lo = vals - 1.96*res["se"][:len(names)]
            ci_hi = vals + 1.96*res["se"][:len(names)]
            x = np.arange(len(names))
            plt.subplot(1,2,2)
            plt.bar(x, vals, facecolor='white', edgecolor='black', linewidth=1.5,
                    yerr=np.vstack([vals-ci_lo, ci_hi-vals]), capsize=5)
            for i in range(len(names)):
                s = pstar(res["p"][i])
                if s:
                    off = 0.05 if vals[i] >= 0 else -0.05
                    va  = 'bottom' if vals[i] >= 0 else 'top'
                    plt.text(i, vals[i] + off, s, ha='center', va=va, fontsize=12)
            plt.axhline(0, color='k', lw=1)
            plt.xticks(x, names)
            plt.ylabel("Std. coefficient (per 1 SD)")
            plt.title(f"{key}: effects after detrending")
            plt.tight_layout(); plt.show()
        import numpy as np
        import matplotlib.pyplot as plt
        from numpy.linalg import lstsq, inv
        from scipy.stats import t as tdist
        
        epochs = ["pre","early","late","rew"]
        preds = ["Hit", "$RPE_{hit}$", "Speed", "$RPE_{speed}$"]
        
        def fit_std(X, y):
            m = np.isfinite(y) & np.all(np.isfinite(X), 1)
            X, y = X[m], y[m]
            # raw fit for R^2 / yhat
            b = lstsq(X, y, rcond=None)[0]
            yhat = X @ b
            R2 = 1 - np.sum((y-yhat)**2)/np.sum((y - np.mean(y))**2)
        
            # standardized (z) for comparable coefs
            Xs = X.copy()
            Xs[:,:-1] = (Xs[:,:-1] - Xs[:,:-1].mean(0)) / Xs[:,:-1].std(0, ddof=0)
            ys = (y - y.mean()) / y.std(ddof=0)
            bz = lstsq(Xs, ys, rcond=None)[0]
        
            resid = ys - Xs @ bz
            n,p = Xs.shape
            dof = n - p
            sigma2 = (resid @ resid) / dof
            XtX_inv = inv(Xs.T @ Xs)
            se = np.sqrt(np.diag(XtX_inv) * sigma2)
            pvals = 2 * tdist.sf(np.abs(bz / se), dof)
        
            return dict(R2=R2, y=y, yhat=yhat, bz=bz[:-1], p=pvals[:-1], mask=m)  # drop intercept for coefs/p
        
        def delta_R2(X, y):
            m = np.isfinite(y) & np.all(np.isfinite(X), 1)
            X, y = X[m], y[m]
            # full
            b = lstsq(X, y, rcond=None)[0]
            yhat = X @ b
            R2_full = 1 - np.sum((y-yhat)**2)/np.sum((y - np.mean(y))**2)
            # ablate each predictor (exclude intercept col = last)
            dR2 = []
            for j in range(X.shape[1]-1):
                Xminus = np.c_[X[:, :j], X[:, j+1:]]  # keep intercept (still last column)
                b2 = lstsq(Xminus, y, rcond=None)[0]
                yhat2 = Xminus @ b2
                R2_minus = 1 - np.sum((y-yhat2)**2)/np.sum((y - np.mean(y))**2)
                dR2.append(R2_full - R2_minus)
            return np.array(dR2), R2_full
        
        # --- collect per-epoch results ---
        coef_mat  = np.zeros((len(epochs), len(preds)))
        p_mat     = np.ones((len(epochs), len(preds)))
        dR2_mat   = np.zeros((len(epochs), len(preds)))
        R2_list   = []
        yhat_dict = {}
        for i, ep in enumerate(epochs):
            res = fit_std(X, Y[ep])
            coef_mat[i,:] = res["bz"]
            p_mat[i,:]    = res["p"]
            yhat_dict[ep] = res["yhat"]
            dR2, R2 = delta_R2(X, Y[ep])
            dR2_mat[i,:] = dR2
            R2_list.append(R2)
        
        # --- Panel A: coefficient heatmap with stars ---
        def pstar(p): return '***' if p<1e-3 else '**' if p<1e-2 else '*' if p<0.05 else ''
        
        plt.figure(figsize=(10, 4.6))
        plt.subplot(1,2,1)
        BETA.append(coef_mat)
        v = np.nanmax(np.abs(coef_mat))
        im = plt.imshow(coef_mat, cmap='coolwarm', vmin=-v, vmax=+v, aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04, label='Standardized β')
        plt.xticks(range(len(preds)), preds)
        plt.yticks(range(len(epochs)), epochs)
        # stars
        for i in range(len(epochs)):
            for j in range(len(preds)):
                s = pstar(p_mat[i,j])
                if s:
                    plt.text(j, i, s, ha='center', va='center', color='k', fontsize=11)
        plt.title('A: Coefficient heatmap (detrended, population avg)')
        
        # --- Panel B: ΔR² heatmap (unique contribution) ---
        plt.subplot(1,2,2)
        mx = np.nanmax(np.abs(dR2_mat))
        im2 = plt.imshow(dR2_mat, cmap='viridis', vmin=0, vmax=mx if mx>0 else 1e-6, aspect='auto')
        plt.colorbar(im2, fraction=0.046, pad=0.04, label='ΔR² (full − minus predictor)')
        plt.xticks(range(len(preds)), preds)
        plt.yticks(range(len(epochs)), epochs)
        plt.title('B: Unique contribution (ΔR²)')
        plt.tight_layout()
        plt.show()
        
        # --- Fit quality overlay: mean_bin_plots for all epochs on one axes ---
        colors = dict(pre='tab:blue', early='tab:orange', late='tab:green', rew='tab:red')
        plt.figure(figsize=(5.2,4.2))
        epoch_colors = {
            "pre":   "#33b983",   # Pretrial
            "early": "#1077f3",   # Early trial
            "late":  "#0050ae",   # Late trial
            "rew":   "#bf8cfc",   # Reward
            "CN":    "#f98517"    # Conditioned neuron
        }
        for ep in epochs:
            pf.mean_bin_plot(Y[ep][np.isfinite(Y[ep]) & np.all(np.isfinite(X),1)],
                             yhat_dict[ep],5,1,1,
                             epoch_colors.get(ep, 'k'))
            plt.plot([], [], color=epoch_colors.get(ep,'k'), label=ep)  # dummy for legend
            plt.legend(frameon=False, title="Epoch")
        mn = min([np.nanmin(Y[e]) for e in epochs])
        mx = max([np.nanmax(Y[e]) for e in epochs])
        
        plt.xlabel('True ΔF/F (population avg)')
        plt.ylabel('Fitted')
        plt.legend(frameon=False, title='Epoch')
        plt.title('Fit calibration (all epochs)')
        plt.tight_layout(); plt.show()
        
        
        
        trl = np.arange(1,len(rpe)+1)
        pre = np.nanmean(F[1:20,:,:],0)
        rewr = np.nanmean(rta[40:80,:,:],0)
        hit = rt!=20
        miss = rt==20
        later  = np.nanmean(rta[20:40,:,:],0)
        early = np.nanmean(F[40:80,:,:],0)
        other = np.nanmean(F[-20:,:,:],0)
        late = pre*0;rew = pre*0
        late[:,hit] = later;
        late[:,miss] = other[:,miss];
        rew[:,hit] = rewr;
        rew[:,miss] = other[:,miss];
        
        
        pre_list.append(pre)
        early_list.append(early)
        late_list.append(late)
        rew_list.append(rew)
        
        trl_list.append(trl)
        rpe_list.append(rpe)
        rt_list.append(rt)
        hit_list.append(hit)
        hit_rpe_list.append(hit_rpe)
        miss_rpe_list.append(miss_rpe)
        
        best_beta = -np.inf
        best_ep   = None
        best_pred = None
        best_Xcol = None
        best_Y    = None
        
        for ep in epochs:
            res = fit_std(X, Y[ep])  # or fit_one depending on which function you use
            bz = res["bz"]           # standardized betas (no intercept)
            
            imax = np.argmax(bz)     # index of biggest positive beta in this epoch
            if bz[imax] > best_beta: # keep track of global best
                best_beta = bz[imax]
                best_ep   = ep
                best_pred = preds[imax]
                best_Xcol = X[:, imax]
                best_Y    = Y[ep]
        
        # --- Make scatter plot for best pair ---
        plt.figure(figsize=(8,4))
        
        # --- Left: time-series traces, scaled to similar amplitude ---
        plt.subplot(121)
        def zscore_trace(x):
            return (x - np.nanmean(x)) / (np.nanstd(x) if np.nanstd(x) > 0 else 1)
        
        plt.plot(zscore_trace(best_Xcol), 'k', label=best_pred)
        plt.plot(zscore_trace(best_Y), color=epoch_colors.get(best_ep, 'r'),
         label=f"{best_ep.capitalize()} ΔF/F")

        plt.legend(frameon=False)
        plt.xlabel("Trial")
        plt.ylabel("Z-scored value")
        plt.title("Predictor vs. Neural Response")
        
        # --- Right: scatter of raw values ---
        plt.subplot(122)
        plt.plot(best_Xcol, best_Y, 'k.')
        plt.xlabel(best_pred)
        plt.ylabel(f"{best_ep.capitalize()} ΔF/F")
        plt.title(f"Strongest effect: {best_pred} during {best_ep}\nβ={best_beta:.2f}")
        
        plt.tight_layout()
        plt.show()





        
    except Exception as e:
        print(type(e).__name__, e)
        continue
#%%
# ================================================================
# Regression with signed vs unsigned RPE terms (Hit removed)
# Options: OLS, Ridge, Lasso
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, inv
from scipy.stats import t as tdist
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler

# ---------- config ----------
method = "Lasso"   # choose: "OLS", "Ridge", or "Lasso"
ridge_alphas = np.logspace(-3, 3, 20)
lasso_alphas = np.logspace(-3, 1, 20)

# ---------- helpers ----------
def zscore_1d(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 0 else 1)

def detrend_trials(Y, order=1, lam=1e-6):
    Y = np.asarray(Y, float)
    trl = np.arange(Y.shape[1])
    trl = (trl - np.nanmean(trl)) / (np.nanstd(trl) if np.nanstd(trl) > 0 else 1.0)
    T = np.column_stack([trl**k for k in range(1, order+1)] + [np.ones_like(trl)])
    Yd = np.full_like(Y, np.nan, float)
    I = np.eye(T.shape[1])
    for r in range(Y.shape[0]):
        y = Y[r, :]
        valid = np.isfinite(y)
        if valid.sum() < (order + 1):
            mu = np.nanmean(y)
            Yd[r, :] = y - (mu if np.isfinite(mu) else 0.0)
            continue
        M = T[valid, :]; yr = y[valid]
        beta = np.linalg.solve(M.T @ M + lam*I, M.T @ yr)
        Yd[r, :] = y - (T @ beta)
    return Yd

def fit_std(X, y):
    """Regression with standardized predictors and response."""
    y = np.ravel(y)
    m = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[m], y[m]

    if method == "Ridge":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        Xs = scaler_X.fit_transform(X[:, :-1])  # exclude intercept
        ys = scaler_y.fit_transform(y[:, None]).ravel()

        ridge = RidgeCV(alphas=ridge_alphas, store_cv_values=True).fit(Xs, ys)
        bz = ridge.coef_
        yhat = scaler_y.inverse_transform(ridge.predict(Xs).reshape(-1, 1)).ravel()
        R2 = ridge.score(Xs, ys)

        return dict(R2=R2, y=y, yhat=yhat, bz=bz, p=[np.nan]*len(bz), mask=m)

    elif method == "Lasso":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        Xs = scaler_X.fit_transform(X[:, :-1])
        ys = scaler_y.fit_transform(y[:, None]).ravel()

        lasso = LassoCV(alphas=lasso_alphas, cv=5, fit_intercept=False).fit(Xs, ys)
        bz = lasso.coef_
        yhat = scaler_y.inverse_transform(lasso.predict(Xs).reshape(-1, 1)).ravel()
        R2 = lasso.score(Xs, ys)

        return dict(R2=R2, y=y, yhat=yhat, bz=bz, p=[np.nan]*len(bz), mask=m)

    else:  # OLS
        b = lstsq(X, y, rcond=None)[0]
        yhat = X @ b
        R2 = 1 - np.sum((y-yhat)**2) / np.sum((y - np.mean(y))**2)

        # standardized
        Xs = X.copy()
        sd = Xs[:,:-1].std(0, ddof=0); sd[sd==0] = 1.0
        Xs[:,:-1] = (Xs[:,:-1] - Xs[:,:-1].mean(0)) / sd
        sdy = y.std(ddof=0) or 1.0
        ys  = (y - y.mean()) / sdy
        bz = lstsq(Xs, ys, rcond=None)[0]

        resid = ys - Xs @ bz
        n,p = Xs.shape
        dof = max(n-p, 1)
        sigma2 = (resid @ resid) / dof
        XtX_inv = inv(Xs.T @ Xs)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        pvals = 2 * tdist.sf(np.abs(bz / se), dof)

        return dict(R2=R2, y=y, yhat=yhat, bz=bz[:-1], p=pvals[:-1], mask=m)

def pstar(p): 
    return '***' if p<1e-3 else '**' if p<1e-2 else '*' if p<0.05 else ''

# ---------- build predictors ----------
rpe_all, rt_all, hit_rpe_all = [], [], []
pre_all, early_all, late_all, rew_all = [], [], [], []

for pre_s, early_s, late_s, rew_s, rpe_s, rt_s, hit_rpe_s in zip(
        pre_list, early_list, late_list, rew_list,
        rpe_list, rt_list, hit_rpe_list):

    pre_all.append(np.nanmean(detrend_trials(pre_s),   axis=0))
    early_all.append(np.nanmean(detrend_trials(early_s), axis=0))
    late_all.append(np.nanmean(detrend_trials(late_s),  axis=0))
    rew_all.append(np.nanmean(detrend_trials(rew_s),   axis=0))

    rpe_all.append(zscore_1d(rpe_s))
    rt_all.append(zscore_1d(10-rt_s))
    hit_rpe_all.append(zscore_1d(hit_rpe_s))

# concat across sessions
rpe     = np.concatenate(rpe_all)
rt      = np.concatenate(rt_all)
hit_rpe = np.concatenate(hit_rpe_all)

# unsigned versions
rpe_abs     = np.abs(rpe)
hit_rpe_abs = np.abs(hit_rpe)

pre   = np.concatenate(pre_all)
early = np.concatenate(early_all)
late  = np.concatenate(late_all)
rew   = np.concatenate(rew_all)

Y = {"pre": pre, "early": early, "late": late, "rew": rew}

# ---------- model variants ----------
models = {
    "Signed only": np.c_[hit_rpe, rt, rpe, np.ones_like(rpe)],
    "Unsigned only": np.c_[hit_rpe_abs, rpe_abs, rt, np.ones_like(rpe)],
    "Both": np.c_[hit_rpe, rpe, hit_rpe_abs, rpe_abs, rt, np.ones_like(rpe)]
}
pred_labels = {
    "Signed only": ["$RPE_{hit}$", "$RPE_{speed}$", "Speed"],
    "Unsigned only": ["|$RPE_{hit}$|", "|$RPE_{speed}$|", "Speed"],
    "Both": ["$RPE_{hit}$", "$RPE_{speed}$", "|$RPE_{hit}$|", "|$RPE_{speed}$|", "Speed"]
}

# ---------- run fits ----------
results = {}
for model_name, X in models.items():
    coef_mat  = np.zeros((len(Y), X.shape[1]-1))  # drop intercept
    p_mat     = np.ones((len(Y), X.shape[1]-1))
    R2_list   = []

    for i, ep in enumerate(Y.keys()):
        res = fit_std(X, Y[ep])
        coef_mat[i, :] = res["bz"]
        p_mat[i, :]    = res["p"]
        R2_list.append(res["R2"])

    results[model_name] = (coef_mat, p_mat, R2_list)

# ---------- plot ----------
for model_name, (coef_mat, p_mat, R2_list) in results.items():
    preds = pred_labels[model_name]
    plt.figure(figsize=(7,5))
    v = np.nanmax(np.abs(coef_mat))
    im = plt.imshow(coef_mat, cmap='coolwarm', vmin=-v, vmax=+v, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Standardized β')
    plt.xticks(range(len(preds)), preds, rotation=30)
    plt.yticks(range(len(Y)), list(Y.keys()))
    for i in range(len(Y)):
        for j in range(len(preds)):
            s = pstar(p_mat[i,j])
            if s:
                plt.text(j, i, s, ha='center', va='center', color='k', fontsize=11)
    plt.title(f'{model_name} ({method})')
    plt.tight_layout()
    plt.show()
