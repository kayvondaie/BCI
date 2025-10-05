# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:03:24 2025

@author: kayvon.daie
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os

sessions = session_counting.counter2(["BCINM_031","BCINM_034"],'010112',has_pophys=False)
#sessions = session_counting.counter2(["BCINM_027","BCINM_017"],'010112',has_pophys=False)
#sessions = session_counting.counter2(["BCINM_021","BCINM_024"],'010112',has_pophys=False)

#%%
from scipy.signal import medfilt, correlate
from scipy.stats import pearsonr
p = r"C:\Users\kayvon.daie\Documents\GitHub\BCI\LC_axon_analysis"
assert os.path.isdir(p), f"Not a directory: {p}"
if p not in sys.path:
    sys.path.insert(0, p)
from axon_helper_module import *
from BCI_data_helpers import *
import bci_time_series as bts

processing_mode = 'all'
si = 10
inds = np.arange(len(sessions)) if processing_mode == 'all' else np.arange(si, si + 1)

# accumulators (population-avg per trial)
pre_list, early_list, late_list, rew_list = [], [], [], []
trl_list, rpe_list, rt_list, hit_list, hit_rpe_list, miss_rpe_list = [], [], [], [], [], []

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
        hit_rpe  = compute_rpe(rt == 20, baseline=1, window=rpe_window, fill_value=50)
        miss_rpe = compute_rpe(rt != 20, baseline=1, window=rpe_window, fill_value=50)
        rpe      = compute_rpe(rt, baseline=5, window=rpe_window, fill_value=50)

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
        valid_reward_frames = valid_reward_frames[:rta.shape[2]]

        # Trial alignment check
        trial_starts = np.where(trial_start_vector > 0)[0]
        assert len(trial_starts) == len(rpe), "Trial count mismatch"

        # --- Build per-epoch responses (pre, early, late, reward) ---
        trl = np.arange(1, len(rpe) + 1)
        pre   = np.nanmean(F[1:20, :, :], 0)
        rewr  = np.nanmean(rta[40:80, :, :], 0)
        hit   = rt != 20
        miss  = rt == 20
        later = np.nanmean(rta[20:40, :, :], 0)
        early = np.nanmean(F[40:80, :, :], 0)
        other = np.nanmean(F[-20:, :, :], 0)

        late = pre * 0
        rew  = pre * 0
        late[:, hit] = later
        late[:, miss] = other[:, miss]
        rew[:, hit]  = rewr
        rew[:, miss] = other[:, miss]

        # --- Append to accumulators ---
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

    except Exception as e:
        print(type(e).__name__, e)
        continue
#%%
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 8,        # base font size
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8
})

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
    "Both": np.c_[hit_rpe, rpe, hit_rpe_abs, rpe_abs, rt, np.ones_like(rpe)]
}
pred_labels = {
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

# ---------- plot (split signed vs unsigned) ----------
for model_name, (coef_mat, p_mat, R2_list) in results.items():
    preds = pred_labels[model_name]

    # indices
    signed_idx   = [0,1,4]  # hit, speed, Speed
    unsigned_idx = [2,3]    # |hit|, |speed|

    coef_signed   = coef_mat[:, signed_idx]
    coef_unsigned = coef_mat[:, unsigned_idx]
    p_signed      = p_mat[:, signed_idx]
    p_unsigned    = p_mat[:, unsigned_idx]

    preds_signed   = [preds[i] for i in signed_idx]
    preds_unsigned = [preds[i] for i in unsigned_idx]

    v = np.nanmax(np.abs(coef_mat))
    n_epochs = len(Y)

    fig = plt.figure(figsize=(4,2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[len(preds_signed), len(preds_unsigned)],
        left=0.25, right=0.9, top=0.9, bottom=0.15, wspace=0.025
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # signed
    im1 = ax1.imshow(coef_signed, cmap='bwr', vmin=-v, vmax=+v,
                     aspect='auto', origin='upper')
    ax1.set_xticks(range(len(preds_signed)))
    ax1.set_xticklabels(preds_signed, rotation=30)
    ax1.set_yticks(np.arange(n_epochs))
    ax1.set_yticklabels(list(Y.keys()))
    ax1.set_ylim(n_epochs-0.5, -0.5)
    ax1.set_title("Signed")
    for i in range(n_epochs):
        for j in range(len(preds_signed)):
            s = pstar(p_signed[i,j])
            if s:
                ax1.text(j, i, s, ha='center', va='center', color='k', fontsize=11)

    # unsigned
    im2 = ax2.imshow(coef_unsigned, cmap='bwr', vmin=-v, vmax=+v,
                     aspect='auto', origin='upper')
    ax2.set_xticks(range(len(preds_unsigned)))
    ax2.set_xticklabels(preds_unsigned, rotation=30)
    ax2.set_yticks([])
    ax2.set_ylim(n_epochs-0.5, -0.5)
    ax2.set_title("Unsigned")
    for i in range(n_epochs):
        for j in range(len(preds_unsigned)):
            s = pstar(p_unsigned[i,j])
            if s:
                ax2.text(j, i, s, ha='center', va='center', color='k', fontsize=11)

    # shared colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Standardized β')

    plt.show()

# ---------- Fit calibration overlay (mean_bin_plot, all epochs on one axes) ----------
X = models["Both"]

yhat_dict = {}
R2_dict = {}
mask_dict = {}

for ep in Y.keys():
    res = fit_std(X, Y[ep])
    yhat_dict[ep] = res["yhat"]
    R2_dict[ep]   = res["R2"]
    mask_dict[ep] = res["mask"]   # <- use the mask returned by fit_std

epoch_colors = {
    "pre":   "#33b983",
    "early": "#1077f3",
    "late":  "#0050ae",
    "rew":   "#bf8cfc",
    "CN":    "#f98517"
}

fig, ax = plt.subplots(figsize=(2.5,2))

for ep in Y.keys():
    pf.mean_bin_plot(Y[ep][mask_dict[ep]], yhat_dict[ep], 5, 1, 1,
                     epoch_colors.get(ep, 'k'))

ax.set_xlabel('True ΔF/F (population avg)', fontsize=8)
ax.set_ylabel('Fit', fontsize=8)
ax.tick_params(axis='both', labelsize=8)

# --- custom text-only legend ---
for i, ep in enumerate(Y.keys()):
    ax.text(1.02, 0.95 - i*0.1, ep, color=epoch_colors[ep],
            transform=ax.transAxes, fontsize=8, va='top', ha='left')

plt.tight_layout()
plt.show()


