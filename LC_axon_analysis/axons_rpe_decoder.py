# -*- coding: utf-8 -*-
"""
Per-session non-negative decoder of trial-by-trial RPE from individual axons.

- One feature per axon (whole-trial mean of F), no epoch splitting.
- No averaging across axons; each axon is its own predictor.
- Lasso with positive=True so all weights are >= 0.
- Per-session fit (axons are not matched across sessions), then aggregate.
- Cross-validated within-session predictions for honest R^2.

Runs the decoder for all three NM types (5HT, NE, Ach) and produces a single
figure of binned true-vs-decoded RPE (one panel per NM) via pf.mean_bin_plot.
"""
import session_counting
import data_dict_create_module_test as ddc
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "figure.titlesize": 8})

# ---------- config ----------
feature_window = "whole" # "whole" (frames 1:end)
rpe_window = 5
n_bins = 5               # bins for pf.mean_bin_plot

nm_sessions = {
    "5HT": ["BCINM_031", "BCINM_034"],
    "NE":  ["BCINM_027", "BCINM_017"],
    "Ach": ["BCINM_021", "BCINM_024"],
}
nm_colors = {"5HT": "#bf8cfc", "NE": "#1077f3", "Ach": "#33b983"}
nm_order  = ["5HT", "NE", "Ach"]
signed_targets = ["rpe_speed", "rpe_hit", "hit", "rt"]

# ---------- path setup for helpers ----------
p = r"C:\Users\kayvon.daie\Documents\GitHub\BCI\LC_axon_analysis"
assert os.path.isdir(p), f"Not a directory: {p}"
if p not in sys.path:
    sys.path.insert(0, p)
from axon_helper_module import *
from BCI_data_helpers import *
import bci_time_series as bts

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_predict

# ---------- helpers ----------
def zscore_cols(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def zscore_1d(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 0 else 1.0)

def build_target(rt, rpe_window, kind):
    if kind == "rpe_speed":
        y = compute_rpe_standard(rt, baseline=5, window=rpe_window, fill_value=50)
    elif kind == "rpe_hit":
        y = compute_rpe_standard((rt != 20).astype(float), baseline=0,
                                 window=rpe_window, fill_value=0)
    elif kind == "abs_rpe_speed":
        y = np.abs(compute_rpe_standard(rt, baseline=5, window=rpe_window, fill_value=50))
    elif kind == "abs_rpe_hit":
        y = np.abs(compute_rpe_standard((rt != 20).astype(float), baseline=0,
                                        window=rpe_window, fill_value=0))
    elif kind == "hit":
        y = (rt != 20).astype(float)
    elif kind == "rt":
        y = np.asarray(rt, dtype=float)
    else:
        raise ValueError(kind)
    return y

def fit_one_nm(nm_type, target_kind):
    """Run per-session non-negative Lasso decoder. Returns list of session records."""
    sessions = session_counting.counter2(nm_sessions[nm_type], '010112', has_pophys=False)
    records = []

    for i in range(len(sessions)):
        try:
            mouse   = sessions['Mouse'][i]
            session = sessions['Session'][i]

            try:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/pophys/'
                data = np.load(os.path.join(folder, f"data_main_{mouse}_{session}_BCI.npy"),
                               allow_pickle=True)
            except Exception:
                folder = f'//allen/aind/scratch/BCI/2p-raw/{mouse}/{session}/'
                data = np.load(os.path.join(folder, f"data_main_{mouse}_{session}_BCI.npy"),
                               allow_pickle=True)

            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']])
            rt[np.isnan(rt)] = 20

            F = data['ch1']['F']  # (T_frames, n_axons, n_trials)
            if F.ndim != 3:
                print(f"[{nm_type} {i}] {mouse}/{session}: unexpected F shape {F.shape}")
                continue

            if feature_window == "whole":
                X_sess = np.nanmean(F[1:, :, :], axis=0).T  # (n_trials, n_axons)
            else:
                raise NotImplementedError(feature_window)

            y_sess = build_target(rt, rpe_window, target_kind)

            if X_sess.shape[0] != len(y_sess):
                n = min(X_sess.shape[0], len(y_sess))
                X_sess, y_sess = X_sess[:n], y_sess[:n]

            good = np.all(np.isfinite(X_sess), axis=1) & np.isfinite(y_sess)
            X_sess, y_sess = X_sess[good], y_sess[good]
            if X_sess.shape[0] < 20 or X_sess.shape[1] < 2:
                print(f"[{nm_type}/{target_kind} {i}] {mouse}/{session}: too few trials/axons")
                continue
            if np.nanstd(y_sess) == 0:
                print(f"[{nm_type}/{target_kind} {i}] {mouse}/{session}: zero variance in target")
                continue

            Xs = zscore_cols(X_sess)
            ys = zscore_1d(y_sess)

            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=cv,
                            positive=True, fit_intercept=True, max_iter=20000)
            lasso.fit(Xs, ys)

            yhat_cv = cross_val_predict(
                LassoCV(alphas=[lasso.alpha_], cv=cv, positive=True,
                        fit_intercept=True, max_iter=20000),
                Xs, ys, cv=cv)
            ss_res = np.sum((ys - yhat_cv) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            cv_r2  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            records.append(dict(
                mouse=mouse, session=session,
                n_trials=Xs.shape[0], n_axons=Xs.shape[1],
                alpha=lasso.alpha_, w=lasso.coef_,
                train_r2=lasso.score(Xs, ys),
                cv_r2=cv_r2, y=ys, yhat_cv=yhat_cv,
            ))
            print(f"[{nm_type}/{target_kind} {i:2d}] {mouse}/{session}: "
                  f"trials={Xs.shape[0]}, axons={Xs.shape[1]}, alpha={lasso.alpha_:.3g}, "
                  f"train R2={lasso.score(Xs, ys):.3f}, CV R2={cv_r2:.3f}, "
                  f"nz_w={(lasso.coef_>0).sum()}/{lasso.coef_.size}")

        except Exception as e:
            print(f"[{nm_type}/{target_kind} {i}] error: {type(e).__name__}: {e}")
            continue

    print(f"  -> {nm_type}/{target_kind}: fit {len(records)} sessions")
    return records

# ---------- run all (NM x target) combinations ----------
all_records = {}  # keyed by (nm, target)
for nm in nm_order:
    for tgt in signed_targets:
        print(f"\n=== {nm} / {tgt} ===")
        all_records[(nm, tgt)] = fit_one_nm(nm, tgt)

#%% ---------- NM x variable correlation matrix (Pearson r on CV preds) ----------
r_matrix = np.full((len(nm_order), len(signed_targets)), np.nan)
for i, nm in enumerate(nm_order):
    for j, tgt in enumerate(signed_targets):
        recs = all_records[(nm, tgt)]
        if len(recs) == 0:
            continue
        y_all    = np.concatenate([r["y"]       for r in recs])
        yhat_all = np.concatenate([r["yhat_cv"] for r in recs])
        if np.std(y_all) == 0 or np.std(yhat_all) == 0:
            continue
        r_matrix[i, j] = np.corrcoef(y_all, yhat_all)[0, 1]

fig, ax = plt.subplots(figsize=(3.6, 2.4))
v = max(0.5, np.nanmax(np.abs(r_matrix)))
im = ax.imshow(r_matrix, cmap='bwr', vmin=-v, vmax=+v, aspect='auto')
ax.set_xticks(range(len(signed_targets)))
ax.set_xticklabels(signed_targets, rotation=30, ha='right')
ax.set_yticks(range(len(nm_order)))
ax.set_yticklabels(nm_order)
for i in range(len(nm_order)):
    for j in range(len(signed_targets)):
        if np.isfinite(r_matrix[i, j]):
            ax.text(j, i, f"{r_matrix[i, j]:.2f}", ha='center', va='center',
                    fontsize=8, color='k' if abs(r_matrix[i, j]) < 0.6 else 'w')
ax.set_title("Pearson r (CV)  |  decoder true-vs-pred")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

#%% ---------- 4 (targets) x 3 (NMs) grid of binned scatter plots ----------
fig, axes = plt.subplots(len(signed_targets), len(nm_order),
                         figsize=(7.5, 7.5), sharex='row', sharey='row')
for j, tgt in enumerate(signed_targets):
    for i, nm in enumerate(nm_order):
        ax = axes[j, i]
        recs = all_records[(nm, tgt)]
        if len(recs) == 0:
            ax.set_title(f"{nm} / {tgt}: no data", fontsize=7)
            continue
        y_all    = np.concatenate([r["y"]       for r in recs])
        yhat_all = np.concatenate([r["yhat_cv"] for r in recs])
        plt.sca(ax)
        pf.mean_bin_plot(y_all, yhat_all, n_bins, 1, 1, nm_colors[nm])
        r_val = np.corrcoef(y_all, yhat_all)[0, 1] if np.std(yhat_all) > 0 else np.nan
        ax.set_title(f"{nm} / {tgt}  r={r_val:.2f}", fontsize=7)
        if j == len(signed_targets) - 1:
            ax.set_xlabel("true (z)")
        if i == 0:
            ax.set_ylabel("decoded (z, CV)")
plt.tight_layout(); plt.show()
