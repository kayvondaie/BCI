# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:25:53 2026

@author: kayvon.daie
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# --- inputs assumed to exist ---
# F: time x neurons x trials
# rt: reaction time (trials,)
# data: dict with 'conditioned_neuron' and 't_bci'
# pf.mean_bin_plot: your helper plotting function

# --- setup ---
cn = data['conditioned_neuron'][0][0]
t = data['t_bci']
ts = np.where(t > 0)[0][0]

# pretrial (t<0) and trial (t>0) averages
pre = np.nanmean(F[:ts, :, :], axis=0)    # neurons x trials
trl = np.nanmean(F[ts:, :, :], axis=0)    # neurons x trials

# predictors: all neurons except CN
non = np.arange(F.shape[1])
non = non[non != cn]

X = pre[non, :].T                 # trials x (neurons-1)
y = trl[cn, :].astype(float)      # trials

# drop trials with NaN target
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

# fill NaNs in X with per-neuron means (keeps your workflow simple)
mu = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
if inds[0].size > 0:
    X[inds] = np.take(mu, inds[1])

# standardize predictors for Lasso
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# sparse regression with CV
cv_folds = min(5, len(y))
lasso = LassoCV(cv=cv_folds, n_jobs=-1, max_iter=20000, random_state=0)
lasso.fit(Xz, y)

beta = lasso.coef_                # sparse weights (len = neurons-1)
yhat = lasso.predict(Xz)          # CV-chosen alpha; in-sample prediction

# --- report selected neurons (photostim candidates) ---
nz = np.where(np.abs(beta) > 1e-8)[0]
stim_neurons = non[nz]
stim_weights = beta[nz]

# rank by absolute weight (or use beta sign if you care about excitatory direction)
order = np.argsort(-np.abs(stim_weights))
stim_neurons = stim_neurons[order]
stim_weights = stim_weights[order]

print(f"CN index: {cn}")
print(f"Using ts={ts} (t_bci crosses 0 at frame {ts})")
print(f"Lasso selected {len(stim_neurons)} / {len(non)} neurons (alpha={lasso.alpha_:.3g})")

# print top candidates
topk = min(20, len(stim_neurons))
for k in range(topk):
    print(f"{k+1:02d}: neuron {stim_neurons[k]}  weight {stim_weights[k]:+.4f}")

# --- plot ---
# If you want *truly* out-of-fold predictions, say so and I'll switch to nested CV.
try:
    pf.mean_bin_plot(yhat, y)
except Exception:
    plt.scatter(yhat, y, s=15, c='k', alpha=0.6)

plt.xlabel("Predicted CN trial activity (Lasso)")
plt.ylabel("Observed CN trial activity")
plt.tight_layout()
plt.show()
