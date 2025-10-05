# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:07:33 2025

@author: kayvon.daie
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Constants
train_inputs =  False
N = 100
dt = 0.01
tau = 2
penalize = 0
λ = 11
t = torch.arange(0, 6 + dt, dt) - 3
T = len(t)
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Neuron groups
num_mod = 40
pre_idx = torch.arange(0, num_mod)
trl_idx = torch.arange(num_mod, num_mod*2)
others_idx = torch.arange(num_mod*2, N)
cond_idx = 85  # neuron to train

# Target activity
target = torch.zeros(T, device=device)
target[t >= 0] = 1.0

# --- CONFIG FLAGS ---
if train_inputs:
    train_recurrent = False
    learning_rate = 1e-2
else:
    train_recurrent = True
    learning_rate = 1e-3

# --- External input setup ---
inp_t = (t < 0).float().unsqueeze(1)  # (T, 1)
noise = torch.randn_like(inp_t, device=device)*.01  # shape (T, 1), same dtype/device

inpweights = torch.zeros(N, device=device)
inpweights[pre_idx] = 1.0
inpweights[trl_idx] = -22.0
inpweights = torch.nn.Parameter(inpweights, requires_grad=bool(train_inputs))

# --- Weight matrix initialization ---
W_np = np.zeros((N, N))
W_np[np.ix_(pre_idx, pre_idx)] = np.random.randn(num_mod, num_mod) * 0.5 + 0.4
W_np[np.ix_(trl_idx, trl_idx)] = np.random.randn(num_mod, num_mod) * 0.5 + 0.7
W_np[np.ix_(pre_idx, trl_idx)] = np.random.randn(num_mod, num_mod) * 1.0 + 1
W_np[np.ix_(trl_idx, pre_idx)] = np.random.randn(num_mod, num_mod) * 1.0 + 0.1
W_np[np.ix_(pre_idx, others_idx)] = np.random.randn(num_mod, N - 2*num_mod) * 2
W_np = W_np / np.max(np.real(np.linalg.eigvals(W_np))) * 0.4
np.fill_diagonal(W_np, 0)

W = torch.tensor(W_np, dtype=torch.float32, device=device, requires_grad=True)
W = torch.nn.Parameter(W, requires_grad=bool(train_recurrent))

trainable_mask = torch.ones((N, N), device=device)
trainable_mask[np.ix_(pre_idx, pre_idx)] = 0
trainable_mask[np.ix_(trl_idx, trl_idx)] = 0
trainable_mask[np.ix_(pre_idx, trl_idx)] = 0
trainable_mask[np.ix_(trl_idx, pre_idx)] = 0
frozen_weights = W.detach().clone() * (1 - trainable_mask)

# --- Optimizer ---
params = []
if train_inputs:
    params.append(inpweights)
if train_recurrent:
    params.append(W)
optimizer = torch.optim.Adam(params, lr=learning_rate)

# --- Training loop ---
L = []
r_epoch = []
inpweight_history = []
for epoch in range(130):
    r_list = []
    r_t = torch.zeros(N, device=device)

    # Recompute external input each epoch to preserve autograd graph
    external = (noise + inp_t) @ inpweights.unsqueeze(0)
    ext_noise = external + torch.randn_like(external) * 0.05

    for ti in range(T):
        r_list.append(r_t)
        inp = r_t @ W + ext_noise[ti]
        dr = (-r_t + torch.relu(inp)) * dt / tau
        r_t = r_t + dr

    r = torch.stack(r_list, dim=0)
    r_epoch.append(r.detach().cpu().numpy())
    if epoch > 30:
        loss = nn.functional.mse_loss(r[300:, cond_idx], target[300:])
    
        if penalize:
            delta_r = r[1:] - r[:-1]
            loss += λ * torch.mean(delta_r ** 2)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if train_recurrent:
            with torch.no_grad():
                W.data = W.data * trainable_mask + frozen_weights
        inpweight_history.append(inpweights.detach().cpu().numpy().copy())
        L.append(loss.item())


plt.figure(figsize=(6, 8))

pop_names = ['Pre', 'Trial', 'Other']
pops = [pre_idx, trl_idx, others_idx]

def compute_avg_W(W_matrix):
    W_avg = np.zeros((3, 3))
    for i, postsyn in enumerate(pops):
        for j, presyn in enumerate(pops):
            W_avg[j,i] = np.mean(W_matrix[np.ix_(postsyn, presyn)])
    return W_avg




    return W_avg

W_initial_avg = compute_avg_W(W_np)
W_final = W.detach().cpu().numpy()
W_final_avg = compute_avg_W(W_final)

for j in range(2):
    plt.subplot(3,2,j+1)
    # Plot average Pre and Trial activity
    if j == 0:
        r = r_epoch[0]
    else:    
        r = r_epoch[-1]
    
    plt.plot(t, np.mean(r[:, pre_idx], axis=1), label='Pre neurons')
    plt.plot(t, np.mean(r[:, trl_idx], axis=1), label='Trial neurons')
    plt.plot(t, np.mean(r[:, others_idx], axis=1), label='Others')
    plt.plot(t, r[:, cond_idx], label='CN')
    
    plt.xlabel('Time')
    plt.ylabel('Mean activity')
    if j == 0:
        plt.legend()

plt.subplot(325)
bef = np.mean(r_epoch[0][500:,:],0)
aft = np.mean(r_epoch[-1][500:,:],0)
plt.plot(bef,aft,'ko')
plt.plot(bef[cond_idx],aft[cond_idx],'mo')
plt.plot(plt.xlim(),plt.xlim())
plt.xlabel('Trial response early')
plt.ylabel('Trial response late')           


w_fit,w_corr = [],[]

def fun(x):
    y = x*(x>0)
    return y

ps_inp = np.zeros((50,))
ps_inp[10] = 1
x_noise = noise.detach().numpy().copy()
WCC = []
for ei in range(2):
    if ei == 0:
        ww = W_np
        inpW = inpweight_history[0]
    else:
        ww = W_final
        inpW = inpweight_history[-1]

    R = np.zeros((51,100,100))
    for ci in range(N):
        stim = np.zeros((N,))
        stim[ci] = 200
        r = np.zeros((51,N))
        for ti in range(50):
            inp = dt/tau*(-r[ti,:] + r[ti,:]@ww + stim*ps_inp[ti] + x_noise[ti] * inpW)
            r[ti+1,:] = r[ti,:] + fun(inp)
        R[:,ci,:] = r

    wcc = np.nanmean(R,0) 
    wcc = wcc - (wcc*np.eye(N))  # remove diagonal
    WCC.append(wcc)
    wcc_flat = wcc.flatten()

    # Build feature matrix
    trl = np.mean(np.stack([np.mean(x[300:,:],0) for x in r_epoch]),0)
    pre = np.mean(np.stack([np.mean(x[0:300,:],0) for x in r_epoch]),0)    
    dpre = np.mean(r_epoch[-1][0:300,:],0) - np.mean(r_epoch[0][0:300,:],0)
    dtrl = np.mean(r_epoch[-1][300:,:],0) - np.mean(r_epoch[0][300:,:],0)
    all_ones = np.ones(trl.shape)

    vectors = [all_ones, trl, pre, dpre, dtrl]
    X = []
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            a = np.outer(vectors[i], vectors[j])
            X.append(a.flatten())
    X[0] = X[0]*0  # zero out all→all
    X = np.stack(X).T

    # Drop constant features, standardize
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    stds = np.nanstd(X, axis=0)
    keep = stds > 1e-8
    X_clean = X[:, keep]

    scaler = StandardScaler()
    X_clean = scaler.fit_transform(X_clean)
    X_clean[np.isnan(X_clean)] = 0

    # Fit Lasso
    model = Lasso(alpha=1e-6, fit_intercept=False, max_iter=10000)
    model.fit(X_clean, wcc_flat)
    beta = model.coef_
    #beta = np.linalg.pinv(X_clean) @ wcc_flat
    
    # Reconstruct full 25-element beta and reshape
    beta_full = np.zeros(25)
    beta_full[keep] = beta
    beta_matrix = beta_full.reshape((5, 5))

    # Compute r-values for each feature (for comparison)
    b = np.zeros(25)
    for i in range(25):
        if np.std(X[:,i]) > 0:
            b[i] = np.corrcoef(wcc_flat, X[:,i])[0,1]
    b_matrix = b.reshape((5,5))

    # Save for comparison
    w_fit.append(beta_matrix)
    w_corr.append(b_matrix)

# Reorder: All=0, Pre=2, ΔPre=3, Trl=1, ΔTrl=4
order = [0, 2, 3, 1, 4]
reordered_labels = ["All", "Pre", "ΔPre", "Trl", "ΔTrl"]

plt.subplot(324)
dw_fit = w_fit[1] - w_fit[0]
dw_fit_reordered = dw_fit[np.ix_(order, order)]
vlim = np.abs(dw_fit_reordered).max()
vlim = 3e-5
sns.heatmap(dw_fit_reordered.T, xticklabels=reordered_labels, yticklabels=reordered_labels,
            annot=False, fmt=".2f", cmap='bwr', vmin=-vlim, vmax=vlim)
plt.xlabel("Presynaptic Feature")
plt.title('$\Delta W_{cc}$')

plt.subplot(323)
dw_fit = w_fit[0]
dw_fit_reordered = dw_fit[np.ix_(order, order)]
vlim = np.abs(dw_fit_reordered).max()
vlim = 3e-5
sns.heatmap(dw_fit_reordered.T, xticklabels=reordered_labels, yticklabels=reordered_labels,
            annot=False, fmt=".2f", cmap='bwr', vmin=-vlim, vmax=vlim)
plt.xlabel("Presynaptic Feature")
plt.title('Initial $W_{cc}$')


plt.subplot(3,2,6)

cce = np.corrcoef(np.stack([np.nanmean(r[300:,:],0) for r in r_epoch[0:20]]).T)
ccl = np.corrcoef(np.stack([np.nanmean(r[300:,:],0) for r in r_epoch[-30:]]).T)
dcc = ccl - cce
x = dcc.flatten()
y = (WCC[1] - WCC[0]).flatten()
ind = np.where((np.isnan(x)==0) & (np.eye(N).flatten()==0))[0]

pf.mean_bin_plot(x[ind],y[ind],5)
plt.xlabel('$\Delta Correlation$');
plt.ylabel('$\Delta W_{cc}$');
plt.tight_layout()
plt.show()


#%%
pop_names = ['Pre', 'Trial', 'Other']
pop_boundaries = [0, 40, 80, 100]  # boundaries for ticks

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# W_np plot
axs[0].imshow(W_np, cmap='bwr', vmin=-np.max(abs(W_np)), vmax=np.max(abs(W_np)))
axs[0].set_title("Initial W")
axs[0].set_xticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[0].set_yticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[0].set_xticklabels(pop_names)
axs[0].set_yticklabels(pop_names)
axs[0].set_xlabel('Postsynaptic')
axs[0].set_ylabel('Presynaptic')

# W_final plot
axs[1].imshow(W_final, cmap='bwr', vmin=-np.max(abs(W_np)), vmax=np.max(abs(W_np)))
axs[1].set_title("Final W")
axs[1].set_xticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[1].set_yticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[1].set_xticklabels(pop_names)
axs[1].set_yticklabels(pop_names)
axs[1].set_xlabel('Postsynaptic')
axs[1].set_ylabel('Presynaptic')

axs[2].imshow(W_final - W_np, cmap='bwr', vmin=-np.max(abs(W_np)), vmax=np.max(abs(W_np)))
axs[2].set_title("$\Delta$ W")
axs[2].set_xticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[2].set_yticks([(pop_boundaries[i] + pop_boundaries[i+1]) // 2 for i in range(3)])
axs[2].set_xticklabels(pop_names)
axs[2].set_yticklabels(pop_names)
axs[2].set_xlabel('Postsynaptic')
axs[2].set_ylabel('Presynaptic')

plt.tight_layout()
plt.show()






