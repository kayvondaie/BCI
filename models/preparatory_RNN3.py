# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:31:37 2025

@author: kayvon.daie
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Constants
N = 100
dt = 0.01
tau = 0.5
penalize = 1;
λ = 0;
t = torch.arange(0, 6 + dt, dt) - 3
T = len(t)
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Neuron groups
pre_idx = torch.arange(0, 40)
trl_idx = torch.arange(40, 80)
others_idx = torch.arange(80, N)
cond_idx = 95  # neuron to train

# Target activity
target = torch.zeros(T, device=device)
target[t >= 0] = 1.0

# External input
external = torch.zeros((T, N), device=device)
external[np.where(t < 0)[0][:, None], pre_idx] = 1.0
external[np.where(t < 0)[0][:, None], trl_idx] = -202.0

# Initial W
W_np = np.zeros((N, N))
W_np[np.ix_(pre_idx, pre_idx)] = np.random.randn(40, 40) * 0.5 + .1
W_np[np.ix_(trl_idx, trl_idx)] = np.random.randn(40, 40) * 0.5 + .1
W_np[np.ix_(pre_idx, trl_idx)] = np.random.randn(40, 40) * 1.0 + .5
W_np[np.ix_(trl_idx, pre_idx)] = np.random.randn(40, 40) * 1.0 + .1
W_np[np.ix_(pre_idx, others_idx)] = np.random.randn(40, 20) * -.2
#W_np = W_np * (np.random.rand(N,N) > .9)
W_np = W_np / np.max(np.real(np.linalg.eigvals(W_np)))*.5
np.fill_diagonal(W_np, 0)

# Convert to torch tensor, set requires_grad on inputs to cond_idx
W = torch.tensor(W_np, dtype=torch.float32, requires_grad=True, device=device)

import torch
import numpy as np

# Create 5×5 matrix of zeros
beta_data_matrix = np.zeros((5, 5))

# Set the two key entries to 1
beta_data_matrix[3, 0] = 1.0  # ΔPre → All
beta_data_matrix[2, 3] = 1.0  # Pre → ΔPre

# Flatten to 25-element vector (row-major)
real_data_beta_vector = beta_data_matrix.flatten()

# Convert to torch tensor
beta_target = torch.tensor(real_data_beta_vector, dtype=torch.float32, device=device)

# Regularization weight
λ_beta = 100000.0  # Tune as needed


# Optimizer
optimizer = optim.Adam([W], learning_rate)
L = []
r_epoch = []

for epoch in range(100):
    r_list = []
    r_t = torch.zeros(N, device=device)
    ext_noise = external + torch.randn_like(external) * 0.05

    for ti in range(T):
        r_list.append(r_t)
        inp = r_t @ W + ext_noise[ti]
        dr = (-r_t + torch.relu(inp)) * dt / tau
        r_t = r_t + dr

    r = torch.stack(r_list, dim=0)

    # ✅ Store CN activity this epoch
    r_epoch.append(r.detach().cpu().numpy())
    loss = nn.functional.mse_loss(r[:, cond_idx], target)
    
    if penalize == 1:
        # --- Define population-level vectors from activity ---
        pre = torch.mean(r[0:300], dim=0)         # Pre window
        trl = torch.mean(r[300:], dim=0)          # Trl window
        dpre = pre - torch.mean(r[0:300], dim=0)  # ΔPre (same as pre - baseline)
        dtrl = trl - torch.mean(r[300:], dim=0)   # ΔTrl
    
        # --- Build outer product design matrix (like X in your analysis) ---
        vectors = [torch.ones_like(pre), trl, pre, dpre, dtrl]
        X_model = []
        for i in range(5):
            for j in range(5):
                outer = torch.ger(vectors[i], vectors[j])
                X_model.append(outer.flatten())
        X_model = torch.stack(X_model, dim=1)  # shape: (N*N, 25)
    
        # --- Compute approximate effective connectivity matrix ---
        wcc_model = torch.ger(torch.mean(r, dim=0), torch.mean(r, dim=0))  # shape: (N, N)
        wcc_flat = wcc_model.flatten()
    
        # --- Solve X_model @ β = wcc_flat (least squares) ---
        beta_model = torch.linalg.lstsq(X_model, wcc_flat.unsqueeze(1)).solution

        beta_model = beta_model[:25, 0]  # Extract β vector (25 entries)
    
        # --- Penalize only the two entries of interest ---
        idxs = torch.tensor([15, 13], device=device)  # ΔPre→All, Pre→ΔPre
        beta_penalty = torch.mean((beta_model[idxs] - beta_target[idxs]) ** 2)
    
        # --- Add to loss ---
        loss += λ_beta * beta_penalty

    


    loss = nn.functional.mse_loss(r[:, cond_idx], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    L.append(loss.item())

plt.figure(figsize=(10, 10))

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



# Initial weights
plt.subplot(323)
im1 = plt.imshow(W_initial_avg, cmap='bwr', vmin=-np.max(abs(W_final_avg)), vmax=np.max(abs(W_final_avg)))
plt.xticks(np.arange(3), pop_names)
plt.yticks(np.arange(3), pop_names)
plt.title('Initial W_{cc}')
plt.xlabel('Presynaptic Pop.')
plt.ylabel('Postsynaptic Pop.')
plt.colorbar(im1, fraction=0.046, pad=0.04)

# Final weights
plt.subplot(324)
im2 = plt.imshow(W_final_avg - W_initial_avg, cmap='bwr', vmin=-np.max(abs(W_final_avg)), vmax=np.max(abs(W_final_avg)))
plt.xticks(np.arange(3), pop_names)
plt.yticks(np.arange(3), pop_names)
plt.title('$\Delta$ W_{cc}')
plt.xlabel('Presynaptic Pop.')
plt.ylabel('Postsynaptic Pop.')
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.tight_layout()

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
    plt.legend()

plt.subplot(325)
bef = np.mean(r_epoch[0][300:,:],0)
aft = np.mean(r_epoch[-1][300:,:],0)
plt.plot(bef,aft,'ko')
plt.plot(bef[cond_idx],aft[cond_idx],'mo')
plt.plot(plt.xlim(),plt.xlim())

plt.subplot(326);
plt.plot(aft - bef,'ko')
plt.plot(cond_idx, aft[cond_idx] - bef[cond_idx],'mo')

plt.tight_layout()
plt.show()
#%%
plt.figure(figsize = (12,4))
w_fit,w_corr = [],[]
def fun(x):
    y = x*(x>0)
    return y

ps_inp = np.zeros((50,));ps_inp[10] = 1;
for ei in range(2):
    if ei == 0:
        ww = W_np
    else:
        ww = W_final
    R = np.zeros((51,100,100));
    for ci in range(N):
        stim = np.zeros((1,N))[0]
        stim[ci] = 20;
        r = np.zeros((51,N));
        for ti in range(50):
            inp = dt/tau*(-r[ti,:] + r[ti,:]@ww + stim*ps_inp[ti])
            r[ti+1,:] = r[ti,:] + fun(inp)
        R[:,ci,:] = r;
    wcc = np.nanmean(R,0) 
    wcc = wcc - (wcc*np.eye(N))
    labels = ["All", "Trl", "Pre", "ΔPre", "ΔTrl"]
    column_labels = [f"{pre} → {post}" for pre in labels for post in labels]

    
    trl = np.mean(r_epoch[-1][550:,:],0)
    pre = np.mean(r_epoch[-1][0:300,:],0)
    dpre = np.mean(r_epoch[-1][0:300,:],0) - np.mean(r_epoch[0][0:300,:],0)
    dtrl = np.mean(r_epoch[-1][550:,:],0) - np.mean(r_epoch[0][300:,:],0)
    all_ones = np.ones(trl.shape);
    vectors = [all_ones,trl,pre,dpre,dtrl]
    X,Xmat = [],[]
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            a = np.outer(vectors[i],vectors[j].T)
            X.append(a.flatten())
            Xmat.append(a);
    X[0] = X[0]*0
    X = np.stack(X).T
    print(X.shape)
    wcc_flat = wcc.flatten()
    
    
    from sklearn.linear_model import LassoCV
    
    # Automatically choose best alpha using cross-validation
    lasso_cv = LassoCV(alphas=np.logspace(-10, 1, 100),  # search over a wide range
                       cv=5,                            # 5-fold cross-validation
                       fit_intercept=False,
                       max_iter=10000)
    lasso_cv.fit(X, wcc_flat)
    
    # Best alpha and final coefficients
    best_alpha = lasso_cv.alpha_
    beta = lasso_cv.coef_
    
    print(f"Best alpha: {best_alpha:.4e}")
    X[np.abs(X)<.05] = 0
    
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-np.nanmean(X[:,i]))/np.nanstd(X[:,i])
    
    b = beta*0
    for i in range(X.shape[1]):
        b[i] = np.corrcoef(wcc_flat,X[:,i])[0,1]
    
    #beta = b
    import seaborn as sns
    import matplotlib.pyplot as plt
    beta_matrix = beta.reshape((5, 5))
    b_matrix = b.reshape((5,5))
    labels = ["All", "Trl", "Pre", "ΔPre", "ΔTrl"]
    #plt.subplot(1,3,ei+1)
    #sns.heatmap(beta_matrix.T, yticklabels=labels, xticklabels=labels, annot=False, fmt=".2f", cmap='bwr')
    #plt.xlabel("Presynaptic Feature")
    #if ei == 0:
    #    plt.ylabel("Postsynaptic Feature")
    
    
    w_fit.append(beta_matrix)
    w_corr.append(b_matrix)

plt.subplot(121)
dw_fit = w_fit[1] - w_fit[0]
sns.heatmap(dw_fit.T, xticklabels=labels, yticklabels=labels, annot=False, fmt=".2f", cmap='bwr')
plt.xlabel("Presynaptic Feature")
plt.title('MLR')

plt.subplot(122)
dw_fit = w_corr[1] - w_corr[0]
sns.heatmap(dw_fit.T, xticklabels=labels, yticklabels = labels,annot=False, fmt=".2f", cmap='bwr')
plt.ylabel("Postsynaptic Feature")
plt.xlabel("Presynaptic Feature")
plt.title('Correlations')
plt.tight_layout()
plt.show()