# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:31:37 2025

@author: kayvon.daie
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
dt = 0.01
tau = 0.5
t = np.arange(0, 6 + dt, dt) - 3  # Time from -3 to 3
T = len(t)

# Neuron group indices
pre_idx = np.arange(0, 40)
trl_idx = np.arange(40, 80)

# Activation function
def relu(x):
    return np.maximum(0, x)

# Weight matrix
W = np.zeros((N, N))

# Pre recurrent weights (sparse positive)
W[np.ix_(pre_idx, pre_idx)] = np.random.randn(40, 40) * 0.5
# Trial recurrent weights
W[np.ix_(trl_idx, trl_idx)] = np.random.randn(40, 40) * 0.5
# Pre → Trial projection (stronger, structured)
W[np.ix_(pre_idx,trl_idx)] = np.random.randn(40, 40) * 1.0

W = W / np.max(np.real(np.linalg.eigvals(W)))

# Optional: zero diagonal (no self-connections)
np.fill_diagonal(W, 0)

# External input: drive Pre neurons before trial
external = np.zeros((T, N))
external[np.where(t < 0)[0][:, None], pre_idx] = 2.0
external[np.where(t < 0)[0][:, None], trl_idx] = -22.0

# State matrix: r[t, neuron]
r = np.zeros((T, N))

# Simulate dynamics
for ti in range(T - 1):
    inp = r[ti] @ W + external[ti]
    dr = (-r[ti] + relu(inp)) * dt / tau
    r[ti + 1] = r[ti] + dr

# Plot average Pre and Trial activity
plt.plot(t, np.mean(r[:, pre_idx], axis=1), label='Pre neurons')
plt.plot(t, np.mean(r[:, trl_idx], axis=1), label='Trial neurons')
plt.axvline(0, color='k', linestyle='--', label='Trial start')
plt.xlabel('Time')
plt.ylabel('Mean activity')
plt.legend()
plt.title('Hand-built RNN with Pre → Trial dynamics')
plt.show()
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Constants
N = 100
dt = 0.01
tau = 0.5
penalize = 0;
λ = .1;
t = torch.arange(0, 6 + dt, dt) - 3
T = len(t)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Neuron groups
pre_idx = torch.arange(0, 40)
trl_idx = torch.arange(40, 80)
others_idx = torch.arange(80, N)
cond_idx = 85  # neuron to train

# Target activity
target = torch.zeros(T, device=device)
target[t >= 0] = 1.0

# External input
external = torch.zeros((T, N), device=device)
external[np.where(t < 0)[0][:, None], pre_idx] = 2.0
external[np.where(t < 0)[0][:, None], trl_idx] = -22.0

# Initial W
W_np = np.zeros((N, N))
W_np[np.ix_(pre_idx, pre_idx)] = np.random.randn(40, 40) * 0.5 + .1
W_np[np.ix_(trl_idx, trl_idx)] = np.random.randn(40, 40) * 0.5 + .1
W_np[np.ix_(pre_idx, trl_idx)] = np.random.randn(40, 40) * 1.0 + .5
W_np[np.ix_(trl_idx, pre_idx)] = np.random.randn(40, 40) * 1.0 + .1
W_np[np.ix_(pre_idx, others_idx)] = np.random.randn(40, 20) * -.2
W_np = W_np / np.max(np.real(np.linalg.eigvals(W_np)))*.5
np.fill_diagonal(W_np, 0)

# Convert to torch tensor, set requires_grad on inputs to cond_idx
W = torch.tensor(W_np, dtype=torch.float32, requires_grad=True, device=device)

# Optimizer
optimizer = optim.Adam([W], lr=1e-2)
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
                # r: shape (T, N)
        delta_r = r[1:] - r[:-1]  # shape (T-1, N)
        smoothness_penalty = torch.mean(delta_r ** 2)
        loss += λ * smoothness_penalty



    loss = nn.functional.mse_loss(r[:, cond_idx], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    L.append(loss.item())



# Plot result
r = r.detach().cpu().numpy()
plt.plot(t, r[:, cond_idx], label=f'Neuron {cond_idx} (conditioned)')
plt.plot(t, target.cpu(), '--', label='Target')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Activity')
plt.legend()
plt.title('Conditioned Neuron Activity After BPTT Training')
plt.show()
#%%
# Define population indices and labels
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


plt.figure(figsize=(10, 4))

# Initial weights
plt.subplot(1, 2, 1)
im1 = plt.imshow(W_initial_avg, cmap='bwr', vmin=-np.max(abs(W_final_avg)), vmax=np.max(abs(W_final_avg)))
plt.xticks(np.arange(3), pop_names)
plt.yticks(np.arange(3), pop_names)
plt.title('Initial Population-Averaged W')
plt.xlabel('Presynaptic Pop.')
plt.ylabel('Postsynaptic Pop.')
plt.colorbar(im1, fraction=0.046, pad=0.04)

# Final weights
plt.subplot(1, 2, 2)
im2 = plt.imshow(W_final_avg - W_initial_avg, cmap='bwr', vmin=-np.max(abs(W_final_avg)), vmax=np.max(abs(W_final_avg)))
plt.xticks(np.arange(3), pop_names)
plt.yticks(np.arange(3), pop_names)
plt.title('Final Population-Averaged W')
plt.xlabel('Presynaptic Pop.')
plt.ylabel('Postsynaptic Pop.')
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
#%%
# Plot average Pre and Trial activity
r = r_epoch[-1]
plt.plot(t, np.mean(r[:, pre_idx], axis=1), label='Pre neurons')
plt.plot(t, np.mean(r[:, trl_idx], axis=1), label='Trial neurons')
plt.plot(t, np.mean(r[:, others_idx], axis=1), label='Others')
plt.plot(t, r[:, cond_idx], label='CN')

plt.xlabel('Time')
plt.ylabel('Mean activity')
plt.legend()
plt.title('Hand-built RNN with Pre → Trial dynamics')
plt.show()
#%%
pop_names = ['Pre', 'Trial', 'Other']
pop_boundaries = [0, 40, 80, 100]  # boundaries for ticks

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

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

plt.tight_layout()
plt.show()
#%%
bef = np.mean(r_epoch[0][300:,:],0)
aft = np.mean(r_epoch[-1][300:,:],0)
plt.scatter(bef,aft)
plt.scatter(bef[cond_idx],aft[cond_idx])

