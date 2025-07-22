# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 23:39:22 2025

@author: kayvon.daie
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.01
t = np.arange(0, 6 + dt, dt)
tau = 0.5
t = t - 3

# Activation function (ReLU)
def fun(x):
    return x * (x > 0)

for j in range(2, 1, -1):
    plt.figure(j)
    if j == 1:
        a = 0
    else:
        a = .5

    # Weight matrix
    # w = np.array([
    #     [1,   2,   0,    a,   a],
    #     [0,   1,   1,    0,   0],
    #     [-2, -3,  -3,   -2,   0],
    #     [0,   0,   0,    0,  a*2],
    #     [0,   0,   0,    0,   0]
    # ], dtype=float)
    w=np.array([
            [ 1,  2,  0,  a*1.5,  0],      # neuron 1  (Pre)
            [ 0,  1,  1,  0,  0],      # neuron 2  (Trl)
            [-2, -3, -3, -2,  0],      # neuron 3  (Trl)
            [ a,  a,  a,  a,  a*1.5],      # neuron 4  (ΔPre)
            [ 0,  0,  0,  0,  0]       # neuron 5  (ΔTrl)
        ], dtype=float)

    # External input
    ts = np.zeros_like(t)
    ts[t < 0] = 1
    ts_v = np.array([0, -1, -1, 0, -1])

    # State variable
    r = np.zeros((len(t), 5))

    # Integration loop
    for ti in range(len(t) - 1):
        input_term = (r[ti, :] @ w
                      + ts[ti] * ts_v * 100
                      + ((t[ti] > -2.5) & (t[ti] < 0)) * np.array([1, 0, 0, 0, 0]))
        r[ti + 1, :] = r[ti, :] + dt / tau * (-r[ti, :] + fun(input_term))

    # Plot activity
    plt.clf()
    plt.subplot(2, 1, 1)
    if j == 2:
        plt.plot(t, r[:, 3], 'r')
        plt.plot(t, r[:, 4], 'b')
    plt.plot(t, r[:, 0], 'r', label='Pre')
    plt.plot(t, r[:, 0:3], 'k')
    
    plt.legend(['DPre','Dtrl'])
    plt.xlabel('Time from trial start (s)')

    # Compute summary of weights
    if j == 1:
        w = w[:3, :3]

    pre = [0]    # index for 'Pre'
    trl = [1, 2] # index for 'Trl'

    wcc = np.zeros((3, 3))
    wcc[0, 1] = np.mean(w[pre, :])
    wcc[0, 2] = np.mean(w[trl, :])

    wcc[1, 0] = np.mean(w[:, pre])
    wcc[2, 0] = np.mean(w[:, trl])

    wcc[1, 1] = np.mean(w[np.ix_(pre, pre)]) - wcc[0, 1]
    wcc[2, 1] = np.mean(w[np.ix_(pre, trl)]) - wcc[0, 1]

    wcc[2, 2] = np.mean(w[np.ix_(trl, trl)]) - wcc[0, 2]
    wcc[1, 2] = np.mean(w[np.ix_(trl, pre)]) - wcc[0, 2]

    # Plot weight summary matrix
    plt.subplot(2, 1, 2)
    im = plt.imshow(wcc, vmin=-2, vmax=2, cmap='bwr')
    plt.xticks([0, 1, 2], ['All', 'Pre', 'Trl'])
    plt.yticks([0, 1, 2], ['All', 'Pre', 'Trl'])
    plt.colorbar(im)

    plt.tight_layout()
    plt.show()
    #%%
import numpy as np
import matplotlib.pyplot as plt


def build_w(a):
    """Weight matrix with the given scalar a."""
    return np.array([
        [ 1,  2,  0,  a,  0],      # neuron 1  (Pre)
        [ 0,  1,  1,  0,  0],      # neuron 2  (Trl)
        [-2, -3, -3, -2,  0],      # neuron 3  (Trl)
        [ a,  a,  a,  a,  a],      # neuron 4  (ΔPre)
        [ 0,  0,  0,  0,  0]       # neuron 5  (ΔTrl)
    ], dtype=float)


def summarize_weight_change_explicit(a_before=0, a_after=1):
    """Return the Δw matrix and the wcc‑like 5 × 5 summary with no loops."""
    w0 = build_w(a_before)
    w1 = build_w(a_after)
    dw = w1 - w0

    # Group indices
    Pre   = [0]
    Trl   = [1, 2]
    dPre  = [3]
    dTrl  = [4]

    # Initialize matrix
    wccΔ = np.zeros((5, 5))

    # Row 0: mean input from group → all targets
    wccΔ[0, 1] = dw[Pre, :].mean()
    wccΔ[0, 2] = dw[Trl, :].mean()
    wccΔ[0, 3] = dw[dPre, :].mean()
    wccΔ[0, 4] = dw[dTrl, :].mean()

    # Column 0: mean output to group from all sources
    wccΔ[1, 0] = dw[:, Pre].mean()
    wccΔ[2, 0] = dw[:, Trl].mean()
    wccΔ[3, 0] = dw[:, dPre].mean()
    wccΔ[4, 0] = dw[:, dTrl].mean()

    # Group-to-group (i,j) minus group j baseline (row 0, col j)
    wccΔ[1, 1] = dw.T[np.ix_(Pre, Pre)].mean()     - wccΔ[0, 1]
    wccΔ[1, 2] = dw.T[np.ix_(Pre, Trl)].mean()     - wccΔ[0, 2]
    wccΔ[1, 3] = dw.T[np.ix_(Pre, dPre)].mean()    - wccΔ[0, 3]
    wccΔ[1, 4] = dw.T[np.ix_(Pre, dTrl)].mean()    - wccΔ[0, 4]

    wccΔ[2, 1] = dw.T[np.ix_(Trl, Pre)].mean()     - wccΔ[0, 1]
    wccΔ[2, 2] = dw.T[np.ix_(Trl, Trl)].mean()     - wccΔ[0, 2]
    wccΔ[2, 3] = dw.T[np.ix_(Trl, dPre)].mean()    - wccΔ[0, 3]
    wccΔ[2, 4] = dw.T[np.ix_(Trl, dTrl)].mean()    - wccΔ[0, 4]

    wccΔ[3, 1] = dw.T[np.ix_(dPre, Pre)].mean()    - wccΔ[0, 1]
    wccΔ[3, 2] = dw.T[np.ix_(dPre, Trl)].mean()    - wccΔ[0, 2]
    wccΔ[3, 3] = dw.T[np.ix_(dPre, dPre)].mean()   - wccΔ[0, 3]
    wccΔ[3, 4] = dw.T[np.ix_(dPre, dTrl)].mean()   - wccΔ[0, 4]

    wccΔ[4, 1] = dw.T[np.ix_(dTrl, Pre)].mean()    - wccΔ[0, 1]
    wccΔ[4, 2] = dw.T[np.ix_(dTrl, Trl)].mean()    - wccΔ[0, 2]
    wccΔ[4, 3] = dw.T[np.ix_(dTrl, dPre)].mean()   - wccΔ[0, 3]
    wccΔ[4, 4] = dw.T[np.ix_(dTrl, dTrl)].mean()   - wccΔ[0, 4]

    labels = ['All', 'Pre', 'Trl', 'ΔPre', 'ΔTrl']
    return dw, wccΔ, labels

dw, wccΔ, labels = summarize_weight_change_explicit()

# Rearranged display order: ['All', 'Pre', 'ΔPre', 'Trl', 'ΔTrl']
plot_order = [0, 1, 3, 2, 4]  # indices in desired order
wccΔ_display = wccΔ[np.ix_(plot_order, plot_order)]
labels_display = [labels[i] for i in plot_order]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(wccΔ_display, cmap='bwr', vmin=-2, vmax=2)
ax.set_xticks(range(len(labels_display)))
ax.set_yticks(range(len(labels_display)))
ax.set_xticklabels(labels_display)
ax.set_yticklabels(labels_display)
plt.colorbar(im, ax=ax, label='Δw  (after – before)')
ax.set_title('Learning‑induced weight‑change summary (wccΔ)')
plt.tight_layout()
plt.show()
