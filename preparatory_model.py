# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 09:32:59 2025

@author: kayvon.daie
"""

import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
dt   = 0.01
tau  = 0.1
t    = np.arange(0, 3+dt, dt)

# === Input signal ===
inp = np.zeros_like(t)
inp[20:200] = 1
w_in = np.array([1, -5, -5, -5])
inp[200:220] = -2

# === Trial weights (not used in your code) ===
wtrl = np.array([[1, 1],
                 [-1, -1]]) * 20

# === Recurrent weights ===
ff  = 0.5
fb  = 1
amp = 3

w = np.array([
    [fb, ff, 0,   0],
    [0,  0,  1,  -1],
    [0,  0,  amp, amp],
    [0,  0, -amp, -amp]
])

# === Simulate ===
r = np.zeros((len(t), 4))     # time × neurons

for i in range(len(t)-1):
    dr = (-r[i] + r[i] @ w + w_in * inp[i]) * (dt / tau)
    r[i+1] = r[i] + dr

# === Plot ===
plt.figure(figsize=(10,4))

# Time courses
plt.subplot(1,2,1)
plt.plot(t, r)
plt.xlabel('time (s)')
plt.ylabel('rate')
plt.title('Neural activity')

# Correlation matrix
plt.subplot(1,2,2)
plt.imshow(np.corrcoef(r.T), aspect='equal', cmap='viridis')
plt.colorbar()
plt.title('corr(r)')

plt.tight_layout()
plt.show()
#%%
