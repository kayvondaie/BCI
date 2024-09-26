# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

# Clear eff
eff = []

nt = 30
ns = np.arange(10, 55, 10)
perf = np.linspace(0.5, 0.9, 5)

eff = np.zeros((len(ns), len(perf)))

for i in range(len(ns)):
    for j in range(len(perf)):
        hit = []
        for iter in range(10000):
            c = np.random.rand(nt, ns[i]) < perf[j]
            hit.append(np.mean(c))
        eff[i, j] = np.mean(hit) - np.percentile(hit, 2.5)

plt.imshow(eff, aspect='auto', cmap='viridis')
plt.colorbar()
plt.xticks([0, len(perf) - 1], [f'{perf[0]:.2f}', f'{perf[-1]:.2f}'])
plt.yticks([0, len(ns) - 1], [f'{ns[0]}', f'{ns[-1]}'])
plt.ylabel('# of sessions')
plt.xlabel('Hit rate on control sessions')
plt.title('Significant effect size\n (bootstrap, p<0.05)')
plt.box(False)
plt.show()
