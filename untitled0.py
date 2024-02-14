# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:26:08 2023

@author: kayvon.daie
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulating some dummy data for a 10-neuron feedforward network
np.random.seed(0)
data = np.random.rand(5, 10)  # 5 instances, 10 neurons each

# Creating a DataFrame for easier plotting
df = pd.DataFrame(data, columns=[f'Neuron {i+1}' for i in range(10)])

# Plotting using parallel coordinates
plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(df, class_column=None, colormap='viridis')
plt.title('Parallel Coordinates Plot for a 10-Neuron Feedforward Network')
plt.xlabel('Neurons')
plt.ylabel('Activity Level')
plt.grid(True)
plt.show()
