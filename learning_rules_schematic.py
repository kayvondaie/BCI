# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:47:04 2023

@author: scanimage
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 2D Gaussian function
def gaussian_2d(x, y, A=1, x0=0, y0=0, sigma_x=5, sigma_y=3):
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

# Define the x and y coordinates
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create a grid of (x, y) coordinates
x, y = np.meshgrid(x, y)

# Calculate the z values (heights) using the 2D Gaussian function
z = gaussian_2d(x, y)

# Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='RdBu_r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)
# Set title
ax.set_title('3D Surface plot of a 2D Gaussian function')
plt.savefig(folder+'cost_function.svg', format='svg')
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 2D Gaussian function
def gaussian_2d(x, y, A=1, x0=-3, y0=2, sigma_x=.5, sigma_y=1):
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

# Define the x and y coordinates
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create a grid of (x, y) coordinates
x, y = np.meshgrid(x, y)

# Calculate the z values (heights) using the 2D Gaussian function
z = gaussian_2d(x, y)

# Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='RdBu_r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)
# Set title
plt.savefig(folder+'photostim_cost_function.svg', format='svg')
plt.show()