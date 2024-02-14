# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:42:06 2024

@author: Kayvon Daie
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
import numpy as np
# List to hold the coordinates of all polygons
polygons = []
ops = np.load('//allen/aind/scratch/BCI/2p-raw/BCINM_006/010324/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
img = ops['meanImg']

# Plot the image
fig, ax = plt.subplots()
ax.imshow(img,vmin=0,vmax=10)

polygon_selector = None

# Function to finalize the current polygon and start a new one
def finalize_polygon(event):
    global polygon_selector
    if polygon_selector:
        polygon_selector.disconnect_events()
        verts = polygon_selector.verts
        if len(verts) > 2:
            polygons.append(verts)
            print(f"Polygon finalized with vertices: {verts}")
            polygon = Polygon(verts, closed=True, edgecolor='red', facecolor='none', lw=2)
            ax.add_patch(polygon)
            fig.canvas.draw()
    if event is None or event.key == 'enter':
        polygon_selector = PolygonSelector(ax, lambda v: None)

# Bind the finalize function to the 'enter' key
fig.canvas.mpl_connect('key_press_event', finalize_polygon)

# Start the first polygon
finalize_polygon(None)

plt.show()


#%%
from PIL import Image
import numpy as np
import cv2

def get_pixels_inside_polygon(polygon, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    y, x = np.where(mask == 1)
    return list(zip(x, y))

def average_pixel_value(polygon, image):
    pixels = get_pixels_inside_polygon(polygon, image.shape)
    pixel_values = [image[y, x] for x, y in pixels]
    return np.mean(pixel_values)

num_tifs = len(ops['tiff_list'])
F = []  # List to hold average values arrays for each TIFF file

for fi in range(num_tifs):
    # Path to your TIFF file
    file_path = ops['data_path'][0] + ops['tiff_list'][fi]
    
    with Image.open(file_path) as img:
        # Create a list to hold each frame as a numpy array, reading every other frame
        frames = [np.array(img.seek(i) or img) for i in range(0, img.n_frames, 2)]  # Step is 2
    
    # Convert the list of frames to a 3D numpy array
    imgs = np.stack(frames, axis=2)# imgs is now a 3D array where imgs[:,:,i] is the i-th frame of the TIFF file
    
    T = imgs.shape[2]
    N = len(polygons)

    # Initialize an empty array
    average_values_per_image = np.zeros((T, N))
    for i in range(T):
        img = imgs[:, :, i]
        for j, polygon in enumerate(polygons):
            average_values_per_image[i, j] = average_pixel_value(polygon, img)
    F.append(average_values_per_image)