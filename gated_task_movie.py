# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:27:15 2023

@author: scanimage
"""

folder = r'D:/KD/BCI_data/BCI_2022/BCI45/051223/'
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
files = ops['tiff_list']
for fi in range(len(files)):
    tif = ScanImageTiffReader(folder+files[fi]).data();
#%%
import imageio
import numpy as np

def normalize_grayscale(image, min_value, max_value):
    """Normalize a grayscale image to a desired range."""
    image_min = np.percentile(image,3)
    image_max = np.percentile(image,90)
    normalized_image = (image - image_min) * ((max_value - min_value) / (image_max - image_min)) + min_value
    return normalized_image

# Assuming your movie data is stored in a 3D NumPy array called "movie_data"
# The shape of the array should be (frames, height, width)

# Create a list to store individual frames
frames = []
min_value = 0
max_value = 200
for fi in range(33,tif.shape[0]):
    im = np.mean((tif[fi-33:fi,:,:])*2,axis = 0);    
    brightness_factor = 5  # Increase or decrease the brightness as desired
    movie_data_brightened = np.clip(im * brightness_factor, 0, 255).astype(np.uint8)

    #image_min = np.min(im)
    #image_max = np.max(im)
    #im = (im - image_min) * ((max_value - min_value) / (image_max - image_min)) + min_value    
    frames.append(im)

# Save the frames as an .mp4 movie
output_file = folder + 'movie.mp4'
imageio.mimwrite(output_file, frames, fps=30, quality=8)

#%%

import cv2
import numpy as np
frames = tif;
# Assuming your frames are stored in a 3D NumPy array called "frames"
# The shape of the array should be (num_frames, xdim, ydim)

# Specify the output file name and properties
output_file = folder+'movie.avi'
fps = 30  # Frames per second

# Get the dimensions of the frames
num_frames, xdim, ydim = frames.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (ydim, xdim))

# Iterate over each frame and write it to the video file
for i in range(num_frames):
    frame = np.uint8(frames[i])  # Convert frame data to uint8 format if needed
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()
#%%
import numpy as np
import cv2
size = 800,800
fps = 60
mx = np.percentile(tif,100)
out = cv2.VideoWriter(folder+'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for fi in range(30,tif.shape[0]):    
    im = np.mean(tif[fi-30:fi,:,:],axis = 0)
    im = im / mx 
    im = (im * 255).astype(np.uint8)
    out.write(im*10)
out.release()
