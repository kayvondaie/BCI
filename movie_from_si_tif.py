import numpy as np
import cv2
import tifffile as tiff
import os

# Path to the input TIF file
tif_path = '//allen/aind/scratch/BCI/2p-raw/751034/rois_00001.tif'

# Read the TIF file
tif_data = tiff.imread(tif_path)

# Keep the first frame and every other frame
selected_frames = tif_data[::2]

# Function to calculate running average with a window size of 10
def running_average(frames, window_size=44):
    smoothed_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        start_idx = max(0, i - window_size + 1)
        smoothed_frames[i] = np.mean(frames[start_idx:i+1], axis=0)
    return smoothed_frames

# Calculate the running average for smoothing
smoothed_frames = running_average(selected_frames)

# Normalize the pixel values to [0, 255] for saving as video
smoothed_frames_norm = ((smoothed_frames - np.min(smoothed_frames)) / 
                        (np.max(smoothed_frames) - np.min(smoothed_frames)) * 255).astype(np.uint8)

# Directory to save the video (same as the TIF directory)
output_dir = os.path.dirname(tif_path)

# Output video file name and path
output_video_path = os.path.join(output_dir, 'output_video.mp4')

# Define the video file parameters
fps = 60
frame_size = (smoothed_frames_norm.shape[2], smoothed_frames_norm.shape[1])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter object to write video
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)

# Write frames to the video file
for frame in smoothed_frames_norm:
    out.write(frame)

# Release the video writer
out.release()

print("Video saved as", output_video_path)
