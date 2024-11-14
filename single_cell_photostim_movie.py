import os
import cv2
import numpy as np
from PIL import Image
import glob
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI93/091224/'
data = ddc.load_data_dict(folder)
#%%
# Define the directory and file pattern
tif_dir = "//allen/aind/scratch/BCI/2p-raw/BCI93/091224/"
tif_pattern = os.path.join(tif_dir, "photostim_slm_*.tif")
output_mp4 = os.path.join(tif_dir, "photostim_avg_trials6.mp4")

# Parameters
frame_rate = 120  # Adjusted frame rate
running_average_window = 10  # Increased window size for smoother averaging
brightness_factor = 4  # Adjust brightness as needed

# Circle parameters
stim_circle_color = (255, 0, 0)  # Teal color in BGR
stim_circle_radius = 7  # Radius of the stimulation spot
stim_circle_thickness = 1  # Thickness of the circle lines (positive value for unfilled circle)
alpha = 1.0  # Since we're drawing unfilled circles, alpha blending is not needed

# Neuron 7 circle parameters
neuron7_circle_color = (0, 0, 255)  # Red color in BGR
neuron7_circle_radius = 7  # Radius of the circle for neuron 7
neuron7_circle_thickness = 1 # Thickness of the circle lines

# Get the list of TIF files
tif_files = sorted(glob.glob(tif_pattern))
if not tif_files:
    print("No TIF files found in the specified directory.")
    exit()

# Load or define the 'data' variable here
# Ensure 'data' is correctly loaded before proceeding
# For example:
# import scipy.io
# data = scipy.io.loadmat('path_to_data_file.mat')

# Check if 'data' is loaded
if 'data' not in locals():
    print("The 'data' variable is not defined. Please load your data before running the script.")
    exit()

# Determine unique neurons in the sequence
unique_neurons = np.unique(data['photostim']['seq']-1)[:30]  # Limit to the first 10 neurons

# Initialize variables to collect all frames and annotations
all_frames_list = []
annotations_list = []  # To store annotations for each frame

# Initialize frame counter
frame_counter = 0

# Process each neuron
for neuron_idx in unique_neurons:
    print(f"Processing neuron {neuron_idx}...")

    # Find trials where this neuron was stimulated
    trials = [i for i, seq in enumerate(data['photostim']['seq']-1) if seq == neuron_idx]

    # Check if trials list is empty
    if not trials:
        print(f"No trials found for neuron {neuron_idx}, skipping.")
        continue

    # Load trials and collect frames
    trial_frames_list = []
    num_frames_list = []
    for trial in trials:
        trial_frames = []
        with Image.open(tif_files[trial]) as tif:
            frame_idx = 0
            try:
                while True:
                    tif.seek(frame_idx)
                    frame = np.array(tif)
                    trial_frames.append(frame)
                    frame_idx += 1
            except EOFError:
                pass
        trial_frames_list.append(np.array(trial_frames))
        num_frames_list.append(len(trial_frames))

    # Determine the minimum frame count
    min_frames = min(num_frames_list)

    # Truncate all trials to the minimum frame count
    trial_frames_list = [frames[:min_frames] for frames in trial_frames_list]

    # Convert list to numpy array
    trial_frames_array = np.array(trial_frames_list)  # Shape: (num_trials, min_frames, height, width)

    # Average trials frame by frame
    averaged_frames = np.nanmean(trial_frames_array, axis=0)  # Shape: (min_frames, height, width)

    # Collect frames for global normalization
    all_frames_list.append(averaged_frames)

    # Get the neuron's position
    distances = data['photostim']['stimDist'][:, neuron_idx]
    a = np.argmin(distances)
    x, y = data['centroidX'][a], data['centroidY'][a]
    x = int(x)
    y = int(y)

    # Create annotations for each frame corresponding to this neuron
    annotations = [(frame_counter + i, x, y) for i in range(averaged_frames.shape[0])]
    annotations_list.extend(annotations)

    # Update frame counter
    frame_counter += averaged_frames.shape[0]

# Concatenate all frames from all neurons
all_frames = np.concatenate(all_frames_list, axis=0)  # Shape: (total_frames, height, width)

# Calculate global minimum and maximum for normalization
global_min = np.min(all_frames)
global_max = np.max(all_frames)
print(f"Global min value: {global_min}, Global max value: {global_max}")

# Normalize the pixel values to [0, 255]
all_frames_norm = ((all_frames - global_min) / (global_max - global_min) * 255)

# Apply brightness adjustment
all_frames_bright = np.clip(all_frames_norm * brightness_factor, 0, 255).astype(np.uint8)

# Apply running average
def running_average(frames, window_size):
    smoothed_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        start_idx = max(0, i - window_size + 1)
        smoothed_frames[i] = np.mean(frames[start_idx:i+1], axis=0)
    return smoothed_frames

smoothed_frames = running_average(all_frames_bright, running_average_window)

# Output video file parameters
height, width = smoothed_frames.shape[1], smoothed_frames.shape[2]
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_mp4, fourcc, frame_rate, frame_size, isColor=True)

# Get neuron 7 coordinates
neuron7_x = int(data['centroidX'][7])
neuron7_y = int(data['centroidY'][7])

# Annotate frames with photostim spots and neuron 7 circle
for idx in range(len(smoothed_frames)):
    # Convert the frame to BGR color
    frame_color = cv2.cvtColor(smoothed_frames[idx], cv2.COLOR_GRAY2BGR)

    # Draw an unfilled red circle around neuron 7 on every frame
    #cv2.circle(frame_color, (neuron7_x, neuron7_y), neuron7_circle_radius, neuron7_circle_color, thickness=neuron7_circle_thickness)

    # Check if there is an annotation for this frame (stimulation spot)
    annotations = [ann for ann in annotations_list if ann[0] == idx]
    for _, x, y in annotations:
        # Draw unfilled teal circle directly on the frame (no alpha blending needed)
        cv2.circle(frame_color, (x, y), stim_circle_radius, stim_circle_color, thickness=stim_circle_thickness)

    # Write the frame to the video
    video_writer.write(frame_color)

# Release the video writer
video_writer.release()

print(f"MP4 file created at: {output_mp4}")
