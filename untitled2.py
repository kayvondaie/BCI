import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory and base filename
base_dir = '//allen/aind/scratch/BCI/2p-raw/BCI75/062024/'
base_name = 'neuron_33_'

# Define the starting number and how many files to process
start_num = 11
num_files = 6  # Including the first file and the next 5

# Initialize a list to hold all frames
all_frames = []

# Loop through the file numbers and read each TIFF file
for i in range(start_num, start_num + num_files):
    file_path = os.path.join(base_dir, f"{base_name}{i:05d}.tif")
    print(f"Loading file: {file_path}")
    tif = tiff.imread(file_path)
    all_frames.append(tif)

# Convert the list of frames into a single NumPy array
all_frames_array = np.concatenate(all_frames, axis=0)

# Calculate the average of all frames
average_frame = np.mean(all_frames_array, axis=0)
#%%
# Display the average frame
plt.imshow(average_frame, cmap='gray',vmin = 0,vmax=100)
plt.colorbar()
plt.title(base_dir)
plt.show()
