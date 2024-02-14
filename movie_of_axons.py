# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:53:31 2024

@author: Kayvon Daie
"""
import numpy as np
import matplotlib.pyplot as plt;import matplotlib as mpl;mpl.rcParams['figure.dpi'] = 300
from PIL import Image
import cv2

folder = '//allen/aind/scratch/BCI/2p-raw/BCINM_006/011724/'
ops = np.load(folder + r'suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()

num_tifs = len(ops['tiff_list'])

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(folder + 'output2.avi', fourcc, 80.0, (800, 800))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder + 'output55667.avi', fourcc, 80.0, (800, 800))

for fi in range(8):
    # Path to your TIFF file
    file_path = ops['data_path'][0] + ops['tiff_list'][fi]
    
    with Image.open(file_path) as img:
        # Create a list to hold each frame as a numpy array, reading every other frame
        frames = [np.array(img.seek(i) or img) for i in range(0, img.n_frames, 2)]  # Step is 2
    
    # Convert the list of frames to a 3D numpy array
    imgs = np.stack(frames, axis=2)# imgs is now a 3D array where imgs[:,:,i] is the i-th frame of the TIFF file


    
    # Define the codec and create VideoWriter object
    for i in range(22,imgs.shape[2]):
        fig, ax = plt.subplots()
        im = ax.imshow(np.mean(imgs[:, :, i-22:i],axis=2), vmin=0, vmax=55, cmap='gray')  # Adjust vmin, vmax, and cmap as needed
        plt.axis('off')  # Hide axis

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)  # Close the figure to free memory

        # Convert buffer to OpenCV image
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        
        # Resize frame to match video size, if necessary
        frame = cv2.resize(frame, (800, 800))
        
        # Write frame to video
        video.write(frame)
    
    # # Write each frame to the video
    # for i in range(20,scaled_imgs.shape[2]):
    #     # Calculate the smoothed frame
    #     start_index = max(i - 20, 0)
    #     frame = np.mean(scaled_imgs[:, :, start_index:i+1], axis=2)
        
    #     # Convert frame to uint8
    #     frame = frame.astype(np.uint8)
        
    #     # Ensure the frame is 3 channels if necessary
    #     if len(frame.shape) < 3:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
    #     out.write(frame)

# Release everything when the job is finished
#out.release()
video.release()

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

# Your data
# imgs = np.random.randint(0, 256, (800, 800, 117), dtype=np.int32)  # Example initialization

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(folder + 'output55.avi', fourcc, 20.0, (800, 800))

for i in range(22,imgs.shape[2]):
    fig, ax = plt.subplots()
    im = ax.imshow(np.mean(imgs[:, :, i-22:i],axis=2), vmin=0, vmax=25, cmap='gray')  # Adjust vmin, vmax, and cmap as needed
    plt.axis('off')  # Hide axis

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Convert buffer to OpenCV image
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(img_array, 1)
    
    # Resize frame to match video size, if necessary
    frame = cv2.resize(frame, (800, 800))
    
    # Write frame to video
    video.write(frame)

# Release the video writer
video.release()
