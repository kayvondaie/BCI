import matplotlib.pyplot as plt
import numpy as np

stat_axons = np.load(folder + r'/suite2p_ch1/plane0/stat.npy', allow_pickle=True)
ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
img = ops['meanImg_chan2']

# Create an empty RGBA image with the same shape as the original image
rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)

# Set the grayscale image to the RGB channels of the RGBA image
normalized_img = img / img.max()  # Normalizing the image to [0, 1] range
rgba_img[..., 0] = normalized_img
rgba_img[..., 1] = normalized_img
rgba_img[..., 2] = normalized_img
rgba_img[..., 3] = 1.0  # Fully opaque initially

# Create an overlay mask for the ROIs
overlay = np.zeros_like(rgba_img)

# Loop through each ROI and fill the overlay
for roi in stat_axons[16:17]:
    ypix = roi['ypix']  # Y-coordinates for the current ROI
    xpix = roi['xpix']  # X-coordinates for the current ROI
    overlay[ypix, xpix, 1] = 1  # Set the green channel to 1 for ROI pixels
    overlay[ypix, xpix, 3] = 0.5  # Set the alpha channel to 0.5 for ROI pixels

# Display the grayscale image
plt.imshow(img, cmap='gray', vmin=0, vmax=20)

# Overlay the RGBA image
plt.imshow(overlay, alpha=1)

# Show the plot with the overlaid mask
plt.axis('off')  # Hide the axes
plt.show()


