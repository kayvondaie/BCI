import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
def reward_triggered_average(data):
    """
    Compute the reward-triggered average for the given dataset.

    Parameters:
    -----------
    data : dict
        Dictionary containing relevant data fields including:
        - 'dt_si': Time step size
        - 'step_time': Steps times
        - 'trial_start': Trial start times
        - 'reward_time': Reward times
        - 'F': Fluorescence data
        - 'SI_start_times': Start times of the SI

    Returns:
    --------
    fr : numpy.ndarray
        Reward-triggered average fluorescence activity.
    """
    # Initialize variables
    dt_si = data['dt_si']
    t = np.arange(0, dt_si * (data['F'].shape[0]), dt_si)
    t = t - 2;
    trial_strt = np.zeros_like(t)
    rew = np.zeros_like(t)
    steps = data['step_time']
    strt = data['trial_start']
    rewT = data['reward_time']
    F = data['F']
    vel = np.zeros((F.shape[0], F.shape[2]))
    offset = int(np.round(data['SI_start_times'][0] / dt_si)[0])
    
    # # Calculate velocity matrix
    # for i in range(len(steps)):
    #     if np.isnan(F[-1, 0, i]):
    #         l = np.where(np.isnan(F[40:, 0, i]) == 1)[0][0] + 39
    #     else:
    #         l = F.shape[0]
    #     v = np.zeros(l)
    #     for si in range(len(steps[i])):
    #         ind = np.where(t > steps[i][si])[0][0]
    #         v[ind] = 1
    #     vel[0:l, i] = v
    
    # Compute trial start
    # for i in range(len(strt)):
    #     ind = np.where(t > strt[i])[0][0]
    #     trial_strt[ind] = 1

    # Compute reward matrix
    rew = np.zeros((F.shape[0], F.shape[2]))
    for i in range(len(rewT)):
        if len(rewT[i]) > 0:
            if rewT[i] < t[-1]:
                ind = np.where(t > rewT[i])[0][0]
                rew[ind, i] = 1


    # Compute positional data
    pos = np.zeros_like(vel)
    for ti in range(pos.shape[1]):
        for i in range(1, pos.shape[0]):  # Start from 1 to avoid accessing -1
            pos[i, ti] = pos[i - 1, ti]
            if vel[i, ti] == 1:
                pos[i, ti] = pos[i - 1, ti] + 1

    # Compute reward-triggered fluorescence
    frew = np.full((240, F.shape[1], F.shape[2]), np.nan)
    for i in range(F.shape[2]):
        ind = np.where(rew[:, i] == 1)[0]
        if len(ind) == 1:  # Only proceed if there is exactly one index
            j = ind[0]
            start = max(j - 60, 0)
            end = min(j + 180, F.shape[0])
            frew[:(end - start), :, i] = F[start:end, :, i]

    # Calculate reward-triggered average
    fr = np.nanmean(frew, axis=2)
    fr_mean = np.nanmean(fr[0:40, :], axis=0)  # Baseline correction
    fr = fr - fr_mean[np.newaxis, :]

    return fr,frew



def trial_start_response(data):
    f = data['F']
    f = np.nanmean(f,axis = 2)
    N = f.shape[1]
    for i in range(N):
        bl = np.nanmean(f[0:19,i])
        f[:,i] = f[:,i] - bl    
    return f


def roi_show(folder,roinums):

    
    stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    img = ops['meanImg']
    
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
    for roi in stat[roinums]:
        ypix = roi['ypix']  # Y-coordinates for the current ROI
        xpix = roi['xpix']  # X-coordinates for the current ROI
        overlay[ypix, xpix, 0] = 1  # Set the green channel to 1 for ROI pixels
        overlay[ypix, xpix, 3] = 0.5  # Set the alpha channel to 0.5 for ROI pixels
    
    # Display the grayscale image
    plt.imshow(img/8, cmap='gray', vmin=0, vmax=10)
    
    # Overlay the RGBA image
    plt.imshow(overlay, alpha=1)
    
    # Show the plot with the overlaid mask
    plt.axis('off')  # Hide the axes
    plt.show()
    

def roi_show_circles(folder, data, roinums, show_numbers=False):
    stat = np.load(folder + r'/suite2p_BCI/plane0/stat.npy', allow_pickle=True)
    ops = np.load(folder + r'/suite2p_BCI/plane0/ops.npy', allow_pickle=True).tolist()
    #fig, ax = plt.subplots()
    ax = plt.gca()
    im = ax.imshow(ops['meanImg'], cmap='gray', vmin=0, vmax=100)
    
    for i in range(len(roinums)):
        x = data['centroidX'][roinums[i]]
        y = data['centroidY'][roinums[i]]
        ax.plot(x, y, 'ro', markerfacecolor='none', markersize=5)
        if show_numbers:
            # Add the ROI number next to the circle
            ax.text(x + 8, y + 11, str(roinums[i]), color='r', fontsize=10)
    
    plt.axis('off')
    plt.show()

    
