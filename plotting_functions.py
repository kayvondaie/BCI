import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pearsonr

def mean_bin_plot(xx, yy, col, pltt, A, color):
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))
    if len(color) == 0:
        color = 'bbbbbbb'
    if pltt < 4:
        pltt = 1
        A = 1
        #color = 'k'
    if col is None:
        col = 5

   
    x = xx
    y = yy / A

    a = np.isnan(x)
    x = x[~a]
    y = y[~a]

    a = np.sort(x, axis=0)
    b = np.argsort(x, axis=0)

    c, p = np.corrcoef(x[b], y[b])
    c = np.diag(c)
    p = np.diag(p)

    x = x[b]
    y = y[b]

    row = len(x) // col
    length = row * col
    x = np.reshape(x[:length], (col, row))
    y = np.reshape(y[:length], (col, row))

  
    X = np.nanmean(x, axis=1)
    Y = np.nanmean(y, axis=1)
    stdEr = np.nanstd(y, axis=1) / np.sqrt(row)

    if pltt == 1:
        
        plt.errorbar(X, Y, stdEr, marker='o', markersize=5,
                     color=color, markerfacecolor=color)

    r, p = pearsonr(X, Y)
    #print('p-value = ' + str(p))
    

    #plt.title('P = ' + str(P))
    return X, Y, p

def tif_display(file_path,strt,skp):
   
    with Image.open(file_path) as img:

        frames = [np.array(img.seek(i) or img) for i in range(strt, img.n_frames, skp)]  # Step is 2        
        # Convert the list of frames to a 3D numpy array
    imgs = np.stack(frames, axis=2)# imgs is now a 3D array where imgs[:,:,i] is the i-th frame of the TIFF file
    return imgs

def show_rois(ops,stat,roi_index):
    import numpy as np
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
    for roi in stat[roi_index]:
        ypix = roi['ypix']  # Y-coordinates for the current ROI
        xpix = roi['xpix']  # X-coordinates for the current ROI
        overlay[ypix, xpix, 0] = 1  # Set the green channel to 1 for ROI pixels
        overlay[ypix, xpix, 3] = 0.5  # Set the alpha channel to 0.5 for ROI pixels
    
    # Display the grayscale image
    plt.imshow(img/8, cmap='gray', vmin=0,vmax=np.percentile(img,30))
    
    # Overlay the RGBA image
    plt.imshow(overlay, alpha=1)
    
    # Show the plot with the overlaid mask
    plt.axis('off')  # Hide the axes
    plt.show()




def show_rois_outline(ops, stat, roi_indices, roi_colors, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()  # get the current Axes if not provided
    import numpy as np
    from skimage.morphology import binary_erosion
    """
    Displays the mean image with outlines of selected ROIs overlaid in
    the specified colors.

    Parameters
    ----------
    ops : dict
        Dictionary containing Suite2p operations and outputs. Must at least
        have 'meanImg'.
    stat : list of dicts
        Each element in 'stat' is a dictionary describing an ROI, typically
        containing 'ypix' and 'xpix' among other keys.
    roi_indices : list of int
        List of ROI indices you want to display.
    roi_colors : list of tuples/list
        List of RGB colors (or RGBA) for each ROI index. Example: [(1,0,0), (0,1,0)].
        Must have the same length as roi_indices.
    """
    # Get the base image and normalize for display
    img = ops['meanImg']
    v30 = np.percentile(img, 30)
    normalized_img = img / np.max(img)

    # Display the base image
    #plt.figure(figsize=(6,6))
    plt.imshow(img/8, cmap='gray', vmin=0, vmax=v30)
    
    # Prepare an RGBA overlay (same size as image)
    overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
    
    # Loop through each ROI index and color
    for idx, color in zip(roi_indices, roi_colors):
        # Extract the y and x pixel indices for this ROI
        roi = stat[idx]
        ypix = roi['ypix']
        xpix = roi['xpix']

        # Create a boolean mask of this ROI
        roi_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        roi_mask[ypix, xpix] = True

        # Erode the mask and XOR it with the original to get just the boundary
        eroded_mask = binary_erosion(roi_mask)
        boundary_mask = roi_mask & (~eroded_mask)

        # Set the overlay color only at boundary pixels
        # color can be an RGB tuple (r, g, b) in [0,1], or RGBA (r, g, b, a)
        # For safety, convert to RGBA internally:
        if len(color) == 3:
            r, g, b = color
            a = 1.0
        elif len(color) == 4:
            r, g, b, a = color
        else:
            raise ValueError("roi_colors entries must be RGB or RGBA tuples.")

        overlay[boundary_mask, 0] = r
        overlay[boundary_mask, 1] = g
        overlay[boundary_mask, 2] = b
        overlay[boundary_mask, 3] = a

    # Overlay the ROI boundaries on top of the grayscale image
    plt.imshow(overlay, alpha=1.0)

    # Clean up plot
    plt.axis('off')
    

def fixed_bin_plot(xx, yy, col=None, pltt=1, A=1, color='b', bins=None):
    """
    Bins data into fixed-width bins and plots the mean and SEM of y in each bin.
    
    Parameters:
        xx (array-like): x-values
        yy (array-like): y-values
        col (int, optional): Number of bins to divide the x-range into. Ignored if `bins` is given.
        pltt (int): If 1, make a plot. If not, just return values.
        A (float): Divides y-values by A (e.g., for normalization).
        color (str): Color for plotting.
        bins (array-like, optional): Custom bin edges (overrides `col`).
    
    Returns:
        X (ndarray): Mean x in each bin
        Y (ndarray): Mean y in each bin
        p (float): p-value of correlation between x and y
    """
    xx = np.ravel(xx).astype(float)
    yy = np.ravel(yy).astype(float) / A
    print(type(A))

    # Remove NaNs
    mask = ~np.isnan(xx) & ~np.isnan(yy)
    x = xx[mask].astype(float)

    y = yy[mask]

    # Correlation
    if len(x) < 2:
        return np.array([]), np.array([]), np.nan

    c_mat = np.corrcoef(x, y)
    c = c_mat[0, 1]
    p = np.corrcoef(x, y)[0, 1]

    # Define bins
    if bins is not None:
        bins = np.asarray(bins)
    else:
        if col is None:
            col = 5
        bins = np.linspace(np.min(x), np.max(x), col + 1)

    bin_indices = np.digitize(x, bins) - 1  # subtract 1 to get 0-based index

    # Compute stats
    X = []
    Y = []
    stdEr = []
    for i in range(len(bins) - 1):
        idx = bin_indices == i
        if np.any(idx):
            X.append(np.mean(x[idx]))
            Y.append(np.mean(y[idx]))
            stdEr.append(np.std(y[idx]) / np.sqrt(np.sum(idx)))
        else:
            X.append(np.nan)
            Y.append(np.nan)
            stdEr.append(np.nan)

    X = np.array(X)
    Y = np.array(Y)
    stdEr = np.array(stdEr)

    if pltt == 1:
        plt.errorbar(X, Y, yerr=stdEr, marker='o', markersize=5,
                     color=color, markerfacecolor=color)

    return X, Y, p


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

def plot_bootstrap_fit(x, y, n_boot=1000, label=None, color='k', alpha_fill=0.2):
    """
    Plot a bootstrap regression fit with 95% confidence interval and return p-value on slope.

    Parameters:
        x, y       : arrays of data
        n_boot     : number of bootstrap samples
        flip_x     : whether to flip x (use -x)
        label      : optional label for the fit
        color      : line color
        alpha_fill : transparency for CI band
        ax         : optional matplotlib axis to plot on
        return_slope : if True, return (slope_median, slope_pval)

    Returns:
        ax                 : matplotlib axis with the plot
        (optional) tuple   : (median_slope, p-value for slope ≠ 0)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ind = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
    x = x[ind]
    y = y[ind]



    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_boot = np.zeros((n_boot, len(x_fit)))
    slope_boot = np.zeros(n_boot)

    for i in range(n_boot):
        xb, yb = resample(x, y)
        coef = np.polyfit(xb, yb, 1)
        slope_boot[i] = coef[0]
        y_boot[i] = np.polyval(coef, x_fit)

    y_median = np.median(y_boot, axis=0)
    y_lower = np.percentile(y_boot, 2.5, axis=0)
    y_upper = np.percentile(y_boot, 97.5, axis=0)

    # p-value for slope ≠ 0
    slope_median = np.median(slope_boot)
    pval = 2 * np.minimum(np.mean(slope_boot > 0), np.mean(slope_boot < 0))

  

    #ax.scatter(x, y, s=10, alpha=0.5, color=color)
    plt.plot(x_fit, y_median, color=color, label=label)
    plt.fill_between(x_fit, y_lower, y_upper, color=color, alpha=alpha_fill)
    return slope_median, pval
    
