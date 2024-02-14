import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

  
    X = np.mean(x, axis=1)
    Y = np.mean(y, axis=1)
    stdEr = np.std(y, axis=1) / np.sqrt(row)

    if pltt == 1:
        
        plt.errorbar(X, Y, stdEr, marker='o', markersize=5,
                     color=color, markerfacecolor=color)


    C = c
    P = p

    plt.title('P = ' + str(P))
    return X, Y

def tif_display(file_path,strt,skp):
   
    with Image.open(file_path) as img:

        frames = [np.array(img.seek(i) or img) for i in range(strt, img.n_frames, skp)]  # Step is 2        
        # Convert the list of frames to a 3D numpy array
    imgs = np.stack(frames, axis=2)# imgs is now a 3D array where imgs[:,:,i] is the i-th frame of the TIFF file
    return imgs