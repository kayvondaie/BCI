import numpy as np
import matplotlib.pyplot as plt

def causal_connectivity(data):
    """
    Calculate causal connectivity given a folder path.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the data.

    Returns:
    --------
    wcc : numpy.ndarray
        Weight matrix showing causal connectivity.
    stimDist : numpy.ndarray
        Stimulation distance matrix.
    amp : numpy.ndarray
        Amplitude differences calculated for photostimulation experiments.
    """
    
    # Initialize variables
    favg_raw = data['photostim']['favg_raw']
    favg = np.zeros(favg_raw.shape)
    stimDist = data['photostim']['stimDist']
    #bl = np.percentile(Ftrace, 50, axis=1)
    N = stimDist.shape[0]
    
    # Process photostimulation data
    for i in range(N):
        favg[:, i] = (favg_raw[:, i] - np.nanmean(favg_raw[0:3, i]))/np.nanmean(favg_raw[0:3, i])
    
    amp = np.nanmean(favg[11:15, :, :], axis=0) - np.nanmean(favg[0:4, :, :], axis=0)
    
    # Compute weighted causal connectivity (wcc)
    wcc = np.zeros((N, N))
    pairwiseDist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            pairwiseDist[j,i] = np.sqrt((data['photostim']['centroidX'][j] - data['photostim']['centroidX'][i])**2 + (data['photostim']['centroidY'][j] - data['photostim']['centroidY'][i])**2)
            ind = np.where((stimDist[j, :] < 10) & (stimDist[i, :] > 15))[0]
            if len(ind) > 0:
                wcc[j, i] = np.nanmean(amp[i, ind])
    
    # Plot causal connectivity heatmap
    ind = np.where(data['iscell'][:, 0] == 1)[0]
    plt.imshow(wcc[np.ix_(ind, ind)], vmin=-1, vmax=1, cmap='seismic')
    plt.xlabel('Post-synaptic')
    plt.ylabel('Pre-synaptic')
    plt.title(data['mouse'] + '  ' + data['session'])
    plt.colorbar()
    plt.show()
    
    return wcc, stimDist, amp, pairwiseDist, favg
