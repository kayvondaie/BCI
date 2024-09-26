import scipy.io
import os
import numpy as np

numtrl = data['F'].shape[2]
BCI_thresholds = np.full((2, numtrl), np.nan)
siHeader = np.load(folder + r'/suite2p_BCI/plane0/siHeader.npy', allow_pickle=True).tolist()

# Determine the base for file names
if isinstance(siHeader['siBase'], str):
    base = siHeader['siBase']
else:
    base = siHeader['siBase'][0]

# Iterate over trials and attempt to load the corresponding threshold files
for i in range(numtrl):
    try:
        st = folder + base + r'_threshold_' + str(i+1) + r'.mat'
        
        # Check if the file exists before trying to load it
        if os.path.exists(st):
            threshold_data = scipy.io.loadmat(st)
            BCI_thresholds[:, i] = threshold_data['BCI_threshold'].flatten()
            
    except:
        pass  # Ignore any exceptions and continue with the next iteration
