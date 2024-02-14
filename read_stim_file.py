# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:35:39 2023

@author: scanimage
"""

import numpy as np

# Read data from file
filename = folder + ops['tiff_list'][0][0:-4] + r'.stim'
hFile = open(filename, 'rb')  # Use 'rb' for reading binary file
phtstimdata = np.fromfile(hFile, dtype=np.float32)
hFile.close()

# Sanity check for file size
datarecordsize = 3
lgth = len(phtstimdata)
if lgth % datarecordsize != 0:
    print('Unexpected size of photostim log file')
    lgth = (lgth // datarecordsize) * datarecordsize
    phtstimdata = phtstimdata[:lgth]

# Reshape the data
phtstimdata = np.reshape(phtstimdata, (lgth // datarecordsize, datarecordsize))

# Extract x, y, and beam power
out = {}
out['X'] = phtstimdata[:, 0]
out['Y'] = phtstimdata[:, 1]
out['Beam'] = phtstimdata[:, 2]