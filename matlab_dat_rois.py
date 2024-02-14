# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 08:47:24 2023

@author: kayvon.daie
"""

import scipy.io
import h5py
folder = '//allen/aind/scratch/2p-working-group/roi-annotations/BCI19/110821/'
file = 'session_110821_analyzed_dat_small__slm12110821.mat'
filename = folder+file
#data = scipy.io.loadmat(folder+file)


with h5py.File(filename, 'r') as file:
    # The file behaves like a Python dictionary
    keys = list(file.keys())
    print(keys)
    
    # To get data from a specific key (equivalent to a variable in the .mat file):
    IM = file['IM'][()]
#%%

with h5py.File(filename, 'r') as hdf5_file:
    # Get the object references from the 'pixelList' attribute of the 'roi' dataset
    pixelList_refs = hdf5_file['roi']['pixelList'][()]
    
    # Get the object references from the 'centroid' attribute of the 'roi' dataset
    centroid_refs = hdf5_file['roi']['centroid'][()]

    roi_pixelLists = []
    for ref in pixelList_refs:
        # Dereference the object reference
        data_pointed_by_ref = hdf5_file[ref[0]]
        
        # Convert to a numpy array and append to the list
        roi_pixelLists.append(data_pointed_by_ref[()])

    roi_centroids = []
    for ref in centroid_refs:
        # Dereference the object reference
        data_pointed_by_ref = hdf5_file[ref[0]]
        
        # Convert to a numpy array and append to the list
        roi_centroids.append(data_pointed_by_ref[()])

print(roi_pixelLists)
print(roi_centroids)
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
fig,ax=plt.subplots()
im = ax.imshow(IM,cmap = 'gray',vmin = 0,vmax = 50)
for i in range(len(roi_centroids)):
    if len(roi_centroids[i]) > 1:
        ax.plot(roi_centroids[i][1],roi_centroids[i][0],'ro',markerfacecolor = 'none')

plt.show()
