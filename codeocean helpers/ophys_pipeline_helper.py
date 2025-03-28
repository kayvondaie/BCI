import numpy as np
import h5py
import json
import sparse
import extract_scanimage_metadata

def run_all_preprocessing(file_dff, file_extraction, file_trial_locations, epoch_data, epoch_ind, tif_file):
    # Load Ftrace
    with h5py.File(file_dff, 'r') as f:
        Ftrace = f['data'][:]

    # Load trial data
    with open(file_trial_locations, 'r') as f:
        tif_data = json.load(f)

    # Get matching tiffs for the specified epoch
    epoch_key = list(epoch_data.keys())[epoch_ind]
    start_idx, end_idx = epoch_data[epoch_key]
    matching_tifs = [
        tif for tif, (start, end) in tif_data.items()
        if not (end < start_idx or start > end_idx)
    ]

    # Extract scanimage metadata and photostim sequence
    siHeader = extract_scanimage_metadata.extract_scanimage_metadata(tif_file)
   
    # Extract spatial masks from sparse HDF5 format
    with h5py.File(file_extraction, 'r') as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]
    pixelmasks = sparse.COO(coords, data, shape).todense()

    # Convert to stat dict
    stat = {}
    for ci in range(pixelmasks.shape[0]):
        ypix, xpix = np.nonzero(pixelmasks[ci])
        stat[ci] = {'ypix': ypix, 'xpix': xpix}

    # Extract full Ftrace from start to end of epoch (used for stimDist_single_cell)
    epoch_start, _ = tif_data[matching_tifs[0]]
    _, epoch_end = tif_data[matching_tifs[-1]]
    F = Ftrace[:, epoch_start:epoch_end] * 20

    # Create ops dictionary (frames per file only, used later)
    ops = {'frames_per_file': [tif_data[tif][1] - tif_data[tif][0] + 1 for tif in matching_tifs]}

    return F, stat, ops, siHeader
