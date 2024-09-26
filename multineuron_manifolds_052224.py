import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

folder = '//allen/aind/scratch/BCI/2p-raw/BCI82/052224/'

def load_integration_rois_files(folder):
    # List to store the NumPy arrays and their file names
    numpy_arrays_with_filenames = []

    # Function to load a CSV file and handle inconsistent rows
    def load_csv_with_inconsistent_rows(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded DataFrame from {file_path} with shape {df.shape}")
            print("DataFrame head:")
            print(df.head())  # Print first few rows for debugging
            return df
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    # Walk through the directory and load the files
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check if the file is a .csv file and contains "IntegrationRois" in its name
            if file.endswith('.csv') and "IntegrationRois" in file:
                # Construct the full path of the file
                file_path = os.path.join(root, file)
                df = load_csv_with_inconsistent_rows(file_path)
                if df is not None:
                    # Convert the DataFrame to a NumPy array
                    array = df.to_numpy()
                    print(f"Array from {file_path} with shape {array.shape}")
                    print("Array head:")
                    print(array[:5])  # Print first few rows of the array for debugging
                    numpy_arrays_with_filenames.append((file, array))
                    print(f"Loaded {file_path} as NumPy array with shape {array.shape}")
                else:
                    print(f"Failed to load {file_path}")
    
    return numpy_arrays_with_filenames

def find_newest_vectors_mat_file(folder):
    newest_file = None
    latest_time = 0
    
    # Walk through the directory and find the newest .mat file with "vectors" in its name
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mat') and "vectors" in file:
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    newest_file = file_path

    if newest_file:
        print(f"Newest .mat file with 'vectors' in its name: {newest_file}")
        # Load the .mat file
        mat_data = loadmat(newest_file)
        return newest_file, mat_data
    else:
        print("No .mat file with 'vectors' in its name found.")
        return None, None

# Code that calls the functions and processes the files
if __name__ == "__main__":
    # Specify the directory
    

    # Load the CSV files
    integration_rois_arrays_with_filenames = load_integration_rois_files(folder)

    # Find and load the newest .mat file
    newest_mat_file, mat_data = find_newest_vectors_mat_file(folder)

    # Example: Accessing the first array and its file name
    if integration_rois_arrays_with_filenames:
        first_file_name, first_array = integration_rois_arrays_with_filenames[0]
        print("First file name:", first_file_name)
        print("First NumPy array:\n", first_array)

    # Example: Accessing data from the .mat file
    if mat_data is not None:
        print("Contents of the newest .mat file:")
        for key in mat_data:
            print(key)

    print("\n".join([file_name for file_name, _ in integration_rois_arrays_with_filenames]))
    ind = input('pick indices of bases for openloop then closedloop')
    ind = np.fromstring(ind[1:-1], sep=',')
#%%
def find_newest_threshold_mat_file(folder):
    newest_file = None
    latest_time = 0
    
    # Walk through the directory and find the newest .mat file with "threshold" in its name
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mat') and "threshold" in file:
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    newest_file = file_path

    if newest_file:
        print(f"Newest .mat file with 'threshold' in its name: {newest_file}")
        # Load the .mat file
        mat_data = loadmat(newest_file)
        return newest_file, mat_data
    else:
        print("No .mat file with 'threshold' in its name found.")
        return None, None

newest_threshold_file, threshold_data = find_newest_threshold_mat_file(folder)

if threshold_data is not None:
    print("Contents of the newest threshold .mat file:")
    for key in threshold_data:
        print(key)

#%%
import numpy as np
import matplotlib.pyplot as plt
import plotting_functions as pf

vector = np.asarray(mat_data['vectors'])
plus_ind = np.where(vector>0)[0]
v_plus = vector*0;
v_plus[plus_ind] = vector[plus_ind]
v_minus = vector*0;
minus_ind = np.where(vector<0)[0]
v_minus[minus_ind] = vector[minus_ind]

thresholds = threshold_data['manifold_thr']

proj = []
projp = []
projn = []
time = []
for i in range(len(ind)):
    I = int(ind[i])
    _, roi_data = integration_rois_arrays_with_filenames[I]
    roi_data = np.nan_to_num(roi_data)    
    proj.append(np.dot(roi_data[:, 2:], vector))
    projp.append(np.dot(roi_data[:, 2:], v_plus))
    projn.append(np.dot(roi_data[:, 2:], v_minus))
    time.append(roi_data[:,0]/60)
    if i > 0:
        time[i] = time[i] + time[i-1][-1]

plt.plot(np.concatenate(time),np.concatenate(proj),'k',linewidth=.3)
plt.plot(time[0],proj[0],'r',linewidth=.3)
plt.plot((0,time[-1][-1]),(thresholds[0][0],thresholds[0][0]))
plt.plot((0,time[-1][-1]),(thresholds[0][1],thresholds[0][1]))
plt.xlabel('Time (min.)')
plt.ylabel('BCI vector projection')
plt.xlim((0,3))
plt.show()

plt.subplot(211)
plt.plot(np.concatenate(time),np.concatenate(projp),'k',linewidth=.3)
plt.plot(time[0],projp[0],'r',linewidth=.3)


plt.subplot(212)
plt.plot(np.concatenate(projn),'k',linewidth=.3)
plt.plot(projn[0],'r',linewidth=.3)
plt.show()

plt.plot(projp[1],-projn[1],'r.',markersize=2)
plt.plot(projp[0],-projn[0],'k.',markersize=2)
plt.ylim((300,1100))
plt.xlim((300,1100))
plt.plot(plt.xlim(),plt.xlim())
plt.show()

pf.mean_bin_plot(projp[1],-projn[1],21,1,1,'r')
pf.mean_bin_plot(projp[0],-projn[0],21,1,1,'k')
#%%
bins = 20;

cco = []
for i in range(len(projp[0])-200):
    ind = np.arange(i, i+199)
    a = np.corrcoef(projp[0][ind].T,projn[0][ind].T)
    cco.append(a[0][1])
ccc = []
for i in range(len(projp[1])-200):
    ind = np.arange(i, i+199)
    a = np.corrcoef(projp[1][ind].T,projn[1][ind].T)
    ccc.append(a[0][1])

cc = []
cc.append(np.asarray(cco))
cc.append(np.asarray(ccc))

plt.plot(np.concatenate(cc),'k',linewidth=.3)
plt.plot(cc[0],'r',linewidth=.3)


#%%

shuff_indn = np.random.permutation(len(projn[0]))
shuff_indp = np.random.permutation(len(projp[0]))
shuff_proj = projp[0][shuff_indp]+projn[0][shuff_indn]
percentiles = np.percentile(shuff_proj, [70, 100])
plt.subplot(211)
plt.plot(np.concatenate(time),np.concatenate(proj),'k',linewidth=.3)
plt.plot((0,time[-1][-1]),(thresholds[0][0],thresholds[0][0]),'y:')
plt.plot((0,time[-1][-1]),(thresholds[0][1],thresholds[0][1]),'y:')
plt.plot(time[0],shuff_proj,'m',linewidth=.3)
plt.plot(time[0],projp[0]+projn[0],'k',linewidth=.3)
plt.plot(time[0],proj[0],'r',linewidth=.3)
plt.plot((0,time[0][-1]),(percentiles[0],percentiles[0]),'y')
plt.plot((0,time[0][-1]),(percentiles[1],percentiles[1]),'y')

plt.subplot(212)
plt.plot(projp[0],-projn[0],'k.',markersize=2)
plt.plot(projp[0][shuff_indp],-projn[0][shuff_indn],'m.',markersize=2)
#%%
plt.subplot(311)
plt.plot(time[0][0:-1],-projn[0][0:-1],linewidth = .3)
plt.plot(time[0][0:-1],projp[0][0:-1],linewidth = .3)
plt.subplot(312)
plt.plot(time[1][0:3000],-projn[1][0:3000],linewidth = .3)
plt.plot(time[1][0:3000],projp[1][0:3000],linewidth = .3)
plt.subplot(313)
plt.plot(time[1][15400:17400],-projn[1][15400:17400],linewidth = .3)
plt.plot(time[1][15400:17400],projp[1][15400:17400],linewidth = .3)

#%%
_, df_open = integration_rois_arrays_with_filenames[1]
_, df_closed= integration_rois_arrays_with_filenames[0]
v_orig = np.abs(vector)
plt.subplot(211)
plt.plot(time[0]*60,np.dot(df_open[:,2:],v_orig),'k',linewidth = .3)
plt.xlabel('Time (s)')
plt.ylabel('First SV')
plt.ylim(700,1900)
plt.subplot(212)
plt.plot(time[0]*60,np.dot(df_open[:,2:],vector),'k',linewidth = .3)
plt.ylim(-600,600)
plt.xlabel('Time (s)')
plt.ylabel('BCI vector')
plt.plot((0,time[0][-1]*60),(thresholds[0][0],thresholds[0][0]),'y:')
plt.plot((0,time[0][-1]*60),(thresholds[0][1],thresholds[0][1]),'y:')
plt.show()

fig = plt.figure(figsize=(6, 6))
a = df_open[:,2:]
plt.imshow(a[:,b].T,vmin = 0,vmax=500,aspect='auto')
# Set x-ticks at positions 0 and 600 (note that data might not have 600 columns)
plt.gca().set_xticks([0, 600])

# Set x-tick labels to '0' and '30'
plt.gca().set_xticklabels(['0', '30'])
plt.xlabel('Time (s)')
plt.ylabel('Neuron #')
plt.title('Open loop imaging')

plt.show()
v = np.concatenate(v_orig);
plt.subplot(121)
plt.plot(v_orig[b],'k')
plt.subplot(122)
plt.plot(vector[b],'k')
#%%
folder_props = folder_props_fun.folder_props_fun(folder)
bases = folder_props['bases']
base = bases[2]
siFiles = folder_props['siFiles']
files = os.listdir(folder)
good = np.zeros([1,np.shape(files)[0]])
for fi in range(0,np.shape(files)[0]):
    str = files[fi]
    a = str.find('.tif')
    if a > -1:
        #b = str.find('_')
        #b2 = str.find('_',b+1)
        #b = max([b,b2]);
        b = max([i for i, char in enumerate(str) if char == '_'])
        b = str[0:b]
        if b == base:
            good[0][fi] = 1
#        if b == base2:
#            good[0][fi] = 1

good = np.where(good == 1)
good = good[1]

tiff_files = [files[i] for i in good]
from PIL import Image
import os

# Function to count frames in a TIFF file
def count_frames(tiff_path):
    with Image.open(tiff_path) as img:
        frame_count = 0
        while True:
            try:
                img.seek(frame_count)
                frame_count += 1
            except EOFError:
                break
    return frame_count

# Iterate through each TIFF file and count the frames
num_frames = []
for i in range(len(tiff_files)):
    num_frames.append(count_frames(folder + tiff_files[i]))    
#%%
F = np.full((1000,df_closed.shape[1]-2,len(num_frames)),np.nan)
Fp = np.full((1000,2,len(num_frames)),np.nan)
first = 0
for ti in range(len(num_frames)):
    ind = np.arange(first-40,first+num_frames[ti]-1)
    a = df_closed[ind,2:]
    F[0:len(a),:,ti] = a
    a = projp[1][ind]
    Fp[0:len(ind),0,ti] = a.reshape(len(ind),)
    a = projn[1][ind]
    Fp[0:len(ind),1,ti] = a.reshape(len(ind),)
    first = ind[-1]+1

a = np.nanmean(Fp,axis=2)    
plt.plot(-a[:,1])
plt.plot(a[:,0])

