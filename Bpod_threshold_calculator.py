import os
import re
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.io
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
# Function to get TIFF and HDF5 files in the folder
def folder_props_fun(folder):
    # Ensure folder path ends with a slash
    if not folder.endswith(os.sep):
        folder = folder + os.sep

    # Get the list of all files in the directory
    files = os.listdir(folder)

    # Filter TIFF and HDF5 files based on their extensions
    siFiles = [f for f in files if re.search(r'\.tif$', f)]
    wsFiles = [f for f in files if re.search(r'\.h5$', f)]

    # Store the results in a dictionary similar to a struct in MATLAB
    folder_props = {
        'siFiles': siFiles,
        'wsFiles': wsFiles,
        'folder': folder
    }
    
    return folder_props

# Variables
mouse = 'BCI85'
date = '082724'
base = 'neuron9'
cn = 8
num_trials = 25

# Folder and base paths
folder = os.path.join(r'\\allen\aind\scratch\BCI\2p-raw', mouse, date)
folder_props = folder_props_fun(folder)  # Get folder properties

# Extract base names from the TIFF files
bases = [x[:x.rfind('_')] for x in folder_props['siFiles']]
siFiles = [x for x, b in zip(folder_props['siFiles'], bases) if b == base]
num_trials = len(siFiles)
siFiles = siFiles[:num_trials]
len_files = np.zeros(len(siFiles))

# Iterate through each TIFF file and count the frames
for i, si_file in enumerate(siFiles):
    start_time = time.time()
    tiff_file_path = os.path.join(folder, si_file)
    with tiff.TiffFile(tiff_file_path) as tiff_obj:
        frame_count = len(tiff_obj.pages)
    len_files[i] = frame_count
    print(f"Time for {si_file}: {time.time() - start_time} seconds")

# Reload folder properties if needed
folder_props = folder_props_fun(folder)

# Load CSV file
csv_file = os.path.join(folder, f"{base}_IntegrationRois_00001.csv")
csv_data = pd.read_csv(csv_file)
roi = csv_data.to_numpy()

# Load .mat file
threshold_data = scipy.io.loadmat(os.path.join(folder, f"{base}_threshold_6.mat"))

# Compute cumulative sum of lengths
strt = np.cumsum(len_files)
#%%

import os
import pandas as pd
from datetime import datetime

# This part follows after the previously converted code and uses existing variables
# from the earlier sections (e.g., mouse, date, etc.)

# Set the folder path for bpod session data
bpod_folder = r'\\allen\aind\scratch\BCI\bpod_session_data\sessions\\'

# Get the directory listing and filter out irrelevant entries (like '.' and '..')
a = [f for f in os.listdir(bpod_folder) if os.path.isdir(os.path.join(bpod_folder, f))]

# Find indices of folders that match the specific date pattern
ind = [i for i, folder in enumerate(a) if folder[:8] == f"20{date[4:6]}{date[0:4]}"]

# Initialize lists for 'yn', 'file_time', and 'csv_files_per_ind'
yn = []
file_time = []
csv_files_per_ind = []

# Iterate over the found indices
for i in range(len(ind)):
    temp = os.path.join(bpod_folder, a[ind[i]])
    file_list = os.listdir(temp)
    
    # Find .csv files in the directory
    csv_files = [f for f in file_list if f.endswith('.csv')]
    
    if csv_files:
        csv_file = os.path.join(temp, csv_files[0])
        
        # Get file modification time
        file_info = os.stat(csv_file)
        file_time.append(datetime.fromtimestamp(file_info.st_mtime).timestamp())

        # Read the CSV file, skipping the first 6 rows
        df = pd.read_csv(csv_file, delimiter=';', skiprows=6)

        # Clean the data, remove unwanted rows
        df = df[df['TYPE'].notna()]  # Keep rows where 'TYPE' is not NaN
        df = df[df['TYPE'] != '|']  # Remove rows where 'TYPE' is '|'
        df = df[df['MSG'].notna()]  # Keep rows where 'MSG' is not NaN
        df = df[df['MSG'].str.strip() != '']  # Remove rows where 'MSG' is empty
        df = df[df['MSG'] != '|']  # Remove rows where 'MSG' is '|'

        # Reset the index after deletion
        df = df.reset_index(drop=True)

        # Find the row where 'SUBJECT-NAME' appears in the 'MSG' column and 'INFO' in the 'TYPE' column
        subj_ind = df[(df['TYPE'] == 'INFO') & (df['MSG'] == 'SUBJECT-NAME')].index

        if len(subj_ind) > 0:
            # Use the '+INFO' column to extract the subject name
            subj = df.loc[subj_ind[0], '+INFO']
            subj = subj[2:subj.find(',')-1]  # Extract the subject name, similar to what you were doing before

        # Compare subject name with 'mouse' and append to 'yn'
        yn.append(1 if subj == mouse else 0)

        # Store csv_files for this index
        csv_files_per_ind.append(csv_files)
    else:
        # If no CSV files, append None to keep lists aligned
        csv_files_per_ind.append([])

# Multiply 'yn' by the corresponding 'file_time'
yn = [yn_val * time_val for yn_val, time_val in zip(yn, file_time)]

# Check if the 'yn' list is non-empty before proceeding
if yn:
    # Get the index of the maximum value in 'yn'
    max_ind = yn.index(max(yn))

    # Ensure that csv_files_per_ind[max_ind] is not empty and the index is within bounds
    if max_ind < len(csv_files_per_ind) and csv_files_per_ind[max_ind]:
        filename = os.path.join(bpod_folder, a[ind[max_ind]], csv_files_per_ind[max_ind][0])
        print(f"Selected file: {filename}")
    else:
        print(f"Error: max_ind {max_ind} exceeds the length of csv_files {len(csv_files_per_ind)} or no csv file available.")
else:
    filename = None
    print("Error: 'yn' list is empty, no file selected.")
#%%
# Read the CSV file, skipping the first 6 rows (headers) and using ';' as the delimiter
raw_data = pd.read_csv(filename, delimiter=';', skiprows=6, header=None)

# Adjust the start row based on where the data starts
start_row = 1  # In Python, indexing starts at 0, so this will be row 2 in MATLAB
data = raw_data.iloc[start_row:, :]

# Find the rows where column 5 is 'Reward_L' and column 1 is 'TRANSITION'
reward_l_rows = (data[4] == 'Reward_L') & (data[0] == 'TRANSITION')

# Filter the data based on the reward_l_rows
filtered_data = data[reward_l_rows]

# Find indices where the condition is met
reward_l_indices = data.index[reward_l_rows].tolist()

# Find the indices where column 1 is 'TRIAL'
trial_start_indices = data.index[data[0] == 'TRIAL'].tolist()

# Initialize 'rew' and 'rt' arrays with NaN
rew = np.full(len(trial_start_indices), np.nan)
rt = np.full(len(trial_start_indices), np.nan)

# Loop through trial start indices and calculate 'rew' and 'rt'
for i in range(len(trial_start_indices) - 1):
    # Get the indices where reward occurred between two trial starts
    ind = [idx for idx in reward_l_indices if trial_start_indices[i] < idx < trial_start_indices[i + 1]]
    
    # Count rewards for this trial
    rew[i] = len(ind)
    
    # If rewards were found, store the reaction time (assuming it's in the third column, as in MATLAB)
    if ind:
        rt[i] = data.iloc[ind[0], 2]  # Adjust based on the actual column where the time is stored

# Handle the last trial (after the last trial start index)
ind_last = [idx for idx in reward_l_indices if idx > trial_start_indices[-1]]
if ind_last:
    rew[-1] = len(ind_last)

# Now 'rew' contains the reward counts and 'rt' contains the reaction times
#%%
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
# Define frm_ind (assuming ROI is already loaded)
frm_ind = np.arange(1, int(np.max(roi[:, 1])) + 1)

# Interpolate roi data
interp_func = interp1d(roi[:, 1], roi, axis=0, kind='linear', fill_value='extrapolate')
roi_interp = interp_func(frm_ind)

# Define the function based on BCI_threshold
fun = lambda x: (x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)) * 3.3

# Initialize variables
strt = 0  # Python is 0-indexed, adjust accordingly
dt_si = np.median(np.diff(roi[:, 0]))
fcn = np.empty((250, len(len_files) - 1))
FCN = np.empty((250, len(len_files) - 1))
t_si = np.empty((250, len(len_files) - 1))
avg = np.empty(len(len_files) - 1)

# Initialize strt at the start of the loop
strt = 0  # Python uses 0-based indexing, corresponding to MATLAB's strt = 1

# Initialize strts array to hold values
strts = np.empty(len(len_files) - 1, dtype=int)  # Initialize with the correct length

# Flatten BCI_threshold to ensure it's 1D
BCI_threshold = threshold_data['BCI_threshold']
BCI_threshold = BCI_threshold.flatten()

# Define the function based on the flattened BCI_threshold
fun = lambda x: np.minimum((x > BCI_threshold[0]) * (x / np.diff(BCI_threshold)[0]) * 3.3, 3.3)


# Loop through the trials
for i in range(len(len_files) - 1):
    strts[i] = strt  # Literal translation of strts(i) = strt
    ind = np.arange(strt, strt + len_files[i], dtype=int)  # Ensure ind is an array of integers

    # Extract and process roi_interp data for fcn and t_si
    a = roi_interp[ind.astype(int), cn + 2]

    # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
    a_padded = np.concatenate([a, np.full(300, np.nan)])
    fcn[:, i] = a_padded[:250]
    FCN[:, i] = a_padded[:250]

    # Repeat for t_si (first column of roi_interp)
    a = roi_interp[ind.astype(int), 0]
    a = a - a[0]  # Shift time values

    # Pad with 300 NaNs as in the MATLAB code, and then select the first 250 elements
    a_padded = np.concatenate([a, np.full(300, np.nan)])
    t_si[:, i] = a_padded[:250]

    strt = strt + len_files[i]  # Update strt for the next trial

    # Determine the stopping point
    if rew[i]:
        stp = np.max(np.where(t_si[:, i] < rt[i])[0])
    else:
        stp = t_si.shape[0]

    # Calculate average for this trial
    avg[i] = np.nanmean(fun(fcn[:stp, i]))

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the decaying exponential model
def exp_model(x, a, l, c):
    return a * np.exp(-x / l) + c

# Clean the data by removing rows where rt or avg are NaN
ind = np.where(~np.isnan(rt[:len(avg)]) & ~np.isnan(avg))[0]

# Prepare the x-axis values for plotting the fit
x = np.linspace(0, np.max(avg), 1000)

# Fit the exponential model to the data
popt, pcov = curve_fit(exp_model, avg[ind], rt[ind], p0=(1, 1, 1))  # Initial guess (1, 1, 1)

# Extract the fit parameters
A, l, c = popt

# Create the fitted values using the model
fit = exp_model(x, A, l, c)

# Plot the data points and the fitted curve
plt.scatter(avg[ind], rt[ind], color='k', facecolor='w', label='Data')
plt.plot(x, fit, label=f'Fit: A={A:.2f}, l={l:.2f}, c={c:.2f}')
plt.xlabel('avg "voltage"')
plt.ylabel('Time to reward (s)')

# Calculate and plot the minimum activity where fit > 10
min_activity = x[np.max(np.where(fit > 10))]
plt.axvline(min_activity, color='k', linestyle=':')

# Annotate the plot with the minimum activity value
plt.text(min_activity + 0.1, 14, f'V = {round(min_activity, 2)}')

# Add final touches and title
plt.title(folder)
plt.legend()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Initialize new_thr and new_slp
pc = np.linspace(0, 90, 100)
new_thr = np.percentile(roi_interp[:, cn + 2], pc)  # Similar to MATLAB's prctile
new_thr = BCI_threshold[0] * np.linspace(0.5, 1.5, 100)
new_slp = BCI_threshold[1] * np.linspace(0.5, 2, 100)

# Initialize avg_new
avg_new = np.zeros((122, 100, 100))
new_rt = np.zeros((122, 100, 100))

# Loop over thresholds and slopes
for ti in range(len(new_thr)):
    for si in range(len(new_slp)):
        # Define the function to compute activity
        fun = lambda x: np.minimum((x > new_thr[ti]) * (x / (new_slp[si] - new_thr[ti])) * 3.3, 3.3)

        
        strt = 0  # Initialize start index
        for i in range(122):  # 122 iterations, based on your final size of avg_new
            strts = strt
            ind = np.arange(strt, strt + len_files[i], dtype=int)  # Indexes as in MATLAB
            
            # Get roi_interp data for fcn and t_si
            a = roi_interp[ind, cn + 2]
            a_padded = np.concatenate([a, np.full(300, np.nan)])  # Pad with 300 NaNs
            fcn = a_padded[:250]
            
            a = roi_interp[ind, 0] - roi_interp[ind, 0][0]  # Time-shifted
            a_padded = np.concatenate([a, np.full(300, np.nan)])  # Pad with 300 NaNs
            t_si = a_padded[:250]
            
            strt += len_files[i]  # Update strt for the next iteration
            
            # Determine stopping point
            if rew[i]:
                stp = np.max(np.where(t_si < rt[i])[0])
            else:
                stp = len(t_si)

            # Compute avg_new
            avg_new[i, ti, si] = np.nanmean(fun(fcn[:stp]))
            new_rt[i,ti,si] = exp_model(np.nanmean(fun(fcn[:stp])), A, l, c)

# Apply the model function to avg_new
#new_rt = model(avg_new)

# Compute hit rate
new_hit = np.mean(new_rt < 10, axis=0)

# Plotting the results
plt.figure()
plt.imshow(new_hit, aspect='auto', origin='upper')
plt.colorbar(label='Hit rate')

# Set Y ticks (Lower threshold)
y_inds = np.arange(0, 100, 20)
y_values = np.round(new_thr[::20])
plt.yticks(y_inds, y_values)

# Set X ticks (Upper threshold)
x_inds = np.arange(0, 100, 20)
x_values = np.round(new_slp[::20] * 100) / 100
plt.xticks(x_inds, x_values)

# Mark the current thresholds
currentx = np.min(np.where(new_slp >= BCI_threshold[1]))
currenty = np.min(np.where(new_thr >= BCI_threshold[0]))
plt.plot(currentx, currenty, 'wo', markersize=10)

# Labels
plt.xlabel('Upper threshold')
plt.ylabel('Lower threshold')
plt.title('Hit Rate Heatmap')
plt.show()
#%%
import matplotlib.pyplot as plt
BCI_threshold = threshold_data['BCI_threshold']
BCI_threshold = BCI_threshold.flatten()
t = np.arange(0, dt_si * FCN.shape[0], dt_si)
plt.figure(figsize=(6, 3))  # Adjust the width and height as needed
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set this to the desired font size
HIT = []
for ii in range(3):
# Increase the figure size to give more space for each subplot
    
    # Loop to create subplots and plot data
    strt = 3
    stp = 4
    if ii == 0:
        AA = BCI_threshold*0
        AA[0] = new_thr[currenty]
        AA[1] = new_slp[currentx]
        hit = avg_new[:,currenty,currentx] 
        color = 'b'
    elif ii == 1:
        AA = BCI_threshold*0
        AA[0] = new_thr[80]
        AA[1] = new_slp[currentx]
        hit = avg_new[:,80,currentx] 
        color = [.5,.5,1]
    elif ii == 2:
        AA = BCI_threshold*0
        AA[0] = new_thr[currenty]
        AA[1] = new_slp[99]
        hit = avg_new[:,currenty,99] 
        color = [.9,.9,1]
    HIT.append(hit)
    fun = lambda x: np.minimum((x > AA[0]) * (x / np.diff(AA)[0]) * 3.3, 3.3)
    
    

    for i in range(stp-strt):
        # First row of subplots
        
        clr = 'k'
        
        
        plt.subplot(2, 3, ii+1)
        plt.xlim(0,10)
        plt.plot(t,FCN[:, i+strt], color=clr, linewidth=1)
        plt.plot(plt.xlim(), [AA[0], AA[0]], color=color)  # 'k' for black line
        plt.plot(plt.xlim(), [AA[1], AA[1]], color=color)  # 'k' for black line
        plt.ylim(50, 1000)
        plt.tight_layout()
            
    
    
        # Remove y-axis labels for all but the first plot in the row
        if i > 0:
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # Second row of subplots
        plt.subplot(2, 3, ii+4)
        plt.xlim(0,10)
        plt.plot(t,fun(FCN[:, i+strt]), color = clr, linewidth=1)        
        plt.ylim(0, 4)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    
        # Remove y-axis labels for all but the first plot in the row
        if i > 0:
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('')
    
    # Optionally, you can adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust these values to change the spacing
    
plt.show()

plt.figure(figsize=(3, 3))  # Adjust the width and height as needed
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Set this to the desired font size

for i in range(3):
    if i == 0:
        color = 'b'
    elif i == 1:
        color = [.5,.5,1]
    elif i == 2:
        color = [.9,.9,1]
    y = HIT[i]
    x = i + np.random.randn(*y.shape) / 10
    plt.plot(x,y,'.',color=color)
plt.plot(plt.xlim(), [min_activity,min_activity], 'k:')  # 'k' for black line
plt.ylabel('cumulative activity')
plt.xticks([0,1,2], ['true', 'medium', 'hard'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)



































